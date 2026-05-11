import glob
import json
import logging
import os
import re
from typing import Union, Optional

import numpy as np
import torch
import wandb
import matplotlib
from torch import nn

from metrics.procrustes import expand_scores_to_individual_layers

matplotlib.use("Agg") # Avoid errors when running without UI
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from metrics.linearity_metric_manager import LinearityMetric
from utils.data_manager import DataManager
from utils.llama_model import LlamaExperimenter
from utils.resnet_model import ResNetExperimenter
import utils.util_functions as utils

logger = logging.getLogger(__name__)
debug_mode = logger.getEffectiveLevel() != logging.DEBUG


def _get_target_layers(model):
    if "resnet" in model.__class__.__name__.lower():
        pattern = r".*conv.*"
    elif "llama" in model.__class__.__name__.lower():
        pattern = r"model\.layers\..*\.self_attn"
    else:
        raise ValueError("Unsupported model type")

    layers = []
    names = []

    for name, module in model.named_modules():
        if re.match(pattern, name):
            layers.append(module)
            names.append(name)

    return layers, names


def _process_output(output, attention_mask=None):
    """
    Convert layer outputs to [B, D]
    """
    if isinstance(output, tuple):
        output = output[0]

    if output.dim() == 4:
        # CNN output [B,C,H,W] -> global avg pool -> [B,C]
        output = output.mean(dim=(2, 3))

    elif output.dim() == 3:
        # Transformer [B,S,H] -> token mean -> [B,H]
        # We want to avoid including padding in computation if possible
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(-1).to(output.dtype)  # [B, S, 1]
            output = (output * attention_mask).sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1)
        else:
            output = output.mean(dim=1)

    elif output.dim() == 2:
        pass

    else:
        raise ValueError(f"Unexpected shape {output.shape}")

    return output.detach()


def _collect_activations(model, x, layers, device):
    acts = []

    def hook_fn(module, inp, out):

        # If there is an attention mask in input, we pass it to processing function
        attention_mask = x["attention_mask"] if isinstance(x, dict) and "attention_mask" in x else None
        if "llama" in model.__class__.__name__.lower() and attention_mask is None:
            logger.warning("Failed to grab attention mask")

        acts.append(_process_output(out, attention_mask).to(device))

    hooks = [layer.register_forward_hook(hook_fn) for layer in layers]

    model.eval()
    with torch.no_grad():
        if isinstance(x, dict):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                model(**x)
        else:
            model(x)

    for h in hooks:
        h.remove()

    return acts


def _linear_cka_fast(X, Y):
    """
    Memory-efficient linear CKA.
    X: [N, Dx]
    Y: [N, Dy]
    """
    X = X.float()
    Y = Y.float()

    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    hsic = torch.norm(X.T @ Y, p="fro") ** 2
    norm_x = torch.norm(X.T @ X, p="fro")
    norm_y = torch.norm(Y.T @ Y, p="fro")

    return (hsic / (norm_x * norm_y + 1e-8)).item()


def cka_similarity_matrix(model_a, model_b, dataloader, device="cuda", tokenizer=None):
    """
    Computes a CKA similarity matrix between layers of two models.

    Args:
        model_a: First model (e.g. teacher)
        model_b: Second model (e.g. student)
        dataloader: DataLoader providing input batches
        device: device to run on
        tokenizer: Optional tokenizer for text data (only needed if models are LLaMA)
    Returns:
        np.ndarray of shape (m, n)
    """
    model_a.to(device).eval()
    model_b.to(device).eval()

    layers_a, names_a = _get_target_layers(model_a)
    layers_b, names_b = _get_target_layers(model_b)

    m, n = len(layers_a), len(layers_b)

    acts_a_all = [[] for _ in range(m)]
    acts_b_all = [[] for _ in range(n)]

    logger.info("Prepped for CKA computation, processing batches...")
    for batch_idx, x in enumerate(dataloader):
        if "resnet" in model_a.__class__.__name__.lower():
            x, _ = x
            x = x.to(device)

        else:
            x = tokenizer(
                x["text"],
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            x = {k: v.to(device) for k, v in x.items()}

        acts_a = _collect_activations(model_a, x, layers_a, device)
        acts_b = _collect_activations(model_b, x, layers_b, device)

        for i in range(m):
            acts_a_all[i].append(acts_a[i].cpu())

        for j in range(n):
            acts_b_all[j].append(acts_b[j].cpu())

        del acts_a, acts_b
        torch.cuda.empty_cache()

    logger.info("Batches processed. Calculating CKA similarity matrix...")

    # concatenate layer activations
    acts_a_all = [torch.cat(v, dim=0) for v in acts_a_all]
    acts_b_all = [torch.cat(v, dim=0) for v in acts_b_all]

    # -------- compute CKA --------
    matrix = np.zeros((m, n), dtype=np.float32)

    for i in range(m):
        Xa = acts_a_all[i]

        for j in range(n):
            Yb = acts_b_all[j]
            matrix[i, j] = _linear_cka_fast(Xa, Yb)

    logger.info("Calculated CKA similarity matrix.")

    return matrix, names_a, names_b

def visualize_cka_similarity_matrix(matrix, save_dir, teacher_layer_names, student_layer_names,
                                    linearity_scores, linearity: str = None):
    """Creates a magma heatmap of the cka similarity matrix using matplotlib.
    Shows row and column indexes to roughly identify layers. Similarity scores are listed in the cells.
    Stores the visualization in ./results with the given filename.
    Args:
        matrix: A numpy array of shape (m, n) containing the cka similarity values between the layers.
        save_dir: The directory to save the heatmap. Saved as "cka_similarity_heatmap.png" in the given directory.
        teacher_layer_names: Names of the teacher layers.
        student_layer_names: Names of the student layers.
        linearity_scores: A list of the linearity scores for each layer (length m).
        linearity: The name of the linearity score type. To override linearity score in relation experiments being read as 'all'
    """
    logger.info(f"Visualizing CKA similarity matrix with shape {matrix.shape} and saving to {save_dir}/cka_similarity_heatmap.png")
    linearity_score = save_dir.split("/")[3] if linearity is None else linearity
    kd_method = save_dir.split("/")[4]
    model_name = save_dir.split("/")[5]
    dataset = save_dir.split("/")[6]

    ticks = [] if "llama" in model_name else [0]
    x_labels = [] if "llama" in model_name else ['conv1']
    y_labels = [] if "llama" in model_name else ['conv1']
    for i, name in enumerate(teacher_layer_names):
        if name in linearity_scores:
            score = linearity_scores[name]
            ticks.append(i)
            y_labels.append(f"{name} ({score:.4f})")
            if name in student_layer_names:
                x_labels.append(name)


    plt.imshow(matrix, cmap='magma', vmin=0, vmax=1, origin='upper')
    plt.colorbar(label=f'CKA Similarity of {model_name} teacher and student layers')
    plt.xlabel('Student model')
    plt.ylabel('Teacher model')
    # Move x-axis to top
    plt.gca().xaxis.tick_top()
    plt.gca().xaxis.set_label_position('top')
    plt.xticks(ticks=ticks[:len(x_labels)], labels=x_labels, rotation=90, fontsize=8)
    plt.yticks(ticks=ticks, labels=y_labels, rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/cka_{model_name}_{linearity_score}_{kd_method}_{dataset}.png")
    plt.close()

def scatterplot_linearity_pruning_scores(linearity_scores: dict, pruning_ratios: dict, save_dir: str, linearity: str = None) -> None:
    """Creates a scatterplot of the linearity compression scores and pruning scores for each layer.
    Points are labeled with their layer index. X-axis will be linearity score, Y-axis will be pruning score.
    Args:
        linearity_scores: A dictionary mapping layer names to linear scores.
        pruning_ratios: A dictionary mapping layer names to pruning scores.
        save_dir: The directory to save the scatterplot. Saved as "linearity_pruning_scatterplot.png" in the given directory.
        linearity: The name of the linearity score type. To override linearity score in relation experiments being read as 'all'
    """
    layer_names = list(set(linearity_scores.keys()).intersection(set(pruning_ratios.keys())))
    logger.info(f"Computing scatterplot for {len(layer_names)} layers out of total {len(linearity_scores) + len(pruning_ratios)} layers.")
    linearity_values = [linearity_scores[name] for name in layer_names]
    pruning_values = [pruning_ratios[name] for name in layer_names] # Invert ratios to show the fraction of pruned weights
    linearity_score = save_dir.split("/")[3] if linearity is None else linearity
    prune_method = save_dir.split("/")[4]
    model_name = save_dir.split("/")[5]
    dataset = save_dir.split("/")[6]

    plt.figure(figsize=(10, 6))
    plt.scatter(linearity_values, pruning_values)

    for i, name in enumerate(layer_names):
        plt.annotate(name, (linearity_values[i], pruning_values[i]))

    plt.xlabel('Linearity Compression Score')
    plt.ylabel('Fraction of pruned weights')
    plt.title('Linearity Compression Scores vs Pruning Ratios for ' + model_name)
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/scatterplot_{model_name}_{linearity_score}_{prune_method}_{dataset}.png")
    plt.close()

def load_linearity_scores_from_disk(path: str, conv_names: Optional[list] = None):
    filepath = str(glob.glob(path, recursive=True)[0]) # We just take the first instance
    logger.info(f"Found save path {path}, attempting to load linearity scores from disk.")
    linearity_scores = dict(json.load(open(filepath, "r")))

    if "procrustes" in filepath and any(not (key[:-1].endswith("conv") or key.endswith("self_attn")) for key in linearity_scores.keys()):
        linearity_scores = expand_scores_to_individual_layers(linearity_scores, "resnet" in filepath, conv_names)

    logger.info(f"Loaded linearity scores from disk: {linearity_scores}")
    return linearity_scores

def run_experiment(model: str, dataset: str, relation_to: str, batch_size: int,
                   epochs: int, lr: float, data_fraction: float, save: bool, seed: int, device: str,
                   skip_finetune_path: Optional[str], pruning_ratio: float=0.1, blocks: Union[None, list]=None, hidden_layer_reduction: int=2):
    """Run the relation to other compression methods experiment.
    Results are logged and stored to wandb if enabled, and models/results are saved to ./results if enabled.
    Args:
        model (str): The model architecture to use (e.g., 'resnet18', 'llama-3.2-1b').
        dataset (str): The dataset to use (e.g., 'imagenet').
        relation_to (str): The compression method identifier to compare against (e.g. 'magnitude_pruning', 'basic_kd')
        batch_size (int): The batch size for training and evaluation.
        epochs (int): The number of epochs for fine-tuning.
        lr (float): The learning rate for the optimizer.
        data_fraction (float): The fraction of the dataset to use for training.
        save (bool): Whether to save the trained models and results.
        seed (int): The random seed for reproducibility.
        device (str): The device to run the experiments on (e.g., 'cpu', 'cuda').
        skip_finetune_path (str): The path to look for a finetuned model saved to disk if skipping is enabled.
        pruning_ratio (float): The ratio of pruning scores to use for each layer.
        blocks (Union[None, list]): The list of blocks to use for distilled resnet.
        hidden_layer_reduction (int): The number of hidden layers to remove for distilled llama.
    """
    save_dir = "./results/rq2/all/" + relation_to + "/" + model + "/" + dataset + "/" + str(seed)
    os.makedirs(save_dir, exist_ok=True)

    # ------------------------------------------------------------
    # Load data and model
    # ------------------------------------------------------------
    if "resnet" in model:
        logger.info(
            f"Running ResNet relation experiment with model={model}, dataset={dataset}, relation_to={relation_to}, batch_size={batch_size}, epochs={epochs}, lr={lr}, data fraction: {data_fraction}, save={save}, seed={seed}, device={device}")
        data_handler = DataManager(dataset_name=dataset, batch_size=batch_size, data_fraction=data_fraction,
                                   seed=seed)
        logger.debug(
            f"Dataset loaded with {len(data_handler.train_set)} training samples and {len(data_handler.val_set)} validation samples.")
        experimenter = ResNetExperimenter(model_name=model, data_handler=data_handler, batch_size=batch_size, epochs=epochs,
                                          learning_rate=lr, device=device, skip_finetune_path=skip_finetune_path)
        if save and not experimenter.skipped:
            # Save finetuned original
            torch.save(experimenter.model.state_dict(), f"{save_dir}/{model}_original.pth")
            logger.info(f"Saved finetuned original model to {save_dir}/{model}_original.pth")
        elif save:
            logger.info(f"Skipping saving finetuned model as it was loaded from disk. Loaded from {skip_finetune_path}.")
    elif "llama" in model:
        logger.info(
            f"Running Llama relation experiment with model: {model}, dataset: {dataset}, relation_to={relation_to}, batch size: {batch_size}, epochs: {epochs}, learning rate: {lr}, data fraction: {data_fraction}, save results: {save}, seed: {seed}, device: {device}")
        data_handler = DataManager(dataset_name=dataset, batch_size=batch_size, data_fraction=data_fraction, model_name=model,
                                   seed=seed)
        logger.debug(
            f"Dataset loaded with {len(data_handler.train_set)} training samples and {len(data_handler.val_set)} validation samples.")
        experimenter = LlamaExperimenter(model_name=model, data_handler=data_handler, batch_size=batch_size,
                                         epochs=epochs, learning_rate=lr, device=device, skip_finetune_path=skip_finetune_path)
        if save and not experimenter.skipped:
            experimenter.model.save_pretrained(f"{save_dir}/original_{model}")
            logger.info(f"Original finetuned model saved to {save_dir}/original_{model}")
        elif save:
            logger.info(f"Skipping saving finetuned model as it was loaded from disk. Loaded from {skip_finetune_path}.")
    else:
        raise ValueError(f"Unknown model: {model}")
    logger.info("Model and data loaded, model fine-tuned.")

    # ------------------------------------------------------------
    # Load linearity scores
    # ------------------------------------------------------------
    conv_names = None
    if "resnet" in experimenter.model:
        conv_names = [name for name, module in experimenter.model.named_modules() if
                      isinstance(module, torch.nn.Conv2d) and "downsample" not in name]

    linearity_scores_fraction = load_linearity_scores_from_disk(f"./results/*/fraction/*/{experimenter.model_name}/{dataset}/{seed}/activation_fractions.json")
    linearity_scores_mean_preactivation = load_linearity_scores_from_disk(f"./results/*/mean_preactivation/*/{experimenter.model_name}/{dataset}/{seed}/mean_preactivations.json")
    linearity_scores_procrustes = load_linearity_scores_from_disk(f"./results/*/procrustes/*/{experimenter.model_name}/{dataset}/{seed}/procrustes_scores.json", conv_names=conv_names)

    # ------------------------------------------------------------
    # Evaluate initial model performance
    # ------------------------------------------------------------
    original_accuracy, original_param_count, original_inference_time, original_gflops = experimenter.validate_model()
    logger.info(f"Original model accuracy: {original_accuracy:.4f}, parameters: {original_param_count}, "
                f"inference time: {original_inference_time:.4f} seconds, gflops: {original_gflops}")

    # --------------------------------------------------------------
    # Compute pruning ratios or student model
    # --------------------------------------------------------------
    prune_dict, student_model = None, None
    match relation_to:
        case 'magnitude_pruning':
            from compression_methods.magnitude_pruning import prune
            prune_dict, compressed_accuracy, compressed_param_count, compressed_inference_time, compressed_gflops = prune(
                experimenter, data_handler, device=device,
                pruning_ratio=pruning_ratio, lr=lr,
                batch_size=batch_size, epochs=epochs)
        case 'hessian_pruning':
            if "llama" in experimenter.model_name:
                raise ValueError("Hessian pruning is not supported for Llama models")

            from compression_methods.hessian_pruning import prune
            prune_dict, compressed_accuracy, compressed_param_count, compressed_inference_time, compressed_gflops = prune(
                experimenter, data_handler, device=device,
                pruning_ratio=pruning_ratio, lr=lr,
                batch_size=batch_size, epochs=epochs)
        case 'taylor_pruning':
            if "llama" in experimenter.model_name:
                raise ValueError("Taylor pruning is not supported for Llama models")

            from compression_methods.taylor_pruning import prune
            prune_dict, compressed_accuracy, compressed_param_count, compressed_inference_time, compressed_gflops = prune(
                experimenter, data_handler, device=device,
                pruning_ratio=pruning_ratio, lr=lr,
                batch_size=batch_size, epochs=epochs)
        case 'wanda_pruning':
            if "resnet" in experimenter.model_name:
                raise ValueError("Wanda pruning is not supported for ResNet models")

            from compression_methods.wanda_pruning import prune
            prune_dict, compressed_accuracy, compressed_param_count, compressed_inference_time, compressed_gflops = prune(
                experimenter, data_handler, device=device,
                pruning_ratio=pruning_ratio, lr=lr,
                batch_size=batch_size, epochs=epochs)
        case 'slicegpt':
            if "resnet" in experimenter.model_name:
                raise ValueError("SliceGPT is not supported for ResNet models")

            from compression_methods.slicegpt import prune
            prune_dict, compressed_accuracy, compressed_param_count, compressed_inference_time, compressed_gflops = prune(
                experimenter, data_handler, device=device,
                pruning_ratio=pruning_ratio, lr=lr,
                batch_size=batch_size, epochs=epochs)
        case 'basic_kd':
            from compression_methods.basic_kd import distill
            if blocks is None and experimenter.model_name == "resnet18":
                blocks = [1, 1, 2, 2]
            elif blocks is None and experimenter.model_name == "resnet34":
                blocks = [2, 3, 6, 3]
            elif blocks is None and experimenter.model_name == "resnet50":
                blocks = [2, 3, 6, 3]

            student_model, compressed_accuracy, compressed_param_count, compressed_inference_time, compressed_gflops = distill(
                experimenter, data_handler, device=device,
                lr=lr, epochs=epochs, blocks=blocks,
                hidden_layer_reduction=hidden_layer_reduction)
        case 'feature_kd':
            from compression_methods.feature_kd import distill
            if blocks is None and experimenter.model_name == "resnet18":
                blocks = [1, 1, 2, 2]
            elif blocks is None and experimenter.model_name == "resnet34":
                blocks = [2, 3, 6, 3]
            elif blocks is None and experimenter.model_name == "resnet50":
                blocks = [2, 3, 6, 3]

            student_model, compressed_accuracy, compressed_param_count, compressed_inference_time, compressed_gflops = distill(
                experimenter, data_handler, device=device,
                lr=lr, epochs=epochs, blocks=blocks,
                hidden_layer_reduction=hidden_layer_reduction)
        case 'born_again_kd':
            from compression_methods.born_again_kd import distill
            if blocks is None and experimenter.model_name == "resnet18":
                blocks = [[1, 1, 2, 2], [1, 1, 1, 2]]
            elif blocks is None and experimenter.model_name == "resnet34":
                blocks = [[2, 3, 6, 3], [2, 3, 5, 3]]
            elif blocks is None and experimenter.model_name == "resnet50":
                blocks = [[2, 3, 6, 3], [2, 3, 5, 3]]

            student_model, compressed_accuracy, compressed_param_count, compressed_inference_time, compressed_gflops = distill(
                experimenter, data_handler, device=device,
                lr=lr, epochs=epochs, blocks_iterations=blocks,
                hidden_layer_reduction_iterations=[2,3])

    logger.info(f"Compressed model evaluated with accuracy: {compressed_accuracy:.4f}, parameters: {compressed_param_count}, "
                f"inference time: {compressed_inference_time:.4f} seconds, gflops: {compressed_gflops}")

    # ------------------------------------------------------------
    # Evaluate compressed model performance
    # ------------------------------------------------------------
    accuracy_loss = utils.accuracy_loss(original_accuracy, compressed_accuracy)
    param_compression_ratio = utils.compression_ratio(original_param_count, compressed_param_count)
    speedup = utils.speedup(original_inference_time, compressed_inference_time)
    gflop_reduction = utils.gflop_reduction(original_gflops, compressed_gflops)
    logger.info(f"Accuracy loss: {accuracy_loss:.4f}, Parameter compression ratio: {param_compression_ratio:.4f}, "
                f"Speedup: {speedup:.4f}x, GFLOP reduction: {gflop_reduction:.4f}")

    # --------------------------------------------------------------
    # Generate either scatterplot or similarity matrix
    # --------------------------------------------------------------
    matrix = None
    teacher_layer_names = None
    student_layer_names = None
    if prune_dict is not None:
        scatterplot_linearity_pruning_scores(linearity_scores_fraction, prune_dict, save_dir, "fraction")
        scatterplot_linearity_pruning_scores(linearity_scores_mean_preactivation, prune_dict, save_dir, "mean_preactivation")
        scatterplot_linearity_pruning_scores(linearity_scores_procrustes, prune_dict, save_dir, "procrustes")
        logger.info("Saved linearity vs pruning scatterplot.")
    if student_model is not None:
        data_loader = DataLoader(data_handler.val_set, batch_size=batch_size, shuffle=False,
                                  num_workers=4,              # try 2–8 depending on CPU
                                  pin_memory=True,            # important for GPU transfer
                                  prefetch_factor=2,          # batches per worker
                                  persistent_workers=True     # avoids worker restart each epoch
                                  )
        matrix, teacher_layer_names, student_layer_names = cka_similarity_matrix(experimenter.model, student_model,
                                                                                data_loader, device=device,
                                                                                tokenizer=data_handler.tokenizer if "llama" in model else None)
        visualize_cka_similarity_matrix(matrix, save_dir, teacher_layer_names, student_layer_names, linearity_scores_fraction, "fraction")
        visualize_cka_similarity_matrix(matrix, save_dir, teacher_layer_names, student_layer_names, linearity_scores_mean_preactivation, "mean_preactivation")
        visualize_cka_similarity_matrix(matrix, save_dir, teacher_layer_names, student_layer_names, linearity_scores_procrustes, "procrustes")
        logger.info("Saved cka similarity heatmap.")

    # ------------------------------------------------------------
    # Log results to wandb and save models/results if enabled
    # ------------------------------------------------------------
    wandb_logging_data = {
        "model": model,
        "dataset": dataset,
        "relation_to": relation_to,
        "seed": seed,
        "linearity_scores_fraction": linearity_scores_fraction,
        "linearity_scores_mean_preactivation": linearity_scores_mean_preactivation,
        "linearity_scores_procrustes": linearity_scores_procrustes,
        "comp_accuracy": compressed_accuracy,
        "comp_param_count": compressed_param_count,
        "comp_inference_time": compressed_inference_time,
        "comp_gflops": compressed_gflops,
        "original_accuracy": original_accuracy,
        "original_param_count": original_param_count,
        "original_inference_time": original_inference_time,
        "original_gflops": original_gflops,
        "accuracy_loss": accuracy_loss,
        "param_compression_ratio": param_compression_ratio,
        "speedup": speedup,
        "gflop_reduction": gflop_reduction,
    }

    if save:
        import json

        json.dump(wandb_logging_data, open(f"{save_dir}/wandb_logging_data.json", "w"), indent=4)
        logger.info(f"Saved wandb logging data to {save_dir}/wandb_logging_data.json")

        if prune_dict is not None:
            json.dump(prune_dict, open(f"{save_dir}/prune_dict.json", "w"), indent=4)
            logger.info(f"Saved prune dict to {save_dir}/prune_dict.json")
        if student_model is not None:
            if "llama" in model:
                # Save llama
                experimenter.model.save_pretrained(f"{save_dir}/compressed_{model}")
            else:
                torch.save(student_model.state_dict(), f"{save_dir}/{model}_distilled.pth")
            logger.info(f"Saved student model to {save_dir}/{model}_distilled.pth")

            # Store cka matrix
            np.save(f"{save_dir}/cka_similarity_matrix.npy", matrix)
            json.dump(teacher_layer_names, open(f"{save_dir}/teacher_layer_names.json", "w"), indent=4)
            json.dump(student_layer_names, open(f"{save_dir}/student_layer_names.json", "w"), indent=4)
            logger.info(f"Saved CKA similarity matrix to {save_dir}/cka_similarity_matrix.npy")

    if prune_dict is not None:
        wandb_logging_data["prune_dict"] = prune_dict
    if student_model is not None:
        wandb_logging_data["student_model"] = student_model
        wandb_logging_data["cka_similarity_matrix"] = matrix

    wandb.log(wandb_logging_data)
    logger.info("Logged results to Weights & Biases")
    logger.info("Experiment completed.")