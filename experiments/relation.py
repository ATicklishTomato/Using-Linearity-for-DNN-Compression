import logging
import os
import re
from typing import Union

import numpy as np
import torch
import wandb
import matplotlib
matplotlib.use("Agg") # Avoid errors when running without UI
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from metrics.linearity_metric_manager import LinearityMetric
from utils.data_manager import DataManager
from utils.llama_model import LlamaExperimenter
from utils.resnet_model import ResNetExperimenter

logger = logging.getLogger(__name__)

def _get_target_layers(model):
    if "resnet" in model.__class__.__name__.lower():
        pattern = r".*conv.*"
    elif "llama" in model.__class__.__name__.lower():
        pattern = r"model\.layers\..*\.self_attn"
    else:
        raise ValueError(...)

    layers = []
    names = []

    for name, module in model.named_modules():
        if re.match(pattern, name):
            layers.append(module)
            names.append(name)

    return layers, names


def _collect_activations(model, x, layers):
    activations = []

    def hook_fn(module, input, output):
        # Flatten spatial dims but keep batch
        if isinstance(output, tuple):
            output = output[0]

        if output.dim() == 4:
            # [B, C, H, W] → [B, C, H*W]
            output = output.flatten(1)

        if output.dim() == 3:
            # [B, S, H] → [B, H]
            output = output.mean(dim=1)

        elif output.dim() == 2:
            # already [B, H]
            pass

        else:
            raise ValueError(f"Unexpected shape: {output.shape}")

        activations.append(output.detach())

    hooks = []
    for layer in layers:
        hooks.append(layer.register_forward_hook(hook_fn))

    if "resnet" in model.__class__.__name__.lower():
        model.eval()
        with torch.no_grad():
            model(x)
    else:
        model.eval()
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                model(**x)

    for h in hooks:
        h.remove()

    return activations


def _center_gram(K):
    """Center Gram matrix."""
    row_mean = K.mean(dim=1, keepdim=True)
    col_mean = K.mean(dim=0, keepdim=True)
    total_mean = K.mean()
    return K - row_mean - col_mean + total_mean


def _linear_cka(X, Y):
    """Compute linear CKA between two activation matrices."""
    # X, Y: (n_samples, features)
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    K = X @ X.T
    L = Y @ Y.T

    Kc = _center_gram(K)
    Lc = _center_gram(L)

    hsic = (Kc * Lc).sum()
    norm_x = torch.norm(Kc)
    norm_y = torch.norm(Lc)

    return (hsic / (norm_x * norm_y + 1e-8)).item()


def cka_similarity_matrix(model_a, model_b, dataloader, device="cuda", max_batches=10, tokenizer=None):
    """
    Computes a CKA similarity matrix between layers of two models.

    Args:
        model_a: First model (e.g. teacher)
        model_b: Second model (e.g. student)
        dataloader: DataLoader providing input batches
        device: device to run on
        max_batches: number of batches to average over
        tokenizer: Optional tokenizer for text data (only needed if models are LLaMA)
    Returns:
        np.ndarray of shape (m, n)
    """
    model_a.to(device).eval()
    model_b.to(device).eval()

    layers_a, names_a = _get_target_layers(model_a)
    layers_b, names_b = _get_target_layers(model_b)

    m, n = len(layers_a), len(layers_b)
    matrix = np.zeros((m, n))

    # accumulate activations over batches
    acts_a_all = [[] for _ in range(m)]
    acts_b_all = [[] for _ in range(n)]

    logger.info("Prepped for CKA computation, starting to collect activations and compute similarity matrix.")
    for i, x in enumerate(dataloader):
        if i >= max_batches:
            break

        if "resnet" in model_a.__class__.__name__.lower() or "resnet" in model_b.__class__.__name__.lower():
            x, _ = x
        else:
            x = tokenizer(x['text'], return_tensors='pt', padding=True, truncation=True)

        x = x.to(device)

        acts_a = _collect_activations(model_a, x, layers_a)
        acts_b = _collect_activations(model_b, x, layers_b)

        for j in range(m):
            acts_a_all[j].append(acts_a[j].cpu())

        for j in range(n):
            acts_b_all[j].append(acts_b[j].cpu())

    logger.info("Activations collected for all batches, now concatenating and computing CKA matrix.")

    # concatenate across batches
    acts_a_all = [torch.cat(a, dim=0) for a in acts_a_all]
    acts_b_all = [torch.cat(b, dim=0) for b in acts_b_all]

    logger.info("Activations concatenated, now computing CKA similarity matrix.")

    # compute CKA matrix
    for i in range(m):
        for j in range(n):
            matrix[i, j] = _linear_cka(acts_a_all[i], acts_b_all[j])

    logger.info("CKA similarity matrix computed.")

    return matrix, names_a, names_b

def visualize_cka_similarity_matrix(matrix, save_dir, layer_names, linearity_scores):
    """Creates a magma heatmap of the cka similarity matrix using matplotlib.
    Shows row and column indexes to roughly identify layers. Similarity scores are listed in the cells.
    Stores the visualization in ./results with the given filename.
    Args:
        matrix: A numpy array of shape (m, n) containing the cka similarity values between the layers.
        save_dir: The directory to save the heatmap. Saved as "cka_similarity_heatmap.png" in the given directory.
        layer_names: Names of the parent model layers.
        linearity_scores: A list of the linearity scores for each layer (length m).
    """
    logger.info(f"Visualizing CKA similarity matrix with shape {matrix.shape} and saving to {save_dir}/cka_similarity_heatmap.png")
    model_name = "llama" if "llama" in save_dir else "resnet"

    ticks = []
    y_labels = []
    for i, name in enumerate(layer_names):
        if name in linearity_scores:
            score = linearity_scores[name]
            ticks.append(i)
            y_labels.append(f"{name.split('.')[-1]} ({score:.4f})")

    plt.imshow(matrix, cmap='magma', vmin=0, vmax=1, origin='upper')
    plt.colorbar(label=f'CKA Similarity of {model_name} layers')
    plt.xlabel('Student model')
    plt.ylabel('Parent model')
    # Move x-axis to top
    plt.gca().xaxis.tick_top()
    plt.gca().xaxis.set_label_position('top')
    plt.xticks(rotation=90)
    plt.yticks(ticks=ticks, labels=y_labels, rotation=45)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/cka_similarity_heatmap.png")
    plt.close()

def scatterplot_linearity_pruning_scores(linearity_scores: dict, pruning_scores: dict, save_dir: str) -> None:
    """Creates a scatterplot of the linearity compression scores and pruning scores for each layer.
    Points are labeled with their layer index. X-axis will be linearity score, Y-axis will be pruning score.
    Args:
        linearity_scores: A dictionary mapping layer names to linear scores.
        pruning_scores: A dictionary mapping layer names to pruning scores.
        save_dir: The directory to save the scatterplot. Saved as "linearity_pruning_scatterplot.png" in the given directory.
    """
    layer_names = list(set(linearity_scores.keys()).intersection(set(pruning_scores.keys())))
    logger.info(f"Computing scatterplot for {len(layer_names)} layers out of total {len(linearity_scores) + len(pruning_scores)} layers.")
    linearity_values = [linearity_scores[name] for name in layer_names]
    pruning_values = [pruning_scores[name] for name in layer_names]

    plt.figure(figsize=(10, 6))
    plt.scatter(linearity_values, pruning_values)

    for i, name in enumerate(layer_names):
        plt.annotate(name, (linearity_values[i], pruning_values[i]))

    plt.xlabel('Linearity Compression Score')
    plt.ylabel('Pruning Score')
    plt.title('Linearity Compression Scores vs Pruning Scores')
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/linearity_pruning_scatterplot.png")
    plt.close()

def run_experiment(model: str, linearity: str, dataset: str, relation_to: str, batch_size: int,
                           epochs: int, lr: float, max_batches: int, save: bool, seed: int, device: str, pruning_ratio: float=0.1,
                   blocks: Union[None, list]=None, hidden_layer_reduction: int=2):
    """Run the relation to other compression methods experiment.
    Results are logged and stored to wandb if enabled, and models/results are saved to ./results if enabled.
    Args:
        model (str): The model architecture to use (e.g., 'resnet18', 'llama-3.2-1b').
        linearity (str): The linearity metric to use (e.g., 'mean_preactivation', 'procrustes', or 'fraction').
        dataset (str): The dataset to use (e.g., 'imagenet').
        relation_to (str): The compression method identifier to compare against (e.g. 'magnitude_pruning', 'basic_kd')
        batch_size (int): The batch size for training and evaluation.
        epochs (int): The number of epochs for fine-tuning.
        lr (float): The learning rate for the optimizer.
        max_batches (int): The maximum number of batches to process during training/evaluation.
        save (bool): Whether to save the trained models and results.
        seed (int): The random seed for reproducibility.
        device (str): The device to run the experiments on (e.g., 'cpu', 'cuda').
        pruning_ratio (float): The ratio of pruning scores to use for each layer.
        blocks (Union[None, list]): The list of blocks to use for distilled resnet.
        hidden_layer_reduction (int): The number of hidden layers to remove for distilled llama.
    """
    short_model = "llama" if "llama" in model else "resnet"
    save_dir = "./results/rq2/" + relation_to + "/" + short_model + "/" + dataset + "/" + str(seed)
    os.makedirs(save_dir, exist_ok=True)

    # ------------------------------------------------------------
    # Load data and model
    # ------------------------------------------------------------
    if "resnet" in model:
        logger.info(
            f"Running ResNet relation experiment with model={model}, linearity={linearity}, dataset={dataset}, relation_to={relation_to}, batch_size={batch_size}, epochs={epochs}, lr={lr}, max_batches={max_batches}, save={save}, seed={seed}, device={device}")
        data_handler = DataManager(dataset_name=dataset, batch_size=batch_size, reduction_fraction=0.1,
                                   seed=seed)  # Reduce to 10% for faster experimentation
        logger.debug(
            f"Dataset loaded with {len(data_handler.train_set)} training samples and {len(data_handler.val_set)} validation samples.")
        experimenter = ResNetExperimenter(model_name=model, data_handler=data_handler, batch_size=batch_size, epochs=epochs,
                                          learning_rate=lr, max_batches=max_batches, device=device)
        if save:
            # Save finetuned original
            torch.save(experimenter.model.state_dict(), f"{save_dir}/{model}_original.pth")
            logger.info(f"Saved finetuned original model to {save_dir}/{model}_original.pth")
    elif "llama" in model:
        logger.info(
            f"Running Llama compression experiment with model: {model}, linearity metric: {linearity}, dataset: {dataset}, relation_to={relation_to}, batch size: {batch_size}, epochs: {epochs}, learning rate: {lr}, max batches: {max_batches}, save results: {save}, seed: {seed}, device: {device}")
        data_handler = DataManager(dataset_name=dataset, batch_size=batch_size, model_name=model,
                                   reduction_fraction=0.1,
                                   seed=seed)  # Reduction fraction is set to 0.1 for faster experimentation, can be adjusted as needed
        logger.debug(
            f"Dataset loaded with {len(data_handler.train_set)} training samples and {len(data_handler.val_set)} validation samples.")
        experimenter = LlamaExperimenter(model_name=model, data_handler=data_handler, batch_size=batch_size,
                                         epochs=epochs, learning_rate=lr, max_batches=max_batches, device=device)
        if save:
            experimenter.model.save_pretrained(f"{save_dir}/original_{model}")
            logger.info(f"Original finetuned model saved to {save_dir}/original_{model}")
    else:
        raise ValueError(f"Unknown model: {model}")
    logger.info("Model and data loaded, model fine-tuned.")

    # ------------------------------------------------------------
    # Compute linearity scores
    # ------------------------------------------------------------
    # We hardcode threshold because we don't care about the split in this case
    metric = LinearityMetric(linearity, model, data_handler, "50%", max_batches, device, save, save_dir)
    linearity_scores = metric.metric_fn(experimenter.model)
    logger.info("Linearity scores computed.")
    logger.debug(f"Linearity scores: {linearity_scores}")
    linear_layers, nonlinear_layers = metric.thresholder(linearity_scores)
    # We recombine the linear and nonlinear splits as we don't care
    linearity_scores = {**linear_layers, **nonlinear_layers}

    # --------------------------------------------------------------
    # Compute pruning ratios or student model
    # --------------------------------------------------------------
    prune_dict, student_model = None, None
    match relation_to:
        case 'magnitude_pruning':
            from compression_methods.magnitude_pruning import prune
            prune_dict, acc, param, infer, gflops = prune(experimenter, data_handler, device=device,
                                                                          pruning_ratio=pruning_ratio, max_batches=max_batches,
                                                                          lr=lr, batch_size=batch_size, epochs=epochs)
        case 'basic_kd':
            from compression_methods.basic_kd import distill
            if blocks is None:
                blocks = [1,1,2,2]
            student_model, acc, param, infer, gflops = distill(experimenter, data_handler,device=device,
                                                                                   lr=lr, epochs=epochs,
                                                                                   max_batches=max_batches,
                                                                                   blocks=blocks,
                                                                                   hidden_layer_reduction=hidden_layer_reduction)

    # --------------------------------------------------------------
    # Generate either scatterplot or similarity matrix
    # --------------------------------------------------------------
    matrix = None
    if prune_dict is not None:
        scatterplot_linearity_pruning_scores(linearity_scores, prune_dict, save_dir)
        logger.info("Saved linearity vs pruning scatterplot.")
    if student_model is not None:
        data_loader = DataLoader(data_handler.val_set, batch_size=batch_size, shuffle=False)
        matrix, parent_layer_names, _ = cka_similarity_matrix(experimenter.model, student_model, data_loader, device=device, max_batches=max_batches, tokenizer=data_handler.tokenizer if "llama" in model else None)
        visualize_cka_similarity_matrix(matrix, save_dir, parent_layer_names, linearity_scores)
        logger.info("Saved cka similarity heatmap.")

    # ------------------------------------------------------------
    # Log results to wandb and save models/results if enabled
    # ------------------------------------------------------------
    if save:
        import json

        # Save linearity scores
        json.dump(linearity_scores, open(f"{save_dir}/linearity_scores.json", "w"), indent=4)
        logger.info(f"Saved linearity scores to {save_dir}/linearity_scores.json")

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
            logger.info(f"Saved CKA similarity matrix to {save_dir}/cka_similarity_matrix.npy")

    wandb_logging_data = {
        "model": model,
        "dataset": dataset,
        "relation_to": relation_to,
        "linearity": linearity,
        "seed": seed,
        "linearity_scores": linearity_scores,
        "comp_acc": acc,
        "comp_param": param,
        "comp_infer": infer,
        "comp_gflops": gflops,
    }

    if prune_dict is not None:
        wandb_logging_data["prune_dict"] = prune_dict
    if student_model is not None:
        wandb_logging_data["student_model"] = student_model
        wandb_logging_data["cka_similarity_matrix"] = matrix

    wandb.log(wandb_logging_data)
    logger.info("Logged results to Weights & Biases")
    logger.info("Experiment completed.")