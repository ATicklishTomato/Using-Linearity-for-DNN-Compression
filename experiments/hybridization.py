import logging
import os
from typing import Union, Optional
import torch
import wandb
from metrics.linearity_metric_manager import LinearityMetric
from utils.data_manager import DataManager
from utils.llama_model import LlamaExperimenter
from utils.resnet_model import ResNetExperimenter
import utils.util_functions as utils

logger = logging.getLogger(__name__)

def run_experiment(model: str, linearity: str, dataset: str, compression_method: str, batch_size: int,
                   epochs: int, lr: float, data_fraction: float, save: bool, seed: int, device: str,
                   skip_finetune_path: Optional[str], pruning_ratio: float=0.1, blocks: Union[None, list]=None,
                   hidden_layer_reduction: int=2):
    """Attempt hybridization with other compression methods experiment.
    Results are logged and stored to wandb if enabled, and models/results are saved to ./results if enabled.
    Args:
        model (str): The model architecture to use (e.g., 'resnet18', 'llama-3.2-1b').
        linearity (str): The linearity metric to use (e.g., 'mean_preactivation', 'procrustes', or 'fraction').
        dataset (str): The dataset to use (e.g., 'imagenet').
        compression_method (str): The compression method identifier to compare against (e.g. 'magnitude_pruning', 'basic_kd')
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
    save_dir = "./results/rq2/" + linearity + "/" + compression_method + "_hybrid/" + model + "/" + dataset + "/" + str(seed)
    os.makedirs(save_dir, exist_ok=True)

    # ------------------------------------------------------------
    # Load data and model
    # ------------------------------------------------------------
    experimenter, data_handler = None, None
    if "resnet" in model:
        logger.info(
            f"Running ResNet relation experiment with model={model}, linearity={linearity}, dataset={dataset}, relation_to={compression_method}, batch_size={batch_size}, epochs={epochs}, lr={lr}, data fraction: {data_fraction}, save={save}, seed={seed}, device={device}")
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
            f"Running Llama compression experiment with model: {model}, linearity metric: {linearity}, dataset: {dataset}, relation_to={compression_method}, batch size: {batch_size}, epochs: {epochs}, learning rate: {lr}, data fraction: {data_fraction}, save results: {save}, seed: {seed}, device: {device}")
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
    # Compute linearity scores
    # ------------------------------------------------------------
    # We hardcode threshold because we don't care about the split in this case
    metric = LinearityMetric(linearity, model, data_handler, "50%", device, save, save_dir)
    linearity_scores = metric.metric_fn(experimenter.model)
    logger.info("Linearity scores computed.")
    logger.debug(f"Linearity scores: {linearity_scores}")
    linear_layers, nonlinear_layers = metric.thresholder(linearity_scores)
    # We recombine the linear and nonlinear splits as we don't care
    linearity_scores = {**linear_layers, **nonlinear_layers}

    # ------------------------------------------------------------
    # Evaluate initial model performance
    # ------------------------------------------------------------
    original_accuracy, original_param_count, original_inference_time, original_gflops = experimenter.validate_model()
    logger.info(f"Original model accuracy: {original_accuracy:.4f}, parameters: {original_param_count}, "
                f"inference time: {original_inference_time:.4f} seconds, gflops: {original_gflops}")

    # --------------------------------------------------------------
    # Distill if that's the hybrid pairing
    # --------------------------------------------------------------
    match compression_method:
        case 'basic_kd':
            from compression_methods.basic_kd import distill
            if blocks is None and experimenter.model_name == "resnet18":
                blocks = [1, 1, 2, 2]
            elif blocks is None and experimenter.model_name == "resnet34":
                blocks = [2, 3, 6, 3]
            elif blocks is None and experimenter.model_name == "resnet50":
                blocks = [2, 3, 6, 3]

            model, compressed_accuracy, compressed_param_count, compressed_inference_time, compressed_gflops = distill(
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

            model, compressed_accuracy, compressed_param_count, compressed_inference_time, compressed_gflops = distill(
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

            model, _, _, _, _ = distill(
                experimenter, data_handler, device=device,
                lr=lr, epochs=epochs, blocks_iterations=blocks,
                hidden_layer_reduction_iterations=[2, 3])

    # ------------------------------------------------------------
    # Do linearity compression
    # ------------------------------------------------------------

    if "resnet" in experimenter.model_name:
        from experiments.resnet_approx_compression import group_contiguous_layers, train_approximation_layers
        all_layers = list(linear_layers.keys()) + list(nonlinear_layers.keys())
        groups = group_contiguous_layers(linear_layers, all_layers, experimenter.model)
        train_approximation_layers(experimenter, data_handler, groups, epochs=epochs, lr=lr,
                                   batch_size=batch_size, device=device)
        logger.info("Linear approximation layers trained and integrated into the model.")
    else:
        from experiments.llama_approx_compression import group_contiguous_layers, train_approximation_layers
        groups = group_contiguous_layers(linear_layers)
        train_approximation_layers(experimenter, data_handler, groups, save_model=save,
                                   epochs=epochs, lr=lr, device=device, save_path=save_dir)
        logger.info("Linear approximation layers trained and integrated into the model.")

    experimenter.finetune()

    compressed_accuracy, compressed_param_count, compressed_inference_time, compressed_gflops = experimenter.validate_model()
    logger.info(f"Merged model accuracy: {compressed_accuracy:.4f}, parameters: {compressed_param_count}, "
                f"inference time: {compressed_inference_time:.4f} seconds, gflops: {compressed_gflops}")



    # ------------------------------------------------------------
    # Apply pruning if that is the hybrid pairing
    # ------------------------------------------------------------
    prune_dict = None
    match compression_method:
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



    # ------------------------------------------------------------
    # Evaluate compressed model performance
    # ------------------------------------------------------------
    accuracy_loss = utils.accuracy_loss(original_accuracy, compressed_accuracy)
    param_compression_ratio = utils.compression_ratio(original_param_count, compressed_param_count)
    speedup = utils.speedup(original_inference_time, compressed_inference_time)
    gflop_reduction = utils.gflop_reduction(original_gflops, compressed_gflops)
    logger.info(f"Accuracy loss: {accuracy_loss:.4f}, Parameter compression ratio: {param_compression_ratio:.4f}, "
                f"Speedup: {speedup:.4f}x, GFLOP reduction: {gflop_reduction:.4f}")


    # ------------------------------------------------------------
    # Log results to wandb and save models/results if enabled
    # ------------------------------------------------------------
    wandb_logging_data = {
        "model": model,
        "dataset": dataset,
        "compression_method": compression_method,
        "linearity": linearity,
        "seed": seed,
        "linearity_scores": linearity_scores,
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

        # Save linearity scores
        json.dump(linearity_scores, open(f"{save_dir}/linearity_scores.json", "w"), indent=4)
        logger.info(f"Saved linearity scores to {save_dir}/linearity_scores.json")

        json.dump(wandb_logging_data, open(f"{save_dir}/wandb_logging_data.json", "w"), indent=4)
        logger.info(f"Saved wandb logging data to {save_dir}/wandb_logging_data.json")

        if prune_dict is not None:
            json.dump(prune_dict, open(f"{save_dir}/prune_dict.json", "w"), indent=4)
            logger.info(f"Saved prune dict to {save_dir}/prune_dict.json")
        if "kd" in compression_method:
            if "llama" in model:
                # Save llama
                experimenter.model.save_pretrained(f"{save_dir}/distilled_{model}")
            else:
                torch.save(model.state_dict(), f"{save_dir}/{model}_distilled.pth")
            logger.info(f"Saved student model to {save_dir}/{model}_distilled.pth")

    if prune_dict is not None:
        wandb_logging_data["prune_dict"] = prune_dict
    if "kd" in compression_method:
        wandb_logging_data["student_model"] = model

    wandb.log(wandb_logging_data)
    logger.info("Logged results to Weights & Biases")
    logger.info("Experiment completed.")