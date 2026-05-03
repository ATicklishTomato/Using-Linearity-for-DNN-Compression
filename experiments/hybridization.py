import logging
import os
import re
from typing import Union, Optional

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
    save_dir = "./results/rq2/" + linearity + "/" + compression_method + "/" + model + "/" + dataset + "/" + str(seed)
    os.makedirs(save_dir, exist_ok=True)

    # ------------------------------------------------------------
    # Load data and model
    # ------------------------------------------------------------
    if "resnet" in model:
        logger.info(
            f"Running ResNet relation experiment with model={model}, linearity={linearity}, dataset={dataset}, relation_to={compression_method}, batch_size={batch_size}, epochs={epochs}, lr={lr}, data fraction: {data_fraction}, save={save}, seed={seed}, device={device}")
        data_handler = DataManager(dataset_name=dataset, batch_size=batch_size, data_fraction=data_fraction,
                                   seed=seed)
        logger.debug(
            f"Dataset loaded with {len(data_handler.train_set)} training samples and {len(data_handler.val_set)} validation samples.")
        experimenter = ResNetExperimenter(model_name=model, data_handler=data_handler, batch_size=batch_size, epochs=epochs,
                                          learning_rate=lr, device=device, skip_finetune_path=skip_finetune_path)
        if save and skip_finetune_path is None:
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
        if save and skip_finetune_path is None:
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
    # Compute pruning ratios or student model
    # --------------------------------------------------------------
    prune_dict, student_model = None, None
    match compression_method:
        case 'magnitude_pruning':
            from compression_methods.magnitude_pruning import prune
            prune_dict, compressed_accuracy, compressed_param_count, compressed_inference_time, compressed_gflops = prune(experimenter, data_handler, device=device,
                                                                          pruning_ratio=pruning_ratio, lr=lr,
                                                          batch_size=batch_size, epochs=epochs)
        case 'basic_kd':
            from compression_methods.basic_kd import distill
            if blocks is None:
                blocks = [1,1,2,2]
            student_model, compressed_accuracy, compressed_param_count, compressed_inference_time, compressed_gflops = distill(experimenter, data_handler,device=device,
                                                                                   lr=lr, epochs=epochs, blocks=blocks,
                                                                                   hidden_layer_reduction=hidden_layer_reduction)

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
    if prune_dict is not None:
        scatterplot_linearity_pruning_scores(linearity_scores, prune_dict, save_dir)
        logger.info("Saved linearity vs pruning scatterplot.")
    if student_model is not None:
        data_loader = DataLoader(data_handler.val_set, batch_size=batch_size, shuffle=False)
        matrix, teacher_layer_names, student_layer_names = cka_similarity_matrix(experimenter.model, student_model,
                                                                                data_loader, device=device,
                                                                                tokenizer=data_handler.tokenizer if "llama" in model else None)
        visualize_cka_similarity_matrix(matrix, save_dir, teacher_layer_names, student_layer_names, linearity_scores)
        logger.info("Saved cka similarity heatmap.")

    # ------------------------------------------------------------
    # Log results to wandb and save models/results if enabled
    # ------------------------------------------------------------
    wandb_logging_data = {
        "model": model,
        "dataset": dataset,
        "relation_to": compression_method,
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

    if prune_dict is not None:
        wandb_logging_data["prune_dict"] = prune_dict
    if student_model is not None:
        wandb_logging_data["student_model"] = student_model
        wandb_logging_data["cka_similarity_matrix"] = matrix

    wandb.log(wandb_logging_data)
    logger.info("Logged results to Weights & Biases")
    logger.info("Experiment completed.")