import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import wandb
from utils.data_manager import DataManager
from utils.resnet_model import ResNetExperimenter
from metrics.linearity_metric_manager import LinearityMetric
import utils.util_functions as utils

logger = logging.getLogger(__name__)

def fold_linear_conv_sequences(
    model,
    linear_layers,
    device='cuda'
):
    """
    Fold Conv-BN-ReLU-Conv sequences when the activation is near-linear.

    Args:
        model (nn.Module): ResNet-like model (unchanged architecture)
        linear_layers (dict): {conv_layer_name: linearity_score}
        device (str): Device to perform folding on (e.g., 'cpu', 'cuda')

    Returns:
        folded_model (nn.Module)
        folded_pairs (list of tuples)
    """
    folded_pairs = []
    model.to(device)

    logger.info("\n[Layer Folding] Starting folding pass")

    # ------------------------------------------------------------
    # Helper: fold BN into Conv
    # ------------------------------------------------------------
    def fold_bn_into_conv(conv, bn):
        """
        Fold a batchnorm layer into its own conv layer by applying its scaling and shifting to the conv weights and biases.
        This effectively removes the batchnorm layer while preserving the same transformations.
        Args:
            conv (nn.Conv2d): Convolutional layer
            bn (nn.BatchNorm2d): Batchnorm layer
        Returns:
            None. Convolutional layer is modified in place
        """
        logger.debug(f"    Folding BatchNorm into Conv ({conv.out_channels} channels)")

        W = conv.weight
        if conv.bias is None:
            bias = torch.zeros(W.size(0), device=W.device)
        else:
            bias = conv.bias

        gamma = bn.weight
        beta = bn.bias
        mean = bn.running_mean
        var = bn.running_var
        eps = bn.eps

        std = torch.sqrt(var + eps)
        W_folded = W * (gamma / std).reshape(-1, 1, 1, 1)
        b_folded = beta + (bias - mean) * gamma / std

        conv.weight.data.copy_(W_folded)
        conv.bias = nn.Parameter(b_folded)

    # ------------------------------------------------------------
    # Helper: fold Conv → Conv
    # ------------------------------------------------------------
    def fold_convs(conv1, conv2):
        """
        Folds to convolutional layers into each other, creating one big convolutional layer that performs the same transformations as the two in direct sequence.
        Args:
            conv1 (nn.Conv2d): Convolutional layer
            conv2 (nn.Conv2d): Convolutional layer
        Returns:
            new_conv (nn.Conv2d): New convolutional layer that performs the same transformations.
        """
        logger.debug(
            f"    Folding Conv layers: "
            f"{conv1.in_channels}→{conv1.out_channels}→{conv2.out_channels}"
        )

        W1 = conv1.weight.data
        W2 = conv2.weight.data

        C_out, C_mid, k2, _ = W2.shape
        _, C_in, k1, _ = W1.shape

        W_fold = torch.zeros(
            (C_out, C_in, k1 + k2 - 1, k1 + k2 - 1),
            device=W1.device,
        )

        for m in range(C_out):
            for i in range(C_in):
                acc = 0
                for j in range(C_mid):
                    acc = acc + F.conv2d(
                        W1[j, i].unsqueeze(0).unsqueeze(0),
                        W2[m, j].unsqueeze(0).unsqueeze(0),
                    ).squeeze()
                W_fold[m, i] = acc

        new_conv = nn.Conv2d(
            in_channels=C_in,
            out_channels=C_out,
            kernel_size=W_fold.shape[-1],
            padding=W_fold.shape[-1] // 2,
            bias=False,
        )
        new_conv.weight.data.copy_(W_fold)
        return new_conv

    # ------------------------------------------------------------
    # Main traversal
    # ------------------------------------------------------------
    for module_name, block in model.named_modules():
        if not isinstance(block, nn.Sequential):
            continue

        logger.debug(f"\n[Inspecting block] {module_name}")

        # Block is a sequential with 2 basic blocks of the ResNet architecture. Iterate over them.
        for idx, module in enumerate(block):
            # logger.debug(f" Inspecting module: {module}")

            if not hasattr(module, 'conv1') or not hasattr(module, 'bn1') or not hasattr(module, 'relu') or not hasattr(module, 'conv2'):
                logger.debug("  Not a Conv-BN-ReLU-Conv block → skipping")
                continue

            conv1 = module.conv1
            bn1 = module.bn1
            relu = module.relu
            conv2 = module.conv2

            if not (
                isinstance(conv1, nn.Conv2d)
                and isinstance(bn1, nn.BatchNorm2d)
                and isinstance(relu, nn.ReLU)
                and isinstance(conv2, nn.Conv2d)
            ):
                logger.debug("  Not Conv-BN-ReLU-Conv → skipping")
                continue

            conv1_name = f"{module_name}.{idx}.conv1"

            logger.debug(f"  Found Conv-BN-ReLU-Conv at {conv1_name}")

            if conv1_name not in linear_layers.keys():
                logger.debug("    Below threshold → ReLU not linear")
                continue

            # Validate foldability
            if (
                conv1.stride != (1, 1)
                or conv2.stride != (1, 1)
                or conv1.groups != 1
                or conv2.groups != 1
            ):
                logger.debug("    Stride/groups incompatible → skipping")
                continue

            logger.debug("    Linearity condition satisfied")
            logger.debug("    Removing BatchNorm and ReLU, folding Convs")

            # ----------------------------------------------------
            # Fold BN → Conv1
            # ----------------------------------------------------
            fold_bn_into_conv(conv1, bn1)

            # ----------------------------------------------------
            # Fold Conv1 → Conv2
            # ----------------------------------------------------
            new_conv = fold_convs(conv1, conv2)
            new_conv.to(device)

            # ----------------------------------------------------
            # Replace modules
            # ----------------------------------------------------
            setattr(module, 'conv1', new_conv)
            setattr(module, 'bn1', nn.Identity())
            setattr(module, 'relu', nn.Identity())
            setattr(module, 'conv2', nn.Identity())

            folded_pairs.append((conv1_name, f"{module_name}.{idx}.conv2"))

            logger.debug("    Folding complete")

    logger.info(f"\n[Layer Folding] Done. Folded {len(folded_pairs)} layer pairs.\n")

    return folded_pairs

def run_experiment(model: str, linearity: str, dataset: str, threshold: str, batch_size: int,
                           epochs: int, lr: float, data_fraction: float, save: bool, seed: int, device: str, sweep: bool=False):
    """Run the ResNet compression experiment. Results are logged and stored to wandb if enabled, and models/results are saved to ./results if enabled.
    Args:
        model (str): The ResNet architecture to use (e.g., 'resnet18').
        linearity (str): The linearity metric to use (e.g., 'mean_preactivation', 'procrustes', or 'fraction').
        dataset (str): The dataset to use (e.g., 'imagenet').
        threshold (str): The threshold for determining linearity (e.g., '75%' or '-0.01').
        batch_size (int): The batch size for training and evaluation.
        epochs (int): The number of epochs for fine-tuning.
        lr (float): The learning rate for the optimizer.
        data_fraction (float): The fraction of the dataset to use for training and evaluation.
        save (bool): Whether to save the trained models and results.
        seed (int): The random seed for reproducibility.
        device (str): The device to run the experiments on (e.g., 'cpu', 'cuda').
        sweep (bool): Flag that indicates whether an additional metric should be computed to use for a W&B sweep.
    """
    save_dir = "./results/rq1/" + threshold.split(".")[-1].split("%")[0] + "/resnet/" + dataset + "/" + str(seed)
    os.makedirs(save_dir, exist_ok=True)

    # ------------------------------------------------------------
    # Load data and model
    # ------------------------------------------------------------
    logger.info(f"Running ResNet compression experiment with model={model}, linearity={linearity}, dataset={dataset}, threshold={threshold}, batch_size={batch_size}, epochs={epochs}, lr={lr}, data fraction={data_fraction}, save={save}, seed={seed}, device={device}")
    data_handler = DataManager(dataset_name=dataset, batch_size=batch_size, data_fraction=data_fraction, seed=seed)
    logger.debug(f"Dataset loaded with {len(data_handler.train_set)} training samples and {len(data_handler.val_set)} validation samples.")
    experimenter = ResNetExperimenter(model_name=model, data_handler=data_handler, batch_size=batch_size, epochs=epochs, learning_rate=lr, device=device)
    logger.info("Model and data loaded, model fine-tuned.")
    if save:
        # Save finetuned original
        torch.save(experimenter.model.state_dict(), f"{save_dir}/{model}_original.pth")
        logger.info(f"Saved finetuned original model to {save_dir}/{model}_original.pth")

    # ------------------------------------------------------------
    # Evaluate initial model performance
    # ------------------------------------------------------------
    original_accuracy, original_param_count, original_inference_time, original_gflops = experimenter.validate_model()
    logger.info(f"Original model accuracy: {original_accuracy:.4f}, parameters: {original_param_count}, "
                f"inference time: {original_inference_time:.4f} seconds, tflops: {original_gflops}")

    # ------------------------------------------------------------
    # Compute linearity scores
    # ------------------------------------------------------------
    metric = LinearityMetric(linearity, model, data_handler, threshold, device, save, save_dir)
    linearity_scores = metric.metric_fn(experimenter.model)
    logger.info("Linearity scores computed.")
    logger.debug(f"Linearity scores: {linearity_scores}")
    linear_layers, nonlinear_layers = metric.thresholder(linearity_scores)
    logger.info(f"Determined linear layers: {linear_layers}")
    logger.info(f"Determined non-linear layers: {nonlinear_layers}")

    # ------------------------------------------------------------
    # Fold layers based on linearity scores
    # ------------------------------------------------------------
    folded_pairs = fold_linear_conv_sequences(experimenter.model, linear_layers)
    logger.debug(f"Folded layer pairs: {folded_pairs}")
    experimenter.finetune()
    logger.info(f"Folded {len(folded_pairs)} blocks in model and fine-tuned")

    # ------------------------------------------------------------
    # Evaluate folded model performance
    # ------------------------------------------------------------
    compressed_accuracy, compressed_param_count, compressed_inference_time, compressed_gflops = experimenter.validate_model()
    logger.info(f"Folded model accuracy: {compressed_accuracy:.4f}, parameters: {compressed_param_count}, "
                f"inference time: {compressed_inference_time:.4f} seconds, tflops: {compressed_gflops}")

    accuracy_loss = utils.accuracy_loss(original_accuracy, compressed_accuracy)
    param_compression_ratio = utils.compression_ratio(original_param_count, compressed_param_count)
    speedup = utils.speedup(original_inference_time, compressed_inference_time)
    gflop_reduction = utils.gflop_reduction(original_inference_time, compressed_inference_time)
    logger.info(f"Accuracy loss: {accuracy_loss:.4f}, Parameter compression ratio: {param_compression_ratio:.4f}, "
                f"Speedup: {speedup:.4f}x, TFLOP reduction: {gflop_reduction:.4f}")

    # ------------------------------------------------------------
    # Log results to wandb and save models/results if enabled
    # ------------------------------------------------------------
    if save:
        import json

        # Save folded model
        torch.save(experimenter.model.state_dict(), f"{save_dir}/{model}_folded.pth")

        # Save results
        results = {
            "original_accuracy": original_accuracy,
            "original_param_count": original_param_count,
            "original_inference_time": original_inference_time,
            "original_gflops": original_gflops,
            "compressed_accuracy": compressed_accuracy,
            "compressed_param_count": compressed_param_count,
            "compressed_inference_time": compressed_inference_time,
            "compressed_gflops": compressed_gflops,
            "compressed_groups": folded_pairs,
            "accuracy_loss": accuracy_loss,
            "param_compression_ratio": param_compression_ratio,
            "speedup": speedup,
            "gflop_reduction": gflop_reduction,
        }
        with open(f"{save_dir}/{model}_folding_results.json", "w") as f:
            json.dump(results, f, indent=4)
        logger.info(f"Saved folded model and results to {save_dir}/{model}_folded.pth and {save_dir}/{model}_folding_results.json")

    logging_data = {
        "original_accuracy": original_accuracy,
        "original_param_count": original_param_count,
        "original_inference_time": original_inference_time,
        "original_gflops": original_gflops,
        "compressed_accuracy": compressed_accuracy,
        "compressed_param_count": compressed_param_count,
        "compressed_inference_time": compressed_inference_time,
        "compressed_gflops": compressed_gflops,
        "compressed_groups": folded_pairs,
        "accuracy_loss": accuracy_loss,
        "param_compression_ratio": param_compression_ratio,
        "speedup": speedup,
        "gflop_reduction": gflop_reduction,
    }

    if sweep:
        # Compute separate metrics that can all be maximized
        accuracy_retention = compressed_accuracy / original_accuracy
        compression_ratio = original_param_count / compressed_param_count
        speedup = original_inference_time / compressed_inference_time

        # Edit these weights as needed to balance importance of metrics
        alpha, beta, gamma = 4, 1, 1

        # Compute combined metric
        compression_score = np.pow(accuracy_retention, alpha) * np.pow(compression_ratio, beta) * np.pow(speedup, gamma)

        # Add to data
        logging_data["compression_score"] = compression_score

    wandb.log(logging_data)
    logger.info("Logged results to Weights & Biases")
    logger.info("Experiment completed.")


