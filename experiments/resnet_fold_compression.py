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

def merge_linear_conv_sequences(
    model,
    linear_layers,
    device='cuda'
):
    """
    Merge Conv-BN-ReLU-Conv sequences when the activation is near-linear.

    Args:
        model (nn.Module): ResNet-like model (unchanged architecture)
        linear_layers (dict): {conv_layer_name: linearity_score}
        device (str): Device to perform merging on (e.g., 'cpu', 'cuda')

    Returns:
        merged_model (nn.Module)
        merged_pairs (list of tuples)
    """
    merged_pairs = []
    model.to(device)

    logger.info("\n[Layer Merging] Starting merging pass")

    # ------------------------------------------------------------
    # Helper: merge BN into Conv
    # ------------------------------------------------------------
    def merge_bn_into_conv(conv, bn):
        """
        Merge a batchnorm layer into its own conv layer by applying its scaling and shifting to the conv weights and biases.
        This effectively removes the batchnorm layer while preserving the same transformations.
        Args:
            conv (nn.Conv2d): Convolutional layer
            bn (nn.BatchNorm2d): Batchnorm layer
        Returns:
            None. Convolutional layer is modified in place
        """
        logger.debug(f"    Mergeing BatchNorm into Conv ({conv.out_channels} channels)")

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
        W_merged = W * (gamma / std).reshape(-1, 1, 1, 1)
        b_merged = beta + (bias - mean) * gamma / std

        conv.weight.data.copy_(W_merged)
        conv.bias = nn.Parameter(b_merged)

    # ------------------------------------------------------------
    # Helper: merge Conv → Conv
    # ------------------------------------------------------------
    def merge_convs(conv1, conv2):
        """
        Merges to convolutional layers into each other, creating one big convolutional layer that performs the same transformations as the two in direct sequence.
        Args:
            conv1 (nn.Conv2d): Convolutional layer
            conv2 (nn.Conv2d): Convolutional layer
        Returns:
            new_conv (nn.Conv2d): New convolutional layer that performs the same transformations.
        """
        logger.debug(
            f"    Mergeing Conv layers: "
            f"{conv1.in_channels}→{conv1.out_channels}→{conv2.out_channels}"
        )

        W1 = conv1.weight.detach()  # [C_mid, C_in, k1, k1]
        W2 = conv2.weight.detach()  # [C_out, C_mid, k2, k2]

        C_mid, C_in, k1, _ = W1.shape
        C_out, C_mid2, k2, _ = W2.shape

        k_merge = k1 + k2 - 1

        # ---------------------------------------------------------
        # Flip W2 because conv2d does cross-correlation
        # Need true kernel convolution for composition
        # ---------------------------------------------------------
        W2_flip = torch.flip(W2, dims=(-1, -2))

        # ---------------------------------------------------------
        # Vectorized kernel composition
        #
        # Treat each (out_channel, mid_channel) kernel in W2 as a
        # conv filter applied to all matching W1[mid_channel, in_channel]
        # ---------------------------------------------------------
        W_merge = torch.zeros(
            (C_out, C_in, k_merge, k_merge),
            device=W1.device,
            dtype=W1.dtype,
        )

        # Process each intermediate channel j
        for j in range(C_mid):
            # Input kernels from conv1:
            # [C_in, 1, k1, k1]
            x = W1[j].unsqueeze(1)

            # Filters from conv2:
            # [C_out, 1, k2, k2]
            w = W2_flip[:, j].unsqueeze(1)

            # Apply all output filters to all input kernels
            # output: [C_in, C_out, k_merge, k_merge]
            out = F.conv2d(
                x,
                w,
                padding=k2 - 1
            )

            # Rearrange to [C_out, C_in, k_merge, k_merge]
            out = out.permute(1, 0, 2, 3)

            W_merge += out

        # ---------------------------------------------------------
        # Build merged conv
        # ---------------------------------------------------------
        new_conv = nn.Conv2d(
            in_channels=C_in,
            out_channels=C_out,
            kernel_size=k_merge,
            padding=k_merge // 2,
            bias=False,
        ).to(W1.device)
        new_conv.weight.data.copy_(W_merge)

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

            # Validate mergeability
            if (
                conv1.stride != (1, 1)
                or conv2.stride != (1, 1)
                or conv1.groups != 1
                or conv2.groups != 1
            ):
                logger.debug("    Stride/groups incompatible → skipping")
                continue

            logger.debug("    Linearity condition satisfied")
            logger.debug("    Removing BatchNorm and ReLU, merging Convs")

            # ----------------------------------------------------
            # Merge BN → Conv1
            # ----------------------------------------------------
            merge_bn_into_conv(conv1, bn1)

            # ----------------------------------------------------
            # Merge Conv1 → Conv2
            # ----------------------------------------------------
            new_conv = merge_convs(conv1, conv2)
            new_conv.to(device)

            # ----------------------------------------------------
            # Replace modules
            # ----------------------------------------------------
            setattr(module, 'conv1', new_conv)
            setattr(module, 'bn1', nn.Identity())
            setattr(module, 'relu', nn.Identity())
            setattr(module, 'conv2', nn.Identity())

            merged_pairs.append((conv1_name, f"{module_name}.{idx}.conv2"))

            logger.debug("    Merging complete")

    logger.info(f"\n[Layer Merging] Done. Merged {len(merged_pairs)} layer pairs.\n")

    return merged_pairs

def run_experiment(model: str, linearity: str, dataset: str, threshold: str, batch_size: int,
                           epochs: int, lr: float, data_fraction: float, save: bool, seed: int, device: str, sweep: bool=False):
    """Run ResNet compression experiment with layer merging. Results are logged and stored to wandb if enabled, and models/results are saved to ./results if enabled.
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
    save_dir = "./results/rq1/" + linearity + "/" + threshold.split(".")[-1].split("%")[0] + "/" + model + "/" + dataset + "/" + str(seed)
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
                f"inference time: {original_inference_time:.4f} seconds, gflops: {original_gflops}")

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
    # Merge layers based on linearity scores
    # ------------------------------------------------------------
    merged_pairs = merge_linear_conv_sequences(experimenter.model, linear_layers)
    logger.debug(f"Merged layer pairs: {merged_pairs}")
    experimenter.finetune()
    logger.info(f"Merged {len(merged_pairs)} blocks in model and fine-tuned")

    # ------------------------------------------------------------
    # Evaluate merged model performance
    # ------------------------------------------------------------
    compressed_accuracy, compressed_param_count, compressed_inference_time, compressed_gflops = experimenter.validate_model()
    logger.info(f"Merged model accuracy: {compressed_accuracy:.4f}, parameters: {compressed_param_count}, "
                f"inference time: {compressed_inference_time:.4f} seconds, gflops: {compressed_gflops}")

    accuracy_loss = utils.accuracy_loss(original_accuracy, compressed_accuracy)
    param_compression_ratio = utils.compression_ratio(original_param_count, compressed_param_count)
    speedup = utils.speedup(original_inference_time, compressed_inference_time)
    gflop_reduction = utils.gflop_reduction(original_gflops, compressed_gflops)
    logger.info(f"Accuracy loss: {accuracy_loss:.4f}, Parameter compression ratio: {param_compression_ratio:.4f}, "
                f"Speedup: {speedup:.4f}x, GFLOP reduction: {gflop_reduction:.4f}")

    # ------------------------------------------------------------
    # Log results to wandb and save models/results if enabled
    # ------------------------------------------------------------
    if save:
        import json

        # Save merged model
        torch.save(experimenter.model.state_dict(), f"{save_dir}/{model}_merged.pth")

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
            "compressed_groups": merged_pairs,
            "accuracy_loss": accuracy_loss,
            "param_compression_ratio": param_compression_ratio,
            "speedup": speedup,
            "gflop_reduction": gflop_reduction,
        }
        with open(f"{save_dir}/{model}_merging_results.json", "w") as f:
            json.dump(results, f, indent=4)
        logger.info(f"Saved merged model and results to {save_dir}/{model}_merged.pth and {save_dir}/{model}_merging_results.json")

    logging_data = {
        "original_accuracy": original_accuracy,
        "original_param_count": original_param_count,
        "original_inference_time": original_inference_time,
        "original_gflops": original_gflops,
        "compressed_accuracy": compressed_accuracy,
        "compressed_param_count": compressed_param_count,
        "compressed_inference_time": compressed_inference_time,
        "compressed_gflops": compressed_gflops,
        "compressed_groups": merged_pairs,
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


