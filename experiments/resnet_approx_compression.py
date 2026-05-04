import os
from functools import reduce
from typing import Tuple, Optional

import numpy as np
import torch
import logging
import wandb
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.data_manager import DataManager
from utils.resnet_model import ResNetExperimenter
from metrics.linearity_metric_manager import LinearityMetric
import utils.util_functions as utils

logger = logging.getLogger(__name__)
debug_mode = logger.getEffectiveLevel() != logging.DEBUG

def group_contiguous_layers(linear_layers, all_layers, model):
    """
    Groups contiguous layeri.j.convk where i is the layer index, j is the block index in the layer, and k is the conv layer index in the block.
    Returns a list of lists of layer indices.
    Args:
        linear_layers (dict): A dictionary of layer names to linearity scores for layers identified as linear.
        all_layers (list): A list of lists of layer names to check for contiguity.
    Returns:
        list of lists: A list where each element is a list of contiguous layer indices that are linear.
    """
    # ------------------------------------------
    # 1. Get sorted candidate layers
    # ------------------------------------------
    indices = sorted(list(linear_layers.keys()))
    ground_truth = sorted(all_layers)

    # ------------------------------------------
    # 2. Basic contiguous grouping (unchanged logic)
    # ------------------------------------------
    raw_groups = []
    current = [indices[0]]

    for prev, curr in zip(indices, indices[1:]):
        if ground_truth.index(prev) + 1 == ground_truth.index(curr):
            current.append(curr)
        else:
            raw_groups.append(current)
            current = [curr]

    raw_groups.append(current)

    # ------------------------------------------
    # 3. Infer conv structure per ResNet block
    # ------------------------------------------
    block_to_convs = {}

    for name, module in model.named_modules():
        # only care about conv layers inside blocks
        if "conv" in name:
            block = ".".join(name.split(".")[:2])  # e.g. layer2.0
            block_to_convs.setdefault(block, set()).add(name)

    # ------------------------------------------
    # 4. Convert raw groups into block-valid groups
    # ------------------------------------------
    final_groups = []

    for group in raw_groups:

        # group by block
        block_map = {}

        for layer in group:
            block = ".".join(layer.split(".")[:2])
            block_map.setdefault(block, []).append(layer)

        for block, layers in block_map.items():

            expected_convs = block_to_convs.get(block, set())
            covered_convs = set(layers)

            # --------------------------------------------------
            # ONLY GROUP if full block is covered
            # --------------------------------------------------
            if expected_convs and covered_convs == expected_convs:
                final_groups.append(layers)
            else:
                # singleton behavior
                for l in layers:
                    final_groups.append([l])

    logger.info(f"Found groups: {final_groups}")

    return final_groups

class LinearConvolutionalBlock(nn.Module):
    def __init__(self, in_shape, out_shape, kernel_size=3):
        """
        in_shape:  (C_in, H_in, W_in)
        out_shape: (C_out, H_out, W_out)
        """
        super().__init__()

        _, self.C_in, self.H_in, self.W_in = in_shape
        _, self.C_out, self.H_out, self.W_out = out_shape
        self.kernel_size = kernel_size

        # Infer stride from spatial sizes
        assert self.H_in % self.H_out == 0
        assert self.W_in % self.W_out == 0

        self.stride_h = self.H_in // self.H_out
        self.stride_w = self.W_in // self.W_out

        # Unfold extracts sliding patches
        self.unfold = nn.Unfold(
            kernel_size=kernel_size,
            stride=(self.stride_h, self.stride_w),
            padding=kernel_size // 2
        )

        # Each patch is flattened to C_in * k * k
        self.in_features = self.C_in * kernel_size * kernel_size
        self.out_features = self.C_out

        # Linear mapping per spatial location
        self.linear = nn.Linear(self.in_features, self.out_features)

    def forward(self, x):
        """
        x: (B, C_in, H_in, W_in)
        returns: (B, C_out, H_out, W_out)
        """
        B = x.shape[0]

        # Extract patches
        patches = self.unfold(x)
        # (B, C_in * k*k, L) where L = H_out * W_out

        patches = patches.transpose(1, 2)
        # (B, L, C_in * k*k)

        # Apply linear mapping
        out = self.linear(patches)
        # (B, L, C_out)

        # Reshape to image grid
        out = out.transpose(1, 2)
        # (B, C_out, L)

        out = out.view(B, self.C_out, self.H_out, self.W_out)

        return out

class IdentityBlock(torch.nn.Module):
    """We need a separate IdentityBlock class to ensure we can handle additional arguments from forward pass"""
    def forward(self, hidden_states, **kwargs):
        return hidden_states

def replace_attention_block(model, layer_group, linear_block):
    """
    Replace the existing convolutional blocks of ResNet with the new linear approximation
    Args:
        model: The ResNet model to modify in place.
        layer_group: A list of layer names that must be replaced
        linear_block: The Linear Block to replace.
    Returns:
        None. Model is modified in place.
    """
    if len(layer_group) == 1:
        parts = layer_group[0].split(".")
        parent = reduce(getattr, parts[:-1], model)
        logger.info(f"Replacing {'.'.join(parts)} with linear block")
        setattr(parent, parts[-1], linear_block)
    else:
        # Replace the whole block
        parts = layer_group[0].split(".")[:2]
        parent = reduce(getattr, parts[:-1], model)
        # replace entire block
        logger.info(f"Replacing {'.'.join(parts)} with linear block")
        setattr(parent, parts[-1], linear_block)

def get_block_input_output(model, model_input, layer_group, device="cuda") -> Tuple[torch.Tensor, torch.Tensor]:
    """Gets the input and output of the specified convolutional layers for training the linear approximation.
    Args:
        model: The ResNet model to modify.
        model_input: The model input to pass through the model
        layer_group: A list of layer names that form a contiguous block to replace.
        device: The device to use.
    Returns:
        group_input: The embedding that gets passed into the block
        group_output: The embedding that gets passed out of the block
    """

    group_input = torch.empty(0) # Placeholder to ensure variable is defined in scope for hook
    group_output = torch.empty(0)

    def input_hook(module, input, output):
        nonlocal group_input
        group_input = input[0].detach()

    def output_hook(module, input, output):
        nonlocal group_output
        group_output = output.detach()

    hooks = [model.get_submodule(layer_group[0]).register_forward_hook(input_hook),
             model.get_submodule(layer_group[-1]).register_forward_hook(output_hook)]

    with torch.no_grad():
        model(model_input)

    for hook in hooks:
        hook.remove()

    return group_input, group_output

def train_block_approximation(
    model,
    layer_group,
    train_dataset,
    device,
    epochs=1,
    lr=2e-4,
    batch_size=64,
):
    """
    Trains a linear approximation layer to mimic a section of a LLama model's attention blocks.
    Args:
        model: The LLaMA model to modify.
        layer_group: A list of layer indices that form a contiguous block to replace.
        train_dataset: The training dataset.
        device: The device to use.
        epochs: The number of epochs to train.
        lr: The learning rate to use.
        batch_size: The batch size for training and evaluation.
    Returns:
        The linear approximation layer trained to mimic the specified attention block layers.
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=4,              # try 2–8 depending on CPU
                              pin_memory=True,            # important for GPU transfer
                              prefetch_factor=2,          # batches per worker
                              persistent_workers=True     # avoids worker restart each epoch
                              )
    model.eval().to(device)

    # Get input size of first layer in group and output of last layer in group
    single_data_point = next(iter(train_loader))[0].to(device) # Get a single data point to determine input and output sizes for the linear block
    x, y_teacher = get_block_input_output(model, single_data_point, layer_group, device=device)
    in_shape = x.shape
    out_shape = y_teacher.shape
    logger.debug(f"Training block approximation for layer group {layer_group} with input size {in_shape} and output size {out_shape}")

    approx = LinearConvolutionalBlock(in_shape, out_shape).to(device)
    approx.train().to(device)

    optimizer = torch.optim.AdamW(approx.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    scaler = torch.amp.GradScaler(device)


    for epoch in range(epochs):
        for i, data in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Training block {layer_group} Epoch {epoch+1}", leave=False, disable=debug_mode):

            inputs, _ = data
            inputs = inputs.to(device)

            x, y_teacher = get_block_input_output(model, inputs, layer_group, device=device)

            y_student = approx(x)
            logger.debug(f"y_student shape: {y_student.shape}, y_teacher shape: {y_teacher.shape}")
            logger.debug(f"y_student NaN: {torch.any(torch.isnan(y_student))}, y_teacher NaN: {torch.any(torch.isnan(y_teacher))}")
            loss = loss_fn(y_student, y_teacher)

            optimizer.zero_grad()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        logger.info(f"Block {layer_group} | Epoch {epoch} | Loss {loss.item():.6f}")
    return approx

def train_approximation_layers(experimenter, data_handler, groups,
                               epochs: int, lr: float, batch_size:int, device: str):
    """Train linear approximations for specified layer groups in the model.
    Args:
        experimenter: The ResNetExperimenter instance containing the model to be compressed and its tokenizer.
        data_handler: The DataHandler instance containing the dataset
        groups: List of layer groups to approximate.
        epochs (int): Number of epochs to train.
        lr (float): Learning rate for training.
        device (str): The device to run the training on (e.g., 'cuda' or 'cpu').
    Returns:
        The compressed model with linear approximations.
    """
    replacements = []
    for layer_group in groups:
        logger.info(f"Training approximation for layer group: {layer_group}")
        linear_block = train_block_approximation(
            experimenter.model,
            layer_group,
            data_handler.train_set,
            device,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size
        )
        replacements.append((experimenter.model, layer_group, linear_block))
    for replacement in replacements:
        replace_attention_block(*replacement)

def run_experiment(model: str, linearity: str, dataset: str, threshold: str, batch_size: int,
                   epochs: int, lr: float, data_fraction: float, save: bool, seed: int, device: str,
                   skip_finetune_path: Optional[str], sweep: bool=False):
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
        skip_finetune_path (str): The path to look for a finetuned model saved to disk if skipping is enabled.
        sweep (bool): Flag that indicates whether an additional metric should be computed to use for a W&B sweep.
    """
    save_dir = "./results/rq1/approx/" + linearity + "/" + threshold.split(".")[-1].split("%")[0] + "/" + model + "/" + dataset + "/" + str(seed)
    os.makedirs(save_dir, exist_ok=True)

    # ------------------------------------------------------------
    # Load data and model
    # ------------------------------------------------------------
    logger.info(f"Running ResNet compression experiment with model={model}, linearity={linearity}, dataset={dataset}, threshold={threshold}, batch_size={batch_size}, epochs={epochs}, lr={lr}, data fraction={data_fraction}, save={save}, seed={seed}, device={device}")
    data_handler = DataManager(dataset_name=dataset, batch_size=batch_size, data_fraction=data_fraction, seed=seed)
    logger.debug(f"Dataset loaded with {len(data_handler.train_set)} training samples and {len(data_handler.val_set)} validation samples.")
    experimenter = ResNetExperimenter(model_name=model, data_handler=data_handler, batch_size=batch_size, epochs=epochs,
                                      learning_rate=lr, device=device, skip_finetune_path=skip_finetune_path)
    logger.info("Model and data loaded, model fine-tuned.")
    if save and not experimenter.skipped:
        # Save finetuned original
        torch.save(experimenter.model.state_dict(), f"{save_dir}/{model}_original.pth")
        logger.info(f"Saved finetuned original model to {save_dir}/{model}_original.pth")
    elif save:
        logger.info(f"Skipped saving finetuned model as it was loaded from disk. Loaded from {skip_finetune_path}.")

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
    # Group contiguous linear layers and create linear approximation layers
    # ------------------------------------------------------------
    all_layers = list(linear_layers.keys()) + list(nonlinear_layers.keys())
    groups = group_contiguous_layers(linear_layers, all_layers, experimenter.model)
    train_approximation_layers(experimenter, data_handler, groups, epochs=epochs, lr=lr,
                               batch_size=batch_size, device=device)
    logger.info("Linear approximation layers trained and integrated into the model.")

    # ------------------------------------------------------------
    # Finetune the compressed model
    # ------------------------------------------------------------
    experimenter.finetune()

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
        torch.save(experimenter.model.state_dict(), f"{save_dir}/{model}_compressed.pth")

        # Save results
        results = {
            "original_accuracy": original_accuracy,
            "original_param_count": original_param_count,
            "original_inference_time": original_inference_time,
            "compressed_accuracy": compressed_accuracy,
            "compressed_param_count": compressed_param_count,
            "compressed_inference_time": compressed_inference_time,
            "compressed_groups": groups,
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
        "compressed_accuracy": compressed_accuracy,
        "compressed_param_count": compressed_param_count,
        "compressed_inference_time": compressed_inference_time,
        "compressed_groups": groups,
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