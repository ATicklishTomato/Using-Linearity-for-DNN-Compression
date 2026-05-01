import os
from functools import reduce
from typing import Tuple

import numpy as np
import torch
import logging
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.data_manager import DataManager
from utils.resnet_model import ResNetExperimenter
from metrics.linearity_metric_manager import LinearityMetric
import utils.util_functions as utils

logger = logging.getLogger(__name__)
debug_mode = logger.getEffectiveLevel() != logging.DEBUG

def group_contiguous_layers(linear_layers, all_layers):
    """
    Groups contiguous layeri.j.convk where i is the layer index, j is the block index in the layer, and k is the conv layer index in the block.
    Returns a list of lists of layer indices.
    Args:
        linear_layers (dict): A dictionary of layer names to linearity scores for layers identified as linear.
        all_layers (list): A list of lists of layer names to check for contiguity.
    Returns:
        list of lists: A list where each element is a list of contiguous layer indices that are linear.
    """
    # get (i,j,k) tuples
    indices = sorted(list(linear_layers.keys()))
    ground_truth = sorted(all_layers)

    groups = []
    current = [indices[0]]

    for prev, curr in zip(indices, indices[1:]):
        prev_idx = ground_truth.index(prev)
        curr_idx = ground_truth.index(curr)
        if prev_idx + 1 == curr_idx:
            current.append(curr)
        else:
            groups.append(current)
            current = [curr]

    groups.append(current)
    logger.debug(f"Grouped contiguous layers: {groups}")
    return groups

class LinearConvolutionalBlock(torch.nn.Module):
    """A simple Linear Convolutional Block that can be trained to mimic parts of a model that behave largely linearly"""
    def __init__(self, hidden_size_in, hidden_size_out):
        super().__init__()
        logger.debug(f"Initializing Linear Convolutional Block with input size {hidden_size_in} and output size {hidden_size_out}")
        self.linear = torch.nn.Linear(hidden_size_in, hidden_size_out, dtype=torch.float32) # Initialize as fp32 first

    def forward(self, hidden_states, **kwargs):
        return self.linear(hidden_states)

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
    first = layer_group[0] # e.g. layer1.0.conv1

    # Replace the first layer in the group with the trained linear block
    parts = first.split(".")
    parent = reduce(getattr, parts[:-1], model)
    setattr(parent, parts[-1], linear_block)

    # Replace other parts with linear block
    for layer in layer_group[1:]:
        parts = layer.split(".")
        parent = reduce(getattr, parts[:-1], model)
        setattr(parent, parts[-1], IdentityBlock()) # replace with identity since the first linear block should already capture the behavior of the group

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
        group_output = output[0].detach()

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
    # Get input size of first layer in group and output of last layer in group
    hidden_size_in = model.get_submodule(layer_group[0]).in_channels * model.get_submodule(layer_group[0]).kernel_size[0] * model.get_submodule(layer_group[0]).kernel_size[1]
    hidden_size_out = model.get_submodule(layer_group[-1]).out_channels * model.get_submodule(layer_group[-1]).kernel_size[0] * model.get_submodule(layer_group[-1]).kernel_size[1]
    logger.debug(f"Training block approximation for layer group {layer_group} with input size {hidden_size_in} and output size {hidden_size_out}")

    approx = LinearConvolutionalBlock(hidden_size_in, hidden_size_out).to(device)

    optimizer = torch.optim.AdamW(approx.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    scaler = torch.amp.GradScaler(device)

    model.eval().to(device)
    approx.train().to(device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        for i, data in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Training block {layer_group} Epoch {epoch+1}", leave=False, disable=debug_mode):

            inputs, _ = data
            inputs= inputs.to(device)

            x, y_teacher = get_block_input_output(model, inputs, layer_group, device=device)

            y_student = approx(x)
            logger.debug(f"y_student shape: {y_student.shape}, y_teacher shape: {y_teacher.shape}")
            logger.debug(f"y_student NaN: {torch.any(torch.isnan(y_student))}, y_teacher NaN: {torch.any(torch.isnan(y_teacher))}")
            loss = loss_fn(y_student, y_teacher.unsqueeze(0))

            optimizer.zero_grad()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        logger.info(f"Block {layer_group} | Epoch {epoch} | Loss {loss.item():.6f}")

    if model.config.dtype == torch.float16:
        logger.debug("Reducing dtype to float16 to match model")
        approx = approx.half()
    return approx

def train_approximation_layers(experimenter, data_handler, groups, save_model: bool,
                               epochs: int, lr: float, batch_size:int, device: str, save_path: str = None):
    """Train linear approximations for specified layer groups in the model.
    Args:
        experimenter: The ResNetExperimenter instance containing the model to be compressed and its tokenizer.
        data_handler: The DataHandler instance containing the dataset
        groups: List of layer groups to approximate.
        save_model (bool): Whether to save the compressed model to disk.
        epochs (int): Number of epochs to train.
        lr (float): Learning rate for training.
        device (str): The device to run the training on (e.g., 'cuda' or 'cpu').
        save_path (str): Path to save the compressed model to disk.
    Returns:
        The compressed model with linear approximations.
    """
    if save_path is None:
        save_path = "./results"

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
        replace_attention_block(experimenter.model, layer_group, linear_block)

    if save_model:
        # Save the compressed model
        experimenter.model.save_pretrained(f"{save_path}/compressed_{experimenter.model_name}")
        logger.info(f"Compressed model saved to {save_path}/compressed_{experimenter.model_name}")

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
    # Group contiguous linear layers and create linear approximation layers
    # ------------------------------------------------------------
    all_layers = list(linear_layers.keys()) + list(nonlinear_layers.keys())
    groups = group_contiguous_layers(linear_layers, all_layers)
    train_approximation_layers(experimenter, data_handler, groups, save_model=save,
                               epochs=epochs, lr=lr, batch_size=batch_size, device=device, save_path=save_dir)
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
        torch.save(experimenter.model.state_dict(), f"{save_dir}/{model}_merged.pth")

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