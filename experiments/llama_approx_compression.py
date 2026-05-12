from typing import Tuple, Optional

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import logging
from metrics.linearity_metric_manager import LinearityMetric
from utils.data_manager import DataManager
from utils.llama_model import LlamaExperimenter
import utils.util_functions as utils

logger = logging.getLogger(__name__)
debug_mode = logger.getEffectiveLevel() != logging.DEBUG

def group_contiguous_layers(linear_layers):
    """
    Groups contiguous model.layers[i].self_attn modules.
    Returns a list of lists of layer indices.
    Args:
        linear_layers (dict): A dictionary of layer names to linearity scores for layers identified as linear.
    Returns:
        list of lists: A list where each element is a list of contiguous layer indices that are linear.
    """
    indices = sorted(
        int(layer.split(".")[2])
        for layer in linear_layers.keys()
    )

    if len(indices) <= 0:
        return []

    groups = []
    current = [indices[0]]

    for prev, curr in zip(indices, indices[1:]):
        if curr == prev + 1:
            current.append(curr)
        else:
            groups.append(current)
            current = [curr]

    groups.append(current)
    logger.debug(f"Grouped contiguous layers: {groups}")
    return groups

class LinearAttentionBlock(torch.nn.Module):
    """A simple Linear Attention Block that can be trained to mimic parts of a model that behave largely linearly"""
    def __init__(self, hidden_size):
        super().__init__()
        logger.debug(f"Initializing Linear Attention Block with hidden size {hidden_size}")
        self.linear = torch.nn.Linear(hidden_size, hidden_size, dtype=torch.float32) # Initialize as fp32 first

    def forward(self, hidden_states, **kwargs):
        return self.linear(hidden_states)

class IdentityBlock(torch.nn.Module):
    """We need a separate IdentityBlock class to ensure we can handle additional arguments from LLama forward pass"""
    def forward(self, hidden_states, **kwargs):
        return hidden_states

def replace_attention_block(model, layer_group, linear_block):
    """
    Replace the existing attention blocks with their linear approximation.
    Args:
        model: The LLaMA model to modify.
        layer_group: A list of layer indices that form a contiguous block to replace.
        linear_block: The Linear Block to replace.
    Returns:
        None. Model is modified in place.
    """
    first = layer_group[0]

    # Replace first layer with trained linear block
    model.model.layers[first] = linear_block

    # Replace remaining layers with identity modules
    for layer_id in layer_group[1:]:
        model.model.layers[layer_id] = IdentityBlock()

def get_block_input_output(model, model_inputs, layer_group, device="cuda") -> Tuple[torch.Tensor, torch.Tensor]:
    """Gets the input and output of the specified attention block layers for training the linear approximation.
    Args:
        model: The LLaMA model to modify.
        model_inputs: The model inputs to pass through the model
        layer_group: A list of layer indices that form a contiguous block to replace.
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

    hooks = [model.model.layers[layer_group[0]].register_forward_hook(input_hook),
             model.model.layers[layer_group[-1]].register_forward_hook(output_hook)]

    with torch.no_grad():
        with torch.amp.autocast(device):
            model(**model_inputs)

    for hook in hooks:
        hook.remove()

    return group_input, group_output


def train_block_approximation(
    model,
    layer_group,
    data_handler,
    device,
    epochs=1,
    lr=2e-4
):
    """
    Trains a linear approximation layer to mimic a section of a LLama model's attention blocks.
    Args:
        model: The LLaMA model to modify.
        layer_group: A list of layer indices that form a contiguous block to replace.
        data_handler: The DataManager instance containing the dataset and its tokenizer.
        device: The device to use.
        epochs: The number of epochs to train.
        lr: The learning rate to use.
    Returns:
        The linear approximation layer trained to mimic the specified attention block layers.
    """
    hidden_size = model.config.hidden_size
    approx = LinearAttentionBlock(hidden_size).to(device)

    optimizer = torch.optim.AdamW(approx.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    scaler = torch.amp.GradScaler(device)
    train_loader = DataLoader(data_handler.train_set, batch_size=data_handler.batch_size, shuffle=True,
                              num_workers=4,  # try 2–8 depending on CPU
                              pin_memory=True,  # important for GPU transfer
                              prefetch_factor=2,  # batches per worker
                              persistent_workers=True  # avoids worker restart each epoch
                              )

    model.eval().to(device)
    approx.train().to(device)

    for epoch in range(epochs):
        for i, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Training block {layer_group} Epoch {epoch+1}", leave=False, disable=debug_mode):

            inputs = data_handler.tokenizer(
                batch["text"],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(device)


            x, y_teacher = get_block_input_output(model, inputs, layer_group, device=device)

            with torch.amp.autocast(device):
                y_student = approx(x)
                logger.debug(f"y_student shape: {y_student.shape}, y_teacher shape: {y_teacher.shape}")
                logger.debug(f"y_student NaN: {torch.any(torch.isnan(y_student))}, y_teacher NaN: {torch.any(torch.isnan(y_teacher))}")
                loss = loss_fn(y_student, y_teacher)

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
                               epochs: int, lr: float, device: str, save_path: str = None):
    """Train linear approximations for specified layer groups in the model.
    Args:
        experimenter: The LlamaExperimenter instance containing the model to be compressed and its tokenizer.
        data_handler: The DataHandler instance containing the dataset and its tokenizer.
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

    replacements = []
    for layer_group in groups:
        logger.info(f"Training approximation for layer group: {layer_group}")
        linear_block = train_block_approximation(
            experimenter.model,
            layer_group,
            data_handler,
            device,
            epochs=epochs,
            lr=lr
        )
        replacements.append((experimenter.model, layer_group, linear_block))
    for replacement in replacements:
        replace_attention_block(*replacement)

    if save_model:
        # Save the compressed model
        experimenter.model.save_pretrained(f"{save_path}/compressed_{experimenter.model_name}")
        logger.info(f"Compressed model saved to {save_path}/compressed_{experimenter.model_name}")


def run_experiment(model: str, linearity: str, dataset: str, threshold: str, batch_size: int,
                   epochs: int, lr: float, data_fraction: float, save: bool, seed: int, device: str,
                   skip_finetune_path: Optional[str], sweep: bool=False):
    """Run the Llama compression experiment. Results are logged and stored to wandb if enabled, and models/results are saved to ./results if enabled.
    Args:
        model (str): The ResNet architecture to use (e.g., 'llama-2-7b').
        linearity (str): The linearity metric to use (e.g., 'mean_preactivation', 'procrustes', or 'fraction').
        dataset (str): The dataset to use (e.g., 'imagenet').
        threshold (str): The threshold for determining linearity (e.g., '75%' or '-0.01').
        batch_size (int): The batch size for training and evaluation.
        epochs (int): The number of epochs for fine-tuning.
        lr (float): The learning rate for the optimizer.
        data_fraction (float): The fraction of the dataset to use for training and evaluation (e.g., 0.05 for 5%).
        save (bool): Whether to save the trained models and results.
        seed (int): The random seed for reproducibility.
        device (str): The device to run the experiments on (e.g., 'cpu', 'cuda').
        skip_finetune_path (str): The path to look for a finetuned model saved to disk if skipping is enabled.
        sweep (bool): Flag that indicates whether an additional metric should be computed to use for a W&B sweep.
    """
    save_dir = "./results/rq1/" + linearity + "/" + threshold.split(".")[-1].split("%")[0] + "/" + model + "/" + dataset + "/" + str(seed)
    os.makedirs(save_dir, exist_ok=True)

    # ------------------------------------------------------------
    # Load data and model
    # ------------------------------------------------------------
    logger.info(
        f"Running Llama compression experiment with model: {model}, linearity metric: {linearity}, dataset: {dataset}, threshold: {threshold}, batch size: {batch_size}, epochs: {epochs}, learning rate: {lr}, data fraction: {data_fraction}, save results: {save}, seed: {seed}, device: {device}")
    data_handler = DataManager(dataset_name=dataset, batch_size=batch_size, data_fraction=data_fraction, model_name=model, seed=seed)
    logger.debug(f"Dataset loaded with {len(data_handler.train_set)} training samples and {len(data_handler.val_set)} validation samples.")
    experimenter = LlamaExperimenter(model_name=model, data_handler=data_handler, batch_size=batch_size, epochs=epochs,
                                     learning_rate=lr, device=device, skip_finetune_path=skip_finetune_path)
    logger.info("Model initialized.")
    if save and not experimenter.skipped:
        # Save original finetuned model
        experimenter.model.save_pretrained(f"{save_dir}/original_{model}")
        logger.info(f"Original finetuned model saved to {save_dir}/original_{model}")
    elif save:
        logger.info(f"Skipping saving finetuned model as it was loaded from disk. Loaded from {skip_finetune_path}.")

    # ------------------------------------------------------------
    # Evaluate initial model performance
    # ------------------------------------------------------------
    original_accuracy, original_param_count, original_inference_time, original_gflops = experimenter.validate_model()
    logger.info(
        f"Original model accuracy: {original_accuracy:.4f}, parameters: {original_param_count}, "
        f"inference time: {original_inference_time:.4f} seconds, GFLOPs: {original_gflops:.4f}")

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
    groups = group_contiguous_layers(linear_layers)
    train_approximation_layers(experimenter, data_handler, groups, save_model=save,
                               epochs=epochs, lr=lr, device=device, save_path=save_dir)
    logger.info("Linear approximation layers trained and integrated into the model.")

    # ------------------------------------------------------------
    # Finetune the compressed model
    # ------------------------------------------------------------
    experimenter.finetune()

    # ------------------------------------------------------------
    # Evaluate compressed model performance
    # ------------------------------------------------------------
    compressed_accuracy, compressed_param_count, compressed_inference_time, compressed_gflops = experimenter.validate_model()
    logger.info(
        f"Compressed model accuracy: {compressed_accuracy:.4f}, parameters: {compressed_param_count}, "
        f"inference time: {compressed_inference_time:.4f} seconds, Gs: {compressed_gflops:.4f}")

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

        # Save compressed model
        experimenter.model.save_pretrained(f"{save_dir}/compressed_{model}")
        logger.info(f"Compressed model saved to {save_dir}/compressed_{model}")

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
        with open(f"{save_dir}/{model}_folding_results.json", "w") as f:
            json.dump(results, f, indent=4)
        logger.info(
            f"Saved compressed model and results to {save_dir}/{model}_compressed.pth and {save_dir}/{model}_folding_results.json")

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