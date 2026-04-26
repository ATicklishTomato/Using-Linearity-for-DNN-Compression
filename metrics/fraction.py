import os
import re
import logging
import json
import torch
from torch.utils.data import DataLoader
from torchvision.models import ResNet
from tqdm import tqdm

logger = logging.getLogger(__name__)
debug_mode = logger.getEffectiveLevel() != logging.DEBUG

def hook_fn(output, linear_outputs, all_ouputs, name):
    """Collect partial results for fraction of neuron activation linearity metric.
    Args:
        output: The output of the neural network model.
        linear_outputs: Dictionary with layer names as keys and output values as values.
        all_ouputs: Dictionary with layer names as keys and output values as values.
        name: The name of the metric.
    """
    if isinstance(output, tuple):
        output = output[0]

    # Count the values in the output that are greater than 0 (i.e., activated neurons)
    activated_neurons = (output > 0).sum().item()
    total_neurons = output.numel()

    logger.debug(f"Found {activated_neurons} neurons in {name} out of {total_neurons} neurons.")

    if name not in linear_outputs.keys():
        linear_outputs[name] = activated_neurons
    else:
        linear_outputs[name] += activated_neurons
    if name not in all_ouputs.keys():
        all_ouputs[name] = total_neurons
    else:
        all_ouputs[name] += total_neurons

def fraction_of_activation(model, data_handler, device='cuda', save=False, save_dir="./results"):
    """Generic function that attempts to encompass both ResNet and Llama. The function attempts to compute the fraction of
    Args:
        model: The neural network model.
        data_handler: The DataManager instance that provides access to the dataset and tokenizer (if applicable).
        device: Device to run the computations on.
        save: Whether to save/load the computed mean preactivations to/from disk.
        save_dir: Path to save the computed mean preactivations to/from disk.
    Returns:
        dict: A dictionary with layer names as keys and mean preactivation values as values.
    """
    logger.info("Starting to compute fraction of activations.")
    is_resnet = isinstance(model, ResNet)
    target_layer_pattern = re.compile(r"^layer\d+\.\d+\.conv\d+$") if is_resnet else re.compile(r"^model\.layers.\d+\.self_attn$")
    logger.info(f"target_layer_pattern: {target_layer_pattern}")

    model.eval().to(device)
    dataset = data_handler.val_set
    hooks = []
    linear_outputs = {}
    all_outputs = {}
    logger.info("Registering hooks.")
    for name, module in model.named_modules():
        if target_layer_pattern.match(name):
            hooks.append(module.register_forward_hook(lambda module, input, output, name=name: hook_fn(output, linear_outputs, all_outputs, name)))

    logger.info("Hooks registered. Starting forward pass.")

    with torch.no_grad():
        if is_resnet:
            data_loader = DataLoader(dataset, batch_size=data_handler.batch_size, shuffle=False, num_workers=0,
                                     pin_memory=True)
            for inputs, _ in tqdm(data_loader, total=len(dataset), desc="Computing mean preactivations", leave=False, disable=debug_mode):
                inputs = inputs.to(device)
                model(inputs)
        else:
            for i in tqdm(range(len(dataset)), desc="Processing samples for preactivations", leave=False, disable=debug_mode):
                inputs = data_handler.tokenizer(dataset[i]['text'], return_tensors='pt', truncation=True, padding='max_length', max_length=512)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    model(**inputs)

    logger.info("Finished forward pass. Removing hooks.")

    # Remove hooks
    for hook in hooks:
        hook.remove()

    logger.info("Hooks removed. Calculating final activation fraction.")

    activation_fractions = {}

    for key, value in linear_outputs.items():
        activation_fractions[key] = value / all_outputs[key]

    if save:
        json.dump(activation_fractions, open(os.path.join(save_dir, "activation_fractions.json"), "w"))
        logger.info(f"Saved activation_fractions to {os.path.join(save_dir, 'activation_fractions.json')}")

    logger.info("Finished computing fraction of activations.")

    return activation_fractions