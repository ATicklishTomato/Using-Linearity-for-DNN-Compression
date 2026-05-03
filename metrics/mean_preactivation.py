import json
import re
import logging
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)
debug_mode = logger.getEffectiveLevel() != logging.DEBUG


def llama_hook(module, input, output, channel_sums, sample_counts, name):
    """This function is to be used as a forward hook in Llama models.
    The hook performs part of the mean of preactivations calculation, as described by Pinson et al. (2024).

    The function updates channel_sums and sample_counts dictionaries in place, which store the running sum of preactivations and the count of samples for each layer, respectively.
    The mean preactivation for each layer can be computed after processing the dataset by dividing the channel_sums by the sample_counts for that layer.

    Args:
        module: The module to which the hook is registered.
        input: The input to the module during the forward pass.
        output: The output from the module during the forward pass.
        channel_sums: Channel sums dictionary that tracks the running sum of preactivations (value) for each layer (key).
        sample_counts: Sample counts dictionary that tracks the number of samples (value) processed for each layer (key).
        name: The name of the layer for which the hook is registered, used as a key in the channel_sums and sample_counts dictionaries.
    Returns:
        None. The function updates the channel_sums and sample_counts dictionaries in place.
        """
    # input[0] shape: [B, T, D]
    x = input[0].detach()

    B, T, D = x.shape

    # Mean over batch + sequence, keep hidden dimension
    per_dim_batch_mean = x.mean(dim=(0, 1))  # [D]

    if name not in channel_sums:
        channel_sums[name] = per_dim_batch_mean * (B * T)
        sample_counts[name] = B * T
    else:
        channel_sums[name] += per_dim_batch_mean * (B * T)
        sample_counts[name] += B * T

def llama_map(mean_preactivations):
    """Map the mean preactivations from the activation layers to the preceding self-attention layers in Llama.
    This is done because we care about the preactivations in the context of the associated self-attention layer.
    Args:
        mean_preactivations: A dictionary with layer names as keys and mean preactivation values as values, where the layer names correspond to the activation layers.
    Returns:
        dict: A dictionary with self-attention layer names as keys and mean preactivations as values.
    """
    mapped_mean_preactivations = {}
    for layer_name, mean_val in mean_preactivations.items():
        match = re.match(r'model\.layers\.(\d+)\.mlp\.act_fn', layer_name)
        if match:
            layer_num = match.group(1)
            mapped_mean_preactivations[f'model.layers.{layer_num}.self_attn'] = mean_val
            logger.debug(f"Mapped model.layers.{layer_num}.self_attn with mean preactivation {mean_val}")
    return mapped_mean_preactivations

def resnet_hook(module, input, output, channel_sums, sample_counts, name):
    """This function is to be used as a forward hook in ResNet models on BatchNorm2d layers (pre-ReLU).
    The hook performs part of the mean of preactivations calculation, as described by Pinson et al. (2024).

    The function updates channel_sums and sample_counts dictionaries in place, which store the running sum of preactivations and the count of samples for each layer, respectively.
    The mean preactivation for each layer can be computed after processing the dataset by dividing the channel_sums by the sample_counts for that layer.

    Args:
        module: The module to which the hook is registered.
        input: The input to the module during the forward pass.
        output: The output from the module during the forward pass.
        channel_sums: Channel sums dictionary that tracks the running sum of preactivations (value) for each layer (key).
        sample_counts: Sample counts dictionary that tracks the number of samples (value) processed for each layer (key).
        name: The name of the layer for which the hook is registered, used as a key in the channel_sums and sample_counts dictionaries.
    Returns:
        None. The function updates the channel_sums and sample_counts dictionaries in place.
        """
    if isinstance(input, tuple):
        # cifar10 has single item tuples around its embeddings for some reason
        input = input[0]
    # output shape: [B, C, H, W]
    B = input.shape[0]

    # spatial + batch mean, but keep channels
    per_channel_batch_mean = input.mean(dim=(0, 2, 3))  # [C]

    if name not in channel_sums:
        channel_sums[name] = per_channel_batch_mean.detach().clone() * B
        sample_counts[name] = B
    else:
        channel_sums[name] += per_channel_batch_mean.detach() * B
        sample_counts[name] += B

def resnet_map(model, mean_preactivations):
    """Map the mean preactivations from the BatchNorm2d layers to the preceding Conv2d layers in ResNet.
    This is done because we care about the preactivations in the context of the associated conv layer.
    Args:
        model: The ResNet model.
        mean_preactivations: A dictionary with layer names as keys and mean preactivation values as values, where the layer names correspond to the BatchNorm2d layers.
    Returns:
        dict: A dictionary with conv layer names as keys and mean preactivations as values.
    """
    mapped_mean_preactivations = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.BatchNorm2d) and "downsample" not in name:
            mean = mean_preactivations.get(name, 0.0)
            layer = name.split('bn')[0]  # Get the layer name before .bn
            index = name.split('bn')[-1]  # Get the index if present
            mapped_mean_preactivations[layer + 'conv' + index] = mean  # Copy to preceding Conv2d layer
        else:
            logger.info(f"Skipping layer {name} for mapping since it's not a BatchNorm2d layer or is a downsample layer.")
    return mapped_mean_preactivations

def mean_preactivations(model, data_handler, device='cuda', save=False, save_dir="./results"):
    """Generic function that attempts to encompass both ResNet and Llama. This is done to avoid code duplication.
    Args:
        model: The neural network model.
        data_handler: The DataManager instance that provides access to the dataset and tokenizer (if applicable).
        device: Device to run the computations on.
        save: Whether to save/load the computed mean preactivations to/from disk.
        save_dir: Path to save the computed mean preactivations to/from disk.
    Returns:
        dict: A dictionary with layer names as keys and mean preactivation values as values.
    """
    save_path = f"{save_dir}/mean_preactivations_{model.__class__.__name__}.pt"
    # if save and os.path.exists(save_path):
    #     logger.info("Loading mean preactivations from disk...")
    #     return torch.load(save_path)

    model.to(device)
    model.eval()
    dataset = data_handler.val_set

    # Store running sums and sample counts per layer
    channel_sums = {}   # layer_name -> tensor [D] for Llama, [C] for ResNet
    sample_counts = {} # layer_name -> int
    hooks = []
    activation_layers = []

    logger.info("Identifying activation layers and registering hooks...")

    # Identify activation layers and register hooks
    if "llama" in save_path.lower():
        # Identify activation layers for Llama
        for name, module in model.named_modules():
            if re.match(r'model\.layers\.\d+\.mlp\.act_fn', name) or re.match(r'model\.layers\.\d+\.post_attention_layernorm', name):
                activation_layers.append((name, module))
        logger.debug("Identified layers, setting hooks...")
        # Register hooks for Llama
        for name, module in activation_layers:
            hooks.append(module.register_forward_hook(lambda module, input, output, name=name: llama_hook(module, input, output, channel_sums, sample_counts, name)))
    else:
        # Identify activation layers for ResNet and register hooks
        for name, module in tqdm(model.named_modules(),
                                 desc="Registering hooks",
                                 leave=False, disable=debug_mode):
            if isinstance(module, torch.nn.BatchNorm2d) and "downsample" not in name:
                hooks.append(module.register_forward_hook(lambda module, input, output, name=name: resnet_hook(module, input, output, channel_sums, sample_counts, name)))

    logger.debug("Hooks registered. Performing forward passes...")

    # Forward pass through the data
    with torch.no_grad():
        if "llama" in save_path.lower():
            for i in tqdm(range(len(dataset)), desc="Processing samples for preactivations", leave=False, disable=debug_mode):
                inputs = data_handler.tokenizer(dataset[i]['text'], return_tensors='pt', truncation=True, padding='max_length', max_length=512)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    model(**inputs)
        else:
            data_loader = DataLoader(dataset, batch_size=data_handler.batch_size, shuffle=False, num_workers=0, pin_memory=True)
            for inputs, _ in tqdm(data_loader, total=len(dataset), desc="Computing mean preactivations", leave=False, disable=debug_mode):
                inputs = inputs.to(device)
                model(inputs)

    logger.debug("Forward passes complete. Computing mean preactivations...")
    # Compute mean preactivations
    mean_preactivations = {}
    for name in channel_sums:
        per_dim_mean = channel_sums[name] / sample_counts[name]  # [D] for Llama, [C] for ResNet
        mean_preactivations[name] = per_dim_mean.mean().item()   # scalar

    logger.debug("Mean preactivations computed.")

    # Remove hooks
    for hook in hooks:
        hook.remove()

    logger.debug("Hooks removed.")

    # Return model to cpu and free vram cache
    model.cpu()
    torch.cuda.empty_cache()

    if "llama" in save_path.lower():
        mean_preactivations = llama_map(mean_preactivations)
    else:
        mean_preactivations = resnet_map(model, mean_preactivations)

    if save:
        json.dump(mean_preactivations, open(f"{save_dir}/mean_preactivations.json", "w"))
        logger.info(f"Saved mean preactivations to {save_dir}/mean_preactivations.json")

    logger.info("Mean preactivations computed.")

    return mean_preactivations