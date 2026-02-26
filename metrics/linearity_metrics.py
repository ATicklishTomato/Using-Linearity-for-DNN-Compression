import re
import torch
import os
import logging

from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)

class LinearityMetric:

    def __init__(self, metric_name: str, model_name: str, data_handler, threshold, max_batches=None, device='cuda', save=False):
        """Encapsulating class that manages the application of the correct metric implementation for a model.

        Args:
            metric_name (str): The name of the linearity metric to compute. Supported metrics: "mean_preactivation", "procrustes", "fraction".
            model_name (str): The name of the model to compute the metric on. Supported models: "Llama-2-7b", "Llama-2-13b", "Resnet-18", "Resnet-34", "Resnet-50".
            data_handler: An instance of DataManager that provides access to the dataset and tokenizer (if applicable).
            threshold: The threshold to use for determining what is(n't) linear. To take a percentile, enter a percentage, e.g. `75%` to consider anything smaller the 75th percentile as non-linear. To take a hard threshold, enter a floating point value, e.g. `-0.01`. Default is 75th percentile.
            max_batches: Maximum number of batches to process during metric computation. If None, process all batches.
            device: Device to run the computations on (e.g., "cpu", "cuda").
            save: Whether to save the computed metric values to disk for faster loading in future runs. If True, the metric values will be saved to a file named `./results/{metric_name}_{model_name}.pt`. If such a file exists, the metric values will be loaded from the file instead of recomputing them.
        """

        self.metric_name = metric_name
        self.model_name = model_name
        self.data_handler = data_handler
        self.thresholder = self.threshold_fn(threshold)
        self.max_batches = max_batches if max_batches is not None else len(data_handler.val_set)
        self.device = device
        self.save = save

        match (model_name, metric_name):
            case ("llama7b", "mean_preactivation") | ("llama13b", "mean_preactivation"):
                self.metric_fn = lambda model: mean_preactivations_llama(model, self.data_handler.tokenizer,
                                                                         self.data_handler.val_set,
                                                                         max_batches=self.max_batches,
                                                                         device=self.device, save=self.save)
            case ("llama7b", "procrustes") | ("llama13b", "procrustes"):
                raise NotImplementedError("Procrustes metric not implemented for Llama yet.")
            case ("llama7b", "fraction") | ("llama13b", "fraction"):
                raise NotImplementedError("Fraction metric not implemented for Llama yet.")
            case ("resnet18", "mean_preactivation") | ("resnet34", "mean_preactivation") | ("resnet50", "mean_preactivation"):
                self.metric_fn = lambda model: mean_preactivations_resnet(model, self.data_handler.val_set,
                                                                          batch_size=self.data_handler.batch_size,
                                                                          device=self.device)
            case("resnet18", "procrustes") | ("resnet34", "procrustes") | ("resnet50", "procrustes"):
                raise NotImplementedError("Procrustes metric not implemented for Resnet yet.")
            case("resnet18", "fraction") | ("resnet34", "fraction") | ("resnet50", "fraction"):
                raise NotImplementedError("Fraction metric not implemented for Resnet yet.")
            case _:
                raise ValueError(f"Unsupported model and metric combination: {model_name} and {metric_name}.")

        logger.info(f"LinearityMetric initialized with model: {model_name}, metric: {metric_name}, threshold: {threshold}, max_batches: {max_batches}, device: {device}, save: {save}.")


    def threshold_fn(self, threshold):
        """Encapsulating a threshold function that either splits based on a percentile or float.
        Args:
            threshold (str): A string that is one of the following: None, a percentage, e.g. `75%`, or a float.
        Returns:
            A function that takes a dictionary of layer names and linearity scores, and splits it into two dictionaries. The first dictionary contains the layers that are considered linear (i.e., those with scores above the threshold), and the second dictionary contains the layers that are considered non-linear (i.e., those with scores below the threshold).
            """
        logger.info(f"Threshold: {threshold}")
        if threshold is None:
            logger.debug(f"Threshold is None, defaulting to 75%")
            thresholder = lambda dictionary: (
                {k: v for k, v in dictionary.items() if v >= torch.quantile(torch.tensor(list(dictionary.values())), 0.75)},
                {k: v for k, v in dictionary.items() if v < torch.quantile(torch.tensor(list(dictionary.values())), 0.75)}
            )
        elif isinstance(threshold, str) and threshold.endswith('%'):
            percentage = float(threshold[:-1]) / 100.0
            logger.debug(f"Threshold is percentage: {percentage}")
            thresholder = lambda dictionary: (
                {k: v for k, v in dictionary.items() if v >= torch.quantile(torch.tensor(list(dictionary.values())), percentage)},
                {k: v for k, v in dictionary.items() if v < torch.quantile(torch.tensor(list(dictionary.values())), percentage)}
            )
        else:
            try:
                float_threshold = float(threshold)
                logger.debug(f"Threshold is float: {float_threshold}")
                thresholder = lambda dictionary: (
                    {k: v for k, v in dictionary.items() if v >= float_threshold},
                    {k: v for k, v in dictionary.items() if v < float_threshold}
                )
            except ValueError:
                raise ValueError(f"Invalid threshold value: {threshold}. Must be None, a percentage string (e.g., '75%'), or a float.")

        return thresholder


def mean_preactivations_llama(model, tokenizer, dataset, max_batches=30, device='cuda', save=False):
    """Compute the mean of preactivations for each activation layer in the model. Function does this by computing the mean preactivation values over a set of input data. For Llama with RMS normalization before and after self-attention, we can't retrieve preactivations from normalization parameters. Thus, we must calculate the mean of the input of the normalization before activation named 'model.layers.n.post_attention_layernorm' and 'model.layers.n.mlp.act_fn' where n is the layer number.
    Args:
        model: The neural network model.
        tokenizer: The tokenizer for the model.
        dataset: The dataset to compute preactivations on.
        max_batches: Maximum number of batches to process from the dataset.
        device: Device to run the computations on.
        save: Whether to save/load the computed mean preactivations to/from disk.

    Returns:
        dict: A dictionary with layer names as keys and mean preactivation values as values.
    """
    save_path = f"./results/mean_preactivations_llama2_7b.pt"
    if save and os.path.exists(save_path):
        logger.info("Loading mean preactivations from disk...")
        return torch.load(save_path)

    model.to(device)
    model.eval()
    activation_layers = []

    # Storage
    channel_sums = {}     # name -> tensor [D]
    sample_counts = {}    # name -> int
    hooks = []

    logger.info("Identifying activation layers and registering hooks...")
    # Identify activation layers
    for name, module in model.named_modules():
        if re.match(r'model\.layers\.\d+\.mlp\.act_fn', name) or re.match(r'model\.layers\.\d+\.post_attention_layernorm', name):
            activation_layers.append((name, module))
    logger.debug("Identified layers, setting hooks...")
    # Define hook to capture preactivations
    def get_preactivation_hook(name):
        def hook(module, input, output):
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

        return hook

    # Register hooks
    for name, module in activation_layers:
        hooks.append(module.register_forward_hook(get_preactivation_hook(name)))
    logger.debug("Hooks registered. Performing forward passes...")

    # One of these is NoneType on the cluster for some reason, so extra checks to avoid errors
    if max_batches is None:
        num_batches = len(dataset)
    elif len(dataset) is None:
        num_batches = max_batches
    else:
        num_batches = min(max_batches, len(dataset))
    # Forward pass through the data
    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="Processing samples for preactivations", leave=False):
            inputs = tokenizer(dataset[i]['text'], return_tensors='pt', truncation=True, padding='max_length', max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            model(**inputs)
    logger.debug("Forward passes complete. Computing mean preactivations...")

    # Compute mean preactivations
    mean_preactivations = {}
    for name in channel_sums:
        per_dim_mean = channel_sums[name] / sample_counts[name]  # [D]
        mean_preactivations[name] = per_dim_mean.mean().item()   # scalar

    logger.debug("Mean preactivations computed.")
    # Remove hooks
    for hook in hooks:
        hook.remove()
    logger.debug("Hooks removed.")

    if save:
        torch.save(mean_preactivations, save_path)
        logger.info("Mean preactivations saved to disk.")

    # Return model to cpu and free vram cache
    model.cpu()
    torch.cuda.empty_cache()

    return mean_preactivations

def mean_preactivations_resnet(model, dataset, batch_size=1, device='cuda', save=False):
    """
    Compute the mean of preactivations for each BatchNorm2d layer (pre-ReLU)
    following the definition:

        p̄^l = (1 / M_l) ∑_i (1 / N) ∑_s z^l_{s,i}

    where z^l_{s,i} is the spatially-averaged preactivation for channel i
    and sample s.

    Args:
        model: The ResNet model.
        dataset: The dataset to compute preactivations on.
        batch_size: Batch size for processing the dataset.
        device: Device to run the computations on.
        save: Whether to save/load the computed mean preactivations to/from disk.
    Returns:
        dict: layer_name -> mean preactivation (scalar)
    """
    save_path = f"./results/mean_preactivations_{model.__class__.__name__}.pt"
    if save and os.path.exists(save_path):
            logger.info("Loading mean preactivations from disk...")
            return torch.load(save_path)

    model.to(device)
    model.eval()
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # Store running sums and sample counts per layer
    channel_sums = {}   # layer_name -> tensor [C]
    sample_counts = {} # layer_name -> int
    hooks = []

    logger.info("Identifying activation layers and registering hooks...")

    def get_hook(name):
        def hook(module, input, output):
            # output shape: [B, C, H, W]
            B = output.shape[0]

            # spatial + batch mean, but keep channels
            per_channel_batch_mean = output.mean(dim=(0, 2, 3))  # [C]

            if name not in channel_sums:
                channel_sums[name] = per_channel_batch_mean.detach().clone() * B
                sample_counts[name] = B
            else:
                channel_sums[name] += per_channel_batch_mean.detach() * B
                sample_counts[name] += B

        return hook

    # Register hooks on BatchNorm layers (pre-ReLU in ResNet18)
    for name, module in tqdm(model.named_modules(),
                             desc="Registering hooks",
                             leave=False):
        if isinstance(module, torch.nn.BatchNorm2d):
            hooks.append(module.register_forward_hook(get_hook(name)))

    logger.info("Hooks registered. Performing forward passes...")

    # Forward passes
    with torch.no_grad():
        for inputs, _ in tqdm(data_loader,
                              desc="Computing mean preactivations",
                              leave=False):
            inputs = inputs.to(device)
            model(inputs)

    logger.info("Mean preactivations computed.")

    # Remove hooks
    for hook in hooks:
        hook.remove()

    logger.info("Hooks removed.")

    # Final averaging: over samples, then over channels
    mean_preactivations = {}
    for name in channel_sums:
        per_channel_mean = channel_sums[name] / sample_counts[name]  # [C]
        mean_preactivations[name] = per_channel_mean.mean().item()   # scalar

    logger.info("Mean preactivations computed.")

    mean_preacts_conv = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            if "downsample" not in name:
                mean = mean_preactivations.get(name, 0.0)
                layer = name.split('bn')[0]  # Get the layer name before .bn
                index = name.split('bn')[-1]  # Get the index if present
                mean_preacts_conv[layer + 'conv' + index] = mean  # Copy to preceding Conv2d layer
            else:
                mean_preacts_conv[name] = mean_preactivations.get(name, 0.0)  # Keep downsample layers as is

    logger.info("Mapped mean preactivations to preceding Conv2d layers.")

    if save:
        torch.save(mean_preacts_conv, save_path)
        logger.info("Mean preactivations saved to disk.")

    return mean_preacts_conv