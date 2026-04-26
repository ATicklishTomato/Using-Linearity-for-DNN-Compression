import os
import re
import json
import logging
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from torchvision.models import ResNet
from tqdm import tqdm

logger = logging.getLogger(__name__)
debug_mode = logger.getEffectiveLevel() != logging.DEBUG


# ============================================================
# Utility functions
# ============================================================

def flatten_representation(tensor):
    """
    Convert arbitrary activations to shape [N, D].

    Cases:
        NLP:    [B, T, D] -> [B*T, D]
        CNN:    [B, C, H, W] -> [B*H*W, C]
        Vector: [B, D] -> [B, D]
    """
    if tensor.dim() == 2:
        return tensor

    elif tensor.dim() == 3:
        # [B,T,D]
        b, t, d = tensor.shape
        return tensor.reshape(b * t, d)

    elif tensor.dim() == 4:
        # [B,C,H,W] -> [B,H,W,C]
        b, c, h, w = tensor.shape
        x = tensor.permute(0, 2, 3, 1).contiguous()
        return x.reshape(b * h * w, c)

    else:
        # fallback
        return tensor.reshape(-1, tensor.shape[-1])


def center_and_normalize(X, eps=1e-8):
    """
    Center columns, then normalize.
    """
    X = X.float()
    X = X - X.mean(dim=0, keepdim=True)
    norm = torch.norm(X, p="fro")
    return X / (norm + eps)


def compute_linearity_score(X, Y):
    """
    Compute:
        1 - min_A ||X_tilde A - Y_tilde||_F^2

    via least squares:
        A = pinv(X_tilde) @ Y_tilde
    """
    X = center_and_normalize(X)
    Y = center_and_normalize(Y)

    # We want to achieve XA = Y, as then the transformation is linear. We can rewrite as X^(-1)XA = X^(-1)Y, aka A = X^(-1)Y
    # Calculating the inverse of X is hard and timeconsuming, so we calculate the pseudoinverse instead to get something close enough.
    A = torch.linalg.pinv(X) @ Y

    residual = X @ A - Y
    error = torch.norm(residual, p="fro") ** 2

    score = 1.0 - error.item()
    return score


# ============================================================
# Hook logic
# ============================================================

def pre_hook_fn(module, inputs, storage, name):
    """
    Save input to block.
    """
    x = inputs[0].detach()
    storage[name]["x"].append(flatten_representation(x).cpu())


def post_hook_fn(module, inputs, output, storage, name):
    """
    Save output from block.
    """
    if isinstance(output, tuple):
        output = output[0]

    y = output.detach()
    storage[name]["y"].append(flatten_representation(y).cpu())


# ============================================================
# Main function
# ============================================================

def procrustes_based_linearity(
    model,
    data_handler,
    device="cuda",
    save=False,
    save_dir="./results",
):
    """
    Compute linearity score for each block. Based off the work of Razzhigaev et al. (2024).

    Args:
        model: neural net model
        data_handler: dataset/tokenizer wrapper
        device: cuda/cpu
        save: save json
        save_dir: output dir

    Returns:
        dict[layer_name] = score
    """

    logger.info("Starting block linearity computation.")

    is_resnet = isinstance(model, ResNet)

    if is_resnet:
        # Residual blocks
        target_layer_pattern = re.compile(r"^model\.layer\d+\.\d+$")
    else:
        # Transformer blocks
        target_layer_pattern = re.compile(r"^model\.layers\.\d+$")

    logger.info(f"Using pattern: {target_layer_pattern.pattern}")

    model.eval().to(device)

    dataset = data_handler.val_set
    storage = defaultdict(lambda: {"x": [], "y": []})
    hooks = []

    # --------------------------------------------------------
    # Register hooks
    # --------------------------------------------------------
    for name, module in model.named_modules():
        if target_layer_pattern.match(name):
            hooks.append(
                module.register_forward_pre_hook(
                    lambda module, inputs, name=name:
                    pre_hook_fn(module, inputs, storage, name)
                )
            )

            hooks.append(
                module.register_forward_hook(
                    lambda module, inputs, output, name=name:
                    post_hook_fn(module, inputs, output, storage, name)
                )
            )

    logger.info("Hooks registered.")

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------
    with torch.no_grad():

        if is_resnet:
            loader = DataLoader(
                dataset,
                batch_size=data_handler.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
            )

            for inputs, _ in tqdm(
                loader,
                desc="ResNet forward pass",
                leave=False,
                disable=debug_mode,
            ):
                inputs = inputs.to(device)
                model(inputs)

        else:
            for i in tqdm(
                range(len(dataset)),
                desc="LLaMA forward pass",
                leave=False,
                disable=debug_mode,
            ):
                tokens = data_handler.tokenizer(
                    dataset[i]["text"],
                    return_tensors="pt",
                    truncation=True,
                    padding="max_length",
                    max_length=512,
                )

                tokens = {k: v.to(device) for k, v in tokens.items()}

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    model(**tokens)

    # --------------------------------------------------------
    # Remove hooks
    # --------------------------------------------------------
    for h in hooks:
        h.remove()

    logger.info("Hooks removed.")

    # --------------------------------------------------------
    # Compute scores
    # --------------------------------------------------------
    scores = {}

    for name in storage.keys():
        if len(storage[name]["x"]) == 0:
            continue

        X = torch.cat(storage[name]["x"], dim=0)
        Y = torch.cat(storage[name]["y"], dim=0)

        # Match feature dims if needed
        d = min(X.shape[1], Y.shape[1])
        X = X[:, :d]
        Y = Y[:, :d]

        score = compute_linearity_score(X, Y)
        scores[name] = score

        logger.info(f"{name}: {score:.6f}")

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------
    if save:
        json.dump(scores, open(os.path.join(save_dir, "procrustes_scores.json"), "w"))
        logger.info(f"Saved to {os.path.join(save_dir, 'procrustes_scores.json')}")

    logger.info("Finished block linearity computation.")
    return scores