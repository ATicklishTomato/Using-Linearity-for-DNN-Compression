import os
import re
import json
import logging
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import ResNet
from tqdm import tqdm

logger = logging.getLogger(__name__)
debug_mode = logger.getEffectiveLevel() != logging.DEBUG


# ============================================================
# Utility functions
# ============================================================

def flatten_representation(X, Y=None):
    """
    Prepare activations for linearity scoring.

    If only X is provided:
        returns flattened X

    If X and Y are provided:
        jointly aligns shapes first, then returns (X_flat, Y_flat)

    Cases
    -----
    NLP:
        [B,T,D] -> [B*T,D]

    CNN same spatial size:
        [B,C,H,W] -> [B*H*W,C]

    CNN different spatial size:
        adaptively pool X to Y spatial size, then flatten

    Vector:
        [B,D] -> [B,D]
    """

    # --------------------------------------------------------
    # Single tensor mode (backward compatible)
    # --------------------------------------------------------
    if Y is None:
        if X.dim() == 2:
            return X

        elif X.dim() == 3:
            b, t, d = X.shape
            return X.reshape(b * t, d)

        elif X.dim() == 4:
            b, c, h, w = X.shape
            X = X.permute(0, 2, 3, 1).contiguous()
            return X.reshape(b * h * w, c)

        else:
            return X.reshape(-1, X.shape[-1])

    # --------------------------------------------------------
    # Joint mode (recommended for scoring)
    # --------------------------------------------------------

    # ---------- vectors ----------
    if X.dim() == 2 and Y.dim() == 2:
        return X, Y

    # ---------- transformers ----------
    if X.dim() == 3 and Y.dim() == 3:
        bx, tx, dx = X.shape
        by, ty, dy = Y.shape

        n = min(tx, ty)
        X = X[:, :n, :]
        Y = Y[:, :n, :]

        X = X.reshape(-1, dx)
        Y = Y.reshape(-1, dy)

        return X, Y

    # ---------- CNNs ----------
    if X.dim() == 4 and Y.dim() == 4:
        _, _, hy, wy = Y.shape

        # align spatial grid BEFORE flattening
        if X.shape[-2:] != Y.shape[-2:]:
            X = F.adaptive_avg_pool2d(X, (hy, wy))

        bx, cx, hx, wx = X.shape
        by, cy, hy, wy = Y.shape

        X = X.permute(0, 2, 3, 1).contiguous().reshape(bx * hx * wx, cx)
        Y = Y.permute(0, 2, 3, 1).contiguous().reshape(by * hy * wy, cy)

        return X, Y

    # ---------- fallback ----------
    X = X.reshape(-1, X.shape[-1])
    Y = Y.reshape(-1, Y.shape[-1])

    n = min(X.shape[0], Y.shape[0])
    return X[:n], Y[:n]


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
    X, Y = flatten_representation(X, Y)
    X = center_and_normalize(X)
    Y = center_and_normalize(Y)

    # We want to achieve XA = Y, as then the transformation is linear. We can rewrite as X^(-1)XA = X^(-1)Y, aka A = X^(-1)Y
    # Calculating the inverse of X is hard and timeconsuming, so we calculate the pseudoinverse instead to get something close enough.
    A = torch.linalg.pinv(X) @ Y

    residual = X @ A - Y
    error = torch.norm(residual, p="fro") ** 2

    score = 1.0 - error.item()
    return score

def expand_scores_to_individual_layers(scores, is_resnet):
    """The scores we get are for blocks. The broader experimental code looks at individual layers.
    This function adapts the labels to reference the individual layers.
    Args:
        scores: dict[block_name, score] The original scores for each block.
        is_resnet: bool Indication whether the scores are for resnet
    Returns:
        dict[layer_name, score] New dict with scores for each layer in the block, copied from the block
    """
    new_scores = {}
    for block_name, score in scores.items():
        if is_resnet:
            new_scores[block_name + ".conv1"] = score
            new_scores[block_name + ".conv2"] = score
        else:
            new_scores[block_name + ".self_attn"] = score

    return new_scores


# ============================================================
# Hook logic
# ============================================================

def hook_fn(module, inputs, output, storage, name):
    """
    Save input and output from block.
    """
    x = inputs[0].detach()
    if len(storage[name]["x"]) == 0:
        storage[name]["x"] = x.cpu()
    else:
        storage[name]["x"] = torch.cat([storage[name]["x"], x.cpu()], dim=0)

    if isinstance(output, tuple):
        output = output[0]

    y = output.detach()
    if len(storage[name]["y"]) == 0:
        storage[name]["y"] = y.cpu()
    else:
        storage[name]["y"] = torch.cat([storage[name]["y"], y.cpu()], dim=0)


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
        target_layer_pattern = re.compile(r"^layer\d+\.\d+$")
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
                module.register_forward_hook(
                    lambda module, inputs, output, name=name:
                    hook_fn(module, inputs, output, storage, name)
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
                num_workers=4,              # try 2–8 depending on CPU
                pin_memory=True,            # important for GPU transfer
                prefetch_factor=2,          # batches per worker
                persistent_workers=True     # avoids worker restart each epoch
            )

            for index, batch in tqdm(
                enumerate(loader),
                total=1000//data_handler.batch_size,  # We only process 1000 samples to avoid overkill
                desc="ResNet forward pass",
                leave=False,
                disable=debug_mode,
            ):
                if index >= 1000//data_handler.batch_size:
                    # We stop after 1000 samples, as concatenating embeddings for the whole dataset is overkill
                    break
                inputs, _ = batch
                inputs = inputs.to(device)
                model(inputs)

        else:
            data_loader = DataLoader(
                dataset,
                batch_size=data_handler.batch_size,
                shuffle=False,
                num_workers=4, pin_memory=True, prefetch_factor=2, persistent_workers=True
            )
            for index, batch in tqdm(
                enumerate(data_loader),
                total=1000//data_handler.batch_size,
                desc="LLaMA forward pass",
                leave=False,
                disable=debug_mode,
            ):
                if index >= 1000//data_handler.batch_size:
                    # We stop after 100 batches, as concatenating embeddings for the whole dataset is overkill
                    break
                tokens = data_handler.tokenizer(
                    batch["text"],
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

        X = storage[name]["x"]
        Y = storage[name]["y"]

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
    return expand_scores_to_individual_layers(scores, is_resnet)