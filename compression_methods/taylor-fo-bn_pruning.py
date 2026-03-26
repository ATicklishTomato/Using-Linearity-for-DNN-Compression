"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

taylor_fo_bn_pruning.py
=======================
Self-contained Taylor-FO-BN pruning (Method 22) for torchvision ResNet models.

Based on:
    "Importance Estimation for Neural Network Pruning" (CVPR 2019)
    Molchanov, Mallya, Tyree, Frosio, Kautz.
    Source repo: https://github.com/NVlabs/Taylor_pruning  (CC BY-NC-SA 4.0)

Algorithm (method 22 = Taylor_gate / Taylor-FO-BN):
    1.  For each "internal" BN layer in a ResNet residual block, register
        forward and backward hooks that capture the BN output activation and
        its gradient.
    2.  Per mini-batch: accumulate per-channel importance =
            |activation_c * gradient_c|  averaged over (batch, H, W).
    3.  Every `frequency` mini-batches: update a running momentum estimate of
        importance, then zero out the `prune_per_iteration` globally
        least-important channels (soft mask step).
    4.  Repeat until `prune_neurons_max` channels have been removed, or until
        `maximum_pruning_iterations` prune steps have been performed.
    5.  Perform structural pruning: physically resize conv weight tensors and
        BN parameter/buffer tensors to remove the zeroed channels.
    6.  Log before/after channel sizes and return a dict of per-layer pruning
        ratios and a summary dict.

Which BN layers are targeted (safe w.r.t. skip connections):
    BasicBlock  (ResNet-18/34): bn1  (sits between conv1 and conv2)
    Bottleneck  (ResNet-50/101/152): bn1 and bn2
        (sit between conv1↔conv2 and conv2↔conv3 respectively)
    The block-output BN (bn2 in BasicBlock, bn3 in Bottleneck) is intentionally
    excluded because it feeds the residual addition and its channel width must
    match the skip-connection output.

Usage example
-------------
    from taylor_fo_bn_pruning import run_taylor_fo_bn_pruning
    import torch, torch.nn as nn
    import torchvision.models as models

    model  = models.resnet50(pretrained=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    result = run_taylor_fo_bn_pruning(
        model         = model,
        data_handler  = loader,
        criterion     = nn.CrossEntropyLoss(),
        device        = torch.device("cuda"),
        pruning_ratio = 0.50,   # remove 50 % of prunable channels globally
    )

    logger.info(result["layer_pruning_ratios"])   # {layer_name: ratio_pruned, ...}
    logger.info(result["summary"])                # totals, per-layer before/after counts
    pruned_model = result["model"]          # structurally pruned nn.Module

Parameters
----------
model : nn.Module
    A torchvision ResNet (resnet18/34/50/101/152).  Must already be in eval or
    train mode as appropriate; the function sets it to train() internally.

data_handler : Iterable[Tuple[Tensor, Tensor]]
    Any iterable that yields (inputs, targets) batches; e.g. a DataLoader.
    Inputs must be on CPU or any device — they will be moved to `device`.

criterion : nn.Module
    Loss function, e.g. nn.CrossEntropyLoss().

device : torch.device
    Target compute device.

pruning_ratio : float
    Fraction of all prunable channels to remove globally.
    E.g. 0.5 removes 50 % of the total prunable channels.

frequency : int  (default 30)
    Number of mini-batches between consecutive prune steps.
    Matches the original "frequency" config key.

prune_per_iteration : int  (default 50)
    Number of channels to prune per prune step.
    Decrease for finer-grained pruning schedules.

maximum_pruning_iterations : int or None  (default None)
    Hard cap on the number of prune steps.  None = no cap (the ratio alone
    controls termination).

pruning_momentum : float  (default 0.9)
    Exponential moving-average coefficient for importance accumulation.

use_momentum : bool  (default True)
    Whether to apply momentum to importance estimates across prune steps.

min_channels_per_layer : int  (default 8)
    Lower bound on remaining channels per layer — prevents complete collapse.

seed : int  (default 0)
    RNG seed for reproducibility.

verbose : bool  (default True)
    logger.info progress to stdout and log via the logging module.

Returns
-------
dict with keys:
    "model"               : the pruned nn.Module (same object, modified in-place)
    "layer_pruning_ratios": {layer_name: fraction_of_channels_pruned}
    "summary"             : list of dicts with per-layer before/after stats
    "total_channels_before": int
    "total_channels_after" : int
    "global_pruning_ratio" : float
"""

from __future__ import annotations

import logging
import math
import random
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from utils.data_manager import DataManager

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Internal type alias
# ─────────────────────────────────────────────────────────────────────────────

# A "triplet" fully describes one prunable BN site:
#   (bn_key, (prev_conv_name, prev_conv), (bn_name, bn), (next_conv_name, next_conv))
_Triplet = Tuple[
    str,
    Tuple[str, nn.Conv2d],
    Tuple[str, nn.BatchNorm2d],
    Tuple[str, nn.Conv2d],
]


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 – discover prunable BN sites in a ResNet
# ─────────────────────────────────────────────────────────────────────────────

def _discover_prunable_triplets(model: nn.Module) -> List[_Triplet]:
    """
    Walk a torchvision ResNet and return one _Triplet per prunable BN layer.

    BasicBlock  -> targets bn1  (between conv1 and conv2)
    Bottleneck  -> targets bn1 and bn2  (between conv1↔conv2 and conv2↔conv3)
    """
    try:
        from torchvision.models.resnet import BasicBlock, Bottleneck
    except ImportError as exc:
        raise ImportError(
            "torchvision is required for ResNet pruning. "
            "Install it with: pip install torchvision"
        ) from exc

    triplets: List[_Triplet] = []

    for block_name, block in model.named_modules():
        if isinstance(block, BasicBlock):
            # conv1 → bn1 (prunable) → ReLU → conv2 → bn2 (block output – skip)
            triplets.append((
                f"{block_name}.bn1",
                (f"{block_name}.conv1", block.conv1),
                (f"{block_name}.bn1",  block.bn1),
                (f"{block_name}.conv2", block.conv2),
            ))

        elif isinstance(block, Bottleneck):
            # conv1 → bn1 (prunable) → ReLU
            # → conv2 → bn2 (prunable) → ReLU
            # → conv3 → bn3 (block output – skip)
            triplets.append((
                f"{block_name}.bn1",
                (f"{block_name}.conv1", block.conv1),
                (f"{block_name}.bn1",  block.bn1),
                (f"{block_name}.conv2", block.conv2),
            ))
            triplets.append((
                f"{block_name}.bn2",
                (f"{block_name}.conv2", block.conv2),
                (f"{block_name}.bn2",  block.bn2),
                (f"{block_name}.conv3", block.conv3),
            ))

    if not triplets:
        raise ValueError(
            "No BasicBlock or Bottleneck modules found.  "
            "Make sure `model` is a torchvision ResNet."
        )

    return triplets


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 – hook management: capture activation and gradient at each BN output
# ─────────────────────────────────────────────────────────────────────────────

class _HookState:
    """Lightweight container for per-BN hook data."""

    def __init__(self, bn_key: str):
        self.key   = bn_key
        self.act   : Optional[Tensor] = None   # forward activation (C,)
        self.grad  : Optional[Tensor] = None   # backward gradient  (C,)


def _register_hooks(
    triplets   : List[_Triplet],
    hook_states: Dict[str, _HookState],
) -> List[torch.utils.hooks.RemovableHandle]:
    """
    Register one forward hook and one backward hook on each target BN module.
    Activations are averaged over (batch, H, W) to get a C-dim vector.
    """
    handles: List[torch.utils.hooks.RemovableHandle] = []

    for _, _, (bn_key, bn_module), _ in triplets:
        state = hook_states[bn_key]

        # --- forward hook: store mean |activation| per channel ---
        def make_fwd(s):
            def fwd_hook(module, inp, out):
                # out: (N, C, H, W)  or  (N, C)
                if out.dim() == 4:
                    s.act = out.detach()          # keep spatial dims; reduce later
                else:
                    s.act = out.detach().unsqueeze(-1).unsqueeze(-1)
            return fwd_hook

        # --- backward hook: store mean |gradient| per channel ---
        def make_bwd(s):
            def bwd_hook(module, grad_in, grad_out):
                g = grad_out[0]
                if g is None:
                    return
                if g.dim() == 4:
                    s.grad = g.detach()
                else:
                    s.grad = g.detach().unsqueeze(-1).unsqueeze(-1)
            return bwd_hook

        handles.append(bn_module.register_forward_hook(make_fwd(state)))
        handles.append(bn_module.register_full_backward_hook(make_bwd(state)))

    return handles


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 – importance accumulation
# ─────────────────────────────────────────────────────────────────────────────

def _accumulate_importance(
    hook_states   : Dict[str, _HookState],
    accumulate    : Dict[str, Tensor],
    iteration_ctr : Dict[str, int],
) -> None:
    """
    For each BN site with valid (act, grad) data, compute
        importance_c = |act_c * grad_c|.mean(batch, H, W)
    and add it to the running accumulator.
    """
    for key, state in hook_states.items():
        if state.act is None or state.grad is None:
            continue

        # (N, C, H, W)
        importance = (state.act * state.grad).abs()        # (N, C, H, W)
        importance = importance.mean(dim=(0, 2, 3))        # (C,)
        importance = importance.cpu()

        if key not in accumulate:
            accumulate[key] = torch.zeros_like(importance)
        accumulate[key] += importance
        iteration_ctr[key] = iteration_ctr.get(key, 0) + 1

        # clear so stale data is never reused
        state.act  = None
        state.grad = None


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 – momentum update and global prune step
# ─────────────────────────────────────────────────────────────────────────────

def _update_averaged(
    accumulate    : Dict[str, Tensor],
    averaged      : Dict[str, Tensor],
    iteration_ctr : Dict[str, int],
    masks         : Dict[str, Tensor],
    use_momentum  : bool,
    momentum_coeff: float,
    prune_step_idx: int,
) -> None:
    """
    Normalise accumulated importance by the iteration count, apply momentum,
    and zero out already-pruned channels.
    """
    for key, acc in accumulate.items():
        n = max(iteration_ctr.get(key, 1), 1)
        contribution = acc / n                             # (C,)

        if not use_momentum or prune_step_idx == 0:
            averaged[key] = contribution
        else:
            averaged[key] = (
                momentum_coeff * averaged[key]
                + (1.0 - momentum_coeff) * contribution
            )

        # zero importance of already-pruned channels so they are never re-selected
        if key in masks:
            averaged[key] = averaged[key] * masks[key].float()

    # reset accumulators for the next window
    for key in list(accumulate.keys()):
        accumulate[key] = torch.zeros_like(accumulate[key])
        iteration_ctr[key] = 0


def _global_prune_step(
    averaged            : Dict[str, Tensor],
    masks               : Dict[str, Tensor],
    prune_per_iteration : int,
    prune_neurons_max   : int,
    total_pruned        : int,
    min_per_layer       : int,
) -> Tuple[int, Dict[str, List[int]]]:
    """
    Globally rank all live channels, prune the `prune_per_iteration` weakest.
    Returns (new_total_pruned, dict_of_pruned_indices_per_layer).
    """
    # Build a flat list of (importance, layer_key, channel_index)
    all_candidates: List[Tuple[float, str, int]] = []

    for key, imp in averaged.items():
        mask = masks.get(key, torch.ones(imp.shape[0], dtype=torch.bool))
        live_channels = mask.nonzero(as_tuple=True)[0].tolist()
        # never prune a layer below `min_per_layer` remaining channels
        n_keep_min = min_per_layer
        n_live     = len(live_channels)
        if n_live <= n_keep_min:
            continue
        for c in live_channels:
            all_candidates.append((imp[c].item(), key, c))

    if not all_candidates:
        return total_pruned, {}

    # Sort ascending (smallest importance first)
    all_candidates.sort(key=lambda x: x[0])

    remaining_budget = prune_neurons_max - total_pruned
    n_to_prune       = min(prune_per_iteration, remaining_budget, len(all_candidates))

    if n_to_prune <= 0:
        return total_pruned, {}

    pruned_this_step: Dict[str, List[int]] = {}
    layer_prune_count: Dict[str, int] = {}

    for _, key, ch in all_candidates[:n_to_prune]:
        mask = masks.get(key)
        if mask is None:
            masks[key] = torch.ones(averaged[key].shape[0], dtype=torch.bool)
            mask = masks[key]

        # Respect per-layer minimum
        n_live_now = mask.sum().item()
        if n_live_now - layer_prune_count.get(key, 0) <= min_per_layer:
            continue

        mask[ch] = False
        averaged[key][ch] = 0.0
        layer_prune_count[key] = layer_prune_count.get(key, 0) + 1
        pruned_this_step.setdefault(key, []).append(ch)

    actual_pruned = sum(len(v) for v in pruned_this_step.values())
    return total_pruned + actual_pruned, pruned_this_step


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 – structural pruning: physically resize tensors
# ─────────────────────────────────────────────────────────────────────────────

def _keep_indices(mask: Tensor) -> Tensor:
    """Return sorted indices where mask == True."""
    return mask.nonzero(as_tuple=True)[0]


def _prune_conv_out_channels(conv: nn.Conv2d, keep_idx: Tensor) -> None:
    """Remove output channels from a Conv2d (no bias assumed for ResNet convs)."""
    with torch.no_grad():
        new_weight = conv.weight.data[keep_idx]
        conv.weight = nn.Parameter(new_weight)
        if conv.bias is not None:
            conv.bias = nn.Parameter(conv.bias.data[keep_idx])
        conv.out_channels = len(keep_idx)


def _prune_conv_in_channels(conv: nn.Conv2d, keep_idx: Tensor) -> None:
    """Remove input channels from a Conv2d."""
    with torch.no_grad():
        # weight shape: (out_ch, in_ch/groups, kH, kW)
        new_weight = conv.weight.data[:, keep_idx]
        conv.weight = nn.Parameter(new_weight)
        conv.in_channels = len(keep_idx)
        # groups stays at 1 for standard convs; depthwise not expected in ResNet stem


def _prune_bn(bn: nn.BatchNorm2d, keep_idx: Tensor) -> None:
    """Remove channels from a BatchNorm2d."""
    with torch.no_grad():
        if bn.weight is not None:
            bn.weight = nn.Parameter(bn.weight.data[keep_idx])
        if bn.bias is not None:
            bn.bias   = nn.Parameter(bn.bias.data[keep_idx])
        if bn.running_mean is not None:
            bn.running_mean = bn.running_mean[keep_idx]
        if bn.running_var is not None:
            bn.running_var  = bn.running_var[keep_idx]
        bn.num_features = len(keep_idx)


def _apply_structural_pruning(
    triplets  : List[_Triplet],
    masks     : Dict[str, Tensor],
    verbose   : bool,
) -> Dict[str, Dict]:
    """
    For each pruned BN site, physically remove channels from the three
    involved tensors: (prev_conv output, bn params/buffers, next_conv input).

    Returns per-layer summary: {bn_key: {"before": C, "after": C', "pruned": n}}
    """
    summary: Dict[str, Dict] = {}

    for bn_key, (prev_name, prev_conv), (bn_name, bn_mod), (next_name, next_conv) in triplets:
        C_before = bn_mod.num_features

        if bn_key not in masks:
            # layer was never pruned
            summary[bn_key] = {"before": C_before, "after": C_before, "pruned": 0,
                               "prev_conv": prev_name, "next_conv": next_name}
            continue

        mask     = masks[bn_key]
        keep_idx = _keep_indices(mask)
        C_after  = len(keep_idx)
        n_pruned = C_before - C_after

        if n_pruned == 0:
            summary[bn_key] = {"before": C_before, "after": C_after, "pruned": 0,
                               "prev_conv": prev_name, "next_conv": next_name}
            continue

        keep_idx_device = keep_idx.to(prev_conv.weight.device)

        _prune_conv_out_channels(prev_conv, keep_idx_device)
        _prune_bn(bn_mod, keep_idx_device)
        _prune_conv_in_channels(next_conv, keep_idx_device)

        ratio = n_pruned / C_before
        summary[bn_key] = {
            "before"    : C_before,
            "after"     : C_after,
            "pruned"    : n_pruned,
            "ratio"     : ratio,
            "prev_conv" : prev_name,
            "next_conv" : next_name,
        }

        if verbose:
            msg = (
                f"[Structural] {bn_key:50s}  "
                f"{C_before:4d} → {C_after:4d}  "
                f"({ratio*100:.1f}% pruned)"
            )
            logger.info(msg)

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def prune(
    model                      : nn.Module,
    data_handler               : DataManager,
    criterion                  : nn.Module,
    device                     : str = 'cuda',
    pruning_ratio              : float = 0.5,
    *,
    frequency                  : int   = 30,
    prune_per_iteration        : int   = 50,
    maximum_pruning_iterations : Optional[int] = None,
    pruning_momentum           : float = 0.9,
    use_momentum               : bool  = True,
    min_channels_per_layer     : int   = 8,
    seed                       : int   = 0,
    verbose                    : bool  = True,
) -> Dict:
    """
    Run Taylor-FO-BN pruning on a ResNet model.  See module docstring for full
    parameter and return-value documentation.
    """
    # ── reproducibility ───────────────────────────────────────────────────────
    torch.manual_seed(seed)
    random.seed(seed)

    # ── validate inputs ───────────────────────────────────────────────────────
    if not (0.0 < pruning_ratio < 1.0):
        raise ValueError(f"pruning_ratio must be in (0, 1), got {pruning_ratio}")

    # ── prepare model ─────────────────────────────────────────────────────────
    model = model.to(device)
    model.train()   # hooks need gradient flow; BatchNorm must be in train mode

    # ── discover prunable triplets ────────────────────────────────────────────
    triplets = _discover_prunable_triplets(model)
    n_layers = len(triplets)

    # Total prunable channels and budget
    total_prunable  = sum(bn_mod.num_features for _, _, (_, bn_mod), _ in triplets)
    prune_neurons_max = int(math.floor(pruning_ratio * total_prunable))

    if maximum_pruning_iterations is None:
        maximum_pruning_iterations = max(
            1, math.ceil(prune_neurons_max / prune_per_iteration)
        )

    if verbose:
        hdr = (
            f"\n{'═'*70}\n"
            f"  Taylor-FO-BN Pruning  (method 22)\n"
            f"  Prunable layers       : {n_layers}\n"
            f"  Total prunable ch.    : {total_prunable}\n"
            f"  Target pruning ratio  : {pruning_ratio*100:.1f}%  "
            f"({prune_neurons_max} channels)\n"
            f"  Max prune iterations  : {maximum_pruning_iterations}\n"
            f"  Channels / iteration  : {prune_per_iteration}\n"
            f"  Accum. frequency      : {frequency} batches\n"
            f"  Momentum (α)          : {pruning_momentum if use_momentum else 'off'}\n"
            f"  Min channels / layer  : {min_channels_per_layer}\n"
            f"{'═'*70}"
        )
        logger.info(hdr)

    # Log initial layer sizes ─────────────────────────────────────────────────
    _log_layer_sizes(triplets, "BEFORE PRUNING", verbose)

    # ── hook infrastructure ───────────────────────────────────────────────────
    hook_states : Dict[str, _HookState] = {
        bn_key: _HookState(bn_key)
        for bn_key, _, _, _ in triplets
    }
    hook_handles = _register_hooks(triplets, hook_states)

    # ── state for the pruning loop ────────────────────────────────────────────
    accumulate    : Dict[str, Tensor] = {}
    averaged      : Dict[str, Tensor] = {}
    masks         : Dict[str, Tensor] = {}
    iter_ctr      : Dict[str, int]    = {}

    total_pruned        = 0
    mini_batch_idx      = 0
    prune_step_idx      = 0

    # ── main gradient-accumulation / prune loop ───────────────────────────────
    try:
        for inputs, targets in data_handler.train_set:
            # ── stop if budget exhausted ─────────────────────────────────────
            if total_pruned >= prune_neurons_max:
                break
            if prune_step_idx >= maximum_pruning_iterations:
                break

            inputs  = inputs.to(device,  non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # ── forward + backward ───────────────────────────────────────────
            model.zero_grad()
            outputs = model(inputs)
            loss    = criterion(outputs, targets)
            loss.backward()

            # ── accumulate importance from this batch ────────────────────────
            _accumulate_importance(hook_states, accumulate, iter_ctr)
            mini_batch_idx += 1

            # ── prune step every `frequency` mini-batches ────────────────────
            if mini_batch_idx % frequency == 0:
                _update_averaged(
                    accumulate, averaged, iter_ctr, masks,
                    use_momentum, pruning_momentum, prune_step_idx,
                )

                total_pruned, pruned_this_step = _global_prune_step(
                    averaged, masks,
                    prune_per_iteration, prune_neurons_max,
                    total_pruned, min_channels_per_layer,
                )
                prune_step_idx += 1

                if verbose:
                    n_step = sum(len(v) for v in pruned_this_step.values())
                    msg = (
                        f"[Prune step {prune_step_idx:4d} | "
                        f"batch {mini_batch_idx:6d}]  "
                        f"pruned this step: {n_step:4d}  |  "
                        f"total: {total_pruned}/{prune_neurons_max} "
                        f"({total_pruned/prune_neurons_max*100:.1f}%)"
                    )
                    logger.info(msg)

    finally:
        # Always remove hooks, even on exception
        for h in hook_handles:
            h.remove()

    # ── structural pruning ────────────────────────────────────────────────────
    if verbose:
        logger.info(f"\n[Structural pruning] applying masks to model weights …")

    summary = _apply_structural_pruning(triplets, masks, verbose)

    # ── log final layer sizes ─────────────────────────────────────────────────
    _log_layer_sizes(triplets, "AFTER PRUNING", verbose)

    # ── build return dict ─────────────────────────────────────────────────────
    layer_pruning_ratios: Dict[str, float] = {}
    for bn_key, info in summary.items():
        layer_pruning_ratios[bn_key] = info.get("ratio", 0.0)

    total_after = sum(info["after"]  for info in summary.values())
    total_before_check = sum(info["before"] for info in summary.values())
    global_ratio = (total_before_check - total_after) / max(total_before_check, 1)

    final_msg = (
        f"\n{'═'*70}\n"
        f"  Pruning complete.\n"
        f"  Total prunable channels before : {total_before_check}\n"
        f"  Total prunable channels after  : {total_after}\n"
        f"  Effective global pruning ratio : {global_ratio*100:.2f}%\n"
        f"  (target was {pruning_ratio*100:.1f}%)\n"
        f"{'═'*70}"
    )
    if verbose:
        logger.info(final_msg)

    return {
        "model"                 : model,
        "layer_pruning_ratios"  : layer_pruning_ratios,
        "summary"               : summary,
        "total_channels_before" : total_before_check,
        "total_channels_after"  : total_after,
        "global_pruning_ratio"  : global_ratio,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Utility: pretty-print layer sizes
# ─────────────────────────────────────────────────────────────────────────────

def _log_layer_sizes(
    triplets : List[_Triplet],
    label    : str,
    verbose  : bool,
) -> None:
    if not verbose:
        return
    sep = f"\n{'─'*70}"
    lines = [sep, f"  {label}", sep]
    for bn_key, (prev_name, prev_conv), (bn_name, bn_mod), (next_name, next_conv) in triplets:
        lines.append(
            f"  {bn_key:50s}  channels: {bn_mod.num_features:4d}  "
            f"(prev_conv out={prev_conv.out_channels}, "
            f"next_conv in={next_conv.in_channels})"
        )
    lines.append(sep)
    msg = "\n".join(lines)
    logger.info(msg)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: compute model parameter count
# ─────────────────────────────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> int:
    """Return the total number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# CLI / smoke test
# ─────────────────────────────────────────────────────────────────────────────

# if __name__ == "__main__":
#     """
#     Quick smoke-test with a ResNet-50 and a synthetic random dataloader.
#     Run:
#         python taylor_fo_bn_pruning.py
#     """
#     import sys
#
#     logging.basicConfig(
#         level=logging.INFO,
#         format="%(asctime)s %(levelname)s %(message)s",
#         handlers=[logging.StreamHandler(sys.stdout)],
#     )
#
#     try:
#         import torchvision.models as tvm
#     except ImportError:
#         logger.info("torchvision is required for the smoke test.  pip install torchvision")
#         sys.exit(1)
#
#     DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     BATCH_SIZE    = 8
#     N_BATCHES     = 200       # synthetic batches to pass through
#     NUM_CLASSES   = 1000
#     PRUNING_RATIO = 0.40      # prune 40 % of internal channels
#
#     logger.info(f"\nDevice: {DEVICE}")
#     logger.info(f"Loading ResNet-50 …")
#     model = tvm.resnet50(weights=None)     # random weights for smoke-test
#     params_before = count_parameters(model)
#     logger.info(f"Parameters before pruning: {params_before:,}")
#
#     # Synthetic dataloader: yields (batch_of_images, batch_of_labels)
#     class _SyntheticLoader:
#         def __init__(self, n_batches, batch_size, num_classes):
#             self.n = n_batches
#             self.bs = batch_size
#             self.nc = num_classes
#
#         def __len__(self):
#             return self.n
#
#         def __iter__(self):
#             for _ in range(self.n):
#                 x = torch.randn(self.bs, 3, 224, 224)
#                 y = torch.randint(0, self.nc, (self.bs,))
#                 yield x, y
#
#     data_handler = _SyntheticLoader(N_BATCHES, BATCH_SIZE, NUM_CLASSES)
#
#     result = prune(
#         model                      = model,
#         data_handler               = data_handler,
#         criterion                  = nn.CrossEntropyLoss(),
#         device                     = DEVICE,
#         pruning_ratio              = PRUNING_RATIO,
#         frequency                  = 20,
#         prune_per_iteration        = 40,
#         maximum_pruning_iterations = None,
#         pruning_momentum           = 0.9,
#         use_momentum               = True,
#         min_channels_per_layer     = 4,
#         seed                       = 42,
#         verbose                    = True,
#     )
#
#     pruned_model  = result["model"]
#     params_after  = count_parameters(pruned_model)
#
#     logger.info(f"\nParameters before  : {params_before:,}")
#     logger.info(f"Parameters after   : {params_after:,}")
#     logger.info(f"Parameter reduction: {(1 - params_after/params_before)*100:.2f}%")
#     logger.info(f"\nPer-layer pruning ratios:")
#     for name, ratio in result["layer_pruning_ratios"].items():
#         logger.info(f"  {name:50s}  {ratio*100:6.2f}%")