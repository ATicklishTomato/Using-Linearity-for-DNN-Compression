"""
Auto-Balanced Filter Pruning (AFP) for ResNet-18 (ImageNet)

Implements:
- Auto-balanced regularization (Eq. 6–13)
- Stage-wise pruning (BasicBlock safe)
- Abreast advancing pruning schedule

Paper:
Auto-Balanced Filter Pruning for Efficient CNNs (AAAI 2018)
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------

def filter_l1(conv: nn.Conv2d):
    """L1 norm per output channel"""
    return conv.weight.data.abs().sum(dim=(1, 2, 3))


def compute_lambda(M, r, eps=1e-12):
    """Eq. (6)"""
    sorted_M, _ = torch.sort(M, descending=True)
    theta = sorted_M[r - 1]

    lambdas = torch.zeros_like(M)
    for i in range(len(M)):
        if M[i] < theta:
            lambdas[i] = 1.0 + torch.log(theta / (M[i] + eps))
        else:
            lambdas[i] = -1.0 - torch.log(M[i] / (theta + eps))
    return lambdas


# ---------------------------------------------------------
# Auto-Balanced Regularization Hook
# ---------------------------------------------------------

class AutoBalancedHook:
    """
    Implements Eq. (12) and auto-balanced tau (Eq. 13)
    """
    def __init__(self, conv, lambdas, alpha):
        self.conv = conv
        self.lambdas = lambdas
        self.alpha = alpha

    def __call__(self, grad):
        W = self.conv.weight.data
        P = self.lambdas > 0
        R = self.lambdas < 0

        S_P = torch.sum(self.lambdas[P] * (W[P] ** 2).sum(dim=(1,2,3)))
        S_R = torch.sum(self.lambdas[R] * (W[R] ** 2).sum(dim=(1,2,3)))

        tau = -self.alpha * (S_P / (S_R + 1e-12))

        reg_grad = torch.zeros_like(W)
        for i in range(W.shape[0]):
            if self.lambdas[i] > 0:
                reg_grad[i] = 2 * self.alpha * self.lambdas[i] * W[i]
            else:
                reg_grad[i] = 2 * tau * self.lambdas[i] * W[i]

        return grad + reg_grad


# ---------------------------------------------------------
# ResNet-18 Stage Handling
# ---------------------------------------------------------

def get_stage_pacesetters(model):
    """One pacesetter conv per stage"""
    return {
        "layer1": model.layer1[0].conv1,
        "layer2": model.layer2[0].conv1,
        "layer3": model.layer3[0].conv1,
        "layer4": model.layer4[0].conv1,
    }


def apply_auto_balanced_regularization(model, target_channels, alpha):
    """
    Registers gradient hooks for pre-training or re-training
    """
    handles = []
    for stage, conv in get_stage_pacesetters(model).items():
        M = filter_l1(conv)
        lambdas = compute_lambda(M, target_channels[stage])
        hook = AutoBalancedHook(conv, lambdas, alpha)
        handles.append(conv.weight.register_hook(hook))
    return handles


# ---------------------------------------------------------
# Structural Pruning (BasicBlock Safe)
# ---------------------------------------------------------

def select_top_channels(conv, r):
    M = filter_l1(conv)
    _, idx = torch.sort(M, descending=True)
    return idx[:r]


def prune_stage(stage: nn.ModuleList, keep_idx):
    """
    Applies identical channel pruning to all blocks in a stage
    """
    for block in stage:
        # conv1: prune output channels
        block.conv1.weight = nn.Parameter(
            block.conv1.weight.data[keep_idx]
        )
        block.conv1.out_channels = len(keep_idx)

        # conv2: prune input channels
        block.conv2.weight = nn.Parameter(
            block.conv2.weight.data[:, keep_idx]
        )
        block.conv2.in_channels = len(keep_idx)

        # downsample if exists
        if block.downsample is not None:
            ds = block.downsample[0]
            ds.weight = nn.Parameter(ds.weight.data[keep_idx])
            ds.out_channels = len(keep_idx)


# ---------------------------------------------------------
# AFP Controller
# ---------------------------------------------------------

class AFPResNet18:
    def __init__(
        self,
        model: nn.Module,
        target_channels: dict,
        alpha: float = 5e-4,
        schedule=(0.5, 0.75, 1.0)
    ):
        self.model = model
        self.target_channels = target_channels
        self.alpha = alpha
        self.schedule = schedule

    def pretrain(self, train_fn, epochs):
        hooks = apply_auto_balanced_regularization(
            self.model, self.target_channels, self.alpha
        )
        train_fn(self.model, epochs)
        for h in hooks:
            h.remove()

    def prune_and_retrain(self, train_fn, retrain_epochs):
        for p in self.schedule:
            for stage_name, stage in zip(
                ["layer1", "layer2", "layer3", "layer4"],
                [
                    self.model.layer1,
                    self.model.layer2,
                    self.model.layer3,
                    self.model.layer4,
                ],
            ):
                conv = stage[0].conv1
                orig = conv.out_channels
                r = max(1, int(orig * (1 - p)))
                keep_idx = select_top_channels(conv, r)
                prune_stage(stage, keep_idx)

            hooks = apply_auto_balanced_regularization(
                self.model, self.target_channels, self.alpha
            )
            train_fn(self.model, retrain_epochs)
            for h in hooks:
                h.remove()
