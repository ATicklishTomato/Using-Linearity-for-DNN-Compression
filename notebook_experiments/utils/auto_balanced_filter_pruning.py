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

def prune_batchnorm(bn: nn.BatchNorm2d, keep_idx):
    bn.num_features = len(keep_idx)

    bn.weight = nn.Parameter(bn.weight.data[keep_idx])
    bn.bias = nn.Parameter(bn.bias.data[keep_idx])

    bn.running_mean = bn.running_mean.data[keep_idx].clone()
    bn.running_var = bn.running_var.data[keep_idx].clone()

def prune_conv_input(conv: nn.Conv2d, keep_idx):
    conv.weight = nn.Parameter(conv.weight.data[:, keep_idx])
    conv.in_channels = len(keep_idx)


def prune_stage(stage: nn.ModuleList, keep_idx):
    """
    Correct ResNet-18 stage pruning:
    - conv1 out
    - conv2 in AND out
    - batchnorms
    - downsample path
    """
    for block in stage:
        # ---- conv1 + bn1 ----
        block.conv1.weight = nn.Parameter(
            block.conv1.weight.data[keep_idx]
        )
        block.conv1.out_channels = len(keep_idx)

        prune_batchnorm(block.bn1, keep_idx)

        # ---- conv2 in + out ----
        block.conv2.weight = nn.Parameter(
            block.conv2.weight.data[keep_idx][:, keep_idx]
        )
        block.conv2.in_channels = len(keep_idx)
        block.conv2.out_channels = len(keep_idx)

        prune_batchnorm(block.bn2, keep_idx)

        # ---- downsample path ----
        if block.downsample is not None:
            ds_conv = block.downsample[0]
            ds_bn = block.downsample[1]

            ds_conv.weight = nn.Parameter(
                ds_conv.weight.data[keep_idx]
            )
            ds_conv.out_channels = len(keep_idx)

            prune_batchnorm(ds_bn, keep_idx)



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
            stages = [self.model.layer1, self.model.layer2, self.model.layer3, self.model.layer4]

            prev_keep_idx = None

            for i, stage in enumerate(stages):
                conv = stage[0].conv1

                # If not first stage, prune input channels from previous stage
                if prev_keep_idx is not None:
                    prune_conv_input(conv, prev_keep_idx)

                # Decide how many channels to keep in this stage
                orig = conv.out_channels
                r = max(1, int(orig * (1 - p)))

                keep_idx = select_top_channels(conv, r)

                # Prune entire stage (blocks + BN + residuals)
                prune_stage(stage, keep_idx)

                # Save for next stage
                prev_keep_idx = keep_idx

            hooks = apply_auto_balanced_regularization(
                self.model, self.target_channels, self.alpha
            )
            train_fn(self.model, retrain_epochs)
            for h in hooks:
                h.remove()
