import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy


def fold_linear_conv_sequences(
    model,
    mean_preacts_conv,
    threshold=0.1,
):
    """
    Fold Conv-BN-ReLU-Conv sequences when the activation is near-linear.

    Args:
        model (nn.Module): ResNet-like model (unchanged architecture)
        mean_preacts_conv (dict): {conv_layer_name: mean preactivation}
        threshold (float): >= threshold => ReLU considered linear

    Returns:
        folded_model (nn.Module)
        folded_pairs (list of tuples)
    """

    model = deepcopy(model)
    folded_pairs = []

    print("\n[Layer Folding] Starting folding pass")
    print(f"[Layer Folding] Linearity threshold: {threshold}\n")

    # ------------------------------------------------------------
    # Helper: fold BN into Conv
    # ------------------------------------------------------------
    def fold_bn_into_conv(conv, bn):
        print(f"    Folding BatchNorm into Conv ({conv.out_channels} channels)")

        W = conv.weight
        if conv.bias is None:
            bias = torch.zeros(W.size(0), device=W.device)
        else:
            bias = conv.bias

        gamma = bn.weight
        beta = bn.bias
        mean = bn.running_mean
        var = bn.running_var
        eps = bn.eps

        std = torch.sqrt(var + eps)
        W_folded = W * (gamma / std).reshape(-1, 1, 1, 1)
        b_folded = beta + (bias - mean) * gamma / std

        conv.weight.data.copy_(W_folded)
        conv.bias = nn.Parameter(b_folded)

    # ------------------------------------------------------------
    # Helper: fold Conv → Conv
    # ------------------------------------------------------------
    def fold_convs(conv1, conv2):
        print(
            f"    Folding Conv layers: "
            f"{conv1.in_channels}→{conv1.out_channels}→{conv2.out_channels}"
        )

        W1 = conv1.weight.data
        W2 = conv2.weight.data

        C_out, C_mid, k2, _ = W2.shape
        _, C_in, k1, _ = W1.shape

        W_fold = torch.zeros(
            (C_out, C_in, k1 + k2 - 1, k1 + k2 - 1),
            device=W1.device,
        )

        for m in range(C_out):
            for i in range(C_in):
                acc = 0
                for j in range(C_mid):
                    acc = acc + F.conv2d(
                        W1[j, i].unsqueeze(0).unsqueeze(0),
                        W2[m, j].unsqueeze(0).unsqueeze(0),
                    ).squeeze()
                W_fold[m, i] = acc

        new_conv = nn.Conv2d(
            in_channels=C_in,
            out_channels=C_out,
            kernel_size=W_fold.shape[-1],
            padding=W_fold.shape[-1] // 2,
            bias=False,
        )
        new_conv.weight.data.copy_(W_fold)
        return new_conv

    # ------------------------------------------------------------
    # Main traversal
    # ------------------------------------------------------------
    for module_name, block in model.named_modules():
        if not isinstance(block, nn.Sequential):
            continue

        print(f"\n[Inspecting block] {module_name}")

        # Block is a sequential with 2 basic blocks of the ResNet architecture. Iterate over them.
        for idx, module in enumerate(block):
            # print(f" Inspecting module: {module}")

            if not hasattr(module, 'conv1') or not hasattr(module, 'bn1') or not hasattr(module, 'relu') or not hasattr(module, 'conv2'):
                print("  Not a Conv-BN-ReLU-Conv block → skipping")
                continue

            conv1 = module.conv1
            bn1 = module.bn1
            relu = module.relu
            conv2 = module.conv2

            if not (
                isinstance(conv1, nn.Conv2d)
                and isinstance(bn1, nn.BatchNorm2d)
                and isinstance(relu, nn.ReLU)
                and isinstance(conv2, nn.Conv2d)
            ):
                print("  Not Conv-BN-ReLU-Conv → skipping")
                continue

            conv1_name = f"{module_name}.{idx}.conv1"
            mean_act = mean_preacts_conv.get(conv1_name, None)

            print(f"  Found Conv-BN-ReLU-Conv at {conv1_name}")

            if mean_act is None:
                print("    No preactivation stats → skipping")
                continue

            print(f"    Mean preactivation: {mean_act:.4f}")

            if mean_act < threshold:
                print("    Below threshold → ReLU not linear")
                continue

            # Validate foldability
            if (
                conv1.stride != (1, 1)
                or conv2.stride != (1, 1)
                or conv1.groups != 1
                or conv2.groups != 1
            ):
                print("    Stride/groups incompatible → skipping")
                continue

            print("    Linearity condition satisfied")
            print("    Removing BatchNorm and ReLU, folding Convs")

            # ----------------------------------------------------
            # Fold BN → Conv1
            # ----------------------------------------------------
            fold_bn_into_conv(conv1, bn1)

            # ----------------------------------------------------
            # Fold Conv1 → Conv2
            # ----------------------------------------------------
            new_conv = fold_convs(conv1, conv2)

            # ----------------------------------------------------
            # Replace modules
            # ----------------------------------------------------
            setattr(module, 'conv1', new_conv)
            setattr(module, 'bn1', nn.Identity())
            setattr(module, 'relu', nn.Identity())
            setattr(module, 'conv2', nn.Identity())

            folded_pairs.append((conv1_name, f"{module_name}.{idx}.conv2"))

            print("    Folding complete")

    print(f"\n[Layer Folding] Done. Folded {len(folded_pairs)} layer pairs.\n")

    return model, folded_pairs


def map_preactivations(model, mean_preacts):
    """Map mean preactivations from BatchNorm layers to their preceding Conv2d layers.
    Args:
        model (nn.Module): The ResNet-like model.
        mean_preacts (dict): A dictionary mapping BatchNorm layer names to their mean preactivation values.
    Returns:
        mean_preacts_conv (dict): A dictionary mapping Conv2d layer names to their mean preactivation values.
    """
    mean_preacts_conv = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            mean = mean_preacts.get(name, 0.0)
            layer = name.split('bn')[0]  # Get the layer name before .bn
            index = name.split('bn')[-1]  # Get the index if present
            mean_preacts_conv[layer + 'conv' + index] = mean  # Copy to preceding Conv2d layer
    return mean_preacts_conv