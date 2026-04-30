import time
from copy import deepcopy

import torch
import torch_pruning as tp
from torch import nn, optim
from torch.utils.data import DataLoader
from torch_pruning.utils import count_ops_and_params
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)
debug_mode = logger.getEffectiveLevel() != logging.DEBUG


def prune(experimenter, data_handler, device='cuda', pruning_ratio=0.5, lr=2e-5, batch_size=64, epochs=10):
    """
    Wrapper function for pruning models based on their architecture.
    Args:
        experimenter:   Experimenter object containing the model to be pruned.
        data_handler:   DataManager object.
        device:         Device to use.
        pruning_ratio:  Pruning ratio.
        lr:            Learning rate to use.
        batch_size:   Batch size to use.
        epochs:        Number of epochs to use.
    Returns:
        dict: A dictionary containing the pruning ratios for each layer.
        accuracy: Accuracy of the pruned model.
        param_count: The number of parameters in the pruned model.
        inference_time: Inference time of the pruned model.
        gflops: GFLOPs of the pruned model.
    """
    model = deepcopy(experimenter.model)
    logger.info(f"Made copy of model: {model}")
    if 'resnet' in experimenter.model_name:
        logger.info("Running ResNet pruning")
        prune_dict = prune_resnet(model, data_handler, device, pruning_ratio)
        logger.info(f"Completed pruning with pruning ratio: {pruning_ratio}")
        finetune_resnet(model, data_handler, lr=lr, batch_size=batch_size, epochs=epochs, device=device)
        acc, param_count, inference_time, gflops = evaluate_resnet(model, data_handler, device)
    else:
        logger.info("Running Llama pruning")
        prune_dict = prune_llama(model, data_handler, device, pruning_ratio)
        logger.info(f"Completed pruning with pruning ratio: {pruning_ratio}")
        finetune_llama(model, data_handler, lr=lr, batch_size=batch_size, epochs=epochs, device=device)
        acc, param_count, inference_time, gflops = evaluate_llama(model, data_handler)

    logger.info("Completed pruning evaluation")
    return prune_dict, acc, param_count, inference_time, gflops

@torch.no_grad()
def prune_resnet(model, data_handler, device='cuda', pruning_ratio=0.5):
    """
    Unstructured magnitude pruning (PAT benchmark version).
    Zeros individual weights in Conv2d and Linear layers.
    """
    model.eval()

    # ---------------------------------------------------------
    # Collect all weights globally
    # ---------------------------------------------------------
    all_weights = []
    layer_map = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.out_features == 1000:
            # Leave final fc be
            continue
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            all_weights.append(module.weight.data.abs().flatten())
            layer_map.append(module)

    all_weights_flat = torch.cat(all_weights)

    total_params = all_weights_flat.numel()
    num_prune = int(pruning_ratio * total_params)

    logger.info(f"Total parameters: {total_params}, pruning: {num_prune}")

    if num_prune == 0:
        return {}

    # ---------------------------------------------------------
    # Global magnitude threshold
    # ---------------------------------------------------------
    threshold = torch.topk(
        all_weights_flat,
        num_prune,
        largest=False
    ).values.max()

    # ---------------------------------------------------------
    # Apply unstructured pruning (MASKING ONLY)
    # ---------------------------------------------------------
    original_counts = {}
    pruned_counts = {}

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.out_features == 1000:
            # Leave final fc be
            continue
        if isinstance(module, nn.Conv2d):
            w = module.weight.data

            original_counts[name] = w.numel()

            mask = w.abs() > threshold
            module.weight.data.mul_(mask)

            pruned_counts[name] = mask.sum().item()

        elif isinstance(module, nn.Linear):
            w = module.weight.data

            original_counts[name] = w.numel()

            mask = w.abs() > threshold
            module.weight.data.mul_(mask)

            pruned_counts[name] = mask.sum().item()

    # ---------------------------------------------------------
    # Compute per-layer sparsity ratios (same return style)
    # ---------------------------------------------------------
    pruned_ratios = {}
    for name in original_counts:
        pruned_ratios[name] = 1 - (pruned_counts[name] / original_counts[name]) # We want fraction of pruned weights

    model.eval()
    torch.cuda.empty_cache()

    logger.info("Completed UNSTRUCTURED pruning (PAT benchmark)")

    return pruned_ratios

def prune_resnet_struct(model, data_handler, device='cuda', pruning_ratio=0.5):
    """
    Instantiate pruner based on example: https://github.com/VainF/Torch-Pruning/blob/master/examples/torchvision_models/torchvision_pruning.py
    Args:
        model:          Model to be pruned.
        data_handler:   DataManager object.
        device:         Device to use.
        pruning_ratio:  Pruning ratio.
    Returns:
        dict: A dictionary containing the pruning ratios for each layer.
    """

    model.to(device).eval()
    ignored_layers = []
    for p in model.parameters():
        p.requires_grad_(True)
    for m in model.modules():
        if isinstance(m, nn.Linear) and m.out_features == 1000:
            ignored_layers.append(m)
    logger.info(f"Prepped {len(ignored_layers)} ignored layers")

    example_inputs = torch.rand((1, *data_handler.train_set[0][0].shape)).to(device)
    importance = tp.importance.GroupMagnitudeImportance(p=1)
    logger.info("Setting up pruner")
    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs=example_inputs,
        importance=importance,
        iterative_steps=1,
        pruning_ratio=pruning_ratio,
        global_pruning=True,
        ignored_layers=ignored_layers,
    )

    logger.info(f"Pruner set up for model name: {model.__class__.__name__}")
    # tp.utils.print_tool.before_pruning(model)

    # Store parameter counts per layer before pruning to compare to pruned later
    original_param_counts = {}
    for name, module in model.named_modules():
        if module not in pruner.ignored_layers:
            # logger.info(module)
            if isinstance(module, nn.Conv2d):
                original_param_counts[name] = module.out_channels
            elif isinstance(module, nn.Linear):
                original_param_counts[name] = module.out_features

    logger.info("Computed original parameter counts for each layer")

    layer_channel_cfg = {}
    for module in model.modules():
        if module not in pruner.ignored_layers:
            # logger.info(module)
            if isinstance(module, nn.Conv2d):
                layer_channel_cfg[module] = module.out_channels
            elif isinstance(module, nn.Linear):
                layer_channel_cfg[module] = module.out_features

    for g in pruner.step(interactive=True):
        g.prune()
    # or
    # pruner.step()

    logger.info("Completed pruning step")

    if isinstance(pruner, (tp.pruner.BNScalePruner, tp.pruner.GroupNormPruner, tp.pruner.GrowingRegPruner)):
        pruner.update_regularizer()  # if the model has been pruned, we need to update the regularizer
        pruner.regularize(model)

    # tp.utils.print_tool.after_pruning(model)

    # Get pruned ratios per layer
    pruned_param_counts = {}
    for name, module in model.named_modules():
        if module not in pruner.ignored_layers:
            if isinstance(module, nn.Conv2d):
                pruned_param_counts[name] = module.out_channels
            elif isinstance(module, nn.Linear):
                pruned_param_counts[name] = module.out_features

    logger.info("Computed pruned parameter counts for each layer")

    pruned_ratios = {}
    for name, original_param_count in original_param_counts.items():
        pruned_ratios[name] = 1 - (pruned_param_counts[name] / original_param_count) # We want fraction of pruned weights

    logger.info("Computed pruned ratios. Finished pruning")

    return pruned_ratios

def finetune_resnet(model, data_handler, lr=2e-5, batch_size=64, epochs=10, device='cuda'):
    """Finetune the ResNet model such that it can be used for linearity metric evaluations."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_loader = DataLoader(data_handler.train_set, batch_size=batch_size, shuffle=True)

    model.to(device).train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        i = 0
        for i, data in tqdm(enumerate(train_loader), total=len(train_loader),
                            desc=f"Finetuning Epoch {epoch + 1}/{epochs}", leave=False, disable=debug_mode):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if i % 100 == 99:  # Print every 100 mini-batches
                logger.info(f"[Epoch {epoch + 1}, Batch {i + 1}] loss: {loss.item():.4f}")

        avg_loss = epoch_loss / (i + 1)
        logger.info(f"Epoch {epoch + 1} completed. Average Loss: {avg_loss:.4f}")

    logger.info("Finished finetuning the ResNet model.")

def evaluate_resnet(model, data_handler, device='cuda'):
        """Validate the ResNet model and compute accuracy, parameter count, inference time, and GFLOPs.
        Args:
            model:          Model to be validated.
            data_handler:   DataManager object.
            device:         Device to use.
        Returns:
            accuracy:           Top-1 accuracy of the model on the validation set.
            param_count:        Number of parameters in the model on the validation set.
            avg_inference_time: Average inference time of the model on the validation set.
            gflops:             GFLOPs during inference.
        """
        model = model.to(device).eval()
        correct = 0
        total = 0
        inference_time = 0
        data_loader = DataLoader(data_handler.val_set, batch_size=data_handler.batch_size, shuffle=False)
        with torch.no_grad():
            for inputs, labels in tqdm(data_loader, total=len(data_loader), desc="Validating ResNet model", leave=False, disable=debug_mode):
                inputs = inputs.to(device)
                labels = labels.to(device)

                start = time.time()
                outputs = model(inputs)
                end = time.time()
                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                inference_time += (end - start)

        accuracy = correct / total

        param_count = sum(p.numel() for p in model.parameters())
        inference_time /= total

        # Compute one more input for TFLOPs computation
        with torch.no_grad():
            example_input = next(iter(data_loader))
            macs, _ = count_ops_and_params(model, example_input[0].to(device))
        gflops = 2 * macs / 1e9  # Convert to GFLOPs

        return accuracy, param_count, inference_time, gflops

def prune_llama(model, data_handler, device='cuda', pruning_ratio=0.5):
    """
    Instantiate pruner based on example: https://github.com/VainF/Torch-Pruning/blob/master/examples/LLMs/prune_llm.py
    Args:
        model:          Model to be pruned.
        data_handler:   DataManager object.
        device:         Device to use.
        pruning_ratio:  Pruning ratio.
    Returns:
        dict: A dictionary containing the pruning ratios for each layer.
    """
    model.eval()

    logger.info("Marked number of heads")

    _is_gqa = (
            model.config.num_attention_heads
            != model.config.num_key_value_heads
    )


    all_weights = []
    layer_weights = []
    layer_map = []

    layers = model.model.layers

    for i, layer in enumerate(layers):
        if layer is model.lm_head:
            # Leave final fc alone
            continue
        mlp = layer.mlp

        gate = mlp.gate_proj if hasattr(mlp, "gate_proj") else mlp.gate_up_proj
        up = mlp.up_proj
        down = mlp.down_proj

        # flatten all FFN weights into one vector per layer
        layer_w = torch.cat([
            gate.weight.flatten(),
            up.weight.flatten(),
            down.weight.flatten()
        ])

        layer_weights.append(layer_w)
        layer_map.append((gate, up, down))

        all_weights.append(layer_w)

    all_weights_flat = torch.cat(all_weights)

    total_params = all_weights_flat.numel()
    num_prune = int(pruning_ratio * total_params)

    logger.info(
        f"Total FFN parameters: {total_params}, "
        f"pruning: {num_prune}"
    )

    if num_prune == 0:
        return {f"layer_{i}": 1.0 for i in range(len(layers))}

    prune_threshold = torch.topk(
        all_weights_flat.abs(),
        num_prune,
        largest=False
    ).values.max()

    pruned_ratios = {}

    for i, layer in enumerate(layers):
        if layer is model.lm_head:
            # Leave final fc alone
            continue
        mlp = layer.mlp

        gate = mlp.gate_proj if hasattr(mlp, "gate_proj") else mlp.gate_up_proj
        up = mlp.up_proj
        down = mlp.down_proj

        before = (
                gate.weight.numel()
                + up.weight.numel()
                + down.weight.numel()
        )

        # create masks
        gate_mask = gate.weight.abs() > prune_threshold
        up_mask = up.weight.abs() > prune_threshold
        down_mask = down.weight.abs() > prune_threshold

        gate.weight.data.mul_(gate_mask)
        up.weight.data.mul_(up_mask)
        down.weight.data.mul_(down_mask)

        after = (
                gate_mask.sum().item()
                + up_mask.sum().item()
                + down_mask.sum().item()
        )

        pruned_ratios[f"model.layers.{i}.self_attn"] =  1 - (after / before) # We want fraction of pruned weights


    model.config.hidden_size = model.lm_head.in_features

    for name, m in model.named_modules():
        if m is model.lm_head:
            # Leave final fc alone
            continue
        if name.endswith("self_attn"):
            if hasattr(m, "q_proj"):
                m.hidden_size = m.q_proj.out_features
            elif hasattr(m, "qkv_proj"):
                m.hidden_size = m.qkv_proj.out_features // 3

            m.num_heads = m.hidden_size // m.head_dim
            model.config.num_attention_heads = m.num_heads

            if not _is_gqa:
                m.num_key_value_heads = m.num_heads
                model.config.num_key_value_heads = m.num_heads

            if hasattr(m, "num_key_value_groups"):
                m.num_key_value_groups = (
                        m.num_heads // model.config.num_key_value_heads
                )

        elif name.endswith("mlp"):
            if hasattr(m, "gate_proj"):
                m.hidden_size = m.gate_proj.in_features
                model.config.intermediate_size = m.gate_proj.out_features
            elif hasattr(m, "gate_up_proj"):
                m.hidden_size = m.gate_up_proj.in_features
                model.config.intermediate_size = (
                        m.gate_up_proj.out_features // 2
                )

    if not _is_gqa:
        model.config.num_key_value_heads = model.config.num_attention_heads

    torch.cuda.empty_cache()
    model.eval()

    logger.info("Completed unstructured pruning")

    return pruned_ratios

def finetune_llama(model, data_handler, lr=2e-5, batch_size=4, epochs=10, device='cuda'):
    """Finetune the LLaMA model such that it can be used for linearity metric evaluations."""

    model.to(device).train()

    optimizer = optim.AdamW(model.parameters(), lr=lr)

    train_loader = DataLoader(data_handler.train_set, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        epoch_loss = 0.0
        batch_idx = 0
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}",
                                               total=len(train_loader), leave=False,
                                               disable=debug_mode)):
            optimizer.zero_grad()

            inputs = data_handler.tokenizer(batch['text'], return_tensors='pt', padding=True, truncation=True).to(
                device)
            labels = inputs.input_ids.clone()
            labels[labels == data_handler.tokenizer.pad_token_id] = -100

            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(**inputs, labels=labels)
                loss = outputs.loss

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

            if batch_idx % 100 == 99:
                logger.info(
                    f"Epoch {epoch + 1}/{epochs} - Batch {batch_idx + 1}/{len(train_loader)} - Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / (batch_idx + 1)
        logger.info(f"Epoch {epoch + 1}/{epochs} - Average Loss: {avg_loss:.4f}")

    logger.info("Finetuning of LLaMA model completed.")

def evaluate_llama(model, data_handler, device='cuda', top_k=5):
    """Validate the LLaMA model and compute accuracy, parameter count, average inference time per token, and GFLOPs.
    Args:
        model: LLaMA model
        data_handler: DataHandler object
        device: torch.device
        top_k (int, optional): The top k accuracy values. Defaults to 5.
    Returns:
        accuracy:           Top-k accuracy of the model on the validation set.
        param_count:        Total number of parameters in the model.
        avg_inference_time: Average inference time per token.
        gflops:             GFLOPs during inference.
    """
    model = model.to(device).eval()
    inference_time = 0
    top_k_correct = 0
    total = 0
    val_loader = DataLoader(data_handler.val_set, batch_size=data_handler.batch_size, shuffle=False)
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, total=len(val_loader), desc="Validating LLaMA model", leave=False, disable=debug_mode)):
            inputs = data_handler.tokenizer(batch['text'], return_tensors='pt', padding=True, truncation=True).to(device)
            labels = inputs.input_ids.clone()

            start_time = time.time()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(**inputs)
            end_time = time.time()
            inference_time += (end_time - start_time)

            # Causal shift
            logits = outputs.logits[:, :-1, :]
            labels = labels[:, 1:]

            _, top_k_preds = torch.topk(logits, k=top_k, dim=-1)

            # Mask out padding
            mask = labels != data_handler.tokenizer.pad_token_id
            correct = (top_k_preds == labels.unsqueeze(-1)).any(dim=-1)

            # Count only correct tokens if not padding token
            top_k_correct += (correct & mask).sum().item()
            total += mask.sum().item()

    accuracy = top_k_correct / total

    param_count = sum(p.numel() for p in model.parameters())
    avg_inference_time = inference_time / total

    # Compute one more input for TFLOPs computation
    with torch.no_grad():
        batch = next(iter(val_loader))

        encoded = data_handler.tokenizer(
            batch['text'],
            return_tensors='pt',
            padding=True,
            truncation=True
        ).to(device)

        inputs = (encoded["input_ids"], encoded["attention_mask"])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            macs, _ = count_ops_and_params(model, inputs)
    gflops = 2 * macs / 1e9  # Convert to GFLOPs

    return accuracy, param_count, avg_inference_time, gflops