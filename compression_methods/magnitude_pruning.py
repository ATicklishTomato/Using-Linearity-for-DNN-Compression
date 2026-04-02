import time
from copy import deepcopy

import torch
import torch_pruning as tp
from torch import nn
from torch.utils.data import DataLoader
from torch_pruning.utils import count_ops_and_params
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)
debug_mode = logger.getEffectiveLevel() != logging.DEBUG


def prune(experimenter, data_handler, device='cuda', pruning_ratio=0.5, max_batches=100):
    """
    Wrapper function for pruning models based on their architecture.
    Args:
        experimenter:   Experimenter object containing the model to be pruned.
        data_handler:   DataManager object.
        device:         Device to use.
        pruning_ratio:  Pruning ratio.
        max_batches:   Maximum number of batches to prune.
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
        acc, param_count, inference_time, gflops = evaluate_resnet(model, data_handler, device, max_batches)
    else:
        logger.info("Running Llama pruning")
        prune_dict = prune_llama(model, data_handler, device, pruning_ratio)
        logger.info(f"Completed pruning with pruning ratio: {pruning_ratio}")
        acc, param_count, inference_time, gflops = evaluate_llama(model, data_handler)

    logger.info("Completed pruning evaluation")
    return prune_dict, acc, param_count, inference_time, gflops

def prune_resnet(model, data_handler, device='cuda', pruning_ratio=0.5):
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

    model.cpu().eval()
    ignored_layers = []
    for p in model.parameters():
        p.requires_grad_(True)
    for m in model.modules():
        if isinstance(m, nn.Linear) and m.out_features == 1000:
            ignored_layers.append(m)

    example_inputs = torch.rand(data_handler.train_set[0][0].shape).to(device)
    importance = tp.importance.GroupMagnitudeImportance(p=1)
    pruner = tp.pruner.GroupNormPruner(
        model,
        example_inputs=example_inputs,
        importance=importance,
        iterative_steps=1,
        pruning_ratio=pruning_ratio,
        global_pruning=False,
        ignored_layers=ignored_layers,
    )

    logger.info("Model Name: {}".format(model.__class__.__name__))
    tp.utils.print_tool.before_pruning(model)

    # Store parameter counts per layer before pruning to compare to pruned later
    original_param_counts = {}
    for module in model.modules():
        if module not in pruner.ignored_layers:
            # logger.info(module)
            if isinstance(module, nn.Conv2d):
                original_param_counts[module] = module.out_channels
            elif isinstance(module, nn.Linear):
                original_param_counts[module] = module.out_features


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

    if isinstance(pruner, (tp.pruner.BNScalePruner, tp.pruner.GroupNormPruner, tp.pruner.GrowingRegPruner)):
        pruner.update_regularizer()  # if the model has been pruned, we need to update the regularizer
        pruner.regularize(model)

    tp.utils.print_tool.after_pruning(model)

    # Get pruned ratios per layer
    pruned_param_counts = {}
    for module in model.modules():
        if module not in pruner.ignored_layers:
            if isinstance(module, nn.Conv2d):
                pruned_param_counts[module] = module.out_channels
            elif isinstance(module, nn.Linear):
                pruned_param_counts[module] = module.out_features

    pruned_ratios = {}
    for name, original_param_count in original_param_counts.items():
        pruned_ratios[name] = original_param_count / pruned_param_counts[name]

    return pruned_ratios

def evaluate_resnet(model, data_handler, device='cuda', max_batches=100):
        """Validate the ResNet model and compute accuracy, parameter count, inference time, and GFLOPs.
        Args:
            model:          Model to be validated.
            data_handler:   DataManager object.
            device:         Device to use.
            max_batches:  Maximum number of batches to use.
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
        num_batches = min(max_batches, len(data_loader))
        with torch.no_grad():
            for inputs, labels in tqdm(data_loader, total=num_batches, desc="Validating ResNet model", leave=False, disable=debug_mode):
                if total >= max_batches * data_handler.batch_size:
                    break
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
        gflops =  2 * (macs / inference_time) / 1e9  # Convert to GFLOPs

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
    tp.utils.print_tool.before_pruning(model)
    text = data_handler.train_set[0]['text']
    inputs = torch.tensor(data_handler.tokenizer.encode(text)).unsqueeze(0).to(device)
    num_heads = {}
    out_channel_groups = {}
    seperate_qkv = False
    for name, m in model.named_modules():
        if name.endswith("self_attn"):
            if hasattr(m, "q_proj"):
                seperate_qkv = True
                num_heads[m.q_proj] = model.config.num_attention_heads
                num_heads[m.k_proj] = model.config.num_key_value_heads
                num_heads[m.v_proj] = model.config.num_key_value_heads
            elif hasattr(m, "qkv_proj"):
                seperate_qkv = False
                num_heads[m.qkv_proj] = model.config.num_attention_heads
        if name.endswith('mlp'):
            if hasattr(m, "gate_up_proj"):
                out_channel_groups[m.gate_up_proj] = 2

    _is_gqa = model.config.num_attention_heads != model.config.num_key_value_heads
    head_pruning_ratio = pruning_ratio
    hidden_size_pruning_ratio = pruning_ratio
    importance = tp.importance.GroupMagnitudeImportance(p=2,
                                                        group_reduction='mean')  # tp.importance.ActivationImportance(p=2, target_types=[torch.nn.Linear])
    pruner = tp.pruner.BasePruner(
        model,
        example_inputs=inputs,
        importance=importance,
        global_pruning=False,
        output_transform=lambda x: x.logits,
        pruning_ratio=hidden_size_pruning_ratio,
        ignored_layers=[model.lm_head],
        num_heads=num_heads,
        prune_num_heads=True,
        prune_head_dims=False,  # we do not prune head dims so that we don't need to prune the ROPE
        head_pruning_ratio=head_pruning_ratio,
        out_channel_groups=out_channel_groups,
        round_to=4,
    )

    # Store parameter counts per layer before pruning to compare to pruned later
    original_param_counts = {}
    for name, m in model.named_modules():
        if name.endswith("self_attn"):
            if hasattr(m, "q_proj"):
                original_param_counts[name] = m.q_proj.out_features
            elif hasattr(m, "qkv_proj"):
                original_param_counts[name] = m.qkv_proj.out_features
        elif name.endswith("mlp"):
            if hasattr(m, "gate_proj"):
                original_param_counts[name] = m.gate_proj.out_features
            elif hasattr(m, "gate_up_proj"):
                original_param_counts[name] = m.gate_up_proj.out_features
            else:
                raise ValueError("Unknown mlp layer")

    for g in pruner.step(interactive=True):
        # logger.info(g)
        g.prune()

    # Update model attributes
    model.config.hidden_size = model.lm_head.in_features
    for name, m in model.named_modules():
        if name.endswith("self_attn"):
            if seperate_qkv:
                m.hidden_size = m.q_proj.out_features
            else:
                m.hidden_size = m.qkv_proj.out_features // 3
            m.num_heads = m.hidden_size // m.head_dim
            model.config.num_attention_heads = m.num_heads
            # m.head_dim = m.q_proj.out_features // m.num_heads
            if not _is_gqa:
                m.num_key_value_heads = m.num_heads
                model.config.num_key_value_heads = m.num_heads
            if hasattr(m, "num_key_value_groups"):
                m.num_key_value_groups = m.num_heads // model.config.num_key_value_heads

        elif name.endswith("mlp"):
            if hasattr(m, "gate_proj"):
                m.hidden_size = m.gate_proj.in_features
                model.config.intermediate_size = m.gate_proj.out_features
            elif hasattr(m, "gate_up_proj"):
                m.hidden_size = m.gate_up_proj.in_features
                model.config.intermediate_size = m.gate_up_proj.out_features // 2
            else:
                raise ValueError("Unknown mlp layer")

    if not _is_gqa:
        model.config.num_key_value_heads = model.config.num_attention_heads
    tp.utils.print_tool.after_pruning(model, do_print=True)
    logger.info(model.config)

    # Get pruned ratios per layer
    pruned_param_counts = {}
    for name, m in model.named_modules():
        if name.endswith("self_attn"):
            if hasattr(m, "q_proj"):
                pruned_param_counts[name] = m.q_proj.out_features
            elif hasattr(m, "qkv_proj"):
                pruned_param_counts[name] = m.qkv_proj.out_features
        elif name.endswith("mlp"):
            if hasattr(m, "gate_proj"):
                pruned_param_counts[name] = m.gate_proj.out_features
            elif hasattr(m, "gate_up_proj"):
                pruned_param_counts[name] = m.gate_up_proj.out_features
            else:
                raise ValueError("Unknown mlp layer")

    pruned_ratios = {}
    for name, original_param_count in original_param_counts.items():
        pruned_ratios[name] = pruned_param_counts[name]/original_param_count

    torch.cuda.empty_cache()
    model.eval()

    return pruned_ratios

def evaluate_llama(model, data_handler, device='cuda', max_batches=100, top_k=5):
    """Validate the LLaMA model and compute accuracy, parameter count, average inference time per token, and GFLOPs.
    Args:
        model: LLaMA model
        data_handler: DataHandler object
        device: torch.device
        max_batches: Maximum number of batches to use
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
    num_batches = min(max_batches, len(val_loader))
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, total=num_batches, desc="Validating LLaMA model", leave=False, disable=debug_mode)):
            if batch_idx >= max_batches:
                break

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
        example_input = next(iter(val_loader))
        example_input = data_handler.tokenizer(example_input['text'], return_tensors='pt', padding=True, truncation=True).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            macs, _ = count_ops_and_params(model, example_input)
    gflops = 2 * (macs / inference_time) / 1e9  # Convert to GFLOPs

    return accuracy, param_count, avg_inference_time, gflops