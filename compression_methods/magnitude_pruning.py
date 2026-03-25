import torch
import torch_pruning as tp
from torch import nn

def prune(experimenter, data_handler, device='cuda', pruning_ratio=0.5):
    """
    Wrapper function for pruning models based on their architecture.
    Args:
        experimenter:   Experimenter object containing the model to be pruned.
        data_handler:   DataManager object.
        device:         Device to use.
        pruning_ratio:  Pruning ratio.
    Returns:
        dict: A dictionary containing the pruning ratios for each layer.
    """

    if 'resnet' in experimenter.model_name:
        return prune_resnet(experimenter.model, data_handler, device, pruning_ratio)
    else:
        return prune_llama(experimenter.model, data_handler, device, pruning_ratio)

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

    print("Model Name: {}".format(model.__class__.__name__))
    tp.utils.print_tool.before_pruning(model)

    # Store parameter counts per layer before pruning to compare to pruned later
    original_param_counts = {}
    for module in model.modules():
        if module not in pruner.ignored_layers:
            # print(module)
            if isinstance(module, nn.Conv2d):
                original_param_counts[module] = module.out_channels
            elif isinstance(module, nn.Linear):
                original_param_counts[module] = module.out_features


    layer_channel_cfg = {}
    for module in model.modules():
        if module not in pruner.ignored_layers:
            # print(module)
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
        # print(g)
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
    print(model.config)

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