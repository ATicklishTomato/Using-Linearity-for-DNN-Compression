from copy import deepcopy

import torch
import torch_pruning as tp
from compression_methods.magnitude_pruning import finetune_resnet, evaluate_resnet
import logging

logger = logging.getLogger(__name__)

def prune(experimenter, data_handler, device='cuda', pruning_ratio=0.5,
                           lr=2e-5, batch_size=64, epochs=10):

    # Importance criteria
    model = deepcopy(experimenter.model)
    example_inputs = data_handler.train_set[0][0].unsqueeze(0).to(device) # a single image, e.g., (1, 3, 224, 224)
    imp = tp.importance.TaylorImportance()

    logger.info("Starting pruning with Taylor importance...")

    # Ignore some layers, e.g., the output layer
    ignored_layers = []
    original_counts = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and module.out_features == 1000:
            ignored_layers.append(module) # DO NOT prune the final classifier!
        if isinstance(module, torch.nn.Conv2d):
            original_counts[name] = module.weight.data.numel() # Store the original number of parameters for later comparison

    # Initialize a pruner
    iterative_steps = epochs
    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs,
        importance=imp,
        iterative_steps=iterative_steps,
        pruning_ratio=pruning_ratio,
        ignored_layers=ignored_layers,
        global_pruning=True,
    )

    logger.info(f"Starting iterative pruning for {iterative_steps} steps with pruning ratio {pruning_ratio}...")

    for i in range(iterative_steps):

        # Taylor expansion requires gradients for importance estimation
        if isinstance(imp, tp.importance.TaylorImportance):
            # A dummy loss, please replace it with your loss function and data!
            loss = model(example_inputs).sum()
            loss.backward() # before pruner.step()

        pruner.step()

    logger.info(f"Finished iterative pruning for {iterative_steps} steps.")

    pruned_ratios = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            pruned_count = module.weight.data.numel()
            pruned_ratios[name] = 1 - (pruned_count / original_counts[name])  # We want fraction of pruned weights

    logger.info(f"Pruned ratios for convolutional layers: {pruned_ratios}")

    finetune_resnet(model, data_handler, device=device, lr=lr, batch_size=batch_size, epochs=epochs)

    acc, params, infer_time, gflops = evaluate_resnet(model, data_handler, device=device)

    return pruned_ratios, acc, params, infer_time, gflops