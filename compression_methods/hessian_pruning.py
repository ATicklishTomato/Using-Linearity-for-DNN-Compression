from copy import deepcopy

import torch
import torch_pruning as tp
from torch import nn
from torch.utils.data import DataLoader

from compression_methods.magnitude_pruning import finetune_resnet, evaluate_resnet
import logging

logger = logging.getLogger(__name__)

def prune(experimenter, data_handler, device='cuda', pruning_ratio=0.5,
                           lr=2e-5, batch_size=64, epochs=10):

    # Importance criteria
    model = deepcopy(experimenter.model).train().to(device)
    example_inputs = data_handler.train_set[0][0].unsqueeze(0).to(device) # a single image, e.g., (1, 3, 224, 224)
    imp = tp.importance.HessianImportance() # Use activation-based importance
    train_loader = DataLoader(data_handler.train_set, batch_size=batch_size, shuffle=True,
                              num_workers=4,  # try 2–8 depending on CPU
                              pin_memory=True,  # important for GPU transfer
                              prefetch_factor=2,  # batches per worker
                              persistent_workers=True  # avoids worker restart each epoch
                              )

    logger.info("Starting pruning with Hessian importance...")

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
    pruner = tp.pruner.MetaPruner(
        model,
        example_inputs,
        importance=imp,
        iterative_steps=iterative_steps,
        pruning_ratio=pruning_ratio,
        ignored_layers=ignored_layers,
        global_pruning=True,
    )

    logger.info(f"Starting iterative pruning for {iterative_steps} steps with pruning ratio {pruning_ratio}...")
    criterion = nn.CrossEntropyLoss(reduction='none')
    for i in range(iterative_steps):

        # ---- collect activations properly ----
        for j, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            imp.zero_grad()  # clear accumulated gradients
            for l in loss:
                model.zero_grad()  # clear gradients
                l.backward(retain_graph=True)  # simgle-sample gradient
                imp.accumulate_grad(model)  # accumulate g^2

            if j >= 3:  # use a few batches, not 1
                break

        # ---- prune ----
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