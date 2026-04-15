import json
import os

import torch
import wandb

from compression_methods.basic_kd import distill
from compression_methods.magnitude_pruning import prune

import logging

from utils.data_manager import DataManager
from utils.llama_model import LlamaExperimenter
from utils.resnet_model import ResNetExperimenter

logger = logging.getLogger(__name__)

def benchmark_compression_methods(model_name, dataset, batch_size, epochs, lr, max_batches, save, seed, device,
                                  pruning_ratio=0.5, blocks=None, hidden_layer_reduction=2, return_for_relation=False):
    """
    Run other compression methods on a given model and dataset to compare performance against linearity-based compression.
    Args:
        model_name (str): The architecture to use (e.g., 'llama-2-7b').
        dataset (str): The dataset to use (e.g., 'imagenet').
        batch_size (int): The batch size for training and evaluation.
        epochs (int): The number of epochs for fine-tuning.
        lr (float): The learning rate for the optimizer.
        max_batches (int): The maximum number of batches to process during training/evaluation.
        save (bool): Whether to save the trained models and results.
        seed (int): The random seed for reproducibility.
        device (str): The device to run the experiments on (e.g., 'cpu', 'cuda').
        pruning_ratio (float): Pruning ratio for pruning (e.g., 0.5).
        blocks: Blocks layout for the student ResNet model. Default is [2,2,2]. Ignored for Llama model.
        hidden_layer_reduction: Number of hidden layers to remove for student Llama model. Default is 2, meaning that 18 layers of Llama-3.2-1b will become 16 layers. Ignored for ResNet model.
        return_for_relation (bool): Whether to return the pruning ratios and student networks for relation analysis. Default is False.
    Returns:
        None if return_for_relation is False, otherwise a dictionary containing the pruning ratios and student networks for relation analysis.
    """
    logger.info(f"Running benchmark compression methods for model {model_name} on dataset {dataset}.")
    model_dir_name = "llama" if "llama" in model_name else "resnet"
    save_dir = "./results/rq1/benchmarks/" + model_dir_name + "/" + dataset + "/" + str(seed)
    os.makedirs(save_dir, exist_ok=True)

    # Load the model and dataset
    data_handler = DataManager(dataset_name=dataset, batch_size=batch_size, model_name=model_name if "llama" in model_name else None,
                               reduction_fraction=0.1, seed=seed)  # Reduction fraction is set to 0.1 for faster experimentation, can be adjusted as needed
    logger.debug(
        f"Dataset loaded with {len(data_handler.train_set)} training samples and {len(data_handler.val_set)} validation samples.")
    if "llama" in model_name:
        experimenter = LlamaExperimenter(model_name=model_name, data_handler=data_handler, batch_size=batch_size, epochs=epochs,
                                         learning_rate=lr, max_batches=max_batches, device=device)
    elif "resnet" in model_name:
        experimenter = ResNetExperimenter(model_name=model_name, data_handler=data_handler, batch_size=batch_size,
                                          epochs=epochs, learning_rate=lr, max_batches=max_batches, device=device)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    logger.info("Model initialized.")

    # Run magnitude pruning
    prune_dict, mag_acc, mag_param, mag_infer, mag_gflops = prune(experimenter, data_handler, device=device,
                                                                  pruning_ratio=pruning_ratio, max_batches=max_batches,
                                                                  lr=lr, batch_size=batch_size, epochs=epochs)
    logger.info(f"Magnitude pruning completed. Acc: {mag_acc}, param: {mag_param}, infer: {mag_infer}, gflops: {mag_gflops}")
    logger.info(prune_dict)

    wandb.log({
        "compression_method": "magnitude",
        "accuracy": mag_acc,
        "params": mag_param,
        "inference_time": mag_infer,
        "gflops": mag_gflops,
    })
    logger.info("Saved magnitude pruning results to Weights & Biases.")

    # Run distillation
    student_model, dist_acc, dist_param, dist_infer, dist_gflops = distill(experimenter, data_handler, device=device,
                                                               lr=lr, epochs=epochs, max_batches=max_batches,
                                                               blocks=blocks, hidden_layer_reduction=hidden_layer_reduction)
    student_model.cpu()
    logger.info(f"Distillation completed. Acc: {dist_acc}, param: {dist_param}, infer: {dist_infer}, gflops: {dist_gflops}")

    wandb.log({
        "compression_method": "classic distillation",
        "accuracy": dist_acc,
        "params": dist_param,
        "inference_time": dist_infer,
        "gflops": dist_gflops,
    })
    logger.info("Saved classic distillation results to Weights & Biases.")

    if save:
        if "llama" in model_name:
            # Save llama
            experimenter.model.save_pretrained(f"{save_dir}/compressed_{student_model}")
        else:
            torch.save(student_model.state_dict(), f"{save_dir}/{model_name}_distilled.pth")
        logger.info(f"Saved models to {save_dir} directory.")

        results = {
            "magnitude_pruning": {
                "prune_dict": prune_dict,
                "accuracy": mag_acc,
                "params": mag_param,
                "inference_time": mag_infer,
                "gflops": mag_gflops,
            },
            "classic_distillation": {
                "accuracy": dist_acc,
                "params": dist_param,
                "inference_time": dist_infer,
                "gflops": dist_gflops,
            }
        }

        with open(f"{save_dir}/{model_name}_compression_benchmark_results.json", "w") as f:
            json.dump(results, f, indent=4)

        logger.info(f"Saved results to {save_dir}/{model_name}_compression_benchmark_results.json")

    logger.info("Benchmark completed.")

    if return_for_relation:
        return {
            "magnitude_pruning": prune_dict,
            "classic_distillation": student_model,
        }
    return None

