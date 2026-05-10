import json
import os
from typing import Optional
import wandb
from compression_methods.basic_kd import distill as basic_distill
from compression_methods.magnitude_pruning import prune as mag_prune
from compression_methods.feature_kd import distill as feature_distill
from compression_methods.born_again_kd import distill as born_again_distill
from compression_methods.taylor_pruning import prune as taylor_prune
from compression_methods.hessian_pruning import prune as hessian_prune
from compression_methods.slicegpt import prune as slicegpt_prune
from compression_methods.wanda_pruning import prune as wanda_prune
import logging
from utils.data_manager import DataManager
from utils.llama_model import LlamaExperimenter
from utils.resnet_model import ResNetExperimenter

logger = logging.getLogger(__name__)

def run_experiment(model_name, dataset, batch_size, epochs, lr, data_fraction, save,
                   seed, device, skip_finetune_path: Optional[str], pruning_ratio=0.1, blocks=None,
                   hidden_layer_reduction=2):
    """
    Run other compression methods on a given model and dataset to compare performance against linearity-based compression.
    Args:
        model_name (str): The architecture to use (e.g., 'llama-2-7b').
        dataset (str): The dataset to use (e.g., 'imagenet').
        batch_size (int): The batch size for training and evaluation.
        epochs (int): The number of epochs for fine-tuning.
        lr (float): The learning rate for the optimizer.
        data_fraction (float): Fraction of data to use in training and evaluation.
        save (bool): Whether to save the trained models and results.
        seed (int): The random seed for reproducibility.
        device (str): The device to run the experiments on (e.g., 'cpu', 'cuda').
        skip_finetune_path (str): The path to look for a finetuned model saved to disk if skipping is enabled.
        pruning_ratio (float): Pruning ratio for pruning (e.g., 0.5).
        blocks: Blocks layout for the student ResNet model. Default is [2,2,2]. Ignored for Llama model.
        hidden_layer_reduction: Number of hidden layers to remove for student Llama model. Default is 2, meaning that 18 layers of Llama-3.2-1b will become 16 layers. Ignored for ResNet model.
        return_for_relation (bool): Whether to return the pruning ratios and student networks for relation analysis. Default is False.
    Returns:
        None if return_for_relation is False, otherwise a dictionary containing the pruning ratios and student networks for relation analysis.
    """
    logger.info(f"Running benchmark compression methods for model {model_name} on dataset {dataset}.")
    save_dir = "./results/rq1/benchmarks/" + model_name + "/" + dataset + "/" + str(seed)
    os.makedirs(save_dir, exist_ok=True)

    # Load the model and dataset
    data_handler = DataManager(dataset_name=dataset, batch_size=batch_size, data_fraction=data_fraction, model_name=model_name if "llama" in model_name else None,
                               seed=seed)
    logger.debug(
        f"Dataset loaded with {len(data_handler.train_set)} training samples and {len(data_handler.val_set)} validation samples.")
    if "llama" in model_name:
        experimenter = LlamaExperimenter(model_name=model_name, data_handler=data_handler, batch_size=batch_size, epochs=epochs,
                                         learning_rate=lr, device=device, skip_finetune_path=skip_finetune_path)
    elif "resnet" in model_name:
        experimenter = ResNetExperimenter(model_name=model_name, data_handler=data_handler, batch_size=batch_size,
                                          epochs=epochs, learning_rate=lr,  device=device, skip_finetune_path=skip_finetune_path)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    logger.info("Model initialized.")

    if blocks is None and experimenter.model_name == "resnet18":
        blocks = [1, 1, 2, 2]
    elif blocks is None and experimenter.model_name == "resnet34":
        blocks = [2, 3, 6, 3]
    elif blocks is None and experimenter.model_name == "resnet50":
        blocks = [2, 3, 6, 3]

    # Run pruning
    prune_dict, acc, param, infer, gflops = mag_prune(experimenter, data_handler, device=device,
                                                                  pruning_ratio=pruning_ratio, lr=lr, batch_size=batch_size, epochs=epochs)
    logger.info(f"Magnitude pruning completed. Acc: {acc}, param: {param}, infer: {infer}, gflops: {gflops}")

    mag_results = {
        "accuracy": acc,
        "params": param,
        "inference_time": infer,
        "gflops": gflops,
        "prune_dict": prune_dict,
    }
    wandb.log({
        "magnitude_pruning": mag_results,
    })
    logger.info("Saved magnitude pruning results to Weights & Biases.")

    taylor_results, hessian_results, slicegpt_results, wanda_results = {}, {}, {}, {}
    if "resnet" in model_name:
        prune_dict, acc, param, infer, gflops = taylor_prune(experimenter, data_handler, device=device,
                                                          pruning_ratio=pruning_ratio, lr=lr, batch_size=batch_size,
                                                          epochs=epochs)
        logger.info(f"Taylor pruning completed. Acc: {acc}, param: {param}, infer: {infer}, gflops: {gflops}")

        taylor_results = {
            "accuracy": acc,
            "params": param,
            "inference_time": infer,
            "gflops": gflops,
            "prune_dict": prune_dict,
        }
        wandb.log({
            "taylor_pruning": taylor_results,
        })
        logger.info("Saved Taylor pruning results to Weights & Biases.")

        prune_dict, acc, param, infer, gflops = hessian_prune(experimenter, data_handler, device=device,
                                                          pruning_ratio=pruning_ratio, lr=lr, batch_size=batch_size,
                                                          epochs=epochs)
        logger.info(f"Hessian pruning completed. Acc: {acc}, param: {param}, infer: {infer}, gflops: {gflops}")

        hessian_results = {
            "accuracy": acc,
            "params": param,
            "inference_time": infer,
            "gflops": gflops,
            "prune_dict": prune_dict,
        }
        wandb.log({
            "hessian_pruning": hessian_results,
        })
        logger.info("Saved Hessian pruning results to Weights & Biases.")
    else:
        prune_dict, acc, param, infer, gflops = slicegpt_prune(experimenter, data_handler, device=device,
                                                          pruning_ratio=pruning_ratio, lr=lr, batch_size=batch_size,
                                                          epochs=epochs)
        logger.info(f"SliceGPT pruning completed. Acc: {acc}, param: {param}, infer: {infer}, gflops: {gflops}")

        slicegpt_results = {
            "accuracy": acc,
            "params": param,
            "inference_time": infer,
            "gflops": gflops,
            "prune_dict": prune_dict,
        }
        wandb.log({
            "slicegpt_pruning": slicegpt_results,
        })
        logger.info("Saved SliceGPT pruning results to Weights & Biases.")

        prune_dict, acc, param, infer, gflops = wanda_prune(experimenter, data_handler, device=device,
                                                          pruning_ratio=pruning_ratio, lr=lr, batch_size=batch_size,
                                                          epochs=epochs)
        logger.info(f"Wanda pruning completed. Acc: {acc}, param: {param}, infer: {infer}, gflops: {gflops}")

        wanda_results = {
            "accuracy": acc,
            "params": param,
            "inference_time": infer,
            "gflops": gflops,
            "prune_dict": prune_dict,
        }
        wandb.log({
            "wanda_pruning": wanda_results,
        })
        logger.info("Saved Wanda pruning results to Weights & Biases.")

    # Run distillation
    _, acc, param, infer, gflops = basic_distill(experimenter, data_handler, device=device,
                                                               lr=lr, epochs=epochs, blocks=blocks, hidden_layer_reduction=hidden_layer_reduction)
    logger.info(f"Distillation completed. Acc: {acc}, param: {param}, infer: {infer}, gflops: {gflops}")

    basic_kd_results = {
                "accuracy": acc,
                "params": param,
                "inference_time": infer,
                "gflops": gflops,
            }

    wandb.log({
        "basic_kd": basic_kd_results,
    })
    logger.info("Saved basic distillation results to Weights & Biases.")

    _, acc, param, infer, gflops = feature_distill(
                experimenter, data_handler, device=device,
                lr=lr, epochs=epochs, blocks=blocks,
                hidden_layer_reduction=hidden_layer_reduction)
    logger.info(f"Feature distillation completed. Acc: {acc}, param: {param}, infer: {infer}, gflops: {gflops}")

    feature_distill_results = {
        "accuracy": acc,
        "params": param,
        "inference_time": infer,
        "gflops": gflops,
    }

    wandb.log({
        "feature_distillation": feature_distill_results,
    })
    logger.info("Saved feature distillation results to Weights & Biases.")

    _, acc, param, infer, gflops = born_again_distill(
                experimenter, data_handler, device=device,
                lr=lr, epochs=epochs, blocks_iterations=blocks,
                hidden_layer_reduction_iterations=[2,3])
    logger.info(f"Born again distillation completed. Acc: {acc}, param: {param}, infer: {infer}, gflops: {gflops}")
    born_again_results = {
        "accuracy": acc,
        "params": param,
        "inference_time": infer,
        "gflops": gflops,
    }

    wandb.log({
        "born_again_distillation": born_again_results,
    })
    logger.info("Saved born again distillation results to Weights & Biases.")

    if save:
        results = {
            "magnitude_pruning": mag_results,
            "basic_kd": basic_kd_results,
            "feature_kd": feature_distill_results,
            "born_again_kd": born_again_results,
        }

        if "resnet" in model_name:
            results["taylor_pruning"] = taylor_results
            results["hessian_pruning"] = hessian_results
        else:
            results["slicegpt_pruning"] = slicegpt_results
            results["wanda_pruning"] = wanda_results

        with open(f"{save_dir}/{model_name}_compression_benchmark_results.json", "w") as f:
            json.dump(results, f, indent=4)

        logger.info(f"Saved results to {save_dir}/{model_name}_compression_benchmark_results.json")

    logger.info("Benchmark completed.")

