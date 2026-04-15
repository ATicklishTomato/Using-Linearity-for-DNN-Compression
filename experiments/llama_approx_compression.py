import numpy as np
import torch
import wandb
from tqdm import tqdm
import os
import logging
from transformers import LlamaForCausalLM
from metrics.linearity_metric_manager import LinearityMetric
from utils.data_manager import DataManager
from utils.llama_model import LlamaExperimenter
import utils.util_functions as utils

logger = logging.getLogger(__name__)
debug_mode = logger.getEffectiveLevel() != logging.DEBUG

def group_contiguous_layers(linear_layers):
    """
    Groups contiguous model.layers[i].self_attn modules.
    Returns a list of lists of layer indices.
    Args:
        linear_layers (dict): A dictionary of layer names to linearity scores for layers identified as linear.
    Returns:
        list of lists: A list where each element is a list of contiguous layer indices that are linear.
    """
    indices = sorted(
        int(layer.split(".")[2])
        for layer in linear_layers.keys()
    )

    groups = []
    current = [indices[0]]

    for prev, curr in zip(indices, indices[1:]):
        if curr == prev + 1:
            current.append(curr)
        else:
            groups.append(current)
            current = [curr]

    groups.append(current)
    logger.debug(f"Grouped contiguous layers: {groups}")
    return groups

class LinearAttentionBlock(torch.nn.Module):
    """A simple Linear Attention Block that can be trained to mimic parts of a model that behave largely linearly"""
    def __init__(self, hidden_size):
        super().__init__()
        logger.debug(f"Initializing Linear Attention Block with hidden size {hidden_size}")
        self.linear = torch.nn.Linear(hidden_size, hidden_size, dtype=torch.float32) # Initialize as fp32 first

    def forward(self, hidden_states, **kwargs):
        return self.linear(hidden_states)

class IdentityBlock(torch.nn.Module):
    """We need a separate IdentityBlock class to ensure we can handle additional arguments from LLama forward pass"""
    def forward(self, hidden_states, **kwargs):
        return hidden_states

def replace_attention_block(model, layer_group, linear_block):
    """
    Replace the existing attention blocks with their linear approximation.
    Args:
        model: The LLaMA model to modify.
        layer_group: A list of layer indices that form a contiguous block to replace.
        linear_block: The Linear Block to replace.
    Returns:
        None. Model is modified in place.
    """
    first = layer_group[0]

    # Replace first layer with trained linear block
    model.model.layers[first] = linear_block

    # Replace remaining layers with identity modules
    for layer_id in layer_group[1:]:
        model.model.layers[layer_id] = IdentityBlock()

def prepare_attention_mask(attention_mask):
    """
    Converts tokenizer attention_mask to a format LLaMA attention accepts.
    Args:
        attention_mask: The attention mask from the tokenizer, typically of shape (batch_size, seq_len).
    Returns:
        A boolean attention mask of shape (batch_size, 1, 1, seq_len) that can be used in the LLaMA attention mechanism.
    """
    # attention_mask: (batch, seq)
    # need shape: (batch, 1, 1, seq) or broadcastable
    return attention_mask[:, None, None, :].to(dtype=torch.bool)


@torch.no_grad()
def get_attention_block_output(model, layer_ids, hidden_states, attention_mask):
    """
    Passes hidden states through the specified attention block layers to get the output for training the linear approximation.
    Args:
        model: The LLaMA model containing the layers.
        layer_ids: A list of layer indices that form the attention block to mimic.
        hidden_states: The input hidden states to the first layer in the block, typically of shape (batch_size, seq_len, hidden_size).
        attention_mask: The attention mask from the tokenizer, typically of shape (batch_size, seq_len).
    Returns:
        The output hidden states after passing through the specified attention block layers, of shape (batch_size, seq_len, hidden_size).
    """
    batch_size, seq_len, _ = hidden_states.shape
    device = hidden_states.device

    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    attn_mask = prepare_attention_mask(attention_mask)

    for layer_id in layer_ids:
        layer = model.model.layers[layer_id]

        residual = hidden_states
        normed = layer.input_layernorm(hidden_states)

        position_embeddings = model.model.rotary_emb(normed, position_ids)

        attn_out, _ = layer.self_attn(
            normed,
            attention_mask=attn_mask,
            position_embeddings=position_embeddings,
            past_key_values=None,
            output_attentions=False,
            use_cache=False,
        )

        hidden_states = residual + attn_out

    return hidden_states


def train_block_approximation(
    model,
    tokenizer,
    layer_group,
    train_dataset,
    device,
    epochs=1,
    lr=2e-4,
    max_batches=200
):
    """
    Trains a linear approximation layer to mimic a section of a LLama model's attention blocks.
    Args:
        model: The LLaMA model to modify.
        tokenizer: The tokenizer to use for tokenization.
        layer_group: A list of layer indices that form a contiguous block to replace.
        train_dataset: The training dataset.
        device: The device to use.
        epochs: The number of epochs to train.
        lr: The learning rate to use.
        max_batches: The maximum number of batches to use.
    Returns:
        The linear approximation layer trained to mimic the specified attention block layers.
    """
    hidden_size = model.config.hidden_size
    approx = LinearAttentionBlock(hidden_size).to(device)

    optimizer = torch.optim.AdamW(approx.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    scaler = torch.amp.GradScaler(device)

    model.eval().to(device)
    approx.train().to(device)

    for epoch in range(epochs):
        for i, batch in tqdm(enumerate(train_dataset), total=min(len(train_dataset), max_batches), desc=f"Training block {layer_group} Epoch {epoch}", leave=False, disable=debug_mode):
            if i >= max_batches:
                break

            inputs = tokenizer(
                batch["text"],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(device)


            with torch.amp.autocast(device):
                with torch.no_grad():
                    x = model.model.embed_tokens(inputs.input_ids)
                    y_teacher = get_attention_block_output(
                        model,
                        layer_group,
                        x,
                        inputs.attention_mask
                    )

                y_student = approx(x)
                logger.debug(f"y_student shape: {y_student.shape}, y_teacher shape: {y_teacher.shape}")
                logger.debug(f"y_student NaN: {torch.any(torch.isnan(y_student))}, y_teacher NaN: {torch.any(torch.isnan(y_teacher))}")
                loss = loss_fn(y_student, y_teacher)

            optimizer.zero_grad()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        logger.info(f"Block {layer_group} | Epoch {epoch} | Loss {loss.item():.6f}")

    if model.config.dtype == torch.float16:
        logger.debug("Reducing dtype to float16 to match model")
        approx = approx.half()
    return approx


def train_approximation_layers(experimenter, data_handler, groups, save_model: bool,
                               epochs: int, lr: float, max_batches: int, device: str, save_path: str = None):
    """Train linear approximations for specified layer groups in the model.
    Args:
        experimenter: The LlamaExperimenter instance containing the model to be compressed and its tokenizer.
        data_handler: The DataHandler instance containing the dataset to be compressed and its tokenizer.
        groups: List of layer groups to approximate.
        save_model (bool): Whether to save the compressed model to disk.
        epochs (int): Number of epochs to train.
        lr (float): Learning rate for training.
        max_batches (int): Maximum number of batches to process during training.
        device (str): The device to run the training on (e.g., 'cuda' or 'cpu').
        save_path (str): Path to save the compressed model to disk.
    Returns:
        The compressed model with linear approximations.
    """
    if save_path is None:
        save_path = "./results"
    if not os.path.exists(f"{save_path}/compressed_{experimenter.model_name}"):
        for layer_group in groups:
            logger.info(f"Training approximation for layer group: {layer_group}")
            linear_block = train_block_approximation(
                experimenter.model,
                data_handler.tokenizer,
                layer_group,
                data_handler.train_set,
                device,
                epochs=epochs,
                lr=lr,
                max_batches=max_batches
            )
            replace_attention_block(experimenter.model, layer_group, linear_block)

        if save_model:
            # Save the compressed model
            experimenter.model.save_pretrained(f"{save_path}/compressed_{experimenter.model_name}")
            logger.info(f"Compressed model saved to {save_path}/compressed_{experimenter.model_name}")
    else:
        experimenter.model = LlamaForCausalLM.from_pretrained(f"{save_path}/compressed_{experimenter.model_name}").to(device)
        logger.info(f"Compressed model loaded from {save_path}/compressed_{experimenter.model_name}")


def run_experiment(model: str, linearity: str, dataset: str, threshold: str, batch_size: int,
                           epochs: int, lr: float, max_batches: int, save: bool, seed: int, device: str, sweep: bool=False):
    """Run the Llama compression experiment. Results are logged and stored to wandb if enabled, and models/results are saved to ./results if enabled.
    Args:
        model (str): The ResNet architecture to use (e.g., 'llama-2-7b').
        linearity (str): The linearity metric to use (e.g., 'mean_preactivation', 'procrustes', or 'fraction').
        dataset (str): The dataset to use (e.g., 'imagenet').
        threshold (str): The threshold for determining linearity (e.g., '75%' or '-0.01').
        batch_size (int): The batch size for training and evaluation.
        epochs (int): The number of epochs for fine-tuning.
        lr (float): The learning rate for the optimizer.
        max_batches (int): The maximum number of batches to process during training/evaluation.
        save (bool): Whether to save the trained models and results.
        seed (int): The random seed for reproducibility.
        device (str): The device to run the experiments on (e.g., 'cpu', 'cuda').
        sweep (bool): Flag that indicates whether an additional metric should be computed to use for a W&B sweep.
    """
    save_dir = "./results/rq1/" + threshold.split(".")[1].split("%")[0] + "/llama/" + dataset

    # ------------------------------------------------------------
    # Load data and model
    # ------------------------------------------------------------
    logger.info(
        f"Running Llama compression experiment with model: {model}, linearity metric: {linearity}, dataset: {dataset}, threshold: {threshold}, batch size: {batch_size}, epochs: {epochs}, learning rate: {lr}, max batches: {max_batches}, save results: {save}, seed: {seed}, device: {device}")
    data_handler = DataManager(dataset_name=dataset, batch_size=batch_size, model_name=model, reduction_fraction=0.1, seed=seed) # Reduction fraction is set to 0.1 for faster experimentation, can be adjusted as needed
    logger.debug(f"Dataset loaded with {len(data_handler.train_set)} training samples and {len(data_handler.val_set)} validation samples.")
    experimenter = LlamaExperimenter(model_name=model, data_handler=data_handler, batch_size=batch_size, epochs=epochs, learning_rate=lr, max_batches=max_batches, device=device, save=save)
    logger.info("Model initialized.")

    # ------------------------------------------------------------
    # Evaluate initial model performance
    # ------------------------------------------------------------
    original_accuracy, original_param_count, original_inference_time, original_gflops = experimenter.validate_model()
    logger.info(
        f"Original model accuracy: {original_accuracy:.4f}, parameters: {original_param_count}, "
        f"inference time: {original_inference_time:.4f} seconds, GFLOPs: {original_gflops:.4f}")

    # ------------------------------------------------------------
    # Compute linearity scores
    # ------------------------------------------------------------
    metric = LinearityMetric(linearity, model, data_handler, threshold, max_batches, device, save)
    linearity_scores = metric.metric_fn(experimenter.model)
    logger.info("Linearity scores computed.")
    logger.debug(f"Linearity scores: {linearity_scores}")
    linear_layers, nonlinear_layers = metric.thresholder(linearity_scores)
    logger.info(f"Determined linear layers: {linear_layers}")
    logger.info(f"Determined non-linear layers: {nonlinear_layers}")

    # ------------------------------------------------------------
    # Group contiguous linear layers and create linear approximation layers
    # ------------------------------------------------------------
    groups = group_contiguous_layers(linear_layers)
    train_approximation_layers(experimenter, data_handler, groups, save_model=save,
                               epochs=epochs, lr=lr, max_batches=max_batches, device=device, save_path=save_dir)
    logger.info("Linear approximation layers trained and integrated into the model.")

    # ------------------------------------------------------------
    # Finetune the compressed model
    # ------------------------------------------------------------
    experimenter.finetune()

    # ------------------------------------------------------------
    # Evaluate compressed model performance
    # ------------------------------------------------------------
    compressed_accuracy, compressed_param_count, compressed_inference_time, compressed_gflops = experimenter.validate_model()
    logger.info(
        f"Compressed model accuracy: {compressed_accuracy:.4f}, parameters: {compressed_param_count}, "
        f"inference time: {compressed_inference_time:.4f} seconds, Gs: {compressed_gflops:.4f}")

    accuracy_loss = utils.accuracy_loss(original_accuracy, compressed_accuracy)
    param_compression_ratio = utils.compression_ratio(original_param_count, compressed_param_count)
    speedup = utils.speedup(original_inference_time, compressed_inference_time)
    gflop_reduction = utils.gflop_reduction(original_gflops, compressed_gflops)
    logger.info(f"Accuracy loss: {accuracy_loss:.4f}, Parameter compression ratio: {param_compression_ratio:.4f}, "
                f"Speedup: {speedup:.4f}x, GFLOP reduction: {gflop_reduction:.4f}")

    # ------------------------------------------------------------
    # Log results to wandb and save models/results if enabled
    # ------------------------------------------------------------
    if save:
        import os
        import json
        os.makedirs(save_dir, exist_ok=True)

        # Save compressed model
        torch.save(experimenter.model.state_dict(), f"{save_dir}/{model}_compressed.pth")

        # Save results
        results = {
            "original_accuracy": original_accuracy,
            "original_param_count": original_param_count,
            "original_inference_time": original_inference_time,
            "compressed_accuracy": compressed_accuracy,
            "compressed_param_count": compressed_param_count,
            "compressed_inference_time": compressed_inference_time,
            "compressed_groups": groups,
            "accuracy_loss": accuracy_loss,
            "param_compression_ratio": param_compression_ratio,
            "speedup": speedup,
            "gflop_reduction": gflop_reduction,
        }
        with open(f"{save_dir}/{model}_folding_results.json", "w") as f:
            json.dump(results, f, indent=4)
        logger.info(
            f"Saved compressed model and results to {save_dir}/{model}_compressed.pth and {save_dir}/{model}_folding_results.json")

    logging_data = {
        "original_accuracy": original_accuracy,
        "original_param_count": original_param_count,
        "original_inference_time": original_inference_time,
        "compressed_accuracy": compressed_accuracy,
        "compressed_param_count": compressed_param_count,
        "compressed_inference_time": compressed_inference_time,
        "compressed_groups": groups,
        "accuracy_loss": accuracy_loss,
        "param_compression_ratio": param_compression_ratio,
        "speedup": speedup,
        "gflop_reduction": gflop_reduction,
    }

    if sweep:
        # Compute separate metrics that can all be maximized
        accuracy_retention = compressed_accuracy / original_accuracy
        compression_ratio = original_param_count / compressed_param_count
        speedup = original_inference_time / compressed_inference_time

        # Edit these weights as needed to balance importance of metrics
        alpha, beta, gamma = 4, 1, 1

        # Compute combined metric
        compression_score = np.pow(accuracy_retention, alpha) * np.pow(compression_ratio, beta) * np.pow(speedup, gamma)

        # Add to data
        logging_data["compression_score"] = compression_score

    wandb.log(logging_data)
    logger.info("Logged results to Weights & Biases")
    logger.info("Experiment completed.")