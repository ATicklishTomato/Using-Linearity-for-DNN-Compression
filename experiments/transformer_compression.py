import time
import torch
import numpy as np
import wandb
from huggingface_hub import login
from datasets import load_dataset
import re
from tqdm import tqdm
import os
import logging
from transformers import LlamaForCausalLM, LlamaTokenizer

logger = logging.getLogger(__name__)

def clean_text(text):
    text = re.sub(r"[^a-zA-Z0-9\s.,!?']", "", text)  # Remove special characters
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text

# Tokenize and clean the dataset
def preprocess(examples, tokenizer):
    examples["text"] = [clean_text(t) for t in examples["text"]]
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)


def load_datasets(tokenizer, dataset_name, batch_size):
    if dataset_name != "tinystories":
        raise NotImplementedError(f"Dataset {dataset_name} not implemented.")

    if not os.path.exists("./data"):
        os.makedirs("./data")

    if os.path.exists("./data/tiny_stories_train") and os.path.exists("./data/tiny_stories_val"):
        train_set = load_dataset("roneneldan/TinyStories", split="train").load_from_disk("./data/tiny_stories_train")
        val_set = load_dataset("roneneldan/TinyStories", split="validation").load_from_disk("./data/tiny_stories_val")
        logger.info("Datasets loaded from disk.")
        return train_set, val_set

    train_set = load_dataset("roneneldan/TinyStories", split="train")
    val_set = load_dataset("roneneldan/TinyStories", split="validation")

    train_set = train_set.map((lambda x: preprocess(x, tokenizer)), batched=True, batch_size=batch_size)
    val_set = val_set.map((lambda x: preprocess(x, tokenizer)), batched=True, batch_size=batch_size)
    logger.debug("Datasets loaded and preprocessed.")

    # Save to disk for faster loading later
    train_set.save_to_disk("./data/tiny_stories_train")
    val_set.save_to_disk("./data/tiny_stories_val")
    logger.info("Datasets saved to disk.")

    return train_set, val_set

def forward_pass_sanity_check(model, tokenizer, dataset, device='cuda', debug=False):
    model.to(device)
    model.eval()

    logger.debug("Preparing inputs for forward pass...")
    inputs = tokenizer(dataset[0]['text'], return_tensors='pt', truncation=True, padding='max_length', max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    logger.debug("Inputs prepared. Performing forward pass...")
    with torch.no_grad():
        outputs = model(**inputs)

    logger.info("Forward pass successful.")
    return outputs

def print_architecture(model):
    for name, module in model.named_modules():
        logger.info(f"{name}: {module.__class__}")

def retrieve_mean_preactivations(model, tokenizer, dataset, max_batches=30, device='cuda', save=False):
    """Compute the mean of preactivations for each activation layer in the model. Function does this by computing the mean preactivation values over a set of input data. For Llama with RMS normalization before and after self-attention, we can't retrieve preactivations from normalization parameters. Thus, we must calculate the mean of the input of the normalization before activation named 'model.layers.n.post_attention_layernorm' and 'model.layers.n.mlp.act_fn' where n is the layer number.
    Args:
        model: The neural network model.
        tokenizer: The tokenizer for the model.
        dataset: The dataset to compute preactivations on.
        max_batches: Maximum number of batches to process from the dataset.
        device: Device to run the computations on.
        save: Whether to save/load the computed mean preactivations to/from disk.

    Returns:
        dict: A dictionary with layer names as keys and mean preactivation values as values.
    """
    save_path = f"./mean_preactivations_llama2_7b.pt"
    if save and os.path.exists(save_path):
        logger.info("Loading mean preactivations from disk...")
        return torch.load(save_path)

    model.to(device)
    model.eval()
    activation_layers = []

    # Storage
    channel_sums = {}     # name -> tensor [D]
    sample_counts = {}    # name -> int
    hooks = []

    logger.info("Identifying activation layers and registering hooks...")
    # Identify activation layers
    for name, module in model.named_modules():
        if re.match(r'model\.layers\.\d+\.mlp\.act_fn', name) or re.match(r'model\.layers\.\d+\.post_attention_layernorm', name):
            activation_layers.append((name, module))
    logger.debug("Identified layers, setting hooks...")
    # Define hook to capture preactivations
    def get_preactivation_hook(name):
        def hook(module, input, output):
            # input[0] shape: [B, T, D]
            x = input[0].detach()

            B, T, D = x.shape

            # Mean over batch + sequence, keep hidden dimension
            per_dim_batch_mean = x.mean(dim=(0, 1))  # [D]

            if name not in channel_sums:
                channel_sums[name] = per_dim_batch_mean * (B * T)
                sample_counts[name] = B * T
            else:
                channel_sums[name] += per_dim_batch_mean * (B * T)
                sample_counts[name] += B * T

        return hook

    # Register hooks
    for name, module in activation_layers:
        hooks.append(module.register_forward_hook(get_preactivation_hook(name)))
    logger.debug("Hooks registered. Performing forward passes...")
    # Forward pass through the data
    with torch.no_grad():
        for i in tqdm(range(min(max_batches, len(dataset))), desc="Processing samples for preactivations", leave=False):
            inputs = tokenizer(dataset[i]['text'], return_tensors='pt', truncation=True, padding='max_length', max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            model(**inputs)
    logger.debug("Forward passes complete. Computing mean preactivations...")

    # Compute mean preactivations
    mean_preactivations = {}
    for name in channel_sums:
        per_dim_mean = channel_sums[name] / sample_counts[name]  # [D]
        mean_preactivations[name] = per_dim_mean.mean().item()   # scalar

    logger.debug("Mean preactivations computed.")
    # Remove hooks
    for hook in hooks:
        hook.remove()
    logger.debug("Hooks removed.")

    if save:
        torch.save(mean_preactivations, save_path)
        logger.info("Mean preactivations saved to disk.")

    return mean_preactivations

def map_mean_preactivations(mean_preactivations):
    # Map the preactivations of the activations to the preceding attention layer for clarity
    mapped_mean_preactivations = {}
    for layer_name, mean_val in mean_preactivations.items():
        match = re.match(r'model\.layers\.(\d+)\.mlp\.act_fn', layer_name)
        if match:
            layer_num = match.group(1)
            mapped_mean_preactivations[f'model.layers.{layer_num}.self_attn'] = mean_val
            logger.debug(f"Mapped model.layers.{layer_num}.self_attn with mean preactivation {mean_val}")

    return mapped_mean_preactivations

def choose_threshold(mean_preactivations, percentile=25):
    """Choose a threshold based on the given percentile of mean preactivation values."""
    values = list(mean_preactivations.values())
    threshold = np.percentile(values, percentile)
    logger.debug(f"Chosen threshold at {percentile}th percentile: {threshold}")
    return threshold

def identify_linear_layers(mean_preactivations, threshold):
    """Identify layers with mean preactivation below the threshold."""
    linear_layers = [layer for layer, mean_val in mean_preactivations.items() if mean_val < threshold]
    logger.debug(f"Identified linear layers: {linear_layers}")
    return linear_layers

def group_contiguous_layers(linear_layers):
    """
    Groups contiguous model.layers[i].self_attn modules.
    Returns a list of lists of layer indices.
    """
    indices = sorted(
        int(layer.split(".")[2])
        for layer in linear_layers
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
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = torch.nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states, **kwargs):
        return self.linear(hidden_states)

class IdentityBlock(torch.nn.Module):
    def forward(self, hidden_states, **kwargs):
        return hidden_states

def replace_attention_block(model, layer_group, linear_block):
    first = layer_group[0]

    # Replace whole forward pass
    model.model.layers[first].forward = linear_block.forward

    # Disable remaining layers
    for layer_id in layer_group[1:]:
        model.model.layers[layer_id].forward = IdentityBlock().forward

def prepare_attention_mask(attention_mask, hidden_states):
    """
    Converts tokenizer attention_mask to a format LLaMA attention accepts.
    """
    # attention_mask: (batch, seq)
    # need shape: (batch, 1, 1, seq) or broadcastable
    return attention_mask[:, None, None, :].to(dtype=torch.bool)


@torch.no_grad()
def get_attention_block_output(model, layer_ids, hidden_states, attention_mask):
    batch_size, seq_len, _ = hidden_states.shape
    device = hidden_states.device

    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    attn_mask = prepare_attention_mask(attention_mask, hidden_states)

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
    hidden_size = model.config.hidden_size
    approx = LinearAttentionBlock(hidden_size).to(device)

    optimizer = torch.optim.AdamW(approx.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    model.eval()
    approx.train()

    for epoch in range(epochs):
        for i, batch in tqdm(enumerate(train_dataset), total=min(len(train_dataset), max_batches), desc=f"Training block {layer_group} Epoch {epoch}", leave=False):
            if i >= max_batches:
                break

            inputs = tokenizer(
                batch["text"],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(device)

            with torch.no_grad():
                x = model.model.embed_tokens(inputs.input_ids)
                y_teacher = get_attention_block_output(
                    model,
                    layer_group,
                    x,
                    inputs.attention_mask
                )

            y_student = approx(x)
            loss = loss_fn(y_student, y_teacher)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        logger.info(f"Block {layer_group} | Epoch {epoch} | Loss {loss.item():.6f}")

    return approx

def finetune_model(model, tokenizer, train_dataset, device, epochs=5, lr=2e-5, max_batches=500):
    wandb.watch(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    model.train()
    for epoch in range(epochs):
        loss = None
        for i, batch in tqdm(enumerate(train_dataset), total=min(len(train_dataset), max_batches), desc=f"Finetuning Epoch {epoch}", leave=False):
            if i >= max_batches:
                break

            inputs = tokenizer(
                batch["text"],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(device)

            labels = inputs.input_ids.clone()
            outputs = model(**inputs)
            logits = outputs.logits

            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        logger.info(f"Finetuning | Epoch {epoch} | Loss {loss.item():.6f}")
        wandb.log({"finetune_loss": loss.item()})

    return model

def evaluate_model(model, tokenizer, val_dataset, device, max_batches=200):
    """Evaluate the model on the validation dataset and return average loss and perplexity."""
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    total_loss = 0.0
    total_time = 0.0
    total_tokens = 0
    top_5_accuracy = 0
    with torch.no_grad():
        for i, batch in tqdm(enumerate(val_dataset), total=min(len(val_dataset), max_batches), desc="Evaluating", leave=False):
            if i >= max_batches:
                break

            inputs = tokenizer(
                batch["text"],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(device)

            labels = inputs.input_ids.clone()

            start_time = time.time()
            outputs = model(**inputs)
            end_time = time.time()

            logits = outputs.logits

            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            num_tokens = (labels != tokenizer.pad_token_id).sum().item()

            total_loss += loss.item() * num_tokens
            total_time += end_time - start_time
            total_tokens += num_tokens
            top_5_accuracy += ((logits.topk(5, dim=-1).indices == labels.unsqueeze(-1)).any(dim=-1) & (labels != tokenizer.pad_token_id)).sum().item()

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    parameter_count = sum(p.numel() for p in model.parameters())
    avg_inference_time = total_time / max_batches
    avg_top_5_accuracy = top_5_accuracy / total_tokens

    return avg_loss, perplexity, parameter_count, avg_inference_time, avg_top_5_accuracy

def train_approximation_layers(device: str, tokenizer, train_dataset, groups, save_model: bool,
                               epochs: int, lr: float, max_batches: int):
    """Train linear approximations for specified layer groups in the model.
    Args:
        device (str): The device to run the training on (e.g., 'cuda' or 'cpu').
        tokenizer: The tokenizer for the model.
        train_dataset: The training dataset.
        groups: List of layer groups to approximate.
        save_model (bool): Whether to save the compressed model to disk.
        epochs (int): Number of epochs to train.
        lr (float): Learning rate for training.
        max_batches (int): Maximum number of batches to process during training.
    Returns:
        The compressed model with linear approximations.
    """

    if not os.path.exists("./results/compressed_llama2_7b"):
        compressed_model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf").to(device)
        for layer_group in groups:
            logger.info(f"Training approximation for layer group: {layer_group}")
            linear_block = train_block_approximation(
                compressed_model,
                tokenizer,
                layer_group,
                train_dataset,
                device,
                epochs=epochs,
                lr=lr,
                max_batches=max_batches
            )
            replace_attention_block(compressed_model, layer_group, linear_block)

        if save_model:
            # Save the compressed model
            compressed_model.save_pretrained("./results/compressed_llama2_7b")
            logger.info("Compressed model saved to ./results/compressed_llama2_7b")
    else:
        compressed_model = LlamaForCausalLM.from_pretrained("./results/compressed_llama2_7b").to(device)
        logger.info("Compressed model loaded from ./results/compressed_llama2_7b")

    return compressed_model

def run_transformer_compression_experiment(model: str, dataset: str, batch_size: int, epochs: int, lr: float,
                                           max_batches: int, save: bool, device: str):
    """Run transformer compression experiment on the specified model and dataset.
    Args:
        model (str): The model name or path.
        dataset (str): The dataset name or path.
        batch_size (int): The batch size for training.
        epochs (int): The number of epochs for training.
        lr (float): The learning rate for training.
        max_batches (int): The maximum number of batches to process during training and evaluation.
        save (bool): Whether to save the results to ./results directory.
        device (str): The device to run the experiment on (e.g., 'cuda' or 'cpu').
    """
    logger.info(f"Running transformer compression experiment for {model} and dataset: {dataset}")

    login(token=open("./hf.login").read().strip())

    logger.debug(f"CUDA Available: {torch.cuda.is_available()}")
    logger.debug(f"GPU: {torch.cuda.get_device_name(0)}")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    logger.debug(f"Matmul allow tf32: {torch.backends.cuda.matmul.allow_tf32}")
    logger.debug(f"CuDNN allow tf32: {torch.backends.cudnn.allow_tf32}")

    if model == "llama7b":
        # Load the Llama-2-7B model and tokenizer
        model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token if not already set to prevent errors
    else:
        raise NotImplementedError(f"Model {model} not implemented.")
    logger.info("Model and tokenizer loaded.")

    # Load and preprocess the dataset
    train_dataset, val_dataset = load_datasets(tokenizer, dataset, batch_size)

    mean_preactivations = retrieve_mean_preactivations(model, tokenizer, val_dataset, max_batches=max_batches)
    mapped_mean_preactivations = map_mean_preactivations(mean_preactivations)

    threshold = choose_threshold(mapped_mean_preactivations, percentile=25)
    linear_layers = identify_linear_layers(mapped_mean_preactivations, threshold)
    groups = group_contiguous_layers(linear_layers)

    compressed_model = train_approximation_layers(device, tokenizer, train_dataset, groups, save_model=save,
                                                    epochs=epochs, lr=lr, max_batches=max_batches)

    compressed_model = finetune_model(
        compressed_model,
        tokenizer,
        train_dataset,
        device,
        epochs=epochs,
        lr=lr,
        max_batches=max_batches
    )

    original_model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf").to(device)
    original_loss, original_ppl, original_params, original_time, original_top_5_accuracy = evaluate_model(
        original_model,
        tokenizer,
        val_dataset,
        device,
        max_batches=max_batches,
    )

    # Evaluate compressed model
    compressed_loss, compressed_ppl, compressed_params, compressed_time, compressed_top_5_accuracy = evaluate_model(
        compressed_model,
        tokenizer,
        val_dataset,
        device,
        max_batches=max_batches,
    )

    size_reduction = (1 - compressed_params / original_params) * 100
    speedup = original_time / compressed_time
    accuracy_loss = original_top_5_accuracy - compressed_top_5_accuracy

    # Log results
    logger.info("\nEvaluation Results:")
    logger.info(
        f"Original Model - Loss: {original_loss:.4f}, Perplexity: {original_ppl:.2f}, " +
        f"Parameters: {original_params}, Avg Inference Time: {original_time:.4f}s")
    logger.info(
        f"Compressed Model - Loss: {compressed_loss:.4f}, Perplexity: {compressed_ppl:.2f}, " +
        f"Parameters: {compressed_params}, Avg Inference Time: {compressed_time:.4f}s")
    logger.info(f"Size Reduction: {size_reduction:.2f}%, Inference Speedup: {speedup:.2f}x")
    logger.info(f"Top-5 Accuracy Loss: {accuracy_loss:.4f}")

    if save:
        # Store results in ./results/transformer_compression_results.csv
        if not os.path.exists("./results"):
            os.makedirs("./results")
        results_file = "./results/transformer_compression_results.csv"
        if not os.path.exists(results_file):
            with open(results_file, "w") as f:
                f.write("Model,Dataset,Original Loss,Original Perplexity,Original Params,Original Inference Time," +
                        "Compressed Loss,Compressed Perplexity,Compressed Params,Compressed Inference Time," +
                        "Size Reduction (%),Inference Speedup (x),Top-5 Accuracy Loss\n")
        with open(results_file, "a") as f:
            f.write(f"{model},{dataset},{original_loss:.4f},{original_ppl:.2f},{original_params}," +
                    f"{original_time:.4f},{compressed_loss:.4f},{compressed_ppl:.2f},{compressed_params}," +
                    f"{compressed_time:.4f},{size_reduction:.2f},{speedup:.2f},{accuracy_loss:.4f}\n")
        logger.info(f"Results saved to {results_file}")

    wandb.log({
        "original_loss": original_loss,
        "original_perplexity": original_ppl,
        "original_params": original_params,
        "original_inference_time": original_time,
        "original_top_5_accuracy": original_top_5_accuracy,
        "compressed_loss": compressed_loss,
        "compressed_perplexity": compressed_ppl,
        "compressed_params": compressed_params,
        "compressed_inference_time": compressed_time,
        "compressed_top_5_accuracy": compressed_top_5_accuracy,
        "size_reduction_percent": size_reduction,
        "inference_speedup_x": speedup,
        "top_5_accuracy_loss": accuracy_loss
    })