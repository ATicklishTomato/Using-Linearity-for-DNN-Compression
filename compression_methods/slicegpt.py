from copy import deepcopy
from utils.slicegpt import data_utils, layernorm_fusion, rotate
from utils.slicegpt.slicing_scheduler import ConstSlicingScheduler
from utils.slicegpt.adapters.llama_adapter import LlamaModelAdapter
from compression_methods.magnitude_pruning import finetune_llama, evaluate_llama
import logging
import torch

logger = logging.getLogger(__name__)
debug_mode = logger.getEffectiveLevel() != logging.DEBUG

def compute_before(model):
    layers = model.model.layers
    layer_sizes = {}
    for i, layer in enumerate(layers):
        attn = layer.self_attn

        # --- Attention sparsity ---
        attn_weights = [
            w.weight.data for w in [attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj]
            if hasattr(w, "weight")
        ]

        attn_total = sum(w.numel() for w in attn_weights)

        layer_sizes[f"model.layers.{i}.self_attn"] = attn_total

    return layer_sizes

def generate_prune_dict(model, before_sizes):
    layers = model.model.layers
    prune_dict = {}
    for i, layer in enumerate(layers):
        attn = layer.self_attn

        # --- Attention sparsity ---
        attn_weights = [
            w.weight.data for w in [attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj]
            if hasattr(w, "weight")
        ]

        attn_total = sum(w.numel() for w in attn_weights)

        prune_dict[f"model.layers.{i}.self_attn"] = 1 - (attn_total / before_sizes[f"model.layers.{i}.self_attn"])

    logger.info(f"Prune dict: {prune_dict}")

    return prune_dict

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

    logger.info("Running Llama pruning")
    layer_sizes_before = compute_before(model)
    model = run_slicegpt(model, data_handler, sparsity=pruning_ratio, device=device)
    prune_dict = generate_prune_dict(model, layer_sizes_before)
    logger.info(f"Completed pruning with pruning ratio: {pruning_ratio}")
    finetune_llama(model, data_handler, lr=lr, batch_size=batch_size, epochs=epochs, device=device)
    acc, param_count, inference_time, gflops = evaluate_llama(model, data_handler)

    logger.info("Completed pruning evaluation")
    return prune_dict, acc, param_count, inference_time, gflops


def run_slicegpt(
    model,
    data_handler,
    sparsity: float,
    device: str = "cuda:0",
):
    """
    Runs SliceGPT slicing on a pre-loaded model.
    Returns the compressed model (smaller dense model, not masked).

    Args:
        model:                  A pre-loaded, fine-tuned HF CausalLM model.
        data_handler:           DataManager object containing the calibration dataset.
        sparsity:               Fraction of hidden dim to remove, e.g. 0.2 = 20%.
        device:                 Device to run calibration/slicing on.
    """
    model.eval()
    model_adapter = LlamaModelAdapter(model)

    # Untie embeddings if needed (Llama-3.2 1B/3B)
    if model.config.tie_word_embeddings:
        model.lm_head.weight = torch.nn.Parameter(
            model.lm_head.weight.clone()
        )

    logger.info("Replacing layers...")
    layernorm_fusion.replace_layers(model_adapter)  # <-- missing

    # Attach rotary_emb to each layer so forward can recompute position embeddings
    for layer in model.model.layers:
        layer.rotary_emb = model.model.rotary_emb

    logger.info("Fusing LayerNorms...")
    layernorm_fusion.fuse_modules(model_adapter)

    logger.info("Loading calibration data...")
    train_loader = data_utils.prepare_dataloader(
        dataset=data_handler.train_set,
        tokenizer=data_handler.tokenizer,
        max_seqlen=512,
        batch_size=1,
        nsamples=128,
        varied_seqlen=False,
        seed=data_handler.seed,
    )

    # Step 4: rotate and slice in one pass
    logger.info(f"Rotating and slicing at sparsity={sparsity}...")
    scheduler = ConstSlicingScheduler(int(model_adapter.hidden_size * (1 - sparsity)))
    rotate.rotate_and_slice(model_adapter, train_loader, scheduler)

    logger.info(f"Done. Hidden dim: {model.config.hidden_size} -> {model_adapter.hidden_size}")

    return model_adapter.model