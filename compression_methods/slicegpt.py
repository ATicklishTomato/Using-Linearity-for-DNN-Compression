from copy import deepcopy
from slicegpt import data_utils, layernorm_fusion, rotate
from slicegpt.slicing_scheduler import ConstSlicingScheduler
from slicegpt.adapters.llama_adapter import LlamaModelAdapter
from compression_methods.magnitude_pruning import finetune_llama, evaluate_llama
import logging
import torch

logger = logging.getLogger(__name__)
debug_mode = logger.getEffectiveLevel() != logging.DEBUG

def compute_before(model):
    layers = model.model.layers
    layer_sizes = {}
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

        layer_sizes[f"model.layers.{i}.self_attn"] = before

    return layer_sizes

def generate_prune_dict(model, before_sizes):
    layers = model.model.layers
    prune_dict = {}
    for i, layer in enumerate(layers):
        if layer is model.lm_head:
            # Leave final fc alone
            continue
        mlp = layer.mlp

        gate = mlp.gate_proj if hasattr(mlp, "gate_proj") else mlp.gate_up_proj
        up = mlp.up_proj
        down = mlp.down_proj

        after = (
                gate.weight.numel()
                + up.weight.numel()
                + down.weight.numel()
        )

        prune_dict[f"model.layers.{i}.self_attn"] = 1 - (after / before_sizes[f"model.layers.{i}.self_attn"])

    return prune_dict

# pip install git+https://github.com/microsoft/TransformerCompression.git
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
    pruned_model = run_slicegpt(model, data_handler, sparsity=pruning_ratio, device=device)
    prune_dict = generate_prune_dict(pruned_model, layer_sizes_before)
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

    print("Fusing LayerNorms...")
    layernorm_fusion.fuse_modules(model_adapter)

    print("Loading calibration data...")
    train_loader = data_utils.prepare_dataloader(
        dataset=data_handler.train_set,
        tokenizer=data_handler.tokenizer,
        max_seqlen=512,
        batch_size=data_handler.batch_size,
        varied_seqlen=False,
        seed=data_handler.seed,
    )

    def to_tensor_loader(dataloader):
        for batch in dataloader:
            yield batch["input_ids"]

    # Step 4: rotate and slice in one pass
    print(f"Rotating and slicing at sparsity={sparsity}...")
    scheduler = ConstSlicingScheduler(int(model_adapter.hidden_size * (1 - sparsity)))
    rotate.rotate_and_slice(model_adapter, to_tensor_loader(train_loader), scheduler)

    print(f"Done. Hidden dim: {model.config.hidden_size} -> {model_adapter.hidden_size}")

    return model_adapter.model