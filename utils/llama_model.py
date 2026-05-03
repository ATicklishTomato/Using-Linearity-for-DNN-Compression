import glob
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
from transformers import LlamaForCausalLM
from transformers.utils.logging import disable_progress_bar
from torch_pruning.utils import count_ops_and_params

from tqdm import tqdm

logger = logging.getLogger(__name__)
debug_mode = logger.getEffectiveLevel() != logging.DEBUG

class LlamaExperimenter:
    def __init__(self, model_name, data_handler, batch_size, epochs, learning_rate, device='cuda', skip_finetune_path = None):
        self.model_name = model_name
        self.data_handler = data_handler
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.device = device
        self.skip_finetune_path = skip_finetune_path
        self.skipped = False

        match model_name:
            case "llama-2-7b":
                self.model = self._initialize_llama_model("meta-llama/Llama-2-7b-hf")
            case "llama-2-13b":
                self.model = self._initialize_llama_model("meta-llama/Llama-2-13b-hf")
            case "llama-3-1b":
                self.model = self._initialize_llama_model("meta-llama/Llama-3.2-1B")
            case "llama-3-3b":
                self.model = self._initialize_llama_model("meta-llama/Llama-3.2-3B")
            case _:
                raise ValueError(f"Unsupported model: {model_name}.")

        if skip_finetune_path is not None:
            try:
                logger.info(f"Skip finetune path is set. Attempting to find finetuned model to load from {skip_finetune_path}")
                directories = glob.glob(self.skip_finetune_path, recursive=True) # Llama stores models in directories, not single files
                directory = str(next((d for d in directories if os.path.isdir(d)), None)) # Just grab the first matching directory
                logger.info(f"Found save file {directory}, attempting to load")
                self.model.from_pretrained(directory).to(self.device)
                self.model.config.pad_token_id = self.data_handler.tokenizer.eos_token_id # Set pad token again to be safe
                logger.info("Loaded finetuned model from file")
                self.skipped = True
            except Exception as e:
                logger.warning(f"Failed to load model due to {e}. Finetuning anyway")
                self.finetune()
        else:
            self.finetune()

    def _initialize_llama_model(self, model_path):
        """Initialize a LLaMA model with the specified path."""
        if logger.getEffectiveLevel() != logging.DEBUG:
            disable_progress_bar()
        model = LlamaForCausalLM.from_pretrained(model_path).to(self.device)
        model.config.pad_token_id = self.data_handler.tokenizer.eos_token_id
        logger.info(f"Initialized LLaMA model from {model_path}.")
        return model

    def finetune(self):
        """Finetune the LLaMA model such that it can be used for linearity metric evaluations."""

        self.model.to(self.device).train()

        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)

        train_loader = DataLoader(self.data_handler.train_set, batch_size=self.batch_size, shuffle=True)
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            batch_idx = 0
            for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}", total=len(train_loader), leave=False, disable=debug_mode)):
                optimizer.zero_grad()

                inputs = self.data_handler.tokenizer(batch['text'], return_tensors='pt', padding=True, truncation=True).to(self.device)
                labels = inputs.input_ids.clone()
                labels[labels == self.data_handler.tokenizer.pad_token_id] = -100

                with torch.autocast("cuda", dtype=torch.bfloat16):
                    outputs = self.model(**inputs, labels=labels)
                    loss = outputs.loss

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()

                if batch_idx % 100 == 99:
                    logger.info(f"Epoch {epoch+1}/{self.epochs} - Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.4f}")

            avg_loss = epoch_loss / (batch_idx + 1)
            logger.info(f"Epoch {epoch+1}/{self.epochs} - Average Loss: {avg_loss:.4f}")

        logger.info("Finetuning of LLaMA model completed.")

    def validate_model(self, top_k=5):
        """Validate the LLaMA model and compute accuracy, parameter count, average inference time per token, and GFLOPs.
        Args:
            top_k (int, optional): The top k accuracy values. Defaults to 5.
        Returns:
            accuracy:           Top-k accuracy of the model on the validation set.
            param_count:        Total number of parameters in the model.
            avg_inference_time: Average inference time per token.
            gflops:             GFLOPs during inference.
        """
        model = self.model.to(self.device).eval()
        inference_time = 0
        top_k_correct = 0
        total = 0
        val_loader = DataLoader(self.data_handler.val_set, batch_size=self.batch_size, shuffle=False)
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, total=len(val_loader), desc="Validating LLaMA model", leave=False, disable=debug_mode)):
                inputs = self.data_handler.tokenizer(batch['text'], return_tensors='pt', padding=True, truncation=True).to(self.device)
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
                mask = labels != self.data_handler.tokenizer.pad_token_id
                correct = (top_k_preds == labels.unsqueeze(-1)).any(dim=-1)

                # Count only correct tokens if not padding token
                top_k_correct += (correct & mask).sum().item()
                total += mask.sum().item()

        accuracy = top_k_correct / total

        param_count = sum(p.numel() for p in model.parameters())
        avg_inference_time = inference_time / total

        # Compute one more input for TFLOPs computation
        with torch.no_grad():
            batch = next(iter(val_loader))

            encoded = self.data_handler.tokenizer(
                batch['text'],
                return_tensors='pt',
                padding=True,
                truncation=True
            ).to(self.device)

            inputs = (encoded["input_ids"], encoded["attention_mask"])

            with torch.autocast("cuda", dtype=torch.bfloat16):
                macs, _ = count_ops_and_params(model, inputs)
        gflops = 2 * macs / 1e9  # Convert to GFLOPs

        return accuracy, param_count, avg_inference_time, gflops