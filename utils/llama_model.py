import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
from transformers import LlamaForCausalLM, LlamaTokenizer

from tqdm import tqdm

logger = logging.getLogger(__name__)

class LlamaExperimenter:
    def __init__(self, model_name, data_handler, batch_size, epochs, learning_rate, max_batches=None, device='cuda', save=False):
        self.model_name = model_name
        self.data_handler = data_handler
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.max_batches = max_batches if max_batches is not None else len(data_handler.train_set)
        self.device = device
        self.save = save

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

        self.finetune()

    def _initialize_llama_model(self, model_path):
        """Initialize a LLaMA model with the specified path."""
        model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to(self.device)
        logger.info(f"Initialized LLaMA model from {model_path}.")
        return model

    def finetune(self):
        """Finetune the LLaMA model such that it can be used for linearity metric evaluations."""

        self.model.to(self.device).train()

        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        loss_fn = nn.CrossEntropyLoss(ignore_index=self.data_handler.tokenizer.pad_token_id)

        train_loader = DataLoader(self.data_handler.train_set, batch_size=self.batch_size, shuffle=True)
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            batch_idx = 0
            for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}", total=min(self.max_batches, len(train_loader)), leave=False)):
                if batch_idx >= self.max_batches:
                    break

                inputs = self.data_handler.tokenizer(batch['text'], return_tensors='pt', padding=True, truncation=True).to(self.device)
                labels = inputs.input_ids.clone()
                outputs = self.model(**inputs, labels=labels)
                logits = outputs.logits

                loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / (batch_idx + 1)
            logger.info(f"Epoch {epoch+1}/{self.epochs} - Average Loss: {avg_loss:.4f}")

        logger.info("Finetuning of LLaMA model completed.")

        if self.save:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            save_path = f"finetuned_{self.model_name}_{timestamp}"
            self.model.save_pretrained(save_path)
            self.data_handler.tokenizer.save_pretrained(save_path)
            logger.info(f"Finetuned model saved to {save_path}.")

    def validate_model(self):
        """Validate the LLaMA model and compute accuracy, parameter count, and average inference time."""
        model = self.model.to(self.device).eval()
        inference_time = 0
        top_5_correct = 0
        total = 0
        val_loader = DataLoader(self.data_handler.val_set, batch_size=self.batch_size, shuffle=False)
        num_batches = min(self.max_batches, len(val_loader))
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, total=num_batches, desc="Validating LLaMA model", leave=False)):
                if batch_idx >= self.max_batches:
                    break

                inputs = self.data_handler.tokenizer(batch['text'], return_tensors='pt', padding=True, truncation=True).to(self.device)
                labels = inputs.input_ids.clone()

                start_time = time.time()
                outputs = model(**inputs)
                end_time = time.time()
                inference_time += (end_time - start_time)

                logits = outputs.logits
                _, top_5_preds = torch.topk(logits, k=5, dim=-1)
                top_5_correct += (top_5_preds == labels.unsqueeze(-1)).any(dim=-1).sum().item()
                total += labels.size(0)

        accuracy = top_5_correct / total

        param_count = sum(p.numel() for p in model.parameters())
        avg_inference_time = inference_time / total

        return accuracy, param_count, avg_inference_time