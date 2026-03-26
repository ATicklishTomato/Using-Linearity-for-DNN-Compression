import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch_pruning.utils import count_ops_and_params
from torchvision import models
import logging

from tqdm import tqdm

logger = logging.getLogger(__name__)
debug_mode = logger.getEffectiveLevel() != logging.DEBUG

class ResNetExperimenter:
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
            case "resnet18":
                self.model = self._initialize_resnet_model(18)
            case "resnet34":
                self.model = self._initialize_resnet_model(34)
            case "resnet50":
                self.model = self._initialize_resnet_model(50)
            case _:
                raise ValueError(f"Unsupported model: {model_name}.")

        self.finetune()

    def _initialize_resnet_model(self, layers):
        """Initialize a ResNet model with the specified number of layers."""
        match layers:
            case 18:
                model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            case 34:
                model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
            case 50:
                model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            case _:
                raise ValueError(f"Unsupported ResNet layers: {layers}.")
        model = model.to(self.device)
        logger.info(f"Initialized ResNet model with {layers} layers.")
        return model

    def finetune(self):
        """Finetune the ResNet model such that it can be used for linearity metric evaluations."""

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        train_loader = DataLoader(self.data_handler.train_set, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            i = 0
            for i, data in tqdm(enumerate(train_loader), total=min(len(train_loader), self.max_batches),
                                desc=f"Finetuning Epoch {epoch+1}/{self.epochs}", leave=False, disable=debug_mode):
                if i >= self.max_batches:
                    break
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                if i % 100 == 99:  # Print every 100 mini-batches
                    logger.info(f"[Epoch {epoch + 1}, Batch {i + 1}] loss: {loss.item():.4f}")

            avg_loss = epoch_loss / (i + 1)
            logger.info(f"Epoch {epoch + 1} completed. Average Loss: {avg_loss:.4f}")

        logger.info("Finished finetuning the ResNet model.")

        if self.save:
            torch.save(self.model.state_dict(), f"./results/{self.model_name}_finetuned.pth")
            logger.info(f"Saved finetuned model to results/{self.model_name}_finetuned.pth")

    def validate_model(self):
        """Validate the ResNet model and compute accuracy, parameter count, inference time, and GFLOPs.
        Returns:
            accuracy:           Top-1 accuracy of the model on the validation set.
            param_count:        Number of parameters in the model on the validation set.
            avg_inference_time: Average inference time of the model on the validation set.
            gflops:             GFLOPs during inference.
        """
        model = self.model.to(self.device).eval()
        correct = 0
        total = 0
        inference_time = 0
        data_loader = DataLoader(self.data_handler.val_set, batch_size=self.batch_size, shuffle=False)
        num_batches = min(self.max_batches, len(data_loader))
        with torch.no_grad():
            for inputs, labels in tqdm(data_loader, total=num_batches, desc="Validating ResNet model", leave=False, disable=debug_mode):
                if total >= self.max_batches * self.batch_size:
                    break
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                start = time.time()
                outputs = model(inputs)
                end = time.time()
                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                inference_time += (end - start)

        accuracy = correct / total

        param_count = sum(p.numel() for p in model.parameters())
        inference_time /= total

        # Compute one more input for TFLOPs computation
        with torch.no_grad():
            example_input = next(iter(data_loader))
            macs, _ = count_ops_and_params(model, example_input[0].to(self.device))
        gflops =  2 * (macs / inference_time) / 1e9  # Convert to GFLOPs

        return accuracy, param_count, inference_time, gflops