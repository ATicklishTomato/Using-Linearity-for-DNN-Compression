import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging

from tqdm import tqdm

logger = logging.getLogger(__name__)

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
            case "Resnet-18":
                self.model = self._initialize_resnet_model(18)
            case "Resnet-34":
                self.model = self._initialize_resnet_model(34)
            case "Resnet-50":
                self.model = self._initialize_resnet_model(50)
            case _:
                raise ValueError(f"Unsupported model: {model_name}.")

        self.finetune()

    def _initialize_resnet_model(self, layers):
        """Initialize a ResNet model with the specified number of layers."""
        from torchvision import models
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
            running_loss = 0.0
            for i, data in tqdm(enumerate(train_loader), total=min(len(train_loader), self.max_batches),
                                desc=f"Finetuning Epoch {epoch+1}/{self.epochs}", leave=False):
                if i >= self.max_batches:
                    break
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % 10 == 9:  # Print every 10 mini-batches
                    logger.info(f"[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 10:.3f}")
                    running_loss = 0.0

        logger.info("Finished finetuning the ResNet model.")

        if self.save:
            torch.save(self.model.state_dict(), f"./results/{self.model_name}_finetuned.pth")
            logger.info(f"Saved finetuned model to results/{self.model_name}_finetuned.pth")