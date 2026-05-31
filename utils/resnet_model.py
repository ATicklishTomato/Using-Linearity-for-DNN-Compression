import glob
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
    def __init__(self, model_name, data_handler, batch_size, epochs, learning_rate,
                 device='cuda', skip_finetune_path=None):
        """Initialize the ResNetExperimenter with the specified parameters.
        Args:
            model_name:         Name of the ResNet model to use (e.g., "resnet18", "resnet34", "resnet50").
            data_handler:       An instance of DataHandler that provides access to the training and validation datasets.
            batch_size:         Batch size for training and validation.
            epochs:             Number of epochs to finetune the model.
            learning_rate:      Learning rate for the optimizer during finetuning.
            device:             Device to run the model on (e.g., 'cuda' or 'cpu').
            skip_finetune_path: Optional path pattern to a pre-finetuned model. If provided, the experimenter will attempt to load this model instead of finetuning from scratch.
        """
        self.model_name = model_name
        self.data_handler = data_handler
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.device = device
        self.skip_finetune_path = skip_finetune_path
        self.skipped = False

        match model_name:
            case "resnet18":
                self.model = self._initialize_resnet_model(18)
            case "resnet34":
                self.model = self._initialize_resnet_model(34)
            case "resnet50":
                self.model = self._initialize_resnet_model(50)
            case _:
                raise ValueError(f"Unsupported model: {model_name}.")

        if skip_finetune_path is not None:
            try:
                logger.info(
                    f"Skip finetune path is set. Attempting to find finetuned model "
                    f"to load from {skip_finetune_path}"
                )
                path = str(glob.glob(self.skip_finetune_path, recursive=True)[0])
                logger.info(f"Found save path {path}, attempting to load")
                self.model.load_state_dict(torch.load(path, weights_only=True))
                logger.info("Loaded finetuned model from file")
                self.skipped = True
            except Exception as e:
                logger.warning(f"Failed to load model due to {e}. Finetuning anyway")
                self.finetune()
        else:
            self.finetune()


    def _initialize_resnet_model(self, layers):
        """Initialize a ResNet model with the specified number of layers.
        Args:
            layers: Number of layers in the ResNet model (e.g., 18, 34, 50).
        Returns:
            A ResNet model initialized with pretrained weights for ImageNet.
        """
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

        train_loader = DataLoader(
            self.data_handler.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True,
            multiprocessing_context='spawn',
        )

        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            i = 0
            for i, data in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Finetuning Epoch {epoch+1}/{self.epochs}",
                leave=False, disable=debug_mode):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                if i % 100 == 99:
                    logger.info(f"[Epoch {epoch + 1}, Batch {i + 1}] loss: {loss.item():.4f}")

            avg_loss = epoch_loss / (i + 1)
            logger.info(f"Epoch {epoch + 1} completed. Average Loss: {avg_loss:.4f}")

        logger.info("Finished finetuning the ResNet model.")

    def validate_model(self):
        """Validate the ResNet model and compute accuracy, parameter count, inference time, and GFLOPs.

        Returns:
            accuracy:           Top-1 accuracy of the model on the validation set.
            param_count:        Number of parameters in the model on the validation set.
            avg_inference_time: Average inference time per sample.
            gflops:             GFLOPs during a single forward pass.
        """
        self.model.to(self.device).eval()
        correct = 0
        total = 0
        inference_time = 0
        data_loader = DataLoader(
            self.data_handler.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=False,
            multiprocessing_context='spawn',
        )

        logger.info("DataLoader initialized.")

        example_input = None
        with torch.no_grad():
            for inputs, labels in tqdm(data_loader, total=len(data_loader), desc="Validating ResNet model", leave=False,
                disable=debug_mode):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                if example_input is None:
                    example_input = inputs

                start = time.time()
                outputs = self.model(inputs)
                end = time.time()
                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                inference_time += end - start

        logger.info("Validation loop completed.")
        accuracy = correct / total

        # count_ops_and_params probes the model with a single forward pass, so
        # it must receive the unwrapped ResNet — DataParallel's scatter/gather
        # logic is not compatible with the single-sample probe it uses internally.
        param_count = sum(p.numel() for p in self.model.parameters())
        inference_time /= total

        with torch.no_grad():
            macs, _ = count_ops_and_params(self.model, example_input.to(self.device))
        gflops = 2 * macs / 1e9

        return accuracy, param_count, inference_time, gflops