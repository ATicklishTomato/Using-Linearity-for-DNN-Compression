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

        # Optionally wrap with DataParallel when a second GPU is available.
        # The primary device is kept as self.device; DataParallel handles
        # scattering batches across all visible CUDA devices automatically.
        if torch.cuda.device_count() > 1:
            logger.info(
                f"multi_gpu=True and {torch.cuda.device_count()} GPUs detected. "
                "Wrapping model with nn.DataParallel."
            )
            self.model = nn.DataParallel(self.model)

        if skip_finetune_path is not None:
            try:
                logger.info(
                    f"Skip finetune path is set. Attempting to find finetuned model "
                    f"to load from {skip_finetune_path}"
                )
                path = str(glob.glob(self.skip_finetune_path, recursive=True)[0])
                logger.info(f"Found save path {path}, attempting to load")
                # load_state_dict must target the underlying model, not the
                # DataParallel wrapper, because the saved weights use plain
                # layer names (e.g. "layer1.0.conv1.weight"), not the
                # "module.*" prefix that DataParallel adds.
                self._unwrap().load_state_dict(torch.load(path, weights_only=True))
                logger.info("Loaded finetuned model from file")
                self.skipped = True
            except Exception as e:
                logger.warning(f"Failed to load model due to {e}. Finetuning anyway")
                self.finetune()
        else:
            self.finetune()

    @property
    def raw_model(self) -> nn.Module:
        return self.model.module if isinstance(self.model, nn.DataParallel) else self.model

    def _unwrap(self) -> nn.Module:
        """Return the raw ResNet, regardless of whether DataParallel is active.

        Use this wherever you need the underlying model rather than the
        wrapper — specifically for load_state_dict, state_dict, and
        count_ops_and_params (which probes the model with a single example
        input and does not expect the DataParallel scatter/gather overhead).
        """
        return self.model.module if isinstance(self.model, nn.DataParallel) else self.model

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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
        # Optimise the underlying parameters; works the same whether or not
        # DataParallel is active because DataParallel does not add parameters.
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        train_loader = DataLoader(
            self.data_handler.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True,
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
        model = self.model.to(self.device).eval()
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
        )

        with torch.no_grad():
            for inputs, labels in tqdm(data_loader, total=len(data_loader), desc="Validating ResNet model", leave=False,
                disable=debug_mode):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                start = time.time()
                outputs = model(inputs)
                end = time.time()
                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                inference_time += end - start

        accuracy = correct / total

        # count_ops_and_params probes the model with a single forward pass, so
        # it must receive the unwrapped ResNet — DataParallel's scatter/gather
        # logic is not compatible with the single-sample probe it uses internally.
        raw_model = self._unwrap()
        param_count = sum(p.numel() for p in raw_model.parameters())
        inference_time /= total

        with torch.no_grad():
            example_input = next(iter(data_loader))
            macs, _ = count_ops_and_params(raw_model, example_input[0].to(self.device))
        gflops = 2 * macs / 1e9

        return accuracy, param_count, inference_time, gflops