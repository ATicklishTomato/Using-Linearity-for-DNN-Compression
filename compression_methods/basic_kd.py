"""
This file contains an implementation of basic knowledge distillation as described
by Hinton et al. (2015) in "Distilling the Knowledge in a Neural Network"
"""
import time

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch_pruning.utils import count_ops_and_params
from torchvision import models
import logging

from tqdm import tqdm

logger = logging.getLogger(__name__)
debug_mode = logger.getEffectiveLevel() != logging.DEBUG


class ForwardKLLoss(torch.nn.Module):
    """
    Class based on https://meta-pytorch.org/torchtune/0.4/tutorials/llama_kd_tutorial.html (last accessed 26-03-2026)
    """
    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, student_logits, teacher_logits, labels) -> torch.Tensor:
        # Implementation from https://github.com/jongwooko/distillm
        # Computes the softmax of the teacher logits
        teacher_prob = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
        # Computes the student log softmax probabilities
        student_logprob = F.log_softmax(student_logits, dim=-1, dtype=torch.float32)
        # Computes the forward KL divergence
        prod_probs = teacher_prob * student_logprob
        # Compute the sum
        x = torch.sum(prod_probs, dim=-1).view(-1)
        # We don't want to include the ignore labels in the average
        mask = (labels != self.ignore_index).int()
        # Loss is averaged over non-ignored targets
        return -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)

def get_student_resnet(blocks=None, block=models.resnet.BasicBlock):
    if blocks is None:
        blocks = [2, 2, 2]

    model = models.resnet.ResNet(block, blocks)

    return model


def train_student_resnet(teacher_model, student_model, data_handler, optimizer, device='cuda', epochs=5, max_batches=100):
    teacher_model.eval()
    student_model.train()

    criterion = ForwardKLLoss()
    train_loader = DataLoader(data_handler.train_set, batch_size=data_handler.batch_size, shuffle=True)
    num_batches = min(max_batches, len(train_loader))
    total = 0
    for epoch in range(epochs):
        total_loss = 0.0
        for inputs, labels in tqdm(train_loader, total=num_batches, desc="Training student ResNet model", leave=False,
                                   disable=debug_mode):
            if total >= max_batches * data_handler.batch_size:
                break
            inputs, labels = inputs.to(device), labels.to(device)

            with torch.no_grad():
                teacher_outputs = teacher_model(inputs)

            student_outputs = student_model(inputs)
            loss = criterion(student_outputs, teacher_outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total += labels.size(0)
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        logger.info(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')

def evaluate_student_resnet(student_model, data_handler, device='cuda', max_batches=100):
    """Validate the student ResNet model and compute accuracy, parameter count, inference time, and GFLOPs.
    Args:
        student_model: ResNet model
        data_handler: DataHandler object
        device: torch.device
        max_batches: maximum number of batches
    Returns:
        accuracy:           Top-1 accuracy of the model on the validation set.
        param_count:        Number of parameters in the model on the validation set.
        avg_inference_time: Average inference time of the model on the validation set.
        gflops:             GFLOPs during inference.
    """
    model = student_model.to(device).eval()
    correct = 0
    total = 0
    inference_time = 0
    data_loader = DataLoader(data_handler.val_set, batch_size=data_handler.batch_size, shuffle=False)
    num_batches = min(max_batches, len(data_loader))
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, total=num_batches, desc="Validating student ResNet model", leave=False,
                                   disable=debug_mode):
            if total >= max_batches * data_handler.batch_size:
                break
            inputs = inputs.to(device)
            labels = labels.to(device)

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
        macs, _ = count_ops_and_params(model, example_input[0].to(device))
    gflops = 2 * (macs / inference_time) / 1e9  # Convert to GFLOPs

    return accuracy, param_count, inference_time, gflops


def distill_student_resnet(experimenter, data_handler, device='cuda', lr=2e-5, max_batches=100):
    """Distill a student ResNet model from a teacher ResNet model.
    Args:
        experimenter: Experimenter object
        data_handler: DataHandler object
        device: torch.device
        lr: learning rate
        max_batches: maximum number of batches
    Returns:
        student_model: ResNet model
        accuracy:           Top-1 accuracy of the model on the validation set.
        param_count:        Number of parameters in the model on the validation set.
        inference_time:    Inference time of the model on the validation set.
        gflops:             GFLOPs during inference.
    """

    teacher_model = experimenter.teacher_model
    student_model = get_student_resnet()

    train_student_resnet(teacher_model, student_model, data_handler,
                         torch.optim.Adam(student_model.parameters(), lr=lr),
                         device=device, max_batches=max_batches)

    accuracy, param_count, inference_time, gflops = evaluate_student_resnet(student_model, data_handler,
                                                                            device=device, max_batches=max_batches)

    return student_model, accuracy, param_count,inference_time, gflops