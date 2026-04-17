"""
This file contains an implementation of basic knowledge distillation as described
by Hinton et al. (2015) in "Distilling the Knowledge in a Neural Network"
"""
import time
from copy import deepcopy

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
        blocks = [1, 2, 2, 2]

    model = models.resnet.ResNet(block, blocks)

    return model


def load_teacher_into_student(teacher, student):
    teacher_dict = teacher.state_dict()
    student_dict = student.state_dict()

    filtered_dict = {}

    for k, v in student_dict.items():
        if k in teacher_dict:
            if teacher_dict[k].shape == v.shape:
                filtered_dict[k] = teacher_dict[k]
            else:
                # skip mismatched layers (important for reduced blocks)
                continue

    student_dict.update(filtered_dict)
    student.load_state_dict(student_dict, strict=False)

    return student


def train_student_resnet(teacher_model, student_model, data_handler, optimizer, device='cuda', epochs=5):
    teacher_model.to(device).eval()
    student_model.to(device).train()

    criterion = ForwardKLLoss()
    train_loader = DataLoader(data_handler.train_set, batch_size=data_handler.batch_size, shuffle=True)
    total = 0
    for epoch in range(epochs):
        total_loss = 0.0
        for inputs, labels in tqdm(train_loader, total=len(train_loader), desc="Training student ResNet model", leave=False,
                                   disable=debug_mode):
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

def evaluate_student_resnet(student_model, data_handler, device='cuda'):
    """Validate the student ResNet model and compute accuracy, parameter count, inference time, and GFLOPs.
    Args:
        student_model: ResNet model
        data_handler: DataHandler object
        device: torch.device
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
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, total=len(data_loader), desc="Validating student ResNet model", leave=False,
                                   disable=debug_mode):
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


def distill_student_resnet(experimenter, data_handler, device='cuda', lr=2e-5, epochs=5, blocks=None):
    """Distill a student ResNet model from a teacher ResNet model.
    Args:
        experimenter: Experimenter object
        data_handler: DataHandler object
        device: torch.device
        lr: learning rate
        epochs: number of epochs
        blocks: The blocks list to be passed to ResNet constructor. Default is None, gets set to [2,2,2], aka one block of 2 less than ResNet-18
    Returns:
        student_model: ResNet model
        accuracy:           Top-1 accuracy of the model on the validation set.
        param_count:        Number of parameters in the model on the validation set.
        inference_time:    Inference time of the model on the validation set.
        gflops:             GFLOPs during inference.
    """

    teacher_model = experimenter.model
    student_model = get_student_resnet(blocks=blocks)
    student_model = load_teacher_into_student(teacher_model, student_model)
    logger.info("Loaded student and teacher ResNet models.")

    train_student_resnet(teacher_model, student_model, data_handler,
                         torch.optim.Adam(student_model.parameters(), lr=lr),
                         device=device, epochs=epochs)
    logger.info("Finished training student ResNet model.")

    accuracy, param_count, inference_time, gflops = evaluate_student_resnet(student_model, data_handler, device=device)
    logger.info(f"Finished evaluating student ResNet model. Accuracy: {accuracy}, Params: {param_count}, Inference Time: {inference_time}, GFLOPs: {gflops}")

    return student_model, accuracy, param_count, inference_time, gflops

def get_student_llama(parent_model, hidden_layer_reduction=2):
    """Get student Llama model from LLAMA model.
    Args:
        parent_model: Llama model
        hidden_layer_reduction: Reduction for the number of hidden layers. 18 layers of Llama-3.2-1b will become 16 layers.
    Returns:
        student_model: Llama model
    """
    # Make a copy of the parent model
    student_model = deepcopy(parent_model)
    # Reduce the number of hidden layers
    student_model.model.layers = student_model.model.layers[:-hidden_layer_reduction]

    return student_model

def train_student_llama(teacher_model, student_model, data_handler, optimizer, device='cuda', epochs=5):
    """Train a student Llama model from a teacher Llama model.
    Args:
        teacher_model: Llama model
        student_model: Llama model
        data_handler: DataHandler object
        optimizer: torch.optim.Optimizer
        device: torch.device
        epochs: number of epochs
    """
    teacher_model.to(device).eval()
    student_model.to(device).train()

    criterion = ForwardKLLoss()
    train_loader = DataLoader(data_handler.train_set, batch_size=data_handler.batch_size, shuffle=True)
    total = 0
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}",
                                               total=len(train_loader), leave=False, disable=debug_mode)):
            optimizer.zero_grad()
            inputs = data_handler.tokenizer(batch['text'], return_tensors='pt', padding=True, truncation=True).to(
                device)
            labels = inputs.input_ids.clone()
            labels[labels == data_handler.tokenizer.pad_token_id] = -100

            with torch.autocast("cuda", dtype=torch.bfloat16):
                with torch.no_grad():
                    teacher_outputs = teacher_model(**inputs, labels=labels)
                student_outputs = student_model(**inputs, labels=labels)
                loss = criterion(student_outputs.logits, teacher_outputs.logits, labels)

            loss.backward()
            nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
            optimizer.step()

            total += labels.size(0)
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        logger.info(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}')

def evaluate_student_llama(student_model, data_handler, device='cuda', top_k=5):
    """Evaluate a student Llama model from a teacher Llama model.
    Args:
        student_model: Llama model
        data_handler: DataManager object
        device: torch.device
        top_k: number of top k to check accuracy
    Returns:
        accuracy: Top-k accuracy of the model on the validation set.
        param_count: Number of parameters in the model.
        inference_time: Inference time of the model on the validation set.
        gflops: GFLOPs of the model on the validation set.
    """
    model = student_model.to(device).eval()
    inference_time = 0
    top_k_correct = 0
    total = 0
    val_loader = DataLoader(data_handler.val_set, batch_size=data_handler.batch_size, shuffle=False)
    with torch.no_grad():
        for batch_idx, batch in enumerate(
                tqdm(val_loader, total=len(val_loader), desc="Validating LLaMA model", leave=False, disable=debug_mode)):
            inputs = data_handler.tokenizer(batch['text'], return_tensors='pt', padding=True, truncation=True).to(
                device)
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
            mask = labels != data_handler.tokenizer.pad_token_id
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

        encoded = data_handler.tokenizer(
            batch['text'],
            return_tensors='pt',
            padding=True,
            truncation=True
        ).to(device)

        inputs = (encoded["input_ids"], encoded["attention_mask"])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            macs, _ = count_ops_and_params(model, inputs)
    gflops = 2 * (macs / inference_time) / 1e9  # Convert to GFLOPs

    return accuracy, param_count, avg_inference_time, gflops

def distill_student_llama(experimenter, data_handler, device='cuda', lr=2e-5, epochs=5,
                          hidden_layer_reduction=2, top_k=5):
    """Evaluate a student Llama model from a teacher Llama model.
    Args:
        experimenter: LlamaExperimenter object with the teacher Llama model.
        data_handler: DataManager object for the Llama model.
        device: torch.device
        lr: learning rate
        epochs: number of epochs
        hidden_layer_reduction: number of hidden layers to remove for student Llama model.
        top_k: number of top k to check accuracy
    Returns:
        student_model: student Llama model trained and evaluated.
        accuracy: Top-k accuracy of the model on the validation set.
        param_count: Number of parameters in the model.
        inference_time: Inference time of the model on the validation set.
        gflops: GFLOPs of the model on the validation set.
    """
    teacher_model = experimenter.model
    student_model = get_student_llama(teacher_model, hidden_layer_reduction=hidden_layer_reduction)
    logger.info("Loaded student and teacher LLaMA models.")

    train_student_llama(teacher_model, student_model, data_handler,
                         torch.optim.Adam(student_model.parameters(), lr=lr),
                         device=device, epochs=epochs)
    logger.info("Finished training student LLaMA model.")

    accuracy, param_count, inference_time, gflops = evaluate_student_llama(student_model, data_handler, device=device, top_k=top_k)
    logger.info(f"Finished evaluating student LLaMA model. Accuracy: {accuracy}, Params: {param_count}, Inference Time: {inference_time}, GFLOPs: {gflops}")

    return student_model, accuracy, param_count, inference_time, gflops

def distill(experimenter, data_handler, device='cuda', lr=2e-5, epochs=5, top_k=5, blocks=None,
            hidden_layer_reduction=2):
    """Wrapper for the two distillation functions. Relevant function is determined based on model_name in experimenter.
    Args:
        experimenter: LlamaExperimenter object or ResNetExperimenter object with the teacher model.
        data_handler: DataManager object for the Llama model or ResNet model.
        device: torch.device
        lr: learning rate
        epochs: number of epochs
        top_k: number of top k to check accuracy. Only used for Llama model.
        blocks: Blocks layout for the student ResNet model. Default is [2,2,2]. Ignored for Llama model.
        hidden_layer_reduction: Number of hidden layers to remove for student Llama model. Default is 2, meaning that 18 layers of Llama-3.2-1b will become 16 layers. Ignored for ResNet model.
    Returns:
        student_model: student Llama model or student ResNet model trained and evaluated.
        accuracy: Top-k accuracy of the Llama model or Top-1 accuracy of the ResNet model on the validation set.
        param_count: Number of parameters in the model.
        inference_time: Inference time of the model on the validation set.
        gflops: GFLOPs of the model on the validation set.
    """
    if "resnet" in experimenter.model_name:
        return distill_student_resnet(experimenter, data_handler, device=device, lr=lr, epochs=epochs, blocks=blocks)
    elif "llama" in experimenter.model_name:
        return distill_student_llama(experimenter, data_handler, device=device, lr=lr, epochs=epochs,
                                     hidden_layer_reduction=hidden_layer_reduction, top_k=top_k)
    else:
        raise ValueError(f"Unknown model name: {experimenter.model_name}")