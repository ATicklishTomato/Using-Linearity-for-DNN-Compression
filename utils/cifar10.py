import torch
from torch.utils.data import Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10


def make_subset(dataset, fraction=0.1, seed=42):
    torch.manual_seed(seed)
    indices = torch.randperm(len(dataset))[:int(len(dataset) * fraction)]
    return Subset(dataset, indices)


def load_datasets(reduction_fraction=0.1, seed=42):
    # ResNets pretrained on ImageNet expect 224x224 RGB images
    # CIFAR-10 images are 32x32, so resize them.
    train_tfms = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    val_tfms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    train_ds = CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=train_tfms
    )
    train_ds = make_subset(train_ds, fraction=reduction_fraction, seed=seed)

    val_ds = CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=val_tfms
    )
    val_ds = make_subset(val_ds, fraction=reduction_fraction, seed=seed)

    return train_ds, val_ds