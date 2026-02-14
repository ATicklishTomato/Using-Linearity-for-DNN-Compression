import numpy as np
from torch.utils.data import Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder


def make_subset(dataset, fraction=0.1, seed=42):
    np.random.seed(seed)
    indices = np.random.choice(
        len(dataset),
        int(len(dataset) * fraction),
        replace=False
    )
    return Subset(dataset, indices)


def load_datasets(reduction_fraction=0.1, seed=42):
    train_tfms = transforms.Compose([
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

    train_ds = ImageFolder("./data/imagenet_train", transform=train_tfms)
    train_ds = make_subset(train_ds, fraction=reduction_fraction, seed=seed)
    val_ds = ImageFolder("./data/imagenet_val", transform=val_tfms)
    val_ds = make_subset(val_ds, fraction=reduction_fraction, seed=seed)

    return train_ds, val_ds