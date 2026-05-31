import torch
from torch.utils.data import Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder


def make_subset(dataset, fraction=0.1, seed=42):
    """Returns a random subset of the given dataset.
    Args:
        dataset: the dataset to be subset
        fraction: fraction of the dataset to be returned
        seed: random seed
    Returns:
        A random subset of the given dataset.
    """
    torch.manual_seed(seed)
    indices = torch.randperm(len(dataset))[:int(len(dataset) * fraction)]
    return Subset(dataset, indices)



def load_datasets(reduction_fraction=0.1, seed=42):
    """Loads the ImageNet dataset and applies the necessary transformations.
    Args:
        reduction_fraction: fraction of the dataset to be returned
        seed: random seed
    Returns:
        A tuple of (train_dataset, val_dataset) where each dataset is a random subset of the original ImageNet dataset.
    """
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