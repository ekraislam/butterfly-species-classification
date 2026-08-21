"""
Dataset Module: Data loading, preprocessing, and augmentation for Butterfly Classification.
Uses torchvision.datasets.ImageFolder on prepared_dataset splits.
"""

import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_transforms():
    """
    Returns training and validation/test transformation pipelines.
    - Train: moderate, realistic data augmentation (crop, flip, small rotation, mild jitter).
    - Val/Test: deterministic resize, center crop, and normalization.
    """
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    return {
        'train': train_transform,
        'validation': val_test_transform,
        'test': val_test_transform
    }

def get_dataloaders(data_dir="prepared_dataset", batch_size=32, num_workers=0):
    """
    Loads train, validation, and test datasets using ImageFolder and returns DataLoaders.
    """
    data_transforms = get_transforms()
    splits = ['train', 'validation', 'test']
    image_datasets = {}
    dataloaders = {}

    for split in splits:
        split_path = os.path.join(data_dir, split)
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"Directory not found: {split_path}")

        image_datasets[split] = datasets.ImageFolder(
            root=split_path,
            transform=data_transforms[split]
        )

        dataloaders[split] = DataLoader(
            image_datasets[split],
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=False
        )

    class_names = image_datasets['train'].classes
    class_to_idx = image_datasets['train'].class_to_idx
    dataset_sizes = {split: len(image_datasets[split]) for split in splits}

    return dataloaders, dataset_sizes, class_names, class_to_idx

if __name__ == "__main__":
    dataloaders, sizes, classes, mapping = get_dataloaders()
    print("=" * 60)
    print("DATASET MODULE VERIFICATION")
    print("=" * 60)
    print(f"Dataset Sizes: {sizes}")
    print(f"Total Classes ({len(classes)}):")
    for cls_name, idx in mapping.items():
        print(f"  [{idx}] {cls_name} (Train: {len(os.listdir(os.path.join('prepared_dataset', 'train', cls_name)))})")
