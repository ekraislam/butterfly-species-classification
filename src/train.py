"""
Training Pipeline Module: ResNet-18 Transfer Learning on 8-Class Butterfly Dataset.
Includes deterministic seeding, pre-training sanity check, AdamW optimizer,
early stopping, validation tracking, checkpointing, and metric curve plotting.
"""

import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import get_dataloaders
from model import create_model, get_parameter_counts

def set_seed(seed: int = 42):
    """Sets random seeds for deterministic execution."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def perform_sanity_check(model, dataloaders, criterion, optimizer, device):
    """
    Performs a 1-batch forward, loss, and backward sanity check before full training.
    """
    print("\n" + "=" * 60)
    print("PRE-TRAINING SANITY CHECK (1 Batch Forward & Backward)")
    print("=" * 60)
    model.train()
    inputs, labels = next(iter(dataloaders['train']))
    inputs, labels = inputs.to(device), labels.to(device)

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    preds = torch.argmax(outputs, dim=1)
    batch_acc = (preds == labels).float().mean().item() * 100

    print(f"  - Batch Size        : {inputs.size(0)}")
    print(f"  - Input Tensor Shape: {list(inputs.shape)}")
    print(f"  - Loss Output       : {loss.item():.4f}")
    print(f"  - Batch Accuracy    : {batch_acc:.2f}%")
    print(f"  - Gradient Check    : PASSED (loss.backward() executed successfully)")
    print(">> SANITY CHECK PASSED: Model is ready for full training <<\n")

def plot_training_curves(history, save_path="results/training_curves.png"):
    """
    Plots and saves loss and accuracy curves for train and validation splits.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss Curve
    ax1.plot(epochs, history['train_loss'], 'b-o', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-s', label='Validation Loss', linewidth=2)
    ax1.set_title('Cross-Entropy Loss vs. Epochs', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Accuracy Curve
    ax2.plot(epochs, history['train_acc'], 'b-o', label='Train Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-s', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Classification Accuracy vs. Epochs', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Training curves saved to: {save_path}")

def train_model(
    data_dir="prepared_dataset",
    num_epochs=10,
    batch_size=32,
    learning_rate=1e-3,
    weight_decay=1e-2,
    patience=4,
    checkpoint_dir="models",
    results_dir="results",
    seed=42
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Load data
    dataloaders, dataset_sizes, class_names, class_to_idx = get_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=0
    )

    num_classes = len(class_names)
    print(f"Dataset summary: Train={dataset_sizes['train']}, Val={dataset_sizes['validation']}, Test={dataset_sizes['test']}")
    print(f"Classes ({num_classes}): {class_names}")

    # Create model
    model = create_model(num_classes=num_classes, pretrained=True, freeze=True)
    model = model.to(device)

    param_counts = get_parameter_counts(model)
    print(f"Model Parameters: Total={param_counts['total']:,} | Trainable={param_counts['trainable']:,} | Frozen={param_counts['frozen']:,}")

    # Loss, Optimizer, Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

    # Pre-training Sanity Check
    perform_sanity_check(model, dataloaders, criterion, optimizer, device)

    # Re-instantiate a clean model so sanity check doesn't bias epoch 1
    model = create_model(num_classes=num_classes, pretrained=True, freeze=True).to(device)
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

    # Training Loop
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    best_model_state = None

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "butterfly_resnet18_best.pth")

    print("=" * 60)
    print(f"BEGINNING TRAINING ({num_epochs} Epochs Maximum)")
    print("=" * 60)
    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        # --- Training Phase ---
        model.train()
        running_train_loss = 0.0
        running_train_correct = 0

        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            preds = torch.argmax(outputs, dim=1)
            running_train_loss += loss.item() * inputs.size(0)
            running_train_correct += torch.sum(preds == labels.data).item()

        scheduler.step()

        epoch_train_loss = running_train_loss / dataset_sizes['train']
        epoch_train_acc = (running_train_correct / dataset_sizes['train']) * 100.0

        # --- Validation Phase ---
        model.eval()
        running_val_loss = 0.0
        running_val_correct = 0

        with torch.no_grad():
            for inputs, labels in dataloaders['validation']:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                preds = torch.argmax(outputs, dim=1)
                running_val_loss += loss.item() * inputs.size(0)
                running_val_correct += torch.sum(preds == labels.data).item()

        epoch_val_loss = running_val_loss / dataset_sizes['validation']
        epoch_val_acc = (running_val_correct / dataset_sizes['validation']) * 100.0
        epoch_time = time.time() - epoch_start

        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)

        print(
            f"Epoch [{epoch:2d}/{num_epochs:2d}] ({epoch_time:4.1f}s) | "
            f"Train Loss: {epoch_train_loss:.4f} - Train Acc: {epoch_train_acc:5.2f}% | "
            f"Val Loss: {epoch_val_loss:.4f} - Val Acc: {epoch_val_acc:5.2f}%"
        )

        # Checkpoint Best Model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_val_acc = epoch_val_acc
            best_epoch = epoch
            patience_counter = 0
            best_model_state = model.state_dict().copy()

            checkpoint_data = {
                'epoch': best_epoch,
                'model_state_dict': best_model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'val_acc': best_val_acc,
                'train_loss': epoch_train_loss,
                'train_acc': epoch_train_acc,
                'class_names': class_names,
                'class_to_idx': class_to_idx,
                'num_classes': num_classes,
                'config': {
                    'architecture': 'resnet18',
                    'pretrained': True,
                    'batch_size': batch_size,
                    'learning_rate': learning_rate,
                    'weight_decay': weight_decay,
                    'seed': seed
                },
                'history': history
            }
            torch.save(checkpoint_data, checkpoint_path)
            print(f"  --> Saved new best checkpoint at Epoch {epoch} (Val Loss: {best_val_loss:.4f}, Val Acc: {best_val_acc:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[Early Stopping Triggered] No improvement in validation loss for {patience} consecutive epochs.")
                break

    total_training_time = time.time() - start_time
    print("=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)
    print(f"Total Time Taken     : {total_training_time:.1f} seconds ({total_training_time/60:.2f} mins)")
    print(f"Best Epoch           : {best_epoch}")
    print(f"Best Validation Loss : {best_val_loss:.4f}")
    print(f"Best Validation Acc  : {best_val_acc:.2f}%")
    print(f"Saved Checkpoint     : {checkpoint_path}")

    # Plot curves
    plot_path = os.path.join(results_dir, "training_curves.png")
    plot_training_curves(history, save_path=plot_path)

    return checkpoint_path, history, best_val_acc, best_val_loss

if __name__ == "__main__":
    train_model()
