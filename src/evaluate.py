"""
Evaluation Module: Model Assessment on Prepared Test Set.
Computes Accuracy, Precision, Recall, F1-Score, Classification Report,
Per-Class Accuracy, and generates a formatted Confusion Matrix Heatmap.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score

from dataset import get_dataloaders
from model import create_model

def evaluate_model(
    checkpoint_path="models/butterfly_resnet18_best.pth",
    data_dir="prepared_dataset",
    results_dir="results"
):
    print("=" * 60)
    print("TEST SET EVALUATION & METRIC REPORTING")
    print("=" * 60)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluation Device: {device}")
    print(f"Loading checkpoint: {checkpoint_path}")

    # Load Checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    class_names = checkpoint['class_names']
    num_classes = checkpoint['num_classes']

    print(f"Loaded model from Best Epoch {checkpoint['epoch']} (Validation Acc: {checkpoint['val_acc']:.2f}%)")
    print(f"Classes ({num_classes}): {class_names}")

    # Instantiate model and load state
    model = create_model(num_classes=num_classes, pretrained=False, freeze=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Load Test DataLoader ONLY
    dataloaders, dataset_sizes, _, _ = get_dataloaders(data_dir=data_dir, batch_size=32, num_workers=0)
    test_loader = dataloaders['test']
    total_test_samples = dataset_sizes['test']

    print(f"Total Test Samples to Evaluate: {total_test_samples} images")
    print("-" * 60)

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # 1. Primary Metrics
    acc = accuracy_score(all_targets, all_preds) * 100.0
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='macro', zero_division=0)
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='weighted', zero_division=0)

    print(f"Test Accuracy        : {acc:.2f}%")
    print(f"Macro Precision      : {macro_p * 100:.2f}%")
    print(f"Macro Recall         : {macro_r * 100:.2f}%")
    print(f"Macro F1-Score       : {macro_f1 * 100:.2f}%")
    print(f"Weighted F1-Score    : {weighted_f1 * 100:.2f}%")
    print("-" * 60)

    # 2. Detailed Classification Report
    print("DETAILED CLASSIFICATION REPORT (PER CLASS):")
    report = classification_report(all_targets, all_preds, target_names=class_names, digits=4)
    print(report)

    # 3. Per-Class Accuracy Breakdown
    print("-" * 60)
    print("PER-CLASS ACCURACY BREAKDOWN:")
    cm = confusion_matrix(all_targets, all_preds)
    for idx, cls_name in enumerate(class_names):
        total_cls = np.sum(all_targets == idx)
        correct_cls = cm[idx, idx]
        cls_acc = (correct_cls / total_cls) * 100.0 if total_cls > 0 else 0.0
        print(f"  [{idx}] {cls_name:<26}: {correct_cls:2d}/{total_cls:2d} correct ({cls_acc:6.2f}%)")

    # 4. Confusion Matrix Heatmap
    os.makedirs(results_dir, exist_ok=True)
    cm_path = os.path.join(results_dir, "confusion_matrix.png")

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        linewidths=0.5,
        linecolor='gray'
    )
    plt.title(f'Confusion Matrix - Butterfly Classifier (Test Accuracy: {acc:.2f}%)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('Ground Truth Label', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print("-" * 60)
    print(f"Confusion matrix heatmap saved to: {cm_path}")

    return {
        'accuracy': acc,
        'macro_precision': macro_p * 100,
        'macro_recall': macro_r * 100,
        'macro_f1': macro_f1 * 100,
        'weighted_f1': weighted_f1 * 100,
        'confusion_matrix_path': cm_path,
        'report': report
    }

if __name__ == "__main__":
    evaluate_model()
