"""
Mobile & Edge Model Exporter: TorchScript (.pt) for PyTorch Mobile, iOS, Android & C++ LibTorch.
Requires zero external dependencies.
"""

import os
import sys
import torch

SRC_DIR = os.path.abspath("src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from model import create_model

def export_to_torchscript():
    checkpoint_path = os.path.join("models", "butterfly_resnet18_best.pth")
    output_path = os.path.join("models", "butterfly_resnet18_torchscript.pt")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    num_classes = len(checkpoint["class_names"])
    model = create_model(num_classes=num_classes, pretrained=False, freeze=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

    print(f"Tracing and exporting TorchScript model: {output_path}...")
    traced_model = torch.jit.trace(model, dummy_input)
    traced_model.save(output_path)

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f">> TorchScript Mobile Export Successful! Saved to {output_path} ({file_size_mb:.2f} MB)")

if __name__ == "__main__":
    export_to_torchscript()
