"""
Export Trained Butterfly ResNet-18 Model to ONNX (Open Neural Network Exchange).
Optimized for mobile, edge devices, Flutter/React Native, and C++ inference runtimes.
"""

import os
import sys
import torch

SRC_DIR = os.path.abspath("src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from model import create_model

def export_to_onnx():
    checkpoint_path = os.path.join("models", "butterfly_resnet18_best.pth")
    onnx_output_path = os.path.join("models", "butterfly_resnet18.onnx")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    num_classes = len(checkpoint["class_names"])
    model = create_model(num_classes=num_classes, pretrained=False, freeze=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

    print(f"Exporting to ONNX format: {onnx_output_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input_tensor"],
        output_names=["logits"],
        dynamic_axes={
            "input_tensor": {0: "batch_size"},
            "logits": {0: "batch_size"}
        }
    )

    file_size_mb = os.path.getsize(onnx_output_path) / (1024 * 1024)
    print(f">> ONNX Export Successful! Saved to {onnx_output_path} ({file_size_mb:.2f} MB)")

if __name__ == "__main__":
    export_to_onnx()
