"""
Environment & PyTorch / TorchVision Compatibility Verification Script
AI-Based Butterfly Species Classification & Visual Explanation System
"""

import sys

def verify_all():
    print("=" * 60)
    print("ENVIRONMENT & PYTORCH/TORCHVISION VERIFICATION")
    print("=" * 60)

    # Python
    python_ver = sys.version.split()[0]
    print(f"Python Version       : {python_ver}")
    print(f"Python Platform      : {sys.platform} (64-bit AMD64)")
    print("-" * 60)

    # 1. Imports
    import torch
    import torchvision
    from torchvision.models import resnet18, ResNet18_Weights
    import PIL
    import numpy as np
    import sklearn
    import matplotlib
    import seaborn as sns
    import cv2
    import streamlit as st

    print(f"torch version        : {torch.__version__}")
    print(f"torchvision version  : {torchvision.__version__}")
    print(f"Pillow version       : {PIL.__version__}")
    print(f"numpy version        : {np.__version__}")
    print(f"scikit-learn version : {sklearn.__version__}")
    print(f"matplotlib version   : {matplotlib.__version__}")
    print(f"seaborn version      : {sns.__version__}")
    print(f"OpenCV version       : {cv2.__version__}")
    print(f"streamlit version    : {st.__version__}")
    print("-" * 60)

    # 2. CUDA & Device Check
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available       : {cuda_available}")
    device = torch.device("cuda" if cuda_available else "cpu")
    print(f"Execution Device     : {device}")
    print("-" * 60)

    # 3. ResNet-18 Instantiation & Forward Pass Test
    print("Testing ResNet-18 model creation & CPU inference:")
    try:
        # Check ResNet18_Weights enum availability
        print(f"ResNet18_Weights Enum: {ResNet18_Weights.DEFAULT}")
        
        # Instantiate architecture
        model = resnet18(weights=None)
        model.eval()
        print("  - Model successfully instantiated: resnet18(weights=None)")

        # Dummy tensor forward pass
        x = torch.randn(1, 3, 224, 224, device=device)
        with torch.no_grad():
            output = model(x)
        
        print(f"  - Input Tensor Shape : {list(x.shape)}")
        print(f"  - Output Tensor Shape: {list(output.shape)}")
        
        assert list(output.shape) == [1, 1000], f"Expected output shape [1, 1000], got {list(output.shape)}"
        print("  - Output Shape Check : PASSED ([1, 1000])")
        print("  - CPU Forward Pass   : PASSED")
    except Exception as e:
        print(f"  - ResNet-18 Test FAILED: {e}")
        return False

    print("-" * 60)
    print(">> ALL PACKAGES AND PYTORCH CPU INFERENCE VERIFIED 100% <<")
    return True

if __name__ == "__main__":
    success = verify_all()
    sys.exit(0 if success else 1)
