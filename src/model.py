"""
Model Architecture Module: ResNet-18 Transfer Learning for Butterfly Classification.
Includes backbone freezing, partial unfreezing, and parameter tracking.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

def freeze_backbone(model: nn.Module):
    """
    Freezes all feature extraction layers in the backbone, keeping only
    the final classification head (model.fc) trainable.
    """
    for name, param in model.named_parameters():
        if "fc" not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

def unfreeze_last_block(model: nn.Module):
    """
    Unfreezes the final residual block (layer4) and the classification head (fc)
    for fine-tuning, while keeping earlier layers (conv1, bn1, layer1-3) frozen.
    """
    for name, param in model.named_parameters():
        if "layer4" in name or "fc" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

def get_parameter_counts(model: nn.Module):
    """
    Returns total parameters, trainable parameters, and non-trainable parameters.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': frozen_params
    }

def create_model(num_classes: int = 8, pretrained: bool = True, freeze: bool = True) -> nn.Module:
    """
    Instantiates ResNet-18 with ImageNet pretrained weights, replaces the
    1000-class classifier head with a num_classes Linear layer, and freezes
    the feature backbone if freeze=True.
    """
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)

    # In ResNet-18, model.fc has in_features=512
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    if freeze:
        freeze_backbone(model)

    return model

if __name__ == "__main__":
    print("=" * 60)
    print("MODEL MODULE VERIFICATION")
    print("=" * 60)
    model = create_model(num_classes=8, pretrained=True, freeze=True)
    counts = get_parameter_counts(model)
    print(f"Total Parameters     : {counts['total']:,}")
    print(f"Trainable Parameters : {counts['trainable']:,}")
    print(f"Frozen Parameters    : {counts['frozen']:,}")
    
    # Test dummy forward pass
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(f"Forward pass output shape: {out.shape} (Expected: [2, 8])")
    assert out.shape == (2, 8), "Shape mismatch!"
    print(">> Model instantiation & forward pass verified <<")
