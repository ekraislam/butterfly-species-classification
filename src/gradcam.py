"""
Grad-CAM Module: Explainable AI with Native PyTorch Hooks for ResNet-18.
Generates Class Activation Maps (CAM) and visual heatmap overlays without external libraries.
Includes strict hook lifecycle management to prevent memory leaks in cached web apps.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image

class GradCAM:
    """
    Native PyTorch implementation of Gradient-weighted Class Activation Mapping (Grad-CAM).
    Designed for ResNet-18 architectures on CPU/GPU.
    Guarantees that PyTorch hooks are registered only during CAM generation and cleanly removed.
    """
    def __init__(self, model, target_layer=None):
        self.model = model
        self.model.eval()

        # Default target layer for ResNet-18 is the final convolutional block in layer4
        if target_layer is None:
            self.target_layer = self.model.layer4[-1]
        else:
            self.target_layer = target_layer

        self.activations = None
        self.gradients = None
        self.handles = []

    def _register_hooks(self):
        """Registers forward and backward hooks on the target layer."""
        self.remove_hooks()  # Ensure no duplicate lingering hooks

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            # grad_output[0] contains the gradients w.r.t layer output
            self.gradients = grad_output[0].detach()

        self.handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        """Removes all registered PyTorch hooks to avoid memory leaks."""
        for handle in self.handles:
            try:
                handle.remove()
            except Exception:
                pass
        self.handles = []
        self.activations = None
        self.gradients = None

    def generate(self, input_tensor, target_class_idx=None, counterfactual=False):
        """
        Generates the 2D Grad-CAM heatmap for a given input tensor and target class.
        Hooks are registered at the start and guaranteed to be removed in a finally block.
        Args:
            input_tensor: 4D Tensor [1, 3, 224, 224]
            target_class_idx: int, target class index (default: argmax prediction)
            counterfactual: bool, if True, generates negative gradients CAM
        Returns:
            heatmap: 2D numpy array [H, W] normalized in [0, 1]
            predicted_idx: int, predicted class index
            score: float, model logit score for the target class
        """
        self._register_hooks()
        try:
            self.model.zero_grad()
            input_tensor = input_tensor.clone()

            # 1. Forward Pass
            output = self.model(input_tensor)

            if target_class_idx is None:
                target_class_idx = torch.argmax(output, dim=1).item()

            score = output[0, target_class_idx]

            # 2. Backward Pass
            score.backward(retain_graph=False)

            # 3. Global Average Pooling of Gradients (Importance Weights)
            if self.gradients is None or self.activations is None:
                raise RuntimeError("Failed to capture activations or gradients. Check hook registration.")

            gradients = self.gradients
            if counterfactual:
                gradients = -gradients

            # Alpha weights: mean across spatial dimensions H, W
            alpha = torch.mean(gradients, dim=(2, 3), keepdim=True)  # Shape: [1, C, 1, 1]

            # 4. Weighted combination of activation maps
            cam = torch.sum(alpha * self.activations, dim=1).squeeze(0)  # Shape: [H, W]

            # 5. Apply ReLU
            cam = F.relu(cam)

            # Convert to numpy
            cam_np = cam.cpu().numpy()

            # 6. Normalize to [0, 1]
            cam_min, cam_max = np.min(cam_np), np.max(cam_np)
            if cam_max - cam_min > 1e-8:
                cam_norm = (cam_np - cam_min) / (cam_max - cam_min)
            else:
                cam_norm = np.zeros_like(cam_np)

            self.model.zero_grad()
            return cam_norm, target_class_idx, score.item()

        finally:
            self.remove_hooks()

    def overlay_heatmap(self, heatmap, original_image, alpha=0.5, colormap=cv2.COLORMAP_JET):
        """
        Overlays the 2D heatmap on the original RGB image.
        Args:
            heatmap: 2D numpy array [0, 1]
            original_image: PIL.Image or numpy RGB array [H, W, 3] in range [0, 255]
            alpha: float, transparency factor for heatmap overlay (0 = only original, 1 = only heatmap)
            colormap: OpenCV colormap constant (default: COLORMAP_JET)
        Returns:
            heatmap_rgb: PIL.Image of the standalone colored heatmap
            overlay_image: PIL.Image of the blended visualization
        """
        if isinstance(original_image, Image.Image):
            orig_rgb = np.array(original_image.convert("RGB"), dtype=np.uint8)
        else:
            orig_rgb = np.array(original_image, dtype=np.uint8)

        img_h, img_w = orig_rgb.shape[:2]

        # Resize heatmap to match image dimensions
        heatmap_resized = cv2.resize(heatmap, (img_w, img_h), interpolation=cv2.INTER_CUBIC)
        heatmap_resized = np.clip(heatmap_resized, 0.0, 1.0)
        heatmap_uint8 = np.uint8(255 * heatmap_resized)

        # Apply colormap (cv2 outputs BGR)
        heatmap_colored_bgr = cv2.applyColorMap(heatmap_uint8, colormap)
        heatmap_colored_rgb = cv2.cvtColor(heatmap_colored_bgr, cv2.COLOR_BGR2RGB)

        # Alpha blending
        blended = np.uint8(alpha * heatmap_colored_rgb + (1.0 - alpha) * orig_rgb)

        return Image.fromarray(heatmap_colored_rgb), Image.fromarray(blended)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_hooks()
