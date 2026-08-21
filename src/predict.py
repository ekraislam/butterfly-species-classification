"""
Prediction Module: ResNet-18 Inference Pipeline for Butterfly Species Classification.
Loads saved checkpoint and processes PIL images on CPU.
"""

import os
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from model import create_model

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class ButterflyPredictor:
    """
    Inference wrapper for the 8-class ResNet-18 Butterfly Classifier.
    """
    def __init__(self, checkpoint_path="models/butterfly_resnet18_best.pth", device=None):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

        self.device = device if device else torch.device("cpu")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.class_names = checkpoint['class_names']
        self.class_to_idx = checkpoint.get('class_to_idx', {name: i for i, name in enumerate(self.class_names)})
        self.num_classes = len(self.class_names)
        self.best_epoch = checkpoint.get('epoch', 'N/A')
        self.val_acc = checkpoint.get('val_acc', 'N/A')

        # Rebuild model and load weights
        self.model = create_model(num_classes=self.num_classes, pretrained=False, freeze=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        # Direct (224, 224) resize without CenterCrop to preserve 100% full specimen framing
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

    def preprocess_image(self, image_input):
        """
        Validates and converts image_input into a 4D PyTorch tensor and display PIL image.
        Preserves 100% full original uncropped framing for visualization.
        """
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image path does not exist: {image_input}")
            img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            img = image_input.convert("RGB")
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}. Expected PIL.Image or filepath.")

        tensor = self.preprocess(img).unsqueeze(0).to(self.device)
        return tensor, img

    def predict(self, image_input, top_k=3):
        """
        Performs inference on a single image.
        Returns:
            dict containing predicted class, confidence, top-k predictions,
            full probability distribution, preprocessed tensor, and display image.
        """
        tensor, display_img = self.preprocess_image(image_input)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1).squeeze(0)

        # Top-K
        top_k = min(top_k, self.num_classes)
        top_k_probs, top_k_indices = torch.topk(probs, top_k)

        top_k_results = [
            (self.class_names[top_k_indices[i].item()], top_k_probs[i].item() * 100.0)
            for i in range(top_k)
        ]

        predicted_idx = top_k_indices[0].item()
        predicted_class = self.class_names[predicted_idx]
        top1_confidence = top_k_probs[0].item() * 100.0

        all_probabilities = {
            self.class_names[i]: probs[i].item() * 100.0
            for i in range(self.num_classes)
        }

        return {
            'predicted_class': predicted_class,
            'predicted_idx': predicted_idx,
            'confidence': top1_confidence,
            'top_k': top_k_results,
            'all_probabilities': all_probabilities,
            'input_tensor': tensor,
            'display_image': display_img
        }
