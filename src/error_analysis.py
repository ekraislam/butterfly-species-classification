"""
Error Analysis Module: Identifies, visualizes, and analyzes misclassified test images.
"""

import os
import shutil
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torchvision import datasets, transforms

from model import create_model

def run_error_analysis(
    checkpoint_path="models/butterfly_resnet18_best.pth",
    data_dir="prepared_dataset",
    results_dir="results"
):
    print("=" * 60)
    print("TEST ERROR ANALYSIS")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 1. Load Checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    class_names = checkpoint['class_names']
    num_classes = checkpoint['num_classes']

    model = create_model(num_classes=num_classes, pretrained=False, freeze=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # 2. Setup Test Dataset with sample filepaths
    val_test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dir = os.path.join(data_dir, "test")
    test_dataset = datasets.ImageFolder(root=test_dir, transform=val_test_transform)
    
    print(f"Loaded {len(test_dataset)} test images across {num_classes} classes.")

    misclassified_dir = os.path.join(results_dir, "misclassified")
    os.makedirs(misclassified_dir, exist_ok=True)

    misclassified_samples = []

    with torch.no_grad():
        for idx in range(len(test_dataset)):
            img_tensor, true_label_idx = test_dataset[idx]
            img_path, _ = test_dataset.samples[idx]
            fname = os.path.basename(img_path)
            true_class = class_names[true_label_idx]

            # Forward pass
            input_batch = img_tensor.unsqueeze(0).to(device)
            logits = model(input_batch)
            probs = F.softmax(logits, dim=1).squeeze(0)

            # Top-3 predictions
            top3_probs, top3_indices = torch.topk(probs, 3)
            pred_idx = top3_indices[0].item()
            pred_class = class_names[pred_idx]
            pred_conf = top3_probs[0].item() * 100.0

            if pred_idx != true_label_idx:
                top3_info = [
                    (class_names[top3_indices[i].item()], top3_probs[i].item() * 100.0)
                    for i in range(3)
                ]
                
                misclassified_samples.append({
                    'index': idx,
                    'filename': fname,
                    'source_path': img_path,
                    'true_class': true_class,
                    'pred_class': pred_class,
                    'pred_conf': pred_conf,
                    'true_conf': probs[true_label_idx].item() * 100.0,
                    'top3': top3_info
                })
                
                # Copy to results/misclassified
                dest_path = os.path.join(misclassified_dir, f"error_{fname}")
                shutil.copy2(img_path, dest_path)

    print(f"\nTotal Misclassified Samples: {len(misclassified_samples)} / {len(test_dataset)}")
    print("-" * 60)

    for i, s in enumerate(misclassified_samples, 1):
        print(f"\nError #{i}: {s['filename']}")
        print(f"  - True Class       : {s['true_class']} (Model probability: {s['true_conf']:.2f}%)")
        print(f"  - Predicted Class  : {s['pred_class']} (Confidence: {s['pred_conf']:.2f}%)")
        print("  - Top-3 Predictions:")
        for rank, (cls, p) in enumerate(s['top3'], 1):
            print(f"      {rank}. {cls:<24}: {p:5.2f}%")

    # 3. Generate Visual Contact Sheet for Misclassified Images
    if misclassified_samples:
        img_w, img_h = 224, 224
        card_w = 480
        card_h = 260
        margin = 20
        total_w = len(misclassified_samples) * (card_w + margin) + margin
        total_h = card_h + 80

        contact_sheet = Image.new("RGB", (total_w, total_h), color=(24, 28, 36))
        draw = ImageDraw.Draw(contact_sheet)

        try:
            font_title = ImageFont.truetype("arial.ttf", 20)
            font_bold = ImageFont.truetype("arialbd.ttf", 15)
            font_text = ImageFont.truetype("arial.ttf", 13)
        except:
            font_title = ImageFont.load_default()
            font_bold = ImageFont.load_default()
            font_text = ImageFont.load_default()

        # Header Title
        draw.text((margin, 15), f"ResNet-18 Test Error Analysis (Total Errors: {len(misclassified_samples)} / 72)", fill=(255, 255, 255), font=font_title)

        for i, s in enumerate(misclassified_samples):
            x_card = margin + i * (card_w + margin)
            y_card = 60

            # Card background
            draw.rectangle([x_card, y_card, x_card + card_w, y_card + card_h - 10], fill=(38, 44, 56), outline=(220, 53, 69), width=2)

            # Paste original image
            with Image.open(s['source_path']) as original_img:
                thumb = original_img.convert("RGB").resize((img_w, img_h))
                contact_sheet.paste(thumb, (x_card + 10, y_card + 10))

            # Info text on the right side of the thumbnail
            text_x = x_card + img_w + 20
            draw.text((text_x, y_card + 15), f"File: {s['filename']}", fill=(255, 255, 255), font=font_bold)
            draw.text((text_x, y_card + 45), f"True Label:", fill=(160, 174, 192), font=font_text)
            draw.text((text_x, y_card + 62), f"{s['true_class']}", fill=(72, 187, 120), font=font_bold)
            
            draw.text((text_x, y_card + 90), f"Predicted Label:", fill=(160, 174, 192), font=font_text)
            draw.text((text_x, y_card + 107), f"{s['pred_class']}", fill=(245, 101, 101), font=font_bold)
            draw.text((text_x, y_card + 128), f"Confidence: {s['pred_conf']:.2f}%", fill=(245, 101, 101), font=font_text)

            draw.text((text_x, y_card + 155), "Top-3 Probabilities:", fill=(203, 213, 225), font=font_bold)
            for r_idx, (t_cls, t_p) in enumerate(s['top3']):
                color = (245, 101, 101) if r_idx == 0 else ((72, 187, 120) if t_cls == s['true_class'] else (203, 213, 225))
                draw.text((text_x, y_card + 175 + r_idx * 18), f"{r_idx+1}. {t_cls[:18]}: {t_p:5.1f}%", fill=color, font=font_text)

        contact_sheet_path = os.path.join(results_dir, "misclassified_contact_sheet.png")
        contact_sheet.save(contact_sheet_path, quality=95)
        print(f"\nContact sheet saved to: {contact_sheet_path}")

    return misclassified_samples

if __name__ == "__main__":
    run_error_analysis()
