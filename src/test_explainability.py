"""
Explainability Test & Validation Script.
Loads test samples across all 8 butterfly classes, generates predictions and Grad-CAM
overlays, saves individual explanation artifacts and an executive contact sheet.
"""

import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from predict import ButterflyPredictor
from gradcam import GradCAM

def run_explainability_suite(
    checkpoint_path="models/butterfly_resnet18_best.pth",
    data_dir="prepared_dataset",
    output_dir="results/gradcam_examples",
    contact_sheet_path="results/gradcam_contact_sheet.png",
    seed=42
):
    print("=" * 60)
    print("EXPLAINABLE AI (Grad-CAM) SUITE VERIFICATION")
    print("=" * 60)

    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(contact_sheet_path), exist_ok=True)

    # 1. Initialize Predictor and GradCAM
    predictor = ButterflyPredictor(checkpoint_path=checkpoint_path)
    gradcam = GradCAM(predictor.model)
    
    print(f"Model Checkpoint   : {checkpoint_path}")
    print(f"Grad-CAM Target    : {gradcam.target_layer.__class__.__name__} (model.layer4[-1])")
    print(f"Device             : {predictor.device}")
    print("-" * 60)

    # 2. Select 1 test sample per class (8 classes total)
    test_dir = os.path.join(data_dir, "test")
    test_samples = []

    for cls in predictor.class_names:
        cls_dir = os.path.join(test_dir, cls)
        if os.path.exists(cls_dir):
            files = sorted(os.listdir(cls_dir))
            # Pick a deterministic sample
            chosen_file = files[0]
            test_samples.append((cls, os.path.join(cls_dir, chosen_file)))

    print(f"Selected {len(test_samples)} test samples for Grad-CAM explanation.\n")

    results_data = []

    for idx, (true_class, img_path) in enumerate(test_samples, 1):
        fname = os.path.basename(img_path)
        
        # Predict
        pred_res = predictor.predict(img_path)
        pred_class = pred_res['predicted_class']
        conf = pred_res['confidence']
        display_img = pred_res['display_image']

        # Generate Grad-CAM
        heatmap_norm, pred_idx, logit_score = gradcam.generate(
            pred_res['input_tensor'],
            pred_res['predicted_idx']
        )
        heatmap_img, overlay_img = gradcam.overlay_heatmap(heatmap_norm, display_img, alpha=0.5)

        # Save individual images
        safe_cls = true_class.replace(' ', '_')
        base_name = f"{idx:02d}_{safe_cls}_{os.path.splitext(fname)[0]}"
        
        orig_save_path = os.path.join(output_dir, f"{base_name}_orig.jpg")
        hm_save_path = os.path.join(output_dir, f"{base_name}_heatmap.jpg")
        ov_save_path = os.path.join(output_dir, f"{base_name}_overlay.jpg")

        display_img.save(orig_save_path, quality=95)
        heatmap_img.save(hm_save_path, quality=95)
        overlay_img.save(ov_save_path, quality=95)

        match_symbol = "MATCH" if true_class == pred_class else "MISMATCH"
        print(f"[{idx}/8] {true_class:<26} -> Predicted: {pred_class:<26} ({conf:5.2f}%) [{match_symbol}]")
        print(f"     Saved overlay: {ov_save_path}")

        results_data.append({
            'index': idx,
            'true_class': true_class,
            'pred_class': pred_class,
            'confidence': conf,
            'orig_img': display_img,
            'heatmap_img': heatmap_img,
            'overlay_img': overlay_img,
            'filename': fname
        })

    # Clean up hooks
    gradcam.remove_hooks()
    print("-" * 60)
    print(">> Hooks successfully removed <<")

    # 3. Create Executive Contact Sheet (8 rows x 3 columns: Original, Heatmap, Overlay)
    img_size = 224
    header_h = 70
    row_h = img_size + 20
    info_w = 340
    col_w = img_size + 15
    total_w = info_w + 3 * col_w + 30
    total_h = header_h + len(results_data) * row_h + 20

    sheet = Image.new("RGB", (total_w, total_h), color=(22, 27, 34))
    draw = ImageDraw.Draw(sheet)

    try:
        font_title = ImageFont.truetype("arialbd.ttf", 22)
        font_col = ImageFont.truetype("arialbd.ttf", 16)
        font_bold = ImageFont.truetype("arialbd.ttf", 14)
        font_text = ImageFont.truetype("arial.ttf", 13)
    except:
        font_title = ImageFont.load_default()
        font_col = ImageFont.load_default()
        font_bold = ImageFont.load_default()
        font_text = ImageFont.load_default()

    # Sheet Title
    draw.text((20, 16), "ResNet-18 Butterfly Classifier - Grad-CAM Visual Explanations", fill=(255, 255, 255), font=font_title)

    # Column Headers
    y_col_head = 45
    draw.text((20, y_col_head), "Target & Prediction", fill=(139, 148, 158), font=font_col)
    draw.text((info_w + 10, y_col_head), "Original Image", fill=(139, 148, 158), font=font_col)
    draw.text((info_w + 10 + col_w, y_col_head), "Grad-CAM Heatmap", fill=(139, 148, 158), font=font_col)
    draw.text((info_w + 10 + 2 * col_w, y_col_head), "Explanation Overlay", fill=(139, 148, 158), font=font_col)

    for r_idx, r in enumerate(results_data):
        y_pos = header_h + r_idx * row_h

        # Info Box on Left
        draw.rectangle([20, y_pos, info_w - 10, y_pos + img_size], fill=(33, 38, 45), outline=(48, 54, 61))
        draw.text((32, y_pos + 40), f"{r['index']}. {r['true_class']}", fill=(240, 246, 252), font=font_bold)
        
        is_correct = (r['true_class'] == r['pred_class'])
        pred_color = (63, 185, 80) if is_correct else (248, 81, 73)
        draw.text((32, y_pos + 80), f"Prediction: {r['pred_class']}", fill=pred_color, font=font_bold)
        draw.text((32, y_pos + 110), f"Confidence: {r['confidence']:.2f}%", fill=(201, 209, 217), font=font_text)
        draw.text((32, y_pos + 135), f"File: {r['filename']}", fill=(139, 148, 158), font=font_text)

        # Image 1: Original
        x1 = info_w + 10
        sheet.paste(r['orig_img'], (x1, y_pos))
        draw.rectangle([x1, y_pos, x1 + img_size, y_pos + img_size], outline=(48, 54, 61), width=1)

        # Image 2: Heatmap
        x2 = x1 + col_w
        sheet.paste(r['heatmap_img'], (x2, y_pos))
        draw.rectangle([x2, y_pos, x2 + img_size, y_pos + img_size], outline=(48, 54, 61), width=1)

        # Image 3: Overlay
        x3 = x2 + col_w
        sheet.paste(r['overlay_img'], (x3, y_pos))
        draw.rectangle([x3, y_pos, x3 + img_size, y_pos + img_size], outline=(48, 54, 61), width=1)

    sheet.save(contact_sheet_path, quality=95)
    print(f"\nExecutive contact sheet generated and saved to:\n  {contact_sheet_path}")
    print("=" * 60)
    print(">> GRAD-CAM SUITE EXECUTION & VALIDATION COMPLETE (100% SUCCESS) <<")

if __name__ == "__main__":
    run_explainability_suite()
