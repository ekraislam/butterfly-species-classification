"""
Test script to generate and verify certificates for all 8 species.
"""
import os
import sys
from PIL import Image

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
APP_DIR = os.path.join(ROOT_DIR, "app")
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from predict import ButterflyPredictor
from gradcam import GradCAM
from utils import generate_report_card

def test_all_certificates():
    predictor = ButterflyPredictor()
    gradcam = GradCAM(predictor.model)
    
    test_dir = "prepared_dataset/test"
    out_dir = "results/all_certificates_test"
    os.makedirs(out_dir, exist_ok=True)
    
    for cls in predictor.class_names:
        cls_dir = os.path.join(test_dir, cls)
        if not os.path.exists(cls_dir):
            continue
        fname = sorted(os.listdir(cls_dir))[0]
        img_path = os.path.join(cls_dir, fname)
        img = Image.open(img_path).convert("RGB")
        pred_res = predictor.predict(img)
        
        with gradcam:
            raw_map, _, _ = gradcam.generate(pred_res['input_tensor'], pred_res['predicted_idx'])
            heatmap_img, overlay_img = gradcam.overlay_heatmap(raw_map, img, alpha=0.55)
            
        report_bytes = generate_report_card(
            original_image=img,
            overlay_image=overlay_img,
            pred_class=pred_res['predicted_class'],
            confidence=pred_res['confidence'],
            top_k=pred_res['top_k']
        )
        
        out_file = os.path.join(out_dir, f"cert_{cls.replace(' ', '_')}.png")
        with open(out_file, "wb") as f:
            f.write(report_bytes)
        print(f"[OK] Generated: {out_file} ({len(report_bytes)} bytes)")

if __name__ == "__main__":
    test_all_certificates()
