"""
Scratch script to render and inspect the redesigned certificate.
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

def test_certificate_render():
    predictor = ButterflyPredictor()
    gradcam = GradCAM(predictor.model)
    
    test_img_path = "prepared_dataset/test/MONARCH/Image_1272.jpg"
    img = Image.open(test_img_path).convert("RGB")
    pred_res = predictor.predict(img)
    
    with gradcam:
        raw_heatmap, _, _ = gradcam.generate(pred_res['input_tensor'], pred_res['predicted_idx'])
        heatmap_img, overlay_img = gradcam.overlay_heatmap(raw_heatmap, img, alpha=0.55)
        
    report_bytes = generate_report_card(
        original_image=img,
        overlay_image=overlay_img,
        pred_class=pred_res['predicted_class'],
        confidence=pred_res['confidence'],
        top_k=pred_res['top_k']
    )
    
    out_path = "results/test_certificate_preview.png"
    with open(out_path, "wb") as f:
        f.write(report_bytes)
        
    print(f"Generated test certificate at: {out_path} ({len(report_bytes)} bytes)")

if __name__ == "__main__":
    test_certificate_render()
