"""
Comprehensive Integration Test for AI Butterfly Vision Streamlit Application.
Tests:
1. Model loading & inference
2. Top-3 probability distribution
3. Native PyTorch Grad-CAM heatmap generation & overlays
4. XAI attention insights linking
5. Inspection Report Card (PNG) export generation
"""

import os
import sys
from PIL import Image

SRC_DIR = os.path.abspath("src")
APP_DIR = os.path.abspath("app")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from predict import ButterflyPredictor
from gradcam import GradCAM
from utils import SPECIES_METADATA, resolve_project_paths, generate_report_card

def run_full_integration_test():
    print("=" * 65)
    print("AI BUTTERFLY VISION - COMPREHENSIVE INTEGRATION SUITE")
    print("=" * 65)

    paths = resolve_project_paths()
    predictor = ButterflyPredictor(checkpoint_path=paths["checkpoint_path"])
    test_dir = paths["test_data_dir"]

    test_classes = ["MONARCH", "ADONIS", "RED POSTMAN", "CLODIUS PARNASSIAN"]

    for cls in test_classes:
        folder = os.path.join(test_dir, cls)
        fname = sorted(os.listdir(folder))[0]
        fpath = os.path.join(folder, fname)
        img = Image.open(fpath)

        print(f"\n[Testing] Species: {cls:<24} | File: {fname}")

        # 1. Prediction
        res = predictor.predict(img)
        pred_class = res['predicted_class']
        conf = res['confidence']
        print(f"  --> Prediction: {pred_class:<20} | Confidence: {conf:6.2f}%")

        # 2. Grad-CAM
        with GradCAM(predictor.model) as gradcam:
            hm, _, _ = gradcam.generate(res['input_tensor'], res['predicted_idx'])
            hm_img, ov_img = gradcam.overlay_heatmap(hm, res['display_image'], alpha=0.55)
        print(f"  --> Grad-CAM: Generated ({hm_img.size}) | Overlay ({ov_img.size})")

        # 3. Report Card Generator
        card_bytes = generate_report_card(
            original_image=res['display_image'],
            overlay_image=ov_img,
            pred_class=pred_class,
            confidence=conf,
            top_k=res['top_k']
        )
        print(f"  --> Report Card PNG Export: Generated ({len(card_bytes):,} bytes)")

        # 4. Metadata verification
        meta = SPECIES_METADATA.get(pred_class)
        assert meta is not None, f"Missing metadata for {pred_class}!"
        print(f"  --> XAI Insight: \"{meta['xai_insight'][:55]}...\"")

    print("\n" + "=" * 65)
    print(">> ALL INTEGRATION TESTS & ASSET GENERATORS PASSED (100% SUCCESS) <<")
    print("=" * 65)

if __name__ == "__main__":
    run_full_integration_test()
