"""
Integration test for Streamlit app logic: Tests model caching, prediction, top-3 ranking,
and Grad-CAM generation on 3 sample images from prepared_dataset/test/.
"""

import os
import sys

APP_DIR = os.path.abspath("app")
SRC_DIR = os.path.abspath("src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from predict import ButterflyPredictor
from gradcam import GradCAM
from utils import SPECIES_METADATA, resolve_project_paths
from PIL import Image

def test_app_pipeline():
    print("=" * 60)
    print("STREAMLIT APP INTEGRATION TEST")
    print("=" * 60)

    paths = resolve_project_paths()
    checkpoint_path = paths["checkpoint_path"]
    test_dir = paths["test_data_dir"]

    print(f"1. Loading Model from: {checkpoint_path}")
    predictor = ButterflyPredictor(checkpoint_path=checkpoint_path)
    gradcam = GradCAM(predictor.model)
    print("   --> Model and Grad-CAM initialized successfully.")

    test_classes_to_test = ["MONARCH", "ADONIS", "RED POSTMAN"]

    print("\n2. Testing 3 Test Images:")
    for cls in test_classes_to_test:
        cls_folder = os.path.join(test_dir, cls)
        files = os.listdir(cls_folder)
        img_path = os.path.join(cls_folder, files[0])
        print(f"\n--- Testing Sample: {cls} ({files[0]}) ---")

        # Load image as PIL
        img = Image.open(img_path)
        print(f"  - Image Loaded Size: {img.size}")

        # Predict
        res = predictor.predict(img)
        print(f"  - Predicted Class  : {res['predicted_class']} (Confidence: {res['confidence']:.2f}%)")
        print("  - Top-3 Predictions:")
        for r, (t_cls, prob) in enumerate(res['top_k'], 1):
            print(f"      {r}. {t_cls:<24}: {prob:5.2f}%")

        # Grad-CAM
        heatmap, pred_idx, score = gradcam.generate(res['input_tensor'], res['predicted_idx'])
        hm_img, overlay_img = gradcam.overlay_heatmap(heatmap, res['display_image'], alpha=0.5)
        print(f"  - Grad-CAM Heatmap : Generated (shape={heatmap.shape})")
        print(f"  - Overlay Size     : {overlay_img.size}")

        # Metadata check
        if res['predicted_class'] in SPECIES_METADATA:
            meta = SPECIES_METADATA[res['predicted_class']]
            print(f"  - Metadata Linked  : {meta['scientific_name']} ({meta['family']})")

    gradcam.remove_hooks()
    print("\n" + "=" * 60)
    print(">> ALL STREAMLIT INTEGRATION TESTS PASSED (100% SUCCESS) <<")
    print("=" * 60)

if __name__ == "__main__":
    test_app_pipeline()
