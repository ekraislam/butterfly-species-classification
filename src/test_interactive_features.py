"""
Unit test for Aspect-Ratio Preservation, Colormap Modes, and Heatmap Modulation.
"""
import os
import sys
import numpy as np
import cv2
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

def test_interactive_heatmap_and_aspect_ratios():
    print("=== Testing Interactive Heatmap & Sizing Suite ===")
    
    predictor = ButterflyPredictor()
    gradcam = GradCAM(predictor.model)
    
    # Test images with different aspect ratios:
    # 1. Landscape 600x300
    # 2. Portrait 300x600
    # 3. Square 400x400
    test_shapes = [(600, 300), (300, 600), (400, 400)]
    
    for w, h in test_shapes:
        dummy_img = Image.new("RGB", (w, h), color=(100, 150, 200))
        pred_res = predictor.predict(dummy_img)
        
        with gradcam:
            raw_map, _, _ = gradcam.generate(pred_res['input_tensor'], pred_res['predicted_idx'])
            
        assert raw_map.shape == (7, 7) or raw_map.ndim == 2, f"Invalid raw map shape: {raw_map.shape}"
        
        # Test multiple colormaps and cutoffs
        colormaps = [
            cv2.COLORMAP_JET,
            cv2.COLORMAP_TURBO,
            cv2.COLORMAP_INFERNO,
            cv2.COLORMAP_VIRIDIS,
            cv2.COLORMAP_PLASMA,
            cv2.COLORMAP_MAGMA
        ]
        
        for cmap in colormaps:
            for alpha in [0.0, 0.3, 0.7, 1.0]:
                for cutoff in [0.0, 0.4, 0.8]:
                    if cutoff > 0.0:
                        mod_map = np.where(raw_map >= cutoff, (raw_map - cutoff) / (1.0 - cutoff + 1e-8), 0.0)
                    else:
                        mod_map = raw_map.copy()
                    
                    gamma = 1.0 + (0.55 - alpha) * 0.8
                    mod_map = np.clip(np.power(mod_map, max(gamma, 0.2)), 0.0, 1.0)
                    
                    heatmap_img, overlay_img = gradcam.overlay_heatmap(
                        mod_map,
                        dummy_img,
                        alpha=alpha,
                        colormap=cmap
                    )
                    
                    assert overlay_img.size == (w, h), f"Overlay size {overlay_img.size} mismatch with original {(w, h)}"
                    assert heatmap_img.size == (w, h), f"Heatmap size {heatmap_img.size} mismatch with original {(w, h)}"
        
        # Test Report Card generation for this shape
        report_bytes = generate_report_card(
            original_image=dummy_img,
            overlay_image=overlay_img,
            pred_class=pred_res['predicted_class'],
            confidence=pred_res['confidence'],
            top_k=pred_res['top_k']
        )
        assert len(report_bytes) > 10000, "Report card generation failed or empty"
        print(f"  [PASSED] Shape {w}x{h} ({'Landscape' if w > h else 'Portrait' if h > w else 'Square'}) verified!")

    print("\n>> ALL INTERACTIVE HEATMAP & ASPECT RATIO TESTS PASSED (100%) <<\n")

if __name__ == "__main__":
    test_interactive_heatmap_and_aspect_ratios()
