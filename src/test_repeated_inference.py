"""
Repeated Inference Stability Test Script.
Runs inference and Grad-CAM on MONARCH, ADONIS, and RED POSTMAN 3 times each,
verifying prediction stability, heatmap stability, and zero hook accumulation.
"""

import os
import sys
import numpy as np
from PIL import Image

SRC_DIR = os.path.abspath("src")
APP_DIR = os.path.abspath("app")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from predict import ButterflyPredictor
from gradcam import GradCAM
from utils import resolve_project_paths

def test_repeated_inference():
    print("=" * 60)
    print("REPEATED INFERENCE & GRAD-CAM STABILITY AUDIT")
    print("=" * 60)

    paths = resolve_project_paths()
    predictor = ButterflyPredictor(checkpoint_path=paths["checkpoint_path"])
    test_dir = paths["test_data_dir"]

    targets = ["MONARCH", "ADONIS", "RED POSTMAN"]
    iterations = 3

    for target in targets:
        folder = os.path.join(test_dir, target)
        sample_file = sorted(os.listdir(folder))[0]
        sample_path = os.path.join(folder, sample_file)
        img = Image.open(sample_path)

        print(f"\n--- Testing Target: {target} ({sample_file}) ---")
        prev_conf = None
        prev_heatmap = None

        for it in range(1, iterations + 1):
            # Check model hook count before inference
            active_hooks_before = len(predictor.model.layer4[-1]._forward_hooks) + len(predictor.model.layer4[-1]._backward_hooks)

            # Predict
            res = predictor.predict(img)
            pred_class = res['predicted_class']
            conf = res['confidence']

            # Grad-CAM
            with GradCAM(predictor.model) as gradcam:
                heatmap, pred_idx, _ = gradcam.generate(res['input_tensor'], res['predicted_idx'])
                hm_img, ov_img = gradcam.overlay_heatmap(heatmap, res['display_image'])

            # Check model hook count after inference
            active_hooks_after = len(predictor.model.layer4[-1]._forward_hooks) + len(predictor.model.layer4[-1]._backward_hooks)

            # Stability checks
            conf_diff = 0.0 if prev_conf is None else abs(conf - prev_conf)
            hm_diff = 0.0 if prev_heatmap is None else np.max(np.abs(heatmap - prev_heatmap))

            print(
                f"  Iteration {it}/{iterations} | Pred: {pred_class:<15} ({conf:5.2f}%) | "
                f"Conf Delta: {conf_diff:6.4f} | Heatmap Delta: {hm_diff:6.4f} | "
                f"Hooks Before/After: {active_hooks_before}/{active_hooks_after}"
            )

            assert active_hooks_before == 0, f"Lingering hooks detected before iteration {it}!"
            assert active_hooks_after == 0, f"Lingering hooks detected after iteration {it}!"
            assert pred_class == target, f"Prediction mismatch on iteration {it}!"
            if prev_conf is not None:
                assert conf_diff < 1e-4, f"Prediction confidence drifted on iteration {it}!"
            if prev_heatmap is not None:
                assert hm_diff < 1e-4, f"Grad-CAM heatmap drifted on iteration {it}!"

            prev_conf = conf
            prev_heatmap = heatmap

    print("\n" + "=" * 60)
    print(">> REPEATED INFERENCE AUDIT PASSED: 100% DETERMINISTIC & ZERO HOOK ACCUMULATION <<")
    print("=" * 60)

if __name__ == "__main__":
    test_repeated_inference()
