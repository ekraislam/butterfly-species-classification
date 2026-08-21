"""
Streamlit Web Application: AI-Based Butterfly Species Classification & Visual Explanation System
Interactive ResNet-18 Inference with Real-time Native Grad-CAM Heatmap Generation.
"""

import os
import sys
from PIL import Image
import streamlit as st
import numpy as np

# Ensure project root and src/ are discoverable regardless of working directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from predict import ButterflyPredictor
from gradcam import GradCAM
from utils import SPECIES_METADATA, resolve_project_paths

# -----------------------------------------------------------------------------
# 1. Page Configuration & Custom Theme
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Butterfly Classifier & Grad-CAM",
    page_icon="🦋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for polished, academic look
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }
    .prediction-badge {
        font-size: 1.5rem;
        font-weight: 700;
        color: #0F766E;
    }
    .confidence-text {
        font-size: 1.1rem;
        font-weight: 600;
        color: #0369A1;
    }
    .info-box {
        background-color: #F0FDF4;
        border-left: 4px solid #16A34A;
        padding: 0.8rem 1rem;
        border-radius: 4px;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. Cached Predictor Resource Loader (Safe Model Caching)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading PyTorch ResNet-18 Model Checkpoint...")
def load_predictor():
    paths = resolve_project_paths()
    checkpoint_path = paths["checkpoint_path"]
    if not os.path.exists(checkpoint_path):
        st.error(f"Model checkpoint not found at: `{checkpoint_path}`")
        return None
    return ButterflyPredictor(checkpoint_path=checkpoint_path)

paths = resolve_project_paths()
predictor = load_predictor()

# -----------------------------------------------------------------------------
# 3. Sidebar: Project Info & Quick Sample Selector
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🦋 About Project")
    st.markdown("""
    **AI-Based Butterfly Species Classification & Visual Explanation System**
    
    A transfer learning system for identifying 8 distinct butterfly species and visualizing neural network decision regions via **Grad-CAM**.
    """)

    st.markdown("---")
    st.markdown("### 📊 Model Architecture & Stats")
    st.markdown("""
    - **Architecture**: ResNet-18 (Transfer Learning)
    - **Classes**: 8 Supported Species
    - **Test Accuracy**: **`97.22%`**
    - **Macro F1-Score**: **`97.12%`**
    - **Inference Mode**: Pure CPU (Lightweight)
    - **Explainability**: Native PyTorch Grad-CAM
    """)

    st.markdown("---")
    st.markdown("### 🧪 Quick Test Samples")
    test_dir = paths["test_data_dir"]
    sample_options = ["(Upload Your Own Image)"]
    sample_map = {}

    if os.path.exists(test_dir):
        for cls_name in sorted(os.listdir(test_dir)):
            cls_path = os.path.join(test_dir, cls_name)
            if os.path.isdir(cls_path):
                files = sorted(os.listdir(cls_path))
                if files:
                    opt_name = f"Sample: {cls_name} ({files[0]})"
                    sample_options.append(opt_name)
                    sample_map[opt_name] = os.path.join(cls_path, files[0])

    selected_sample = st.selectbox("Select a benchmark sample to test:", sample_options)

# -----------------------------------------------------------------------------
# 4. Main Section: Header & Image Input
# -----------------------------------------------------------------------------
st.markdown('<div class="main-title">🦋 AI-Based Butterfly Species Classification</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">8-Class Butterfly Recognition with Explainable AI (Grad-CAM)</div>', unsafe_allow_html=True)

if predictor is None:
    st.error("Model failed to load. Please verify `models/butterfly_resnet18_best.pth` exists.")
    st.stop()

# Determine active image source
active_image = None
active_filename = None

uploaded_file = st.file_uploader(
    "Upload a Butterfly Photo (JPG, JPEG, PNG)",
    type=["jpg", "jpeg", "png"],
    help="Upload an image of a butterfly to identify its species."
)

if uploaded_file is not None:
    try:
        active_image = Image.open(uploaded_file).convert("RGB")
        active_filename = uploaded_file.name
    except Exception as e:
        st.error(f"Error reading uploaded image: {e}")
elif selected_sample != "(Upload Your Own Image)" and selected_sample in sample_map:
    sample_file_path = sample_map[selected_sample]
    try:
        active_image = Image.open(sample_file_path).convert("RGB")
        active_filename = os.path.basename(sample_file_path)
    except Exception as e:
        st.error(f"Error loading sample image: {e}")

# -----------------------------------------------------------------------------
# 5. Analysis & Results Section
# -----------------------------------------------------------------------------
if active_image is not None:
    col_input, col_action = st.columns([1, 2])
    with col_input:
        st.image(active_image, caption=f"Selected Input: {active_filename}", use_container_width=True)

    with col_action:
        st.markdown("#### Ready to Analyze")
        st.write("Click below to run ResNet-18 classification and generate explainable Grad-CAM heatmaps.")
        analyze_clicked = st.button("🔍 Analyze Butterfly & Generate Explanation", type="primary", use_container_width=True)

    if analyze_clicked or st.session_state.get("auto_run", True):
        with st.spinner("Processing neural network inference and gradient activation maps..."):
            try:
                # 1. Run Classification
                pred_res = predictor.predict(active_image)
                pred_class = pred_res['predicted_class']
                confidence = pred_res['confidence']
                top_k = pred_res['top_k']
                all_probs = pred_res['all_probabilities']

                # 2. Run Grad-CAM with Safe On-Demand Lifecycle
                with GradCAM(predictor.model) as gradcam_engine:
                    heatmap_norm, _, _ = gradcam_engine.generate(
                        pred_res['input_tensor'],
                        pred_res['predicted_idx']
                    )

                st.markdown("---")

                # Layout: Left column = Classification, Right column = Grad-CAM
                res_col1, res_col2 = st.columns([1.1, 1.3])

                with res_col1:
                    st.markdown("### 🎯 Classification Result")
                    
                    # Result Card
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="color: #64748B; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.05em;">Identified Species</div>
                        <div class="prediction-badge">{pred_class}</div>
                        <div class="confidence-text">Confidence: {confidence:.2f}%</div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.progress(float(min(confidence / 100.0, 1.0)))

                    # Top-3 Breakdown
                    st.markdown("#### 🏆 Top-3 Predictions")
                    for rank, (cls_name, prob) in enumerate(top_k, 1):
                        bar_col, text_col = st.columns([3, 1])
                        with bar_col:
                            st.write(f"**{rank}. {cls_name}**")
                            st.progress(float(prob / 100.0))
                        with text_col:
                            st.write(f"**{prob:.2f}%**")

                    # Species facts snippet if available
                    if pred_class in SPECIES_METADATA:
                        meta = SPECIES_METADATA[pred_class]
                        st.markdown(f"""
                        <div class="info-box">
                            <strong>Scientific Name:</strong> <em>{meta['scientific_name']}</em><br>
                            <strong>Family:</strong> {meta['family']}<br>
                            <strong>Key Trait:</strong> {meta['key_features']}
                        </div>
                        """, unsafe_allow_html=True)

                with res_col2:
                    st.markdown("### 🔬 Explainable AI (Grad-CAM)")
                    st.caption("The highlighted regions indicate image areas that contributed strongly to the model's prediction.")

                    # Overlay Transparency Slider
                    blend_alpha = st.slider("Heatmap Overlay Intensity", min_value=0.2, max_value=0.8, value=0.5, step=0.05)
                    with GradCAM(predictor.model) as gradcam_engine:
                        heatmap_img, overlay_img = gradcam_engine.overlay_heatmap(heatmap_norm, pred_res['display_image'], alpha=blend_alpha)

                    cam_tab1, cam_tab2, cam_tab3 = st.tabs(["✨ Overlay View", "🌡️ Heatmap Only", "🖼️ Original Cropped"])
                    with cam_tab1:
                        st.image(overlay_img, caption=f"Grad-CAM Overlay ({pred_class})", use_container_width=True)
                    with cam_tab2:
                        st.image(heatmap_img, caption="Raw Class Activation Heatmap", use_container_width=True)
                    with cam_tab3:
                        st.image(pred_res['display_image'], caption="Preprocessed Input (224x224)", use_container_width=True)

            except Exception as e:
                st.error(f"An error occurred during inference/Grad-CAM computation: {e}")

else:
    st.info("👋 Upload a butterfly image or pick a test sample from the sidebar to begin analysis.")

# -----------------------------------------------------------------------------
# 6. Supported Species Reference Guide
# -----------------------------------------------------------------------------
st.markdown("---")
with st.expander("📚 Supported Butterfly Species Reference & Facts", expanded=False):
    st.markdown("This model is trained to recognize the following **8 butterfly species** with high precision:")
    spec_cols = st.columns(2)
    for idx, (s_name, s_info) in enumerate(SPECIES_METADATA.items()):
        col_target = spec_cols[idx % 2]
        with col_target:
            st.markdown(f"""
            **{idx+1}. {s_name}** (*{s_info['scientific_name']}*)  
            - **Family**: {s_info['family']}  
            - **Appearance**: {s_info['appearance']}  
            - **Distribution**: {s_info['distribution']}  
            - **Distinctive Trait**: {s_info['key_features']}  
            """)
            st.markdown("---")
