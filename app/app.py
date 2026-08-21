"""
AI Butterfly Vision: Ultra-Premium Next-Gen Interface
A Breathtaking, Award-Winning Dark Cyber-Glass AI Showcase for Butterfly Species Classification & Grad-CAM Explainable AI.
Designed & Developed by Ohi.
"""

import os
import sys
from PIL import Image
import streamlit as st
import numpy as np

# Ensure project root and src/ are discoverable
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from predict import ButterflyPredictor
from gradcam import GradCAM
from utils import SPECIES_METADATA, resolve_project_paths, generate_report_card

# -----------------------------------------------------------------------------
# 1. Page Configuration & Masterpiece Custom CSS (Luxury Dark Glassmorphism)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Butterfly Vision • Next-Gen Explainable AI",
    page_icon="🦋",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=Space+Grotesk:wght@400;500;600;700&display=swap');
    
    /* Global Reset & Typography */
    html, body, [class*="css"], .stApp {
        font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    
    code, pre, .stat-num, .gauge-val {
        font-family: 'Space Grotesk', monospace !important;
    }
    
    /* Dynamic Cosmic Mesh Background */
    .stApp {
        background-color: #030712 !important;
        background-image: 
            radial-gradient(at 50% 0%, rgba(56, 189, 248, 0.18) 0px, transparent 60%),
            radial-gradient(at 100% 30%, rgba(139, 92, 246, 0.12) 0px, transparent 50%),
            radial-gradient(at 0% 70%, rgba(16, 185, 129, 0.10) 0px, transparent 50%),
            radial-gradient(at 80% 90%, rgba(244, 63, 94, 0.08) 0px, transparent 50%) !important;
        background-attachment: fixed !important;
        color: #F8FAFC !important;
    }
    
    /* Hide Default Streamlit Clutter */
    #MainMenu, footer, header, .stDeployButton, [data-testid="stDeployButton"], [data-testid="stToolbar"], [data-testid="stDecoration"] {
        display: none !important;
        visibility: hidden !important;
    }
    
    /* Hero Header */
    .hero-container {
        text-align: center;
        padding: 2.8rem 1rem 1.4rem 1rem;
        position: relative;
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.15), rgba(139, 92, 246, 0.15));
        border: 1px solid rgba(56, 189, 248, 0.4);
        color: #38BDF8;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        padding: 7px 22px;
        border-radius: 999px;
        margin-bottom: 1.2rem;
        box-shadow: 0 0 30px rgba(56, 189, 248, 0.25);
        backdrop-filter: blur(10px);
    }
    .hero-title {
        font-size: 3.8rem;
        font-weight: 900;
        letter-spacing: -0.04em;
        line-height: 1.08;
        background: linear-gradient(135deg, #FFFFFF 20%, #CBD5E1 55%, #38BDF8 95%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.8rem;
        text-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }
    .hero-subtitle {
        font-size: 1.22rem;
        color: #94A3B8;
        font-weight: 400;
        max-width: 720px;
        margin: 0 auto;
        line-height: 1.6;
    }

    /* Floating Metric Strip */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin: 1.8rem 0 2.5rem 0;
    }
    .metric-card {
        background: rgba(15, 23, 42, 0.55);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 18px;
        padding: 18px 16px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; height: 2px;
        background: linear-gradient(90deg, transparent, rgba(56, 189, 248, 0.6), transparent);
    }
    .metric-card:hover {
        transform: translateY(-4px);
        border-color: rgba(56, 189, 248, 0.45);
        box-shadow: 0 15px 35px rgba(56, 189, 248, 0.15);
    }
    .metric-val {
        font-size: 1.85rem;
        font-weight: 800;
        background: linear-gradient(135deg, #38BDF8 0%, #34D399 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2px;
    }
    .metric-lbl {
        font-size: 0.74rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #94A3B8;
    }

    /* Main Action Button (Luminous Gradient with Glow) */
    button[kind="primary"] {
        background: linear-gradient(135deg, #0EA5E9 0%, #06B6D4 50%, #10B981 100%) !important;
        color: #FFFFFF !important;
        font-weight: 800 !important;
        border: 1px solid rgba(56, 189, 248, 0.6) !important;
        border-radius: 14px !important;
        box-shadow: 0 6px 25px rgba(14, 165, 233, 0.45) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-size: 0.96rem !important;
        padding: 0.75rem 1.6rem !important;
    }
    button[kind="primary"]:hover {
        transform: translateY(-2px) scale(1.01) !important;
        box-shadow: 0 10px 35px rgba(16, 185, 129, 0.6) !important;
        border-color: rgba(52, 211, 153, 0.9) !important;
    }

    /* Secondary Gallery Buttons (Glass Glow Chips) */
    button[kind="secondary"] {
        background: rgba(22, 32, 54, 0.55) !important;
        color: #E2E8F0 !important;
        border: 1px solid rgba(255, 255, 255, 0.09) !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        font-size: 0.88rem !important;
        transition: all 0.25s ease !important;
        padding: 0.6rem 0.9rem !important;
    }
    button[kind="secondary"]:hover {
        border-color: #38BDF8 !important;
        color: #FFFFFF !important;
        background: rgba(56, 189, 248, 0.18) !important;
        box-shadow: 0 0 20px rgba(56, 189, 248, 0.3) !important;
        transform: translateY(-2px) !important;
    }

    /* Download AI Report Button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, rgba(20, 30, 50, 0.9), rgba(10, 15, 28, 0.95)) !important;
        color: #38BDF8 !important;
        border: 1px solid rgba(56, 189, 248, 0.45) !important;
        font-weight: 800 !important;
        border-radius: 14px !important;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.4) !important;
        transition: all 0.25s ease !important;
        font-size: 0.95rem !important;
        padding: 0.75rem 1.4rem !important;
    }
    .stDownloadButton > button:hover {
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.25), rgba(16, 185, 129, 0.25)) !important;
        border-color: #38BDF8 !important;
        color: #FFFFFF !important;
        box-shadow: 0 0 30px rgba(56, 189, 248, 0.5) !important;
        transform: translateY(-2px) !important;
    }

    /* Prediction Result Capsule */
    .result-capsule {
        background: linear-gradient(145deg, rgba(20, 30, 52, 0.75), rgba(10, 16, 30, 0.85));
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 22px 24px;
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.45);
        margin-bottom: 1.2rem;
    }
    .result-tag {
        font-size: 0.74rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.16em;
        color: #38BDF8;
        margin-bottom: 4px;
    }
    .result-name {
        font-size: 2.4rem;
        font-weight: 900;
        color: #FFFFFF;
        letter-spacing: -0.03em;
        line-height: 1.05;
        margin-bottom: 4px;
    }
    .result-meta {
        font-size: 1.08rem;
        font-style: italic;
        color: #94A3B8;
    }
    
    /* High-Tech Circular Meter */
    .hud-gauge {
        width: 86px;
        height: 86px;
        border-radius: 50%;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        border: 4px solid #10B981;
        background: radial-gradient(circle, rgba(16, 185, 129, 0.2) 0%, rgba(10, 15, 25, 0.8) 100%);
        box-shadow: 0 0 25px rgba(16, 185, 129, 0.35);
        text-align: center;
        flex-shrink: 0;
    }
    .gauge-val {
        font-size: 1.22rem;
        font-weight: 900;
        color: #FFFFFF;
        line-height: 1;
    }
    .gauge-lbl {
        font-size: 0.62rem;
        text-transform: uppercase;
        color: #34D399;
        font-weight: 800;
        letter-spacing: 0.08em;
        margin-top: 3px;
    }

    /* Top-3 Ranking Row */
    .rank-capsule {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: rgba(22, 32, 54, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 14px;
        padding: 12px 16px;
        margin-bottom: 10px;
        transition: all 0.2s ease;
    }
    .rank-capsule:hover {
        background: rgba(22, 32, 54, 0.65);
        border-color: rgba(56, 189, 248, 0.3);
    }
    .rank-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 26px;
        height: 26px;
        background: rgba(56, 189, 248, 0.15);
        color: #38BDF8;
        font-weight: 800;
        font-size: 0.82rem;
        border-radius: 8px;
        margin-right: 12px;
    }
    .rank-title {
        font-weight: 700;
        font-size: 0.98rem;
        color: #F1F5F9;
    }
    .rank-score {
        font-weight: 800;
        font-size: 0.98rem;
        color: #38BDF8;
    }

    /* XAI Diagnostic Callout */
    .xai-callout {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.7), rgba(20, 30, 52, 0.5));
        border-left: 4px solid #38BDF8;
        border-radius: 14px;
        padding: 16px 20px;
        margin-top: 1.2rem;
        font-size: 0.94rem;
        line-height: 1.6;
        color: #CBD5E1;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
    }
    .xai-hdr {
        font-weight: 800;
        color: #38BDF8;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 6px;
    }

    /* Custom Styled Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(15, 23, 42, 0.75);
        padding: 6px;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 12px;
        color: #94A3B8;
        font-weight: 700;
        font-size: 0.9rem;
        padding: 9px 18px;
        transition: all 0.2s ease;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.2), rgba(139, 92, 246, 0.2)) !important;
        color: #38BDF8 !important;
        border: 1px solid rgba(56, 189, 248, 0.4) !important;
        box-shadow: 0 0 15px rgba(56, 189, 248, 0.2);
    }

    /* Supported Species Showcase Cards (Luminous Pods) */
    .species-pod {
        background: rgba(18, 26, 44, 0.55);
        backdrop-filter: blur(14px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 18px 14px;
        text-align: center;
        height: 100%;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    .species-pod:hover {
        transform: translateY(-4px);
        border-color: rgba(56, 189, 248, 0.5);
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.5);
        background: rgba(22, 34, 58, 0.8);
    }
    .pod-name {
        font-weight: 800;
        font-size: 1.0rem;
        color: #FFFFFF;
        margin-bottom: 4px;
    }
    .pod-sci {
        font-size: 0.84rem;
        font-style: italic;
        color: #38BDF8;
        margin-bottom: 4px;
    }
    .pod-fam {
        font-size: 0.76rem;
        font-weight: 600;
        color: #94A3B8;
    }

    /* Footer Signature */
    .app-footer {
        text-align: center;
        padding: 3.5rem 1rem 2.2rem 1rem;
        margin-top: 4rem;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
        color: #64748B;
        font-size: 0.9rem;
    }
    .footer-author {
        background: linear-gradient(135deg, #38BDF8, #34D399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 1.05rem;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. Cached Resource Loader (Safe Model Caching)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Initializing Neural AI Engine...")
def load_predictor():
    paths = resolve_project_paths()
    checkpoint_path = paths["checkpoint_path"]
    if not os.path.exists(checkpoint_path):
        st.error(f"Checkpoint not found at `{checkpoint_path}`")
        return None
    return ButterflyPredictor(checkpoint_path=checkpoint_path)

paths = resolve_project_paths()
predictor = load_predictor()

# -----------------------------------------------------------------------------
# 3. Hero Header
# -----------------------------------------------------------------------------
st.markdown("""
<div class="hero-container">
    <div class="hero-badge">🦋 ResNet18 • Deep Transfer Learning • Grad-CAM XAI</div>
    <div class="hero-title">AI Butterfly Vision</div>
    <div class="hero-subtitle">World-Class Butterfly Species Recognition with Real-Time Spatial Neural Gradient Explanations</div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 4. Floating Performance Metric Strip
# -----------------------------------------------------------------------------
st.markdown("""
<div class="metric-grid">
    <div class="metric-card">
        <div class="metric-val">97.22%</div>
        <div class="metric-lbl">Test Set Accuracy</div>
    </div>
    <div class="metric-card">
        <div class="metric-val">97.12%</div>
        <div class="metric-lbl">Macro F1-Score</div>
    </div>
    <div class="metric-card">
        <div class="metric-val">8 Species</div>
        <div class="metric-lbl">Taxonomic Classes</div>
    </div>
    <div class="metric-card">
        <div class="metric-val">Grad-CAM</div>
        <div class="metric-lbl">Explainable AI</div>
    </div>
</div>
""", unsafe_allow_html=True)

if predictor is None:
    st.error("⚠️ Failed to load AI model. Please verify `models/butterfly_resnet18_best.pth` exists.")
    st.stop()

# -----------------------------------------------------------------------------
# 5. Triple Input Studio: 1-Click Gallery | Upload Photo | Live Camera Capture
# -----------------------------------------------------------------------------
st.markdown("### 📸 Select or Capture Butterfly Specimen")

if "selected_image" not in st.session_state:
    st.session_state.selected_image = None
if "selected_filename" not in st.session_state:
    st.session_state.selected_filename = None

test_dir = paths["test_data_dir"]
benchmark_samples = {}
if os.path.exists(test_dir):
    for cls in predictor.class_names:
        cls_dir = os.path.join(test_dir, cls)
        if os.path.isdir(cls_dir):
            files = sorted(os.listdir(cls_dir))
            if files:
                benchmark_samples[cls] = os.path.join(cls_dir, files[0])

input_tab1, input_tab2, input_tab3 = st.tabs([
    "⚡ 1-Click Benchmark Gallery",
    "📤 Upload Butterfly Image",
    "📷 Live Camera Capture"
])

with input_tab1:
    st.caption("Click any specimen card below to instantly trigger neural inference and gradient explainability:")
    sample_cols = st.columns(4)
    for idx, (cls_name, fpath) in enumerate(benchmark_samples.items()):
        col_target = sample_cols[idx % 4]
        with col_target:
            if st.button(f"🦋 {cls_name}", key=f"btn_sample_{idx}", use_container_width=True):
                st.session_state.selected_image = Image.open(fpath).convert("RGB")
                st.session_state.selected_filename = f"{cls_name} ({os.path.basename(fpath)})"

with input_tab2:
    uploaded_file = st.file_uploader(
        "Drop or browse a butterfly image (JPG, JPEG, PNG)",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear image of a butterfly to analyze its species."
    )
    if uploaded_file is not None:
        try:
            st.session_state.selected_image = Image.open(uploaded_file).convert("RGB")
            st.session_state.selected_filename = uploaded_file.name
        except Exception as e:
            st.error(f"Error reading image: {e}")

with input_tab3:
    st.caption("Point your webcam or mobile camera at a butterfly photo or book:")
    camera_photo = st.camera_input("Take a butterfly snapshot", label_visibility="collapsed")
    if camera_photo is not None:
        try:
            st.session_state.selected_image = Image.open(camera_photo).convert("RGB")
            st.session_state.selected_filename = "Live_Camera_Capture.jpg"
        except Exception as e:
            st.error(f"Error capturing camera snapshot: {e}")

# -----------------------------------------------------------------------------
# 6. Analysis & Visual Diagnostic Suite
# -----------------------------------------------------------------------------
active_image = st.session_state.selected_image
active_filename = st.session_state.selected_filename

if active_image is not None:
    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
    
    # Image Command Bar
    cmd_col1, cmd_col2 = st.columns([1, 2.2])
    with cmd_col1:
        st.image(active_image, caption=f"Active Specimen: {active_filename}", use_container_width=True)
    
    with cmd_col2:
        st.markdown("### Ready for Neural Diagnostics")
        st.write("Extract deep convolutional feature maps with ResNet-18 and backpropagate gradients via layer4 Grad-CAM hooks.")
        run_analysis = st.button("✨ Run Neural Analysis & Explainability", type="primary", use_container_width=True)

    if run_analysis or st.session_state.get("auto_run", True):
        with st.spinner("Executing neural feature mapping and gradient backpropagation..."):
            try:
                # 1. Inference
                pred_res = predictor.predict(active_image)
                pred_class = pred_res['predicted_class']
                confidence = pred_res['confidence']
                top_k = pred_res['top_k']
                display_img = pred_res['display_image']

                # 2. Grad-CAM
                with GradCAM(predictor.model) as gradcam_engine:
                    heatmap_norm, _, _ = gradcam_engine.generate(
                        pred_res['input_tensor'],
                        pred_res['predicted_idx']
                    )

                st.markdown("<div style='height: 22px;'></div>", unsafe_allow_html=True)

                # Two-Column Results Suite
                col_left, col_right = st.columns([1.15, 1.35], gap="large")

                # --- LEFT COLUMN: Prediction Cockpit & Biological Facts ---
                with col_left:
                    st.markdown("### 🎯 Classification Result")
                    
                    meta = SPECIES_METADATA.get(pred_class, {
                        "scientific_name": "Unknown",
                        "family": "Insecta",
                        "color_primary": "#38BDF8",
                        "xai_insight": "Model focused on discriminative visual wing patterns."
                    })

                    gauge_color = "#10B981" if confidence >= 80.0 else "#F59E0B"
                    gauge_lbl = "HIGH" if confidence >= 80.0 else "MODERATE"

                    st.markdown(f"""
                    <div class="result-capsule" style="border-top: 4px solid {meta['color_primary']};">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                            <div>
                                <div class="result-tag">IDENTIFIED SPECIES</div>
                                <div class="result-name">{pred_class}</div>
                                <div class="result-meta">{meta['scientific_name']} • {meta['family']}</div>
                            </div>
                            <div class="hud-gauge" style="border-color: {gauge_color};">
                                <div class="gauge-val">{confidence:.1f}%</div>
                                <div class="gauge-lbl" style="color: {gauge_color};">{gauge_lbl}</div>
                            </div>
                        </div>
                        <div style="background: rgba(255,255,255,0.08); height: 8px; border-radius: 999px; overflow: hidden; margin-top: 14px;">
                            <div style="width: {min(confidence, 100.0):.1f}%; height: 100%; border-radius: 999px; background: linear-gradient(90deg, #38BDF8, {meta['color_primary']});"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Top-3 Probabilities
                    st.markdown("#### 🏆 Top-3 Ranked Probabilities")
                    for rank, (cls_name, prob) in enumerate(top_k, 1):
                        st.markdown(f"""
                        <div class="rank-capsule">
                            <div style="display: flex; align-items: center;">
                                <span class="rank-badge">{rank}</span>
                                <span class="rank-title">{cls_name}</span>
                            </div>
                            <span class="rank-score">{prob:.2f}%</span>
                        </div>
                        """, unsafe_allow_html=True)

                    # XAI Diagnostic Callout
                    st.markdown(f"""
                    <div class="xai-callout">
                        <div class="xai-hdr">🔬 Neural Attention Diagnostic</div>
                        {meta['xai_insight']}
                    </div>
                    """, unsafe_allow_html=True)

                    # Biological Metadata Expander
                    with st.expander("📖 View Biological Taxonomy & Habitat", expanded=False):
                        st.markdown(f"""
                        - **Visual Markers**: {meta['appearance']}
                        - **Geographic Range**: {meta['distribution']}
                        - **Ecological Feature**: {meta['key_features']}
                        """)

                # --- RIGHT COLUMN: Explainable AI (Grad-CAM) Studio ---
                with col_right:
                    st.markdown("### 🔬 Explainable AI Studio (Grad-CAM)")
                    st.caption("The highlighted regions indicate image areas that contributed strongly to the model's prediction.")

                    # Overlay Intensity Slider
                    blend_alpha = st.slider(
                        "Heatmap Blend Intensity (α)",
                        min_value=0.2,
                        max_value=0.85,
                        value=0.55,
                        step=0.05,
                        help="Adjust opacity of the Grad-CAM thermal overlay"
                    )

                    with GradCAM(predictor.model) as gradcam_engine:
                        heatmap_img, overlay_img = gradcam_engine.overlay_heatmap(
                            heatmap_norm,
                            display_img,
                            alpha=blend_alpha
                        )

                    cam_tab1, cam_tab2, cam_tab3, cam_tab4 = st.tabs([
                        "✨ Explanation Overlay",
                        "🌡️ Thermal Heatmap",
                        "🖼️ Original Input",
                        "🔍 Split Comparison"
                    ])

                    with cam_tab1:
                        st.image(
                            overlay_img,
                            caption=f"Grad-CAM Attention Map on {pred_class} (α = {blend_alpha:.2f})",
                            use_container_width=True
                        )
                    with cam_tab2:
                        st.image(
                            heatmap_img,
                            caption="Raw Class Activation Heatmap (Jet Colormap)",
                            use_container_width=True
                        )
                    with cam_tab3:
                        st.image(
                            display_img,
                            caption="Preprocessed Model Input (224x224)",
                            use_container_width=True
                        )
                    with cam_tab4:
                        comp_col1, comp_col2 = st.columns(2)
                        with comp_col1:
                            st.image(display_img, caption="Original Input", use_container_width=True)
                        with comp_col2:
                            st.image(overlay_img, caption="Grad-CAM Overlay", use_container_width=True)

                    # Export / Downloadable Inspection Report Card
                    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
                    report_bytes = generate_report_card(
                        original_image=display_img,
                        overlay_image=overlay_img,
                        pred_class=pred_class,
                        confidence=confidence,
                        top_k=top_k
                    )
                    st.download_button(
                        label="📥 Download AI Inspection Report Card (PNG)",
                        data=report_bytes,
                        file_name=f"Butterfly_AI_Report_{pred_class.replace(' ', '_')}.png",
                        mime="image/png",
                        use_container_width=True
                    )

            except Exception as e:
                st.error(f"Error during AI analysis: {e}")

else:
    st.info("💡 Choose a butterfly specimen above via 1-Click Gallery, Image Upload, or Live Camera to start AI analysis.")

# -----------------------------------------------------------------------------
# 7. Supported Butterfly Species Showcase (Luminous Pods)
# -----------------------------------------------------------------------------
st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)
st.markdown("### 🌿 Supported Butterfly Species Taxonomy (8 Classes)")
st.markdown("The neural network is specialized to distinguish the following 8 butterfly species:")

species_items = list(SPECIES_METADATA.items())
row1_cols = st.columns(4)
for i in range(4):
    name, s_meta = species_items[i]
    with row1_cols[i]:
        st.markdown(f"""
        <div class="species-pod" style="border-top: 3px solid {s_meta['color_primary']};">
            <div class="pod-name">{i+1}. {name}</div>
            <div class="pod-sci">{s_meta['scientific_name']}</div>
            <div class="pod-fam">{s_meta['family'].split('(')[0].strip()}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

row2_cols = st.columns(4)
for i in range(4, 8):
    name, s_meta = species_items[i]
    with row2_cols[i - 4]:
        st.markdown(f"""
        <div class="species-pod" style="border-top: 3px solid {s_meta['color_primary']};">
            <div class="pod-name">{i+1}. {name}</div>
            <div class="pod-sci">{s_meta['scientific_name']}</div>
            <div class="pod-fam">{s_meta['family'].split('(')[0].strip()}</div>
        </div>
        """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 8. Footer & Credits
# -----------------------------------------------------------------------------
st.markdown("""
<div class="app-footer">
    <div>Designed & Developed by <span class="footer-author">Ohi</span></div>
    <div style="margin-top: 8px; font-size: 0.84rem; color: #94A3B8;">
        Deep Transfer Learning (ResNet-18) • Native PyTorch Grad-CAM • TorchScript Mobile & Edge Architecture
    </div>
</div>
""", unsafe_allow_html=True)
