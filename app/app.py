"""
AI Butterfly Vision: Streamlit Web Application
An Ultra-High-Contrast, Crystal-Clear Typography & Eye-Comfort Interface for Butterfly Species Classification & Grad-CAM Explainable AI.
Designed & Developed by Ohi.
"""

import os
import sys
import base64
import hashlib
import textwrap
from io import BytesIO
from PIL import Image
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import cv2

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
from translations import UI_TEXT, SPECIES_DOSSIER

# -----------------------------------------------------------------------------
# 1. Page Configuration & Masterpiece Ultra-High Contrast CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Butterfly Vision • Explainable AI",
    page_icon="🦋",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Language State Initialization
if "app_lang" not in st.session_state:
    st.session_state.app_lang = "EN"

def t(key):
    lang = st.session_state.get("app_lang", "EN")
    return UI_TEXT.get(lang, {}).get(key, UI_TEXT.get("EN", {}).get(key, key))

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;600;700;800;900&family=Space+Grotesk:wght@700;800&display=swap');
    
    /* Global Reset with Bold, Large Typography */
    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        font-size: 17px !important;
    }
    
    code, pre, .stat-num, .gauge-val {
        font-family: 'Space Grotesk', monospace !important;
    }
    
    /* Clean Soft Background */
    .stApp {
        background-color: #EEF2F6 !important;
        color: #0F172A !important;
    }
    
    /* Hide Default Streamlit Clutter */
    #MainMenu, footer, header, .stDeployButton, [data-testid="stDeployButton"], [data-testid="stToolbar"], [data-testid="stDecoration"] {
        display: none !important;
        visibility: hidden !important;
    }
    
    /* ========================================================================= */
    /* ALL LABELS, WIDGETS, AND SLIDERS: 100% DEEP BLACK & ROYAL BLUE TRACK      */
    /* ========================================================================= */
    label, 
    [data-testid="stWidgetLabel"], 
    [data-testid="stWidgetLabel"] p, 
    [data-testid="stWidgetLabel"] span,
    div[data-testid="stSlider"] label,
    div[data-testid="stSlider"] label p,
    div[data-testid="stSlider"] [data-testid="stWidgetLabel"] p {
        color: #0F172A !important;
        -webkit-text-fill-color: #0F172A !important;
        font-weight: 900 !important;
        font-size: 1.15rem !important;
    }

    /* Slider Value Floating Badge */
    div[data-testid="stSlider"] div[data-testid="stThumbValue"] {
        background: #0284C7 !important;
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
        font-weight: 900 !important;
        font-size: 1.0rem !important;
        padding: 2px 10px !important;
        border-radius: 8px !important;
        box-shadow: 0 2px 8px rgba(2, 132, 199, 0.3) !important;
    }

    /* Slider Knob */
    div[data-testid="stSlider"] [role="slider"] {
        background-color: #0284C7 !important;
        border: 3px solid #FFFFFF !important;
        box-shadow: 0 3px 10px rgba(2, 132, 199, 0.5) !important;
        width: 22px !important;
        height: 22px !important;
    }

    /* Slider Track */
    div[data-testid="stSlider"] div[data-baseweb="slider"] > div > div:first-child {
        background: linear-gradient(90deg, #0284C7, #0D9488) !important;
        height: 8px !important;
        border-radius: 999px !important;
    }
    
    div[data-testid="stSlider"] div[data-baseweb="slider"] > div {
        background-color: #CBD5E1 !important;
        height: 8px !important;
        border-radius: 999px !important;
    }

    /* ========================================================================= */
    /* BULLETPROOF UNIVERSAL TABS (100% FIT • ZERO HORIZONTAL SCROLL CHEVRONS)   */
    /* ========================================================================= */
    
    /* Tab Container Bar */
    div[role="tablist"],
    [data-baseweb="tab-list"],
    div[data-testid="stTabs"] > div:first-child {
        display: flex !important;
        flex-wrap: nowrap !important;
        justify-content: space-between !important;
        align-items: center !important;
        gap: 8px !important;
        background: #F1F5F9 !important;
        padding: 6px !important;
        border-radius: 16px !important;
        border: 2px solid #CBD5E1 !important;
        overflow: hidden !important;
        width: 100% !important;
    }
    
    /* All Individual Tab Items */
    button[role="tab"],
    div[role="tab"],
    [data-baseweb="tab"],
    .stTabs [data-baseweb="tab"] {
        flex: 1 1 0 !important;
        min-width: 0 !important;
        border-radius: 11px !important;
        padding: 10px 10px !important;
        font-size: 0.95rem !important;
        font-weight: 800 !important;
        transition: all 0.2s ease !important;
        border: none !important;
        text-align: center !important;
        justify-content: center !important;
        white-space: normal !important;
        line-height: 1.25 !important;
    }
    
    /* UNSELECTED TABS */
    button[role="tab"][aria-selected="false"],
    div[role="tab"][aria-selected="false"],
    [data-baseweb="tab"][aria-selected="false"] {
        background-color: #FFFFFF !important;
        border: 1.5px solid #CBD5E1 !important;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.04) !important;
    }
    
    button[role="tab"][aria-selected="false"]:hover,
    div[role="tab"][aria-selected="false"]:hover,
    [data-baseweb="tab"][aria-selected="false"]:hover {
        border-color: #0284C7 !important;
        background-color: #F0F9FF !important;
    }

    button[role="tab"][aria-selected="false"] *,
    div[role="tab"][aria-selected="false"] *,
    [data-baseweb="tab"][aria-selected="false"] *,
    button[role="tab"][aria-selected="false"] p,
    div[role="tab"][aria-selected="false"] p,
    [data-baseweb="tab"][aria-selected="false"] p {
        color: #334155 !important;
        -webkit-text-fill-color: #334155 !important;
        font-weight: 800 !important;
        font-size: 0.95rem !important;
    }
    
    /* SELECTED TAB */
    button[role="tab"][aria-selected="true"],
    div[role="tab"][aria-selected="true"],
    [data-baseweb="tab"][aria-selected="true"] {
        background-color: #0284C7 !important;
        border: 1.5px solid #0284C7 !important;
        box-shadow: 0 4px 12px rgba(2, 132, 199, 0.3) !important;
    }
    
    button[role="tab"][aria-selected="true"] *,
    div[role="tab"][aria-selected="true"] *,
    [data-baseweb="tab"][aria-selected="true"] *,
    button[role="tab"][aria-selected="true"] p,
    div[role="tab"][aria-selected="true"] p,
    [data-baseweb="tab"][aria-selected="true"] p {
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
        font-weight: 900 !important;
        font-size: 0.95rem !important;
    }
    
    /* REMOVE ALL STREAMLIT RED HIGHLIGHT LINES, SCROLL BUTTONS & DARK GRADIENT OVERFLOWS */
    [data-baseweb="tab-highlight"],
    [data-baseweb="tab-border"],
    [data-baseweb="tab-scroll-button"],
    [data-baseweb="tab-list"] > button,
    div[data-testid="stTabs"] hr,
    div[data-testid="stTabs"] [data-baseweb="tab-highlight"],
    div[data-testid="stTabs"] [data-baseweb="tab-border"],
    div[data-testid="stTabs"] button[aria-label*="scroll"],
    div[data-testid="stTabs"] [data-testid="stBaseButton-header"],
    div[data-testid="stTabs"] [data-baseweb="tab-list"] > button,
    [data-baseweb="tab-list"] button[aria-label*="scroll"] {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
        width: 0 !important;
        opacity: 0 !important;
        pointer-events: none !important;
        background-color: transparent !important;
        border: none !important;
    }

    /* HIDE / REMOVE STREAMLIT FLOATING BLACK IMAGE CORNER TOOLBAR */
    [data-testid="stElementToolbar"],
    div[data-testid="stImage"] [data-testid="stElementToolbar"],
    div[data-testid="stImage"] button[title="View fullscreen"] {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        pointer-events: none !important;
    }

    /* ========================================================================= */
    /* ULTRA-PREMIUM LUXURY PILL / SEGMENTED CAPSULE SWITCHER (LANGUAGE SELECTOR) */
    /* ========================================================================= */
    div[data-testid="stRadio"] {
        display: flex !important;
        justify-content: flex-end !important;
        align-items: center !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    
    div[data-testid="stRadio"] > label,
    div[data-testid="stRadio"] [data-testid="stWidgetLabel"] {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
        margin: 0 !important;
    }

    div[data-testid="stRadio"] div[role="radiogroup"],
    div[data-testid="stRadio"] > div:last-child {
        display: inline-flex !important;
        flex-direction: row !important;
        background: #FFFFFF !important;
        padding: 4px 6px !important;
        border-radius: 999px !important;
        border: 2px solid #CBD5E1 !important;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.08), inset 0 1px 2px rgba(255, 255, 255, 0.9) !important;
        gap: 6px !important;
        align-items: center !important;
    }

    /* Target all radio option labels */
    div[data-testid="stRadio"] label[data-baseweb="radio"],
    div[data-testid="stRadio"] label[role="radio"],
    div[data-testid="stRadio"] label {
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        margin: 0 !important;
        padding: 7px 20px !important;
        border-radius: 999px !important;
        cursor: pointer !important;
        border: 1.5px solid transparent !important;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
        background: transparent !important;
        position: relative !important;
        user-select: none !important;
    }

    /* Completely hide all circular radio dots / default inputs / marks / svgs */
    div[data-testid="stRadio"] label > div:first-child,
    div[data-testid="stRadio"] [data-baseweb="radio"] > div:first-child,
    div[data-testid="stRadio"] [data-testid="stRadio-item"] > div:first-child,
    div[data-testid="stRadio"] [data-testid="stRadio-item"] label > div:first-child,
    div[data-testid="stRadio"] input[type="radio"],
    div[data-testid="stRadio"] input[type="radio"] + div,
    div[data-testid="stRadio"] div[class*="Radio"],
    div[data-testid="stRadio"] div[class*="radio"],
    div[data-testid="stRadio"] div[class*="RadioMark"],
    div[data-testid="stRadio"] div[class*="RadioIndicator"],
    div[data-testid="stRadio"] div[class*="StyledRadio"],
    div[data-testid="stRadio"] svg {
        display: none !important;
        visibility: hidden !important;
        width: 0 !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        opacity: 0 !important;
        pointer-events: none !important;
        position: absolute !important;
    }

    /* Unselected text styling */
    div[data-testid="stRadio"] label div[data-testid="stMarkdownContainer"] p,
    div[data-testid="stRadio"] label span,
    div[data-testid="stRadio"] label p,
    div[data-testid="stRadio"] label div {
        color: #1E293B !important;
        -webkit-text-fill-color: #1E293B !important;
        font-weight: 800 !important;
        font-size: 0.95rem !important;
        letter-spacing: -0.01em !important;
        margin: 0 !important;
        padding: 0 !important;
        transition: color 0.2s ease !important;
    }

    /* Unselected Hover state */
    div[data-testid="stRadio"] label:hover {
        background: #F1F5F9 !important;
        border-color: #E2E8F0 !important;
        transform: translateY(-1px) !important;
    }
    
    div[data-testid="stRadio"] label:hover p,
    div[data-testid="stRadio"] label:hover span {
        color: #0284C7 !important;
        -webkit-text-fill-color: #0284C7 !important;
    }

    /* SELECTED / ACTIVE OPTION CAPSULE */
    div[data-testid="stRadio"] label:has(input:checked),
    div[data-testid="stRadio"] label[aria-checked="true"],
    div[data-testid="stRadio"] label:has([aria-checked="true"]),
    div[data-testid="stRadio"] label:has(input[type="radio"]:checked) {
        background: linear-gradient(135deg, #0284C7 0%, #0369A1 100%) !important;
        border: 1.5px solid #0284C7 !important;
        box-shadow: 0 4px 14px rgba(2, 132, 199, 0.4), inset 0 1px 1px rgba(255, 255, 255, 0.4) !important;
        transform: translateY(-1px) !important;
    }

    /* SELECTED / ACTIVE OPTION TEXT: SOLID BRILLIANT WHITE */
    div[data-testid="stRadio"] label:has(input:checked) *,
    div[data-testid="stRadio"] label:has(input:checked) p,
    div[data-testid="stRadio"] label:has(input:checked) span,
    div[data-testid="stRadio"] label:has(input:checked) div,
    div[data-testid="stRadio"] label:has(input:checked) [data-testid="stMarkdownContainer"] p,
    div[data-testid="stRadio"] label[aria-checked="true"] *,
    div[data-testid="stRadio"] label[aria-checked="true"] p,
    div[data-testid="stRadio"] label[aria-checked="true"] span,
    div[data-testid="stRadio"] label[aria-checked="true"] div,
    div[data-testid="stRadio"] label[aria-checked="true"] [data-testid="stMarkdownContainer"] p,
    div[data-testid="stRadio"] label:has(input[type="radio"]:checked) * {
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
        font-weight: 900 !important;
        text-shadow: 0 1px 3px rgba(0, 0, 0, 0.35) !important;
    }

    /* ========================================================================= */
    /* MASTERPIECE SPECIMEN STUDIO IMAGE FRAMING (EXPANDED & BEAUTIFULLY SCALED) */
    /* ========================================================================= */
    div[data-testid="stImage"] {
        background: #FFFFFF !important;
        border: 2px solid #CBD5E1 !important;
        border-radius: 22px !important;
        padding: 16px !important;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08) !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;
        align-items: center !important;
        overflow: hidden !important;
        margin-top: 14px !important;
        width: 100% !important;
        transition: all 0.25s ease !important;
    }
    
    div[data-testid="stImage"]:hover {
        border-color: #0284C7 !important;
        box-shadow: 0 14px 35px rgba(2, 132, 199, 0.18) !important;
    }
    
    div[data-testid="stImage"] img {
        width: 100% !important;
        max-width: 100% !important;
        max-height: 580px !important;
        height: auto !important;
        object-fit: contain !important;
        border-radius: 16px !important;
        display: block !important;
        margin: 0 auto !important;
    }

    div[data-testid="stImage"] [data-testid="stImageCaption"],
    div[data-testid="stImage"] [data-testid="stCaptionContainer"] {
        text-align: center !important;
        font-weight: 800 !important;
        color: #0F172A !important;
        font-size: 1.02rem !important;
        margin-top: 12px !important;
        letter-spacing: 0.02em !important;
        padding: 0 4px !important;
    }
    
    /* Compact Camera Box */
    [data-testid="stCameraInput"] {
        max-width: 520px !important;
        margin: 0 auto !important;
        border-radius: 20px !important;
        overflow: hidden !important;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.15) !important;
        border: 2px solid #94A3B8 !important;
    }

    /* ========================================================================= */
    /* MASTERPIECE DRAG & DROP FILE UPLOADER (ZERO BLACK • PURE WHITE & SKY BLUE) */
    /* ========================================================================= */
    [data-testid="stFileUploader"],
    [data-testid="stFileUploader"] > div,
    [data-testid="stFileUploader"] section,
    [data-testid="stFileUploadDropzone"],
    section[data-testid="stFileUploadDropzone"],
    div[data-testid="stFileUploaderDropzone"],
    div[data-baseweb="file-uploader"],
    div[data-baseweb="file-uploader"] > div {
        background: #FFFFFF !important;
        background-color: #FFFFFF !important;
        border-radius: 20px !important;
    }

    [data-testid="stFileUploader"] section,
    section[data-testid="stFileUploadDropzone"],
    div[data-testid="stFileUploaderDropzone"],
    div[data-baseweb="file-uploader"] > div {
        border: 2.5px dashed #0284C7 !important;
        padding: 24px 20px !important;
        text-align: center !important;
        box-shadow: 0 8px 24px rgba(2, 132, 199, 0.08) !important;
        transition: all 0.25s ease !important;
    }
    
    [data-testid="stFileUploader"] section:hover,
    section[data-testid="stFileUploadDropzone"]:hover,
    div[data-testid="stFileUploaderDropzone"]:hover {
        background: #F0F9FF !important;
        background-color: #F0F9FF !important;
        border-color: #0369A1 !important;
        border-style: solid !important;
        box-shadow: 0 14px 32px rgba(2, 132, 199, 0.18) !important;
        transform: translateY(-2px) !important;
    }

    /* All text inside uploader: Deep Solid Black */
    [data-testid="stFileUploader"] span,
    [data-testid="stFileUploader"] p,
    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploader"] div,
    [data-testid="stFileUploadDropzone"] span,
    [data-testid="stFileUploadDropzone"] p,
    [data-testid="stFileUploadDropzone"] small,
    [data-testid="stFileUploaderDropzoneInstructions"] span,
    [data-testid="stFileUploaderDropzoneInstructions"] small,
    [data-testid="stFileUploaderDropzoneInstructions"] div {
        color: #0F172A !important;
        -webkit-text-fill-color: #0F172A !important;
        font-weight: 800 !important;
        font-size: 1.05rem !important;
    }

    /* Upload Button: Solid Royal Blue */
    [data-testid="stFileUploader"] button,
    [data-testid="stFileUploadDropzone"] button,
    div[data-testid="stFileUploader"] button[kind="secondary"] {
        background: #0284C7 !important;
        background-color: #0284C7 !important;
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 900 !important;
        font-size: 1.05rem !important;
        padding: 10px 24px !important;
        box-shadow: 0 4px 14px rgba(2, 132, 199, 0.3) !important;
        transition: all 0.2s ease !important;
    }
    [data-testid="stFileUploader"] button:hover,
    [data-testid="stFileUploadDropzone"] button:hover,
    div[data-testid="stFileUploader"] button[kind="secondary"]:hover {
        background: #0369A1 !important;
        background-color: #0369A1 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 20px rgba(2, 132, 199, 0.45) !important;
    }
    [data-testid="stFileUploader"] button *,
    [data-testid="stFileUploadDropzone"] button * {
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
        font-weight: 900 !important;
    }
    
    /* ========================================================================= */
    /* MASTERPIECE FLUID HERO CARD (RESPONSIVE GLASSMORPHISM)                    */
    /* ========================================================================= */
    .hero-left-card {
        background: linear-gradient(135deg, #FFFFFF 0%, #F0F9FF 60%, #E0F2FE 100%);
        border: 2.5px solid #94A3B8;
        border-radius: 24px;
        padding: 20px 22px;
        box-shadow: 0 14px 32px rgba(2, 132, 199, 0.10), inset 0 0 20px rgba(255, 255, 255, 0.8);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        min-height: 290px;
        height: 290px;
        box-sizing: border-box;
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #0284C7;
        color: #FFFFFF;
        font-size: 0.78rem;
        font-weight: 900;
        padding: 4px 12px;
        border-radius: 999px;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        width: fit-content;
        margin-bottom: 6px;
    }
    .hero-title {
        font-family: 'Space Grotesk', 'Plus Jakarta Sans', sans-serif;
        font-size: 2.05rem;
        font-weight: 900;
        color: #0F172A;
        line-height: 1.15;
        margin: 3px 0 5px 0;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        font-size: 0.90rem;
        font-weight: 700;
        color: #334155;
        line-height: 1.38;
        margin-bottom: 8px;
    }
    .hero-tag-strip {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: auto;
    }
    .hero-tag {
        display: inline-flex;
        align-items: center;
        background: #FFFFFF;
        border: 1.5px solid #94A3B8;
        color: #0F172A;
        font-size: 0.78rem;
        font-weight: 800;
        padding: 4px 12px;
        border-radius: 999px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
        transition: all 0.2s ease;
    }
    .hero-tag:hover {
        border-color: #0284C7;
        color: #0284C7;
        transform: translateY(-1px);
    }

    /* ========================================================================= */
    /* MASTERPIECE 3D BOTANICAL VIVARIUM (HIGH-FIDELITY SPECIES & NIGHT MODE)    */
    /* ========================================================================= */
    .sanctuary-terrarium {
        background: linear-gradient(135deg, #FFFFFF 0%, #F0FDF4 40%, #E0F2FE 100%);
        border: 2.5px solid #94A3B8;
        border-radius: 28px;
        padding: 24px;
        height: 290px;
        position: relative;
        overflow: hidden;
        box-shadow: 0 16px 36px rgba(2, 132, 199, 0.12), inset 0 0 20px rgba(255, 255, 255, 0.8);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        transition: background 0.6s ease, border-color 0.6s ease, box-shadow 0.6s ease;
    }

    /* Bioluminescent Night Mode */
    .sanctuary-terrarium.sanctuary-night {
        background: linear-gradient(135deg, #050B14 0%, #0B192C 60%, #062820 100%) !important;
        border-color: #10B981 !important;
        box-shadow: 0 16px 40px rgba(16, 185, 129, 0.25), inset 0 0 30px rgba(16, 185, 129, 0.15) !important;
    }

    /* Day / Night Sanctuary Toggle Button */
    .sanctuary-theme-toggle {
        position: absolute;
        top: 14px;
        right: 16px;
        z-index: 25;
        background: rgba(255, 255, 255, 0.92);
        border: 1.5px solid #CBD5E1;
        color: #0F172A;
        font-size: 0.78rem;
        font-weight: 800;
        padding: 5px 13px;
        border-radius: 999px;
        cursor: pointer;
        box-shadow: 0 3px 8px rgba(0,0,0,0.06);
        transition: all 0.25s ease;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    .sanctuary-theme-toggle:hover {
        transform: scale(1.06);
        border-color: #0284C7;
        box-shadow: 0 4px 12px rgba(2, 132, 199, 0.2);
    }
    .sanctuary-night .sanctuary-theme-toggle {
        background: rgba(15, 23, 42, 0.88);
        border-color: #10B981;
        color: #34D399;
        box-shadow: 0 0 14px rgba(16, 185, 129, 0.4);
    }

    /* Ambient Bioluminescent Sparkles */
    .sanctuary-glow {
        position: absolute;
        top: -30px;
        right: -30px;
        width: 160px;
        height: 160px;
        background: radial-gradient(circle, rgba(2, 132, 199, 0.25) 0%, rgba(2, 132, 199, 0) 70%);
        border-radius: 50%;
        animation: glow-pulse 4s ease-in-out infinite alternate;
        pointer-events: none;
        transition: background 0.6s ease;
    }
    .sanctuary-night .sanctuary-glow {
        background: radial-gradient(circle, rgba(16, 185, 129, 0.35) 0%, rgba(16, 185, 129, 0) 70%);
    }
    
    @keyframes glow-pulse {
        0% { transform: scale(0.9); opacity: 0.5; }
        100% { transform: scale(1.3); opacity: 0.9; }
    }

    /* 3D Butterfly Stage */
    .butterfly-stage {
        position: relative;
        width: 100%;
        height: 155px;
        perspective: 800px;
    }

    /* Shared Carrier Group for Tooltips & Hover Scaling */
    .carrier-group {
        cursor: pointer;
        transform-style: preserve-3d;
        transition: filter 0.3s ease;
        will-change: transform;
    }
    .carrier-group.is-chasing {
        animation: none !important;
        transition: none !important;
    }
    .carrier-group:hover {
        z-index: 40 !important;
    }

    /* Interactive Specimen Tooltips */
    .specimen-tooltip {
        position: absolute;
        bottom: 110%;
        left: 50%;
        transform: translateX(-50%) translateY(8px) scale(0.9);
        opacity: 0;
        pointer-events: none;
        background: rgba(255, 255, 255, 0.96);
        backdrop-filter: blur(12px);
        border: 1.5px solid #0284C7;
        border-radius: 12px;
        padding: 8px 12px;
        box-shadow: 0 10px 25px rgba(2, 132, 199, 0.25);
        white-space: nowrap;
        z-index: 50;
        transition: all 0.25s cubic-bezier(0.2, 0.8, 0.2, 1);
        text-align: center;
    }
    .carrier-group:hover .specimen-tooltip {
        opacity: 1;
        transform: translateX(-50%) translateY(0) scale(1);
    }
    .sanctuary-night .specimen-tooltip {
        background: rgba(15, 23, 42, 0.94);
        border-color: #10B981;
        box-shadow: 0 10px 25px rgba(16, 185, 129, 0.4);
    }
    .st-name {
        font-size: 0.86rem;
        font-weight: 900;
        color: #0F172A;
        margin-bottom: 2px;
    }
    .sanctuary-night .st-name {
        color: #F8FAFC;
    }
    .st-sci {
        font-size: 0.74rem;
        font-style: italic;
        color: #0284C7;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .sanctuary-night .st-sci {
        color: #38BDF8;
    }
    .st-badge {
        font-size: 0.68rem;
        font-weight: 800;
        background: #ECFDF5;
        color: #059669;
        border: 1px solid #10B981;
        border-radius: 999px;
        padding: 2px 8px;
        display: inline-block;
    }
    .sanctuary-night .st-badge {
        background: rgba(6, 78, 59, 0.6);
        color: #34D399;
    }

    /* 3D Wing Flapping Mechanics (Precise Perspective) */
    .butterfly-svg-unit {
        display: flex;
        align-items: center;
        transform-style: preserve-3d;
        filter: drop-shadow(0 4px 10px rgba(0, 0, 0, 0.15));
        transition: filter 0.3s ease;
    }
    .wing-left-svg {
        transform-origin: 100% 50%;
        animation: wing-flap-left 0.25s ease-in-out infinite alternate;
    }
    .wing-right-svg {
        transform-origin: 0% 50%;
        animation: wing-flap-right 0.25s ease-in-out infinite alternate;
    }

    @keyframes wing-flap-left {
        0% { transform: rotateY(0deg) rotateZ(0deg); }
        100% { transform: rotateY(-65deg) rotateZ(-5deg); }
    }
    @keyframes wing-flap-right {
        0% { transform: rotateY(0deg) rotateZ(0deg); }
        100% { transform: rotateY(65deg) rotateZ(5deg); }
    }

    /* 1. Monarch Carrier */
    .monarch-carrier {
        position: absolute;
        top: 25px;
        left: 28%;
        animation: monarch-flight 6.5s ease-in-out infinite alternate;
        transform-style: preserve-3d;
    }
    @keyframes monarch-flight {
        0% { transform: translate3d(0, 0, 0) rotate(4deg); }
        30% { transform: translate3d(25px, -18px, 15px) rotate(-6deg); }
        70% { transform: translate3d(-15px, 14px, -10px) rotate(5deg); }
        100% { transform: translate3d(18px, -10px, 8px) rotate(-2deg); }
    }
    .sanctuary-night .monarch-carrier {
        filter: drop-shadow(0 0 10px #F97316) drop-shadow(0 0 20px rgba(249, 115, 22, 0.4));
    }

    /* 2. Adonis Blue Carrier */
    .adonis-carrier {
        position: absolute;
        top: 55px;
        right: 22%;
        animation: adonis-flight 5.2s ease-in-out infinite alternate;
        transform-style: preserve-3d;
    }
    @keyframes adonis-flight {
        0% { transform: translate3d(0, 0, 0) rotate(-5deg); }
        40% { transform: translate3d(-28px, -20px, 12px) rotate(8deg); }
        80% { transform: translate3d(12px, 16px, -8px) rotate(-4deg); }
        100% { transform: translate3d(-16px, -10px, 5px) rotate(3deg); }
    }
    .sanctuary-night .adonis-carrier {
        filter: drop-shadow(0 0 12px #38BDF8) drop-shadow(0 0 22px rgba(56, 189, 248, 0.5));
    }

    /* 3. Emerald Cattleheart Carrier */
    .emerald-carrier {
        position: absolute;
        top: 15px;
        left: 8%;
        animation: emerald-flight 7.0s ease-in-out infinite alternate;
        transform-style: preserve-3d;
    }
    @keyframes emerald-flight {
        0% { transform: translate3d(0, 0, 0) rotate(8deg); }
        50% { transform: translate3d(22px, 22px, 8px) rotate(-10deg); }
        100% { transform: translate3d(-10px, 8px, -5px) rotate(4deg); }
    }
    .sanctuary-night .emerald-carrier {
        filter: drop-shadow(0 0 12px #10B981) drop-shadow(0 0 24px rgba(16, 185, 129, 0.5));
    }

    /* 4. Red Postman Carrier */
    .postman-carrier {
        position: absolute;
        top: 70px;
        left: 45%;
        animation: postman-flight 5.8s ease-in-out infinite alternate;
        transform-style: preserve-3d;
    }
    @keyframes postman-flight {
        0% { transform: translate3d(0, 0, 0) rotate(-4deg); }
        45% { transform: translate3d(-18px, -16px, 10px) rotate(6deg); }
        100% { transform: translate3d(20px, 12px, -8px) rotate(-3deg); }
    }
    .sanctuary-night .postman-carrier {
        filter: drop-shadow(0 0 12px #EF4444) drop-shadow(0 0 20px rgba(239, 68, 68, 0.4));
    }

    /* 5. Southern Dogface Carrier */
    .dogface-carrier {
        position: absolute;
        top: 22px;
        right: 8%;
        animation: dogface-flight 6.2s ease-in-out infinite alternate;
        transform-style: preserve-3d;
    }
    @keyframes dogface-flight {
        0% { transform: translate3d(0, 0, 0) rotate(6deg); }
        55% { transform: translate3d(-15px, 20px, 12px) rotate(-8deg); }
        100% { transform: translate3d(10px, -12px, -6px) rotate(4deg); }
    }
    .sanctuary-night .dogface-carrier {
        filter: drop-shadow(0 0 12px #FBBF24) drop-shadow(0 0 22px rgba(251, 191, 36, 0.5));
    }

    /* Botanical Wildflowers Habitat */
    .flower-habitat {
        position: absolute;
        bottom: 58px;
        left: 0;
        width: 100%;
        height: 75px;
        pointer-events: none;
        display: flex;
        justify-content: space-between;
        padding: 0 15px;
        z-index: 4;
    }
    .botanical-flower {
        width: 68px;
        height: 75px;
        animation: flower-breathe 4.5s ease-in-out infinite alternate;
        transform-origin: bottom center;
        filter: drop-shadow(0 4px 10px rgba(0, 0, 0, 0.12));
        cursor: pointer;
        pointer-events: auto;
        transition: transform 0.2s ease;
    }
    .botanical-flower:hover {
        transform: scale(1.15) rotate(5deg);
    }
    .flower-alt {
        animation-delay: -2.2s;
        animation-duration: 5.2s;
    }

    @keyframes flower-breathe {
        0% { transform: rotate(-4deg) scale(0.96); }
        100% { transform: rotate(4deg) scale(1.04); }
    }

    /* Rising Honey Nectar Sparkles */
    .nectar-particle {
        position: absolute;
        width: 7px;
        height: 7px;
        background: #FBBF24;
        border-radius: 50%;
        box-shadow: 0 0 10px #F59E0B;
        animation: nectar-float 3.2s ease-in-out infinite;
    }
    .p1 { bottom: 30px; left: 35px; animation-delay: 0s; }
    .p2 { bottom: 50px; left: 48px; animation-delay: 1.1s; }
    .p3 { bottom: 30px; right: 35px; animation-delay: 0.5s; }
    .p4 { bottom: 50px; right: 48px; animation-delay: 1.7s; }

    @keyframes nectar-float {
        0% { transform: translateY(0) scale(0.5); opacity: 0; }
        50% { opacity: 0.95; transform: translateY(-35px) scale(1.2); }
        100% { transform: translateY(-70px) scale(0.3); opacity: 0; }
    }

    /* Bioluminescent Fireflies for Night Mode */
    .firefly-layer {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        opacity: 0;
        transition: opacity 0.8s ease;
        z-index: 8;
    }
    .sanctuary-night .firefly-layer {
        opacity: 1;
    }
    .firefly-dot {
        position: absolute;
        width: 5px;
        height: 5px;
        background: #6EE7B7;
        border-radius: 50%;
        box-shadow: 0 0 10px #34D399, 0 0 20px #10B981;
        animation: firefly-drift 5s ease-in-out infinite alternate;
    }
    .ff1 { top: 25%; left: 20%; animation-delay: 0s; }
    .ff2 { top: 60%; left: 40%; animation-delay: 1.2s; }
    .ff3 { top: 35%; left: 75%; animation-delay: 2.4s; }
    .ff4 { top: 70%; left: 85%; animation-delay: 0.8s; }

    @keyframes firefly-drift {
        0% { opacity: 0.2; transform: translate(0, 0) scale(0.7); }
        50% { opacity: 1; transform: translate(25px, -25px) scale(1.3); }
        100% { opacity: 0.3; transform: translate(-18px, -45px) scale(0.8); }
    }

    /* Dynamic Cursor Honey Nectar Droplet */
    #cursor-nectar {
        position: absolute;
        width: 28px;
        height: 28px;
        border-radius: 50%;
        background: radial-gradient(circle, #FDE047 15%, #F59E0B 70%, #D97706 100%);
        box-shadow: 0 0 18px #F59E0B, 0 0 30px rgba(245, 158, 11, 0.6);
        pointer-events: none;
        opacity: 0;
        transform: translate(-50%, -50%) scale(0.5);
        transition: opacity 0.3s ease, transform 0.2s ease;
        z-index: 30;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 14px;
    }
    .sanctuary-terrarium:hover #cursor-nectar {
        opacity: 1;
        transform: translate(-50%, -50%) scale(1);
    }

    /* Terrarium Bottom Badge */
    .terrarium-footer {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: rgba(255, 255, 255, 0.90);
        backdrop-filter: blur(10px);
        border: 1.5px solid #CBD5E1;
        border-radius: 14px;
        padding: 8px 16px;
        z-index: 10;
        transition: background 0.6s ease, border-color 0.6s ease;
    }
    .sanctuary-night .terrarium-footer {
        background: rgba(15, 23, 42, 0.88) !important;
        border-color: rgba(16, 185, 129, 0.4) !important;
    }
    .terrarium-title {
        font-size: 0.92rem;
        font-weight: 900;
        color: #0F172A;
        letter-spacing: 0.04em;
        transition: color 0.6s ease;
    }
    .sanctuary-night .terrarium-title {
        color: #F8FAFC !important;
    }
    .terrarium-live {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.84rem;
        font-weight: 900;
        color: #059669;
    }
    .sanctuary-night .terrarium-live {
        color: #34D399 !important;
    }
    .live-dot {
        width: 8px;
        height: 8px;
        background: #10B981;
        border-radius: 50%;
        box-shadow: 0 0 8px #10B981;
        animation: pulse-dot 1.5s infinite;
    }
    @keyframes pulse-dot {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.4; transform: scale(0.8); }
    }

    /* Floating Metric Strip */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin: 1.2rem 0 2.2rem 0;
    }
    .metric-card {
        background: #FFFFFF;
        border: 2px solid #94A3B8;
        border-radius: 18px;
        padding: 18px 14px;
        text-align: center;
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.08);
        transition: all 0.25s ease;
        min-height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        border-color: #0284C7;
        box-shadow: 0 12px 28px rgba(2, 132, 199, 0.2);
    }
    .metric-val {
        font-size: 2.2rem;
        font-weight: 900;
        color: #0284C7;
        margin-bottom: 4px;
        line-height: 1.1;
    }
    .metric-lbl {
        font-size: 0.94rem;
        font-weight: 900;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #0F172A;
        min-height: 38px;
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
        line-height: 1.2;
    }

    /* Primary Action Button (Big & Bold) */
    button[kind="primary"],
    div[data-testid="stButton"] button[kind="primary"] {
        background: #0284C7 !important;
        color: #FFFFFF !important;
        font-weight: 900 !important;
        border: none !important;
        border-radius: 14px !important;
        box-shadow: 0 6px 20px rgba(2, 132, 199, 0.35) !important;
        transition: all 0.25s ease !important;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        font-size: 1.10rem !important;
        padding: 10px 24px !important;
        min-height: 56px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        text-align: center !important;
        line-height: 1.2 !important;
    }
    button[kind="primary"]:hover,
    div[data-testid="stButton"] button[kind="primary"]:hover {
        background: #0369A1 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 28px rgba(2, 132, 199, 0.45) !important;
    }

    /* Secondary & Sample Gallery Buttons (Grid Uniform Alignment) */
    button[kind="secondary"],
    div[data-testid="stButton"] button[kind="secondary"],
    div[data-testid="stButton"] button {
        background: #FFFFFF !important;
        color: #0F172A !important;
        border: 2px solid #CBD5E1 !important;
        border-radius: 14px !important;
        font-weight: 800 !important;
        font-size: 0.95rem !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05) !important;
        padding: 8px 12px !important;
        min-height: 60px !important;
        height: 60px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        text-align: center !important;
        line-height: 1.25 !important;
        white-space: normal !important;
        word-break: break-word !important;
    }
    button[kind="secondary"] *,
    div[data-testid="stButton"] button[kind="secondary"] *,
    div[data-testid="stButton"] button * {
        color: #0F172A !important;
        -webkit-text-fill-color: #0F172A !important;
        font-weight: 800 !important;
        text-align: center !important;
    }
    button[kind="secondary"]:hover,
    div[data-testid="stButton"] button[kind="secondary"]:hover,
    div[data-testid="stButton"] button:hover {
        border-color: #0284C7 !important;
        color: #0284C7 !important;
        background: #E0F2FE !important;
        box-shadow: 0 6px 18px rgba(2, 132, 199, 0.25) !important;
        transform: translateY(-2px) !important;
    }
    button[kind="secondary"]:hover *,
    div[data-testid="stButton"] button[kind="secondary"]:hover * {
        color: #0284C7 !important;
        -webkit-text-fill-color: #0284C7 !important;
    }

    /* Download AI Report Button */
    .stDownloadButton > button {
        background: #0284C7 !important;
        color: #FFFFFF !important;
        border: none !important;
        font-weight: 900 !important;
        border-radius: 14px !important;
        box-shadow: 0 6px 20px rgba(2, 132, 199, 0.35) !important;
        transition: all 0.2s ease !important;
        font-size: 1.10rem !important;
        padding: 10px 24px !important;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        min-height: 56px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    .stDownloadButton > button:hover {
        background: #0369A1 !important;
        box-shadow: 0 10px 28px rgba(2, 132, 199, 0.45) !important;
        transform: translateY(-2px) !important;
    }

    /* Prediction Result Capsule */
    .result-capsule {
        background: #FFFFFF;
        border: 2.5px solid #94A3B8;
        border-radius: 24px;
        padding: 26px;
        box-shadow: 0 15px 35px rgba(15, 23, 42, 0.1);
        margin-bottom: 1.4rem;
    }
    .result-tag {
        font-size: 0.95rem;
        font-weight: 900;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #0284C7;
        margin-bottom: 6px;
    }
    .result-name {
        font-size: 2.7rem;
        font-weight: 900;
        color: #0F172A;
        letter-spacing: -0.03em;
        line-height: 1.1;
        margin-bottom: 6px;
    }
    .result-meta {
        font-size: 1.25rem;
        font-weight: 700;
        font-style: italic;
        color: #1E293B;
    }
    
    /* High-Tech Circular Meter */
    .hud-gauge {
        width: 102px;
        height: 102px;
        border-radius: 50%;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        border: 5px solid #10B981;
        background: #ECFDF5;
        box-shadow: 0 0 25px rgba(16, 185, 129, 0.3);
        text-align: center;
        flex-shrink: 0;
    }
    .gauge-val {
        font-size: 1.55rem;
        font-weight: 900;
        color: #065F46;
        line-height: 1;
    }
    .gauge-lbl {
        font-size: 0.76rem;
        text-transform: uppercase;
        color: #059669;
        font-weight: 900;
        letter-spacing: 0.08em;
        margin-top: 3px;
    }

    /* Top-3 Ranking Row */
    .rank-capsule {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: #FFFFFF;
        border: 2px solid #CBD5E1;
        border-radius: 16px;
        padding: 16px 20px;
        margin-bottom: 11px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
        transition: all 0.2s ease;
    }
    .rank-capsule:hover {
        border-color: #0284C7;
        background: #F8FAFC;
    }
    .rank-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 32px;
        height: 32px;
        background: #E0F2FE;
        color: #0369A1;
        font-weight: 900;
        font-size: 1.05rem;
        border-radius: 8px;
        margin-right: 16px;
    }
    .rank-title {
        font-weight: 900;
        font-size: 1.2rem;
        color: #0F172A;
    }
    .rank-score {
        font-weight: 900;
        font-size: 1.25rem;
        color: #0284C7;
    }

    /* XAI Diagnostic Callout */
    .xai-callout {
        background: #E0F2FE;
        border-left: 6px solid #0284C7;
        border-radius: 16px;
        padding: 20px 22px;
        margin-top: 1.4rem;
        font-size: 1.12rem;
        font-weight: 600;
        line-height: 1.65;
        color: #0F172A;
        box-shadow: 0 4px 16px rgba(2, 132, 199, 0.1);
    }
    .xai-hdr {
        font-weight: 900;
        color: #0369A1;
        font-size: 1.05rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 6px;
    }

    /* ========================================================================= */
    /* MASTERPIECE PHOTOGRAPHIC SPECIES CARDS (MUSEUM GRADE - UNIFIED GRID)      */
    /* ========================================================================= */
    .species-card-v2 {
        background: #FFFFFF !important;
        border: 2px solid #CBD5E1 !important;
        border-radius: 20px !important;
        padding: 18px 12px 14px 12px !important;
        text-align: center !important;
        min-height: 285px !important;
        height: 285px !important;
        box-shadow: 0 6px 20px rgba(15, 23, 42, 0.05) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        position: relative !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: space-between !important;
        margin-bottom: 8px !important;
    }
    .species-card-v2:hover {
        transform: translateY(-5px) !important;
        box-shadow: 0 16px 32px rgba(2, 132, 199, 0.18) !important;
        border-color: #0284C7 !important;
    }
    .species-num-badge {
        position: absolute;
        top: 10px;
        right: 12px;
        background: #F1F5F9;
        color: #0369A1;
        font-weight: 900;
        font-size: 0.82rem;
        padding: 2px 8px;
        border-radius: 8px;
        border: 1px solid #CBD5E1;
    }
    .species-thumb-container {
        width: 96px;
        height: 96px;
        border-radius: 50%;
        padding: 3px;
        background: #FFFFFF;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        margin: 4px auto 10px auto;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: transform 0.3s ease;
        flex-shrink: 0;
    }
    .species-card-v2:hover .species-thumb-container {
        transform: scale(1.06);
    }
    .species-thumb-img {
        width: 90px;
        height: 90px;
        border-radius: 50%;
        object-fit: cover;
    }
    .card-title-name {
        font-size: 1.05rem !important;
        font-weight: 900 !important;
        color: #0F172A !important;
        margin: 2px 0 4px 0 !important;
        line-height: 1.25 !important;
        min-height: 48px !important;
        height: 48px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        text-align: center !important;
        overflow: hidden !important;
    }
    .card-sci-name {
        font-size: 0.88rem !important;
        font-weight: 700 !important;
        font-style: italic !important;
        color: #0284C7 !important;
        margin-bottom: 6px !important;
        min-height: 20px !important;
        height: 20px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    .card-fam-tag {
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        background: #F8FAFC !important;
        border: 1.5px solid #E2E8F0 !important;
        color: #475569 !important;
        font-size: 0.80rem !important;
        font-weight: 800 !important;
        padding: 3px 12px !important;
        border-radius: 999px !important;
        letter-spacing: 0.01em !important;
        min-height: 26px !important;
        height: 26px !important;
    }

    /* Footer */
    .app-footer {
        text-align: center;
        padding: 3.5rem 1rem 2.2rem 1rem;
        margin-top: 4rem;
        border-top: 2px solid #94A3B8;
        color: #1E293B;
        font-size: 1.05rem;
        font-weight: 700;
    }
    .footer-author {
        color: #0284C7;
        font-weight: 900;
        font-size: 1.25rem;
    }

    /* ========================================================================= */
    /* 100% BULLETPROOF RESPONSIVE DESIGN (MOBILE, TABLET, LAPTOP, DESKTOP)       */
    /* ========================================================================= */
    .main .block-container {
        max-width: 1360px !important;
        padding-top: 1.2rem !important;
        padding-bottom: 3rem !important;
        transition: padding 0.3s ease !important;
    }

    @media (max-width: 992px) {
        .main .block-container {
            padding-left: 1.2rem !important;
            padding-right: 1.2rem !important;
            padding-top: 1.0rem !important;
        }
        
        .metric-grid {
            grid-template-columns: repeat(2, 1fr) !important;
            gap: 12px !important;
        }
    }

    @media (max-width: 768px) {
        html, body, [class*="css"], .stApp {
            font-size: 15px !important;
        }

        .main .block-container {
            padding-left: 0.8rem !important;
            padding-right: 0.8rem !important;
            padding-top: 0.8rem !important;
        }

        /* Top Install App Button */
        button[key="btn_top_install_app"],
        div[data-testid="stButton"] button[key="btn_top_install_app"] {
            background: #FFFFFF !important;
            border: 2px solid #0284C7 !important;
            color: #0284C7 !important;
            border-radius: 999px !important;
            padding: 5px 12px !important;
            font-size: 0.86rem !important;
            font-weight: 800 !important;
            min-height: 40px !important;
            height: 40px !important;
            box-shadow: 0 2px 8px rgba(2, 132, 199, 0.12) !important;
            transition: all 0.2s ease !important;
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
        }
        button[key="btn_top_install_app"]:hover {
            background: #E0F2FE !important;
            border-color: #0369A1 !important;
            transform: translateY(-1px) !important;
        }

        /* Top Navigation Header on Mobile */
        div[data-testid="stRadio"] {
            justify-content: center !important;
            margin-top: 6px !important;
            width: 100% !important;
        }
        div[data-testid="stRadio"] div[role="radiogroup"],
        div[data-testid="stRadio"] > div:last-child {
            width: 100% !important;
            justify-content: center !important;
        }
        div[data-testid="stRadio"] label {
            flex: 1 1 0 !important;
            padding: 6px 12px !important;
            font-size: 0.88rem !important;
            text-align: center !important;
        }

        /* Metric Grid: 2x2 on Mobile */
        .metric-grid {
            grid-template-columns: repeat(2, 1fr) !important;
            gap: 10px !important;
            margin: 1rem 0 1.8rem 0 !important;
        }
        .metric-card {
            padding: 14px 8px !important;
            min-height: 96px !important;
            border-radius: 14px !important;
        }
        .metric-val {
            font-size: 1.7rem !important;
        }
        .metric-lbl {
            font-size: 0.80rem !important;
            min-height: 28px !important;
            line-height: 1.15 !important;
        }

        /* Tabs on Mobile */
        div[role="tablist"],
        [data-baseweb="tab-list"],
        div[data-testid="stTabs"] > div:first-child {
            gap: 4px !important;
            padding: 4px !important;
        }
        button[role="tab"],
        div[role="tab"],
        .stTabs [data-baseweb="tab"] {
            padding: 8px 4px !important;
            font-size: 0.82rem !important;
        }

        /* Buttons Uniform Scaling on Mobile */
        button[kind="secondary"],
        div[data-testid="stButton"] button[kind="secondary"],
        div[data-testid="stButton"] button {
            min-height: 52px !important;
            height: auto !important;
            font-size: 0.88rem !important;
            padding: 6px 8px !important;
        }

        button[kind="primary"],
        div[data-testid="stButton"] button[kind="primary"] {
            min-height: 50px !important;
            font-size: 1.0rem !important;
            padding: 8px 16px !important;
        }

        /* Camera input fluid width */
        [data-testid="stCameraInput"] {
            width: 100% !important;
            max-width: 100% !important;
        }

        /* Diagnostic Result Section */
        .result-capsule {
            padding: 18px 14px !important;
            border-radius: 18px !important;
        }
        .result-name {
            font-size: 1.9rem !important;
        }
        .result-meta {
            font-size: 1.05rem !important;
        }
        .hud-gauge {
            width: 82px !important;
            height: 82px !important;
            border-width: 4px !important;
        }
        .gauge-val {
            font-size: 1.25rem !important;
        }
        .gauge-lbl {
            font-size: 0.68rem !important;
        }

        /* Top-3 Ranking Hierarchy */
        .rank-capsule {
            padding: 12px 14px !important;
        }
        .rank-title {
            font-size: 1.0rem !important;
        }
        .rank-score {
            font-size: 1.05rem !important;
        }
        .rank-badge {
            width: 26px !important;
            height: 26px !important;
            font-size: 0.90rem !important;
            margin-right: 10px !important;
        }

        /* Species Roster Cards (#1 to #8) */
        .species-card-v2 {
            min-height: 250px !important;
            height: auto !important;
            padding: 14px 8px !important;
            margin-bottom: 8px !important;
        }
        .species-thumb-container {
            width: 80px !important;
            height: 80px !important;
            margin: 4px auto 8px auto !important;
        }
        .species-thumb-img {
            width: 74px !important;
            height: 74px !important;
        }
        .card-title-name {
            font-size: 0.92rem !important;
            min-height: 38px !important;
            height: auto !important;
        }
        .card-sci-name {
            font-size: 0.80rem !important;
            min-height: 18px !important;
            height: auto !important;
        }
        .card-fam-tag {
            font-size: 0.72rem !important;
            padding: 2px 8px !important;
        }

        /* XAI Callout Box */
        .xai-callout {
            padding: 14px 16px !important;
            font-size: 0.96rem !important;
        }

        /* Footer */
        .app-footer {
            padding: 2.2rem 0.8rem 1.8rem 0.8rem !important;
            margin-top: 2.5rem !important;
            font-size: 0.92rem !important;
        }
    }

    @media (max-width: 480px) {
        .metric-grid {
            grid-template-columns: repeat(2, 1fr) !important;
            gap: 8px !important;
        }
        .metric-val {
            font-size: 1.45rem !important;
        }
        .metric-lbl {
            font-size: 0.72rem !important;
        }
        .result-name {
            font-size: 1.55rem !important;
        }
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
# -----------------------------------------------------------------------------
# Top Navigation & Language Switcher
# -----------------------------------------------------------------------------
top_c1, top_c2 = st.columns([3.0, 1.0], vertical_alignment="center")
with top_c1:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 12px; padding: 2px 0;">
        <span style="font-size: 1.6rem; filter: drop-shadow(0 2px 4px rgba(0,0,0,0.08));">🦋</span>
        <span style="font-size: 1.3rem; font-weight: 900; color: #0F172A; letter-spacing: -0.02em;">AI Butterfly Vision Lab</span>
        <span style="background: linear-gradient(135deg, #ECFDF5 0%, #D1FAE5 100%); color: #059669; font-weight: 800; font-size: 0.76rem; padding: 3px 12px; border-radius: 999px; border: 1.5px solid #10B981; box-shadow: 0 2px 6px rgba(16,185,129,0.15);">● LIVE XAI v2.4</span>
    </div>
    """, unsafe_allow_html=True)
with top_c2:
    lang_options = ["🇬🇧 English", "🇧🇩 বাংলা"]
    current_idx = 0 if st.session_state.app_lang == "EN" else 1
    selected_lang_str = st.radio(
        "Language Selector",
        options=lang_options,
        index=current_idx,
        horizontal=True,
        label_visibility="collapsed",
        key="lang_switcher_radio"
    )
    target_lang = "BN" if "বাংলা" in selected_lang_str else "EN"
    if target_lang != st.session_state.app_lang:
        st.session_state.app_lang = target_lang
        st.rerun()

# -----------------------------------------------------------------------------
# 3. Hero Header (Masterpiece Split Layout & 3D Botanical Vivarium)
# -----------------------------------------------------------------------------
t_badge = t("hero_badge")
t_title = t("app_title")
t_subtitle = t("app_subtitle")
t_tag1 = t("hero_tag_species")
t_tag2 = t("hero_tag_xai")
t_tag3 = t("hero_tag_acc")
t_tag4 = t("hero_tag_cert")
t_day = "☀️ দিনের বাগান" if st.session_state.app_lang == "BN" else "☀️ Day Meadow"
t_night = "🌙 রাতের দ্যুতি" if st.session_state.app_lang == "BN" else "🌙 Night Glow"
t_terrarium_title = "🌺 মধু ও ফুলের বাগান" if st.session_state.app_lang == "BN" else "🌺 NECTAR & FLOWER SANCTUARY"
t_terrarium_live = "৫টি জীবন্ত প্রজাপতি সক্রিয়" if st.session_state.app_lang == "BN" else "5 SPECIES ACTIVE"

master_hero_html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@600;700;800;900&family=Space+Grotesk:wght@700;800&display=swap');
* {{ box-sizing: border-box; margin: 0; padding: 0; user-select: none; }}
body {{
    font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
    background: transparent;
    overflow: hidden;
    padding: 0;
    margin: 0;
}}
.hero-split-grid {{
    display: grid;
    grid-template-columns: 1.06fr 1fr;
    gap: 16px;
    align-items: stretch;
    width: 100%;
    height: 195px;
    box-sizing: border-box;
}}
@media (max-width: 860px) {{
    .hero-split-grid {{
        grid-template-columns: 1fr !important;
        height: auto !important;
        gap: 12px !important;
    }}
    .hero-left-card {{
        height: auto !important;
        min-height: 190px !important;
        padding: 16px 16px !important;
    }}
    .hero-title {{
        font-size: 1.55rem !important;
    }}
    .sanctuary-terrarium {{
        height: 200px !important;
    }}
}}

.hero-left-card {{
    background: linear-gradient(135deg, #FFFFFF 0%, #F0F9FF 60%, #E0F2FE 100%);
    border: 2px solid #94A3B8;
    border-radius: 20px;
    padding: 15px 18px;
    box-shadow: 0 10px 24px rgba(2, 132, 199, 0.08), inset 0 0 16px rgba(255, 255, 255, 0.8);
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    height: 100%;
    box-sizing: border-box;
}}
.hero-badge {{
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: #0284C7;
    color: #FFFFFF;
    font-size: 0.72rem;
    font-weight: 900;
    padding: 2px 10px;
    border-radius: 999px;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    width: fit-content;
    margin-bottom: 2px;
}}
.hero-title {{
    font-family: 'Space Grotesk', 'Plus Jakarta Sans', sans-serif;
    font-size: 1.72rem;
    font-weight: 900;
    color: #0F172A;
    line-height: 1.12;
    margin: 2px 0 3px 0;
    letter-spacing: -0.02em;
}}
.hero-subtitle {{
    font-size: 0.85rem;
    font-weight: 700;
    color: #334155;
    line-height: 1.30;
    margin-bottom: 0px;
}}
.hero-tag-strip {{
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: auto;
}}
.hero-tag {{
    display: inline-flex;
    align-items: center;
    background: #FFFFFF;
    border: 1.5px solid #94A3B8;
    color: #0F172A;
    font-size: 0.72rem;
    font-weight: 800;
    padding: 3px 9px;
    border-radius: 999px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.03);
}}

.sanctuary-terrarium {{
    background: linear-gradient(135deg, #FFFFFF 0%, #F0FDF4 40%, #E0F2FE 100%);
    border: 2px solid #94A3B8;
    border-radius: 20px;
    padding: 10px 14px;
    height: 100%;
    position: relative;
    overflow: hidden;
    box-shadow: 0 10px 24px rgba(2, 132, 199, 0.08), inset 0 0 16px rgba(255, 255, 255, 0.8);
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    cursor: crosshair;
    box-sizing: border-box;
    transition: background 0.6s ease, border-color 0.6s ease, box-shadow 0.6s ease;
}}
.sanctuary-terrarium.sanctuary-night {{
    background: linear-gradient(135deg, #050B14 0%, #0B192C 60%, #062820 100%) !important;
    border-color: #10B981 !important;
    box-shadow: 0 16px 40px rgba(16, 185, 129, 0.25), inset 0 0 30px rgba(16, 185, 129, 0.15) !important;
}}

.sanctuary-theme-toggle {{
    position: absolute;
    top: 8px;
    right: 10px;
    z-index: 50;
    background: rgba(255, 255, 255, 0.94);
    border: 1.5px solid #CBD5E1;
    color: #0F172A;
    font-size: 0.76rem;
    font-weight: 800;
    padding: 2px 9px;
    border-radius: 999px;
    cursor: pointer;
    box-shadow: 0 3px 8px rgba(0,0,0,0.06);
    transition: all 0.25s ease;
    display: inline-flex;
    align-items: center;
    gap: 5px;
}}
.sanctuary-theme-toggle:hover {{
    transform: scale(1.06);
    border-color: #0284C7;
}}
.sanctuary-night .sanctuary-theme-toggle {{
    background: rgba(15, 23, 42, 0.92) !important;
    border-color: #10B981 !important;
    color: #F8FAFC !important;
}}
.sanctuary-glow {{
    position: absolute;
    top: -30px;
    right: -30px;
    width: 160px;
    height: 160px;
    background: radial-gradient(circle, rgba(2, 132, 199, 0.25) 0%, rgba(2, 132, 199, 0) 70%);
    border-radius: 50%;
    pointer-events: none;
}}
.sanctuary-night .sanctuary-glow {{
    background: radial-gradient(circle, rgba(16, 185, 129, 0.35) 0%, rgba(16, 185, 129, 0) 70%);
}}

.firefly-layer {{
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
    opacity: 0;
    transition: opacity 0.8s ease;
    z-index: 8;
}}
.sanctuary-night .firefly-layer {{ opacity: 1; }}
.firefly-dot {{
    position: absolute;
    width: 6px;
    height: 6px;
    background: #6EE7B7;
    border-radius: 50%;
    box-shadow: 0 0 12px #34D399, 0 0 24px #10B981;
    animation: firefly-drift 5.5s ease-in-out infinite alternate;
}}
.ff1 {{ top: 20%; left: 18%; animation-delay: 0s; }}
.ff2 {{ top: 55%; left: 42%; animation-delay: 1.2s; }}
.ff3 {{ top: 28%; left: 70%; animation-delay: 2.4s; }}
.ff4 {{ top: 65%; left: 88%; animation-delay: 0.8s; }}
@keyframes firefly-drift {{
    0% {{ opacity: 0.2; transform: translate(0, 0) scale(0.7); }}
    50% {{ opacity: 1; transform: translate(28px, -28px) scale(1.35); }}
    100% {{ opacity: 0.3; transform: translate(-20px, -50px) scale(0.8); }}
}}

.butterfly-stage {{
    position: relative;
    width: 100%;
    height: 110px;
    perspective: 800px;
}}
.carrier-group {{
    position: absolute;
    top: 0;
    left: 0;
    cursor: pointer;
    transform-style: preserve-3d;
    will-change: transform;
    z-index: 20;
    transition: filter 0.3s ease;
}}

.sanctuary-night .monarch-carrier {{ filter: drop-shadow(0 0 14px #F97316); }}
.sanctuary-night .adonis-carrier {{ filter: drop-shadow(0 0 16px #38BDF8); }}
.sanctuary-night .emerald-carrier {{ filter: drop-shadow(0 0 16px #10B981); }}
.sanctuary-night .postman-carrier {{ filter: drop-shadow(0 0 14px #EF4444); }}
.sanctuary-night .dogface-carrier {{ filter: drop-shadow(0 0 14px #FBBF24); }}

.butterfly-svg-unit {{
    display: flex;
    align-items: center;
    transform-style: preserve-3d;
    filter: drop-shadow(0 4px 10px rgba(0, 0, 0, 0.15));
}}
.wing-left-svg {{
    transform-origin: 100% 50%;
    animation: wing-flap-left 0.22s ease-in-out infinite alternate;
}}
.wing-right-svg {{
    transform-origin: 0% 50%;
    animation: wing-flap-right 0.22s ease-in-out infinite alternate;
}}
@keyframes wing-flap-left {{
    0% {{ transform: rotateY(0deg); }}
    100% {{ transform: rotateY(-65deg); }}
}}
@keyframes wing-flap-right {{
    0% {{ transform: rotateY(0deg); }}
    100% {{ transform: rotateY(65deg); }}
}}

/* Lush 4-Flower Botanical Garden Layer */
.flower-habitat {{
    position: absolute;
    bottom: 20px;
    left: 0;
    width: 100%;
    height: 52px;
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    padding: 0 12px;
    z-index: 6;
    pointer-events: none;
}}
.botanical-flower {{
    width: 40px;
    height: 48px;
    animation: flower-sway 4.8s ease-in-out infinite alternate;
    transform-origin: bottom center;
    filter: drop-shadow(0 4px 10px rgba(0, 0, 0, 0.12));
    cursor: pointer;
    pointer-events: auto;
    transition: transform 0.25s ease;
}}
.botanical-flower:hover {{
    transform: scale(1.18) rotate(6deg);
}}
.fl-orchid {{ animation-delay: 0s; }}
.fl-lotus {{ animation-delay: -1.8s; }}
.fl-sunburst {{ animation-delay: -3.2s; }}
.fl-hibiscus {{ animation-delay: -0.9s; }}

@keyframes flower-sway {{
    0% {{ transform: rotate(-4.5deg) scale(0.96); }}
    100% {{ transform: rotate(4.5deg) scale(1.04); }}
}}

/* Fairy Dust Particle Engine */
.fairy-dust-spark {{
    position: absolute;
    width: 4.5px;
    height: 4.5px;
    border-radius: 50%;
    pointer-events: none;
    z-index: 15;
}}

/* Click Ripple Shockwave */
.ripple-wave {{
    position: absolute;
    border-radius: 50%;
    border: 2px solid #F59E0B;
    background: radial-gradient(circle, rgba(245, 158, 11, 0.25) 0%, rgba(245, 158, 11, 0) 75%);
    pointer-events: none;
    transform: translate(-50%, -50%) scale(0.1);
    animation: ripple-spread 0.85s cubic-bezier(0.2, 0.8, 0.2, 1) forwards;
    z-index: 25;
}}
.sanctuary-night .ripple-wave {{
    border-color: #38BDF8;
    background: radial-gradient(circle, rgba(56, 189, 248, 0.3) 0%, rgba(56, 189, 248, 0) 75%);
}}
@keyframes ripple-spread {{
    0% {{ transform: translate(-50%, -50%) scale(0.1); opacity: 1; }}
    100% {{ transform: translate(-50%, -50%) scale(2.8); opacity: 0; }}
}}

.nectar-particle {{
    position: absolute;
    width: 7px;
    height: 7px;
    background: #FBBF24;
    border-radius: 50%;
    box-shadow: 0 0 10px #F59E0B;
    pointer-events: none;
}}

#cursor-nectar {{
    position: absolute;
    width: 26px;
    height: 26px;
    border-radius: 50%;
    background: radial-gradient(circle, #FDE047 15%, #F59E0B 70%, #D97706 100%);
    box-shadow: 0 0 18px #F59E0B, 0 0 30px rgba(245, 158, 11, 0.6);
    pointer-events: none;
    opacity: 0;
    transform: translate(-50%, -50%) scale(0.6);
    transition: opacity 0.2s ease, transform 0.2s ease;
    z-index: 45;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 13px;
}}
.sanctuary-terrarium:hover #cursor-nectar {{
    opacity: 1;
    transform: translate(-50%, -50%) scale(1);
}}

.terrarium-footer {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: rgba(255, 255, 255, 0.90);
    backdrop-filter: blur(10px);
    border: 1.5px solid #CBD5E1;
    border-radius: 14px;
    padding: 7px 14px;
    z-index: 10;
    transition: all 0.6s ease;
}}
.sanctuary-night .terrarium-footer {{
    background: rgba(15, 23, 42, 0.88) !important;
    border-color: rgba(16, 185, 129, 0.4) !important;
}}
.terrarium-title {{
    font-size: 0.88rem;
    font-weight: 900;
    color: #0F172A;
    letter-spacing: 0.03em;
}}
.sanctuary-night .terrarium-title {{ color: #F8FAFC !important; }}
.terrarium-live {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 0.80rem;
    font-weight: 900;
    color: #059669;
}}
.sanctuary-night .terrarium-live {{ color: #34D399 !important; }}
.live-dot {{
    width: 8px;
    height: 8px;
    background: #10B981;
    border-radius: 50%;
    box-shadow: 0 0 8px #10B981;
    animation: pulse-dot 1.5s infinite;
}}
@keyframes pulse-dot {{
    0%, 100% {{ opacity: 1; transform: scale(1); }}
    50% {{ opacity: 0.4; transform: scale(0.8); }}
}}
</style>
</head>
<body>
<div class="hero-split-grid">
    <!-- Left Hero Card -->
    <div class="hero-left-card">
        <div>
            <div class="hero-badge">{t_badge}</div>
            <div class="hero-title">{t_title}</div>
            <div class="hero-subtitle">{t_subtitle}</div>
        </div>
        <div class="hero-tag-strip">
            <span class="hero-tag">{t_tag1}</span>
            <span class="hero-tag">{t_tag2}</span>
            <span class="hero-tag">{t_tag3}</span>
            <span class="hero-tag">{t_tag4}</span>
        </div>
    </div>

    <!-- Right 3D Botanical Vivarium -->
    <div class="sanctuary-terrarium" id="sanctuaryTerrarium">
            <button class="sanctuary-theme-toggle" id="sanctuaryThemeToggle" title="Toggle Day / Night Bioluminescence">{t_day}</button>
            <div class="sanctuary-glow"></div>
            
            <div class="firefly-layer">
                <div class="firefly-dot ff1"></div>
                <div class="firefly-dot ff2"></div>
                <div class="firefly-dot ff3"></div>
                <div class="firefly-dot ff4"></div>
            </div>
            
            <div class="butterfly-stage">
                <!-- 1. Emerald Cattleheart -->
                <div class="carrier-group emerald-carrier" title="Emerald Cattleheart (Parides sesostris)">
                    <div class="butterfly-svg-unit">
                        <svg class="wing-left-svg" viewBox="0 0 46 50" width="42" height="46">
                            <path d="M44,24 C40,8 26,2 10,8 C2,14 4,28 12,38 C20,46 30,48 44,28 Z" fill="#0F172A" stroke="#022C22" stroke-width="2" />
                            <path d="M38,18 C30,12 20,16 16,22 C18,28 28,32 36,24 Z" fill="#10B981" filter="drop-shadow(0 0 4px #34D399)" />
                            <circle cx="20" cy="38" r="3" fill="#EF4444" filter="drop-shadow(0 0 3px #F87171)" />
                            <circle cx="28" cy="42" r="2.5" fill="#DC2626" />
                        </svg>
                        <svg viewBox="0 0 14 50" width="12" height="46" style="z-index:10;">
                            <path d="M7,12 Q3,4 1,3 M7,12 Q11,4 13,3" stroke="#0F172A" stroke-width="1.4" fill="none" />
                            <circle cx="7" cy="14" r="3" fill="#0F172A" />
                            <ellipse cx="7" cy="22" rx="2.5" ry="4.5" fill="#022C22" />
                            <ellipse cx="7" cy="32" rx="2" ry="7" fill="#0F172A" />
                        </svg>
                        <svg class="wing-right-svg" viewBox="0 0 46 50" width="42" height="46">
                            <path d="M2,24 C6,8 20,2 36,8 C44,14 42,28 34,38 C26,46 16,48 2,28 Z" fill="#0F172A" stroke="#022C22" stroke-width="2" />
                            <path d="M8,18 C16,12 26,16 30,22 C28,28 18,32 10,24 Z" fill="#10B981" filter="drop-shadow(0 0 4px #34D399)" />
                            <circle cx="26" cy="38" r="3" fill="#EF4444" filter="drop-shadow(0 0 3px #F87171)" />
                            <circle cx="18" cy="42" r="2.5" fill="#DC2626" />
                        </svg>
                    </div>
                </div>

                <!-- 2. Monarch -->
                <div class="carrier-group monarch-carrier" title="Monarch (Danaus plexippus)">
                    <div class="butterfly-svg-unit">
                        <svg class="wing-left-svg" viewBox="0 0 50 55" width="46" height="50">
                            <defs>
                                <linearGradient id="monarchLeftGrad" x1="100%" y1="50%" x2="0%" y2="50%">
                                    <stop offset="0%" stop-color="#EA580C" />
                                    <stop offset="70%" stop-color="#C2410C" />
                                    <stop offset="100%" stop-color="#0F172A" />
                                </linearGradient>
                            </defs>
                            <path d="M48,25 C45,10 32,2 14,8 C4,12 2,24 10,36 C16,45 28,52 48,32 Z" fill="url(#monarchLeftGrad)" stroke="#0F172A" stroke-width="2.2" />
                            <path d="M48,25 Q32,20 16,14 M48,25 Q28,26 12,28 M48,25 Q32,36 18,44 M48,25 Q38,45 28,48" stroke="#0F172A" stroke-width="1.6" fill="none" />
                            <circle cx="8" cy="14" r="1.5" fill="#FFFFFF" />
                            <circle cx="5" cy="22" r="1.5" fill="#FFFFFF" />
                            <circle cx="8" cy="30" r="1.5" fill="#FFFFFF" />
                            <circle cx="14" cy="40" r="1.5" fill="#FFFFFF" />
                        </svg>
                        <svg viewBox="0 0 16 55" width="14" height="50" style="z-index:10;">
                            <path d="M8,12 Q4,4 1,3 M8,12 Q12,4 15,3" stroke="#0F172A" stroke-width="1.5" fill="none" />
                            <circle cx="1" cy="3" r="1.2" fill="#EA580C" />
                            <circle cx="15" cy="3" r="1.2" fill="#EA580C" />
                            <circle cx="8" cy="14" r="3.5" fill="#0F172A" />
                            <ellipse cx="8" cy="22" rx="3" ry="5" fill="#1E293B" />
                            <ellipse cx="8" cy="34" rx="2.5" ry="9" fill="#0F172A" />
                        </svg>
                        <svg class="wing-right-svg" viewBox="0 0 50 55" width="46" height="50">
                            <defs>
                                <linearGradient id="monarchRightGrad" x1="0%" y1="50%" x2="100%" y2="50%">
                                    <stop offset="0%" stop-color="#EA580C" />
                                    <stop offset="70%" stop-color="#C2410C" />
                                    <stop offset="100%" stop-color="#0F172A" />
                                </linearGradient>
                            </defs>
                            <path d="M2,25 C5,10 18,2 36,8 C46,12 48,24 40,36 C34,45 22,52 2,32 Z" fill="url(#monarchRightGrad)" stroke="#0F172A" stroke-width="2.2" />
                            <path d="M2,25 Q18,20 34,14 M2,25 Q22,26 38,28 M2,25 Q18,36 32,44 M2,25 Q12,45 22,48" stroke="#0F172A" stroke-width="1.6" fill="none" />
                            <circle cx="42" cy="14" r="1.5" fill="#FFFFFF" />
                            <circle cx="45" cy="22" r="1.5" fill="#FFFFFF" />
                            <circle cx="42" cy="30" r="1.5" fill="#FFFFFF" />
                            <circle cx="36" cy="40" r="1.5" fill="#FFFFFF" />
                        </svg>
                    </div>
                </div>

                <!-- 3. Red Postman -->
                <div class="carrier-group postman-carrier" title="Red Postman (Heliconius erato)">
                    <div class="butterfly-svg-unit">
                        <svg class="wing-left-svg" viewBox="0 0 46 50" width="40" height="44">
                            <path d="M44,24 C40,6 24,0 8,6 C0,12 2,26 10,36 C18,45 28,48 44,28 Z" fill="#0F172A" stroke="#7F1D1D" stroke-width="1.8" />
                            <path d="M38,14 C28,10 16,16 12,22 C14,26 26,24 36,18 Z" fill="#EF4444" filter="drop-shadow(0 0 3px #DC2626)" />
                        </svg>
                        <svg viewBox="0 0 14 50" width="12" height="44" style="z-index:10;">
                            <path d="M7,10 Q3,3 1,2 M7,10 Q11,3 13,2" stroke="#0F172A" stroke-width="1.4" fill="none" />
                            <circle cx="7" cy="12" r="3" fill="#0F172A" />
                            <ellipse cx="7" cy="20" rx="2.5" ry="4.5" fill="#7F1D1D" />
                            <ellipse cx="7" cy="30" rx="2" ry="7" fill="#0F172A" />
                        </svg>
                        <svg class="wing-right-svg" viewBox="0 0 46 50" width="40" height="44">
                            <path d="M2,24 C6,6 22,0 38,6 C46,12 44,26 36,36 C28,45 18,48 2,28 Z" fill="#0F172A" stroke="#7F1D1D" stroke-width="1.8" />
                            <path d="M8,14 C18,10 30,16 34,22 C32,26 20,24 10,18 Z" fill="#EF4444" filter="drop-shadow(0 0 3px #DC2626)" />
                        </svg>
                    </div>
                </div>

                <!-- 4. Adonis Blue -->
                <div class="carrier-group adonis-carrier" title="Adonis Blue (Polyommatus bellargus)">
                    <div class="butterfly-svg-unit">
                        <svg class="wing-left-svg" viewBox="0 0 42 46" width="38" height="42">
                            <defs>
                                <radialGradient id="adonisLeftGrad" cx="100%" cy="50%" r="90%">
                                    <stop offset="0%" stop-color="#38BDF8" />
                                    <stop offset="60%" stop-color="#0284C7" />
                                    <stop offset="90%" stop-color="#0369A1" />
                                    <stop offset="100%" stop-color="#0F172A" />
                                </radialGradient>
                            </defs>
                            <path d="M40,22 C36,8 24,2 10,8 C2,14 4,26 12,35 C20,42 30,44 40,26 Z" fill="url(#adonisLeftGrad)" stroke="#0F172A" stroke-width="1.8" />
                            <path d="M40,22 Q26,18 14,14 M40,22 Q24,24 10,26 M40,22 Q26,32 14,36" stroke="#0284C7" stroke-width="1.2" fill="none" opacity="0.6" />
                        </svg>
                        <svg viewBox="0 0 12 46" width="10" height="42" style="z-index:10;">
                            <path d="M6,10 Q2,3 1,2 M6,10 Q10,3 11,2" stroke="#0F172A" stroke-width="1.3" fill="none" />
                            <circle cx="6" cy="12" r="2.8" fill="#0F172A" />
                            <ellipse cx="6" cy="19" rx="2.2" ry="4" fill="#0369A1" />
                            <ellipse cx="6" cy="28" rx="1.8" ry="6" fill="#0F172A" />
                        </svg>
                        <svg class="wing-right-svg" viewBox="0 0 42 46" width="38" height="42">
                            <defs>
                                <radialGradient id="adonisRightGrad" cx="0%" cy="50%" r="90%">
                                    <stop offset="0%" stop-color="#38BDF8" />
                                    <stop offset="60%" stop-color="#0284C7" />
                                    <stop offset="90%" stop-color="#0369A1" />
                                    <stop offset="100%" stop-color="#0F172A" />
                                </radialGradient>
                            </defs>
                            <path d="M2,22 C6,8 18,2 32,8 C40,14 38,26 30,35 C22,42 12,44 2,26 Z" fill="url(#adonisRightGrad)" stroke="#0F172A" stroke-width="1.8" />
                            <path d="M2,22 Q16,18 28,14 M2,22 Q18,24 32,26 M2,22 Q16,32 28,36" stroke="#0284C7" stroke-width="1.2" fill="none" opacity="0.6" />
                        </svg>
                    </div>
                </div>

                <!-- 5. Southern Dogface -->
                <div class="carrier-group dogface-carrier" title="Southern Dogface (Zerene cesonia)">
                    <div class="butterfly-svg-unit">
                        <svg class="wing-left-svg" viewBox="0 0 42 46" width="38" height="42">
                            <defs>
                                <radialGradient id="dogfaceLeftGrad" cx="100%" cy="50%" r="90%">
                                    <stop offset="0%" stop-color="#FDE047" />
                                    <stop offset="70%" stop-color="#F59E0B" />
                                    <stop offset="95%" stop-color="#0F172A" />
                                </radialGradient>
                            </defs>
                            <path d="M40,22 C36,8 24,2 10,8 C2,14 4,26 12,35 C20,42 30,44 40,26 Z" fill="url(#dogfaceLeftGrad)" stroke="#B45309" stroke-width="1.8" />
                            <circle cx="22" cy="18" r="3.5" fill="#0F172A" />
                        </svg>
                        <svg viewBox="0 0 12 46" width="10" height="42" style="z-index:10;">
                            <path d="M6,10 Q2,3 1,2 M6,10 Q10,3 11,2" stroke="#0F172A" stroke-width="1.3" fill="none" />
                            <circle cx="6" cy="12" r="2.8" fill="#0F172A" />
                            <ellipse cx="6" cy="19" rx="2.2" ry="4" fill="#B45309" />
                            <ellipse cx="6" cy="28" rx="1.8" ry="6" fill="#0F172A" />
                        </svg>
                        <svg class="wing-right-svg" viewBox="0 0 42 46" width="38" height="42">
                            <defs>
                                <radialGradient id="dogfaceRightGrad" cx="0%" cy="50%" r="90%">
                                    <stop offset="0%" stop-color="#FDE047" />
                                    <stop offset="70%" stop-color="#F59E0B" />
                                    <stop offset="95%" stop-color="#0F172A" />
                                </radialGradient>
                            </defs>
                            <path d="M2,22 C6,8 18,2 32,8 C40,14 38,26 30,35 C22,42 12,44 2,26 Z" fill="url(#dogfaceRightGrad)" stroke="#B45309" stroke-width="1.8" />
                            <circle cx="20" cy="18" r="3.5" fill="#0F172A" />
                        </svg>
                    </div>
                </div>

                <div id="cursor-nectar">🍯</div>

                <div class="flower-habitat">
                    <!-- Flower 1: Pink Tropical Orchid -->
                    <div class="botanical-flower fl-orchid" title="Click to release Orchid Pollen!">
                        <svg viewBox="0 0 100 110" width="58" height="72">
                            <defs>
                                <radialGradient id="orchidGrad" cx="50%" cy="50%" r="50%">
                                    <stop offset="0%" stop-color="#F472B6" />
                                    <stop offset="60%" stop-color="#DB2777" />
                                    <stop offset="100%" stop-color="#831843" />
                                </radialGradient>
                                <radialGradient id="orchidGold" cx="50%" cy="50%" r="50%">
                                    <stop offset="0%" stop-color="#FEF08A" />
                                    <stop offset="80%" stop-color="#EAB308" />
                                </radialGradient>
                            </defs>
                            <path d="M50,75 Q50,95 50,110" stroke="#059669" stroke-width="4" fill="none" />
                            <path d="M50,85 Q65,80 70,88 Q58,95 50,90" fill="#10B981" />
                            <ellipse cx="50" cy="30" rx="15" ry="22" fill="url(#orchidGrad)" transform="rotate(-30 50 30)" opacity="0.95" />
                            <ellipse cx="50" cy="30" rx="15" ry="22" fill="url(#orchidGrad)" transform="rotate(30 50 30)" opacity="0.95" />
                            <ellipse cx="32" cy="55" rx="16" ry="13" fill="url(#orchidGrad)" />
                            <ellipse cx="68" cy="55" rx="16" ry="13" fill="url(#orchidGrad)" />
                            <ellipse cx="50" cy="62" rx="18" ry="15" fill="#BE185D" />
                            <circle cx="50" cy="48" r="7" fill="url(#orchidGold)" filter="drop-shadow(0 0 6px #F59E0B)" />
                        </svg>
                    </div>

                    <!-- Flower 2: Mystic Blue Lotus -->
                    <div class="botanical-flower fl-lotus" title="Click to release Lotus Pollen!">
                        <svg viewBox="0 0 100 110" width="58" height="72">
                            <defs>
                                <radialGradient id="lotusGrad" cx="50%" cy="50%" r="50%">
                                    <stop offset="0%" stop-color="#67E8F9" />
                                    <stop offset="60%" stop-color="#0284C7" />
                                    <stop offset="100%" stop-color="#0F172A" />
                                </radialGradient>
                            </defs>
                            <path d="M50,75 Q50,95 50,110" stroke="#059669" stroke-width="4" fill="none" />
                            <path d="M50,85 Q35,80 30,88 Q42,95 50,90" fill="#10B981" />
                            <ellipse cx="50" cy="32" rx="12" ry="26" fill="url(#lotusGrad)" opacity="0.95" />
                            <ellipse cx="50" cy="32" rx="12" ry="26" fill="url(#lotusGrad)" transform="rotate(-35 50 32)" opacity="0.9" />
                            <ellipse cx="50" cy="32" rx="12" ry="26" fill="url(#lotusGrad)" transform="rotate(35 50 32)" opacity="0.9" />
                            <ellipse cx="32" cy="56" rx="15" ry="12" fill="url(#lotusGrad)" />
                            <ellipse cx="68" cy="56" rx="15" ry="12" fill="url(#lotusGrad)" />
                            <circle cx="50" cy="48" r="6" fill="#38BDF8" filter="drop-shadow(0 0 8px #38BDF8)" />
                        </svg>
                    </div>

                    <!-- Flower 3: Golden Sunburst Blossom -->
                    <div class="botanical-flower fl-sunburst" title="Click to release Sunburst Pollen!">
                        <svg viewBox="0 0 100 110" width="58" height="72">
                            <defs>
                                <radialGradient id="sunPetal" cx="50%" cy="50%" r="50%">
                                    <stop offset="0%" stop-color="#FEF08A" />
                                    <stop offset="65%" stop-color="#F59E0B" />
                                    <stop offset="100%" stop-color="#B45309" />
                                </radialGradient>
                                <radialGradient id="sunCenter" cx="50%" cy="50%" r="50%">
                                    <stop offset="0%" stop-color="#78350F" />
                                    <stop offset="90%" stop-color="#451A03" />
                                </radialGradient>
                            </defs>
                            <path d="M50,75 Q50,95 50,110" stroke="#059669" stroke-width="4" fill="none" />
                            <path d="M50,85 Q65,80 70,88 Q58,95 50,90" fill="#10B981" />
                            <g transform="translate(50,42)">
                                <ellipse cx="0" cy="-22" rx="6.5" ry="13" fill="url(#sunPetal)" />
                                <ellipse cx="0" cy="-22" rx="6.5" ry="13" fill="url(#sunPetal)" transform="rotate(45)" />
                                <ellipse cx="0" cy="-22" rx="6.5" ry="13" fill="url(#sunPetal)" transform="rotate(90)" />
                                <ellipse cx="0" cy="-22" rx="6.5" ry="13" fill="url(#sunPetal)" transform="rotate(135)" />
                                <ellipse cx="0" cy="-22" rx="6.5" ry="13" fill="url(#sunPetal)" transform="rotate(180)" />
                                <ellipse cx="0" cy="-22" rx="6.5" ry="13" fill="url(#sunPetal)" transform="rotate(225)" />
                                <ellipse cx="0" cy="-22" rx="6.5" ry="13" fill="url(#sunPetal)" transform="rotate(270)" />
                                <ellipse cx="0" cy="-22" rx="6.5" ry="13" fill="url(#sunPetal)" transform="rotate(315)" />
                                <circle cx="0" cy="0" r="12" fill="url(#sunCenter)" filter="drop-shadow(0 0 6px #F59E0B)" />
                                <circle cx="0" cy="0" r="4.5" fill="#FBBF24" opacity="0.85" />
                            </g>
                        </svg>
                    </div>

                    <!-- Flower 4: Ruby Red Hibiscus -->
                    <div class="botanical-flower fl-hibiscus" title="Click to release Hibiscus Pollen!">
                        <svg viewBox="0 0 100 110" width="58" height="72">
                            <defs>
                                <radialGradient id="hibiscusGrad" cx="50%" cy="50%" r="50%">
                                    <stop offset="0%" stop-color="#F87171" />
                                    <stop offset="60%" stop-color="#DC2626" />
                                    <stop offset="100%" stop-color="#7F1D1D" />
                                </radialGradient>
                            </defs>
                            <path d="M50,75 Q50,95 50,110" stroke="#059669" stroke-width="4" fill="none" />
                            <path d="M50,85 Q35,80 30,88 Q42,95 50,90" fill="#10B981" />
                            <ellipse cx="50" cy="28" rx="14" ry="20" fill="url(#hibiscusGrad)" />
                            <ellipse cx="28" cy="42" rx="14" ry="18" fill="url(#hibiscusGrad)" transform="rotate(-30 28 42)" />
                            <ellipse cx="72" cy="42" rx="14" ry="18" fill="url(#hibiscusGrad)" transform="rotate(30 72 42)" />
                            <ellipse cx="38" cy="62" rx="13" ry="16" fill="url(#hibiscusGrad)" />
                            <ellipse cx="62" cy="62" rx="13" ry="16" fill="url(#hibiscusGrad)" />
                            <circle cx="50" cy="48" r="6.5" fill="#FDE047" filter="drop-shadow(0 0 6px #EF4444)" />
                        </svg>
                    </div>
                </div>
            </div>
            
            <div class="terrarium-footer">
                <span class="terrarium-title">{t_terrarium_title}</span>
                <span class="terrarium-live"><span class="live-dot"></span> {t_terrarium_live}</span>
            </div>
        </div>
</div>

<script>
(function() {{
    const terrarium = document.getElementById('sanctuaryTerrarium');
    const themeBtn = document.getElementById('sanctuaryThemeToggle');
    const nectar = document.getElementById('cursor-nectar');
    if (!terrarium) return;

    if (themeBtn) {{
        themeBtn.onclick = function(e) {{
            e.stopPropagation();
            const isNight = terrarium.classList.toggle('sanctuary-night');
            themeBtn.innerHTML = isNight ? '{t_night}' : '{t_day}';
        }};
    }}

    function createPollenBurst(x, y, count) {{
        // Spawn Shockwave Ripple
        const wave = document.createElement('div');
        wave.className = 'ripple-wave';
        wave.style.left = x + 'px';
        wave.style.top = y + 'px';
        terrarium.appendChild(wave);
        setTimeout(() => {{ wave.remove(); }}, 850);

        // Spawn Pollen Sparks
        for (let i = 0; i < count; i++) {{
            const spark = document.createElement('div');
            spark.className = 'nectar-particle';
            spark.style.left = x + 'px';
            spark.style.top = y + 'px';
            spark.style.background = '#FEF08A';
            spark.style.boxShadow = '0 0 14px #F59E0B';
            const tx = (Math.random() * 100 - 50);
            const ty = (Math.random() * -75 - 15);
            spark.style.transform = 'translate(' + tx + 'px, ' + ty + 'px) scale(' + (Math.random() * 1.5 + 0.8) + ')';
            spark.style.transition = 'all 1.2s cubic-bezier(0.2, 0.8, 0.2, 1)';
            spark.style.opacity = '1';
            terrarium.appendChild(spark);
            setTimeout(() => {{ spark.remove(); }}, 1200);
        }}
    }}

    function spawnFairyDust(x, y, color) {{
        const dust = document.createElement('div');
        dust.className = 'fairy-dust-spark';
        dust.style.left = x + 'px';
        dust.style.top = y + 'px';
        dust.style.background = color;
        dust.style.boxShadow = '0 0 8px ' + color + ', 0 0 14px ' + color;
        dust.style.opacity = '0.9';
        dust.style.transition = 'transform 1.1s ease-out, opacity 1.1s ease-out';
        terrarium.appendChild(dust);

        requestAnimationFrame(() => {{
            const driftX = (Math.random() * 16 - 8);
            const driftY = (Math.random() * 18 + 6);
            dust.style.transform = 'translate(' + driftX + 'px, ' + driftY + 'px) scale(0.2)';
            dust.style.opacity = '0';
        }});

        setTimeout(() => {{ dust.remove(); }}, 1100);
    }}

    const flowers = terrarium.querySelectorAll('.botanical-flower');
    flowers.forEach(fl => {{
        fl.onclick = function(e) {{
            e.stopPropagation();
            const tRect = terrarium.getBoundingClientRect();
            const flRect = fl.getBoundingClientRect();
            createPollenBurst(flRect.left - tRect.left + flRect.width / 2, flRect.top - tRect.top + flRect.height / 2, 10);
        }};
    }});

    const carriers = [
        {{ name: 'Emerald', el: terrarium.querySelector('.emerald-carrier'), homeX: 45, homeY: 35, x: 45, y: 35, bank: 0, glow: '#10B981' }},
        {{ name: 'Monarch', el: terrarium.querySelector('.monarch-carrier'), homeX: 145, homeY: 25, x: 145, y: 25, bank: 0, glow: '#F97316' }},
        {{ name: 'Postman', el: terrarium.querySelector('.postman-carrier'), homeX: 245, homeY: 48, x: 245, y: 48, bank: 0, glow: '#EF4444' }},
        {{ name: 'Adonis', el: terrarium.querySelector('.adonis-carrier'), homeX: 345, homeY: 28, x: 345, y: 28, bank: 0, glow: '#38BDF8' }},
        {{ name: 'Dogface', el: terrarium.querySelector('.dogface-carrier'), homeX: 435, homeY: 32, x: 435, y: 32, bank: 0, glow: '#FBBF24' }}
    ].filter(b => b.el !== null);

    let mouseX = -999;
    let mouseY = -999;
    let isHovering = false;
    let startTime = performance.now();
    let frameCount = 0;

    terrarium.onmousemove = function(e) {{
        const rect = terrarium.getBoundingClientRect();
        mouseX = e.clientX - rect.left;
        mouseY = e.clientY - rect.top;
        isHovering = true;

        if (nectar) {{
            nectar.style.left = mouseX + 'px';
            nectar.style.top = mouseY + 'px';
            nectar.style.opacity = '1';
        }}
    }};

    terrarium.onmouseenter = function(e) {{
        const rect = terrarium.getBoundingClientRect();
        mouseX = e.clientX - rect.left;
        mouseY = e.clientY - rect.top;
        isHovering = true;
    }};

    terrarium.onmouseleave = function() {{
        isHovering = false;
        mouseX = -999;
        mouseY = -999;
        if (nectar) nectar.style.opacity = '0';
    }};

    terrarium.onclick = function(e) {{
        const rect = terrarium.getBoundingClientRect();
        createPollenBurst(e.clientX - rect.left, e.clientY - rect.top, 10);
    }};

    function animateBalancedSwarm(now) {{
        const time = (now - startTime) * 0.0022;
        frameCount++;

        // Determine ONLY the single closest curious butterfly
        let primaryFollower = -1;

        if (isHovering && mouseX > 0 && mouseY > 0) {{
            let minDist = 99999;
            carriers.forEach((b, idx) => {{
                const d = Math.hypot(mouseX - b.x, mouseY - b.y);
                if (d < minDist) {{
                    minDist = d;
                    primaryFollower = idx;
                }}
            }});
        }}

        // 1. Calculate Organic Positions & Motion
        carriers.forEach((b, i) => {{
            const hoverX = Math.sin(time * 1.6 + i * 1.4) * 14;
            const hoverY = Math.cos(time * 2.0 + i * 1.2) * 8;

            let targetX = b.homeX + hoverX;
            let targetY = b.homeY + hoverY;
            let moveSpeed = 0.035;

            if (i === primaryFollower) {{
                // Gentle orbiting flight next to the honey pot
                targetX = mouseX - 22 + Math.cos(time * 2.6) * 26;
                targetY = mouseY - 32 + Math.sin(time * 2.6) * 16;
                moveSpeed = 0.075;
            }}

            const prevX = b.x;
            const prevY = b.y;
            b.x += (targetX - b.x) * moveSpeed;
            b.y += (targetY - b.y) * moveSpeed;

            // Aerodynamic Level Banking
            const vx = b.x - prevX;
            const targetBank = Math.max(-16, Math.min(16, vx * 4.8));
            b.bank += (targetBank - b.bank) * 0.12;

            // Spawn Fairy Dust flight trails
            if (frameCount % 6 === 0 && (Math.abs(vx) > 0.3 || i === primaryFollower)) {{
                const isNight = terrarium.classList.contains('sanctuary-night');
                const particleColor = isNight ? b.glow : '#FDE047';
                spawnFairyDust(b.x + 22, b.y + 24, particleColor);
            }}
        }});

        // 2. Separation Physics (Guarantees zero stacking / zero clumping)
        for (let i = 0; i < carriers.length; i++) {{
            for (let j = i + 1; j < carriers.length; j++) {{
                const b1 = carriers[i];
                const b2 = carriers[j];
                const diffX = b1.x - b2.x;
                const diffY = b1.y - b2.y;
                const dist = Math.hypot(diffX, diffY);
                const minDist = 72;

                if (dist < minDist && dist > 0) {{
                    const overlap = (minDist - dist) * 0.09;
                    const nx = diffX / dist;
                    const ny = diffY / dist;
                    b1.x += nx * overlap;
                    b1.y += ny * overlap;
                    b2.x -= nx * overlap;
                    b2.y -= ny * overlap;
                }}
            }}
        }}

        // 3. Render 3D Positions
        carriers.forEach(b => {{
            b.el.style.transform = 'translate3d(' + b.x.toFixed(1) + 'px, ' + b.y.toFixed(1) + 'px, 20px) rotate(' + b.bank.toFixed(1) + 'deg)';
        }});

        requestAnimationFrame(animateBalancedSwarm);
    }}

    requestAnimationFrame(animateBalancedSwarm);

    function sendFrameHeight() {{
        try {{
            const height = document.body.scrollHeight || document.documentElement.scrollHeight;
            if (window.parent) {{
                window.parent.postMessage({{type: 'streamlit:setFrameHeight', height: height + 10}}, '*');
            }}
        }} catch(e) {{}}
    }}
    window.addEventListener('load', sendFrameHeight);
    window.addEventListener('resize', sendFrameHeight);
    setTimeout(sendFrameHeight, 300);
    setTimeout(sendFrameHeight, 1000);
}})();
</script>
</body>
</html>
"""

components.html(master_hero_html, height=202, scrolling=False)

# -----------------------------------------------------------------------------
# 4. Floating Performance Metric Strip
# -----------------------------------------------------------------------------
st.markdown(f"""
<div class="metric-grid">
    <div class="metric-card">
        <div class="metric-val">{t('metric_acc_val')}</div>
        <div class="metric-lbl">{t('metric_acc_lbl')}</div>
    </div>
    <div class="metric-card">
        <div class="metric-val">{t('metric_f1_val')}</div>
        <div class="metric-lbl">{t('metric_f1_lbl')}</div>
    </div>
    <div class="metric-card">
        <div class="metric-val">{t('metric_species_val')}</div>
        <div class="metric-lbl">{t('metric_species_lbl')}</div>
    </div>
    <div class="metric-card">
        <div class="metric-val">{t('metric_xai_val')}</div>
        <div class="metric-lbl">{t('metric_xai_lbl')}</div>
    </div>
</div>
""", unsafe_allow_html=True)

if predictor is None:
    st.error("⚠️ Failed to load AI model. Please verify `models/butterfly_resnet18_best.pth` exists.")
    st.stop()

# -----------------------------------------------------------------------------
# 5. Triple Input Studio: 1-Click Gallery | Upload Photo | Live Camera Capture
# -----------------------------------------------------------------------------
st.markdown(f"## {t('studio_title')}")

if "selected_image" not in st.session_state:
    st.session_state.selected_image = None
if "selected_filename" not in st.session_state:
    st.session_state.selected_filename = None
if "active_source_id" not in st.session_state:
    st.session_state.active_source_id = None
if "analysis_cache" not in st.session_state:
    st.session_state.analysis_cache = None
if "show_report_modal" not in st.session_state:
    st.session_state.show_report_modal = False

test_dir = paths["test_data_dir"]
benchmark_samples = {}
species_b64_thumbnails = {}

if os.path.exists(test_dir):
    for cls in predictor.class_names:
        cls_dir = os.path.join(test_dir, cls)
        if os.path.isdir(cls_dir):
            files = sorted(os.listdir(cls_dir))
            if files:
                fpath = os.path.join(cls_dir, files[0])
                benchmark_samples[cls] = fpath
                try:
                    t_img = Image.open(fpath).convert("RGB").resize((140, 140), Image.Resampling.LANCZOS)
                    t_buf = BytesIO()
                    t_img.save(t_buf, format="JPEG", quality=92)
                    species_b64_thumbnails[cls] = base64.b64encode(t_buf.getvalue()).decode()
                except Exception:
                    species_b64_thumbnails[cls] = ""

input_tab1, input_tab2, input_tab3 = st.tabs([
    t("tab_gallery"),
    t("tab_upload"),
    t("tab_camera")
])

with input_tab1:
    st.markdown(f"<p style='font-size: 1.15rem; font-weight: 800; color: #0F172A; margin-bottom: 12px;'>{t('gallery_prompt')}</p>", unsafe_allow_html=True)
    sample_items = list(benchmark_samples.items())
    
    # Row 1: Items 1 to 4
    g_row1 = st.columns(4)
    for idx in range(min(4, len(sample_items))):
        cls_name, fpath = sample_items[idx]
        with g_row1[idx]:
            disp_label = SPECIES_DOSSIER.get(cls_name, {}).get(f"name_{st.session_state.app_lang.lower()}", cls_name)
            if st.button(f"🦋 {disp_label}", key=f"btn_sample_{idx}", use_container_width=True):
                src_id = f"gallery_{cls_name}_{os.path.basename(fpath)}"
                if st.session_state.get("active_source_id") != src_id:
                    st.session_state.selected_image = Image.open(fpath).convert("RGB")
                    st.session_state.selected_filename = f"{cls_name} ({os.path.basename(fpath)})"
                    st.session_state.active_source_id = src_id
                    st.session_state.analysis_cache = None
                    st.session_state.show_report_modal = False

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
    
    # Row 2: Items 5 to 8
    g_row2 = st.columns(4)
    for idx in range(4, min(8, len(sample_items))):
        cls_name, fpath = sample_items[idx]
        with g_row2[idx - 4]:
            disp_label = SPECIES_DOSSIER.get(cls_name, {}).get(f"name_{st.session_state.app_lang.lower()}", cls_name)
            if st.button(f"🦋 {disp_label}", key=f"btn_sample_{idx}", use_container_width=True):
                src_id = f"gallery_{cls_name}_{os.path.basename(fpath)}"
                if st.session_state.get("active_source_id") != src_id:
                    st.session_state.selected_image = Image.open(fpath).convert("RGB")
                    st.session_state.selected_filename = f"{cls_name} ({os.path.basename(fpath)})"
                    st.session_state.active_source_id = src_id
                    st.session_state.analysis_cache = None
                    st.session_state.show_report_modal = False

with input_tab2:
    st.markdown(f"""
    <div style="text-align: center; padding: 14px 10px 10px 10px;">
        <div style="font-size: 1.25rem; font-weight: 900; color: #0F172A; margin-bottom: 4px;">
            {t('upload_title')}
        </div>
        <div style="font-size: 1.0rem; font-weight: 700; color: #0284C7;">
            {t('upload_sub')}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Upload Image File",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
        help=t("upload_help")
    )
    if uploaded_file is not None:
        src_id = f"upload_{uploaded_file.name}_{uploaded_file.size}"
        if st.session_state.get("active_source_id") != src_id:
            try:
                st.session_state.selected_image = Image.open(uploaded_file).convert("RGB")
                st.session_state.selected_filename = uploaded_file.name
                st.session_state.active_source_id = src_id
                st.session_state.analysis_cache = None
                st.session_state.show_report_modal = False
            except Exception as e:
                st.error(f"Error reading uploaded image: {e}")

with input_tab3:
    st.markdown(f"<p style='font-size: 1.15rem; font-weight: 800; color: #0F172A; text-align: center; margin-bottom: 14px;'>{t('camera_prompt')}</p>", unsafe_allow_html=True)
    
    if "camera_active" not in st.session_state:
        st.session_state.camera_active = False

    cam_col1, cam_col2, cam_col3 = st.columns([1, 2, 1])
    with cam_col2:
        if not st.session_state.camera_active:
            st.markdown(f"""
            <div style="text-align: center; padding: 28px 20px; background: #F8FAFC; border: 2.5px dashed #94A3B8; border-radius: 24px; margin-bottom: 14px; box-shadow: 0 4px 16px rgba(15, 23, 42, 0.04);">
                <div style="font-size: 2.6rem; margin-bottom: 10px;">📸</div>
                <div style="font-size: 1.18rem; font-weight: 900; color: #0F172A; margin-bottom: 6px;">
                    {t('cam_ready_title')}
                </div>
                <div style="font-size: 0.90rem; color: #64748B; font-weight: 700; line-height: 1.4; margin-bottom: 18px;">
                    {t('cam_ready_sub')}
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button(t("btn_activate_cam"), key="btn_activate_cam", type="primary", use_container_width=True):
                st.session_state.camera_active = True
                st.rerun()
        else:
            top_cam_c1, top_cam_c2 = st.columns([2.5, 1.5], vertical_alignment="center")
            with top_cam_c1:
                st.markdown(f"<div style='font-weight: 800; font-size: 0.95rem; color: #059669;'>● {t('cam_ready_title')}</div>", unsafe_allow_html=True)
            with top_cam_c2:
                if st.button(t("btn_deactivate_cam"), key="btn_stop_cam", use_container_width=True):
                    st.session_state.camera_active = False
                    st.rerun()

            camera_photo = st.camera_input(t("camera_label"), label_visibility="collapsed")
            if camera_photo is not None:
                photo_bytes = camera_photo.getvalue()
                src_id = f"cam_{hashlib.md5(photo_bytes).hexdigest()[:12]}"
                if st.session_state.get("active_source_id") != src_id:
                    try:
                        st.session_state.selected_image = Image.open(BytesIO(photo_bytes)).convert("RGB")
                        st.session_state.selected_filename = "Live_Camera_Capture.jpg"
                        st.session_state.active_source_id = src_id
                        st.session_state.analysis_cache = None
                        st.session_state.show_report_modal = False
                    except Exception as e:
                        st.error(f"Error capturing camera snapshot: {e}")

# -----------------------------------------------------------------------------
# 6. Analysis & Visual Diagnostic Suite (With On-Demand Report Inspector)
# -----------------------------------------------------------------------------
active_image = st.session_state.selected_image
active_filename = st.session_state.selected_filename

if active_image is not None:
    img_w, img_h = active_image.size
    aspect_val = img_w / max(img_h, 1)
    if aspect_val >= 1.25:
        ratio_badge = "Landscape"
    elif aspect_val <= 0.8:
        ratio_badge = "Portrait"
    else:
        ratio_badge = "Square"

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    
    cmd_col1, cmd_col2 = st.columns([1.15, 1.0], gap="large")
    with cmd_col1:
        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; align-items: center; background: #FFFFFF; border: 2px solid #CBD5E1; border-radius: 14px; padding: 10px 16px; margin-bottom: 8px;">
            <div style="font-size: 0.92rem; font-weight: 900; color: #0284C7; text-transform: uppercase; letter-spacing: 0.05em;">
                {t('active_specimen')}
            </div>
            <div style="display: flex; gap: 8px;">
                <span style="background: #E0F2FE; color: #0369A1; font-weight: 800; font-size: 0.80rem; padding: 3px 9px; border-radius: 999px; border: 1px solid #BAE6FD;">
                    📐 {img_w} × {img_h} px
                </span>
                <span style="background: #F1F5F9; color: #475569; font-weight: 800; font-size: 0.80rem; padding: 3px 9px; border-radius: 999px; border: 1px solid #CBD5E1;">
                    {ratio_badge}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.image(active_image, caption=f"{active_filename} ({img_w}×{img_h} px)", use_container_width=True)
    
    with cmd_col2:
        st.markdown(f"""
        <div style="background: #FFFFFF; border: 2px solid #CBD5E1; border-radius: 20px; padding: 22px; box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08); height: 100%; display: flex; flex-direction: column; justify-content: space-between;">
            <div>
                <div style="display: inline-flex; align-items: center; gap: 6px; background: #ECFDF5; color: #059669; font-weight: 900; font-size: 0.82rem; padding: 4px 12px; border-radius: 999px; border: 1px solid #A7F3D0; margin-bottom: 10px;">
                    {t('ready_badge')}
                </div>
                <div style="font-size: 1.45rem; font-weight: 900; color: #0F172A; margin-bottom: 8px;">
                    {t('deep_feat_title')}
                </div>
                <div style="font-size: 1.05rem; font-weight: 600; color: #475569; line-height: 1.55; margin-bottom: 20px;">
                    {t('deep_feat_desc')}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        btn_c1, btn_c2 = st.columns([2.0, 1.2])
        with btn_c1:
            run_analysis = st.button(t("btn_run_analysis"), type="primary", use_container_width=True)
        with btn_c2:
            if st.button(t("btn_reset"), use_container_width=True, help=t("btn_reset_help")):
                st.session_state.selected_image = None
                st.session_state.selected_filename = None
                st.session_state.active_source_id = None
                st.session_state.analysis_cache = None
                st.session_state.show_report_modal = False
                st.rerun()

    if run_analysis:
        with st.spinner(t("spinner_msg")):
            try:
                pred_res = predictor.predict(active_image)
                with GradCAM(predictor.model) as gradcam_engine:
                    raw_heatmap, _, _ = gradcam_engine.generate(
                        pred_res['input_tensor'],
                        pred_res['predicted_idx']
                    )
                st.session_state.analysis_cache = {
                    "pred_class": pred_res['predicted_class'],
                    "confidence": pred_res['confidence'],
                    "top_k": pred_res['top_k'],
                    "display_img": pred_res['display_image'],
                    "raw_heatmap": raw_heatmap,
                }
                st.session_state.show_report_modal = False
            except Exception as e:
                st.error(f"Error during AI analysis: {e}")

    # RENDER PERSISTENT ANALYSIS
    if st.session_state.analysis_cache is not None:
        cached = st.session_state.analysis_cache
        pred_class = cached["pred_class"]
        confidence = cached["confidence"]
        top_k = cached["top_k"]
        display_img = cached["display_img"]
        raw_heatmap = cached["raw_heatmap"]

        st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)
        col_left, col_right = st.columns([1.15, 1.35], gap="large")

        with col_left:
            st.markdown(f"## {t('result_heading')}")
            
            dossier_data = SPECIES_DOSSIER.get(pred_class, {})
            lang_key = st.session_state.app_lang.lower()
            disp_species_name = dossier_data.get(f"name_{lang_key}", pred_class)
            disp_sci_name = dossier_data.get("scientific_name", "Unknown")
            disp_family_name = dossier_data.get(f"family_{lang_key}", dossier_data.get("family_en", "Lepidoptera"))
            color_primary = dossier_data.get("color_primary", "#0284C7")
            insight_text = dossier_data.get(f"ai_attention_{lang_key}", dossier_data.get("ai_attention_en", "Model neural attention is focused on visual wing patterns."))

            gauge_color = "#10B981" if confidence >= 80.0 else "#F59E0B"
            gauge_lbl = ("উচ্চ" if confidence >= 80.0 else "মাঝারি") if st.session_state.app_lang == "BN" else ("HIGH" if confidence >= 80.0 else "MODERATE")

            st.markdown(f"""
            <div class="result-capsule" style="border-top: 7px solid {color_primary};">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                    <div>
                        <div class="result-tag">{t('identified_species')}</div>
                        <div class="result-name">{disp_species_name}</div>
                        <div class="result-meta">{disp_sci_name} • {disp_family_name}</div>
                    </div>
                    <div class="hud-gauge" style="border-color: {gauge_color};">
                        <div class="gauge-val">{confidence:.1f}%</div>
                        <div class="gauge-lbl" style="color: {gauge_color};">{gauge_lbl}</div>
                    </div>
                </div>
                <div style="background: #CBD5E1; height: 12px; border-radius: 999px; overflow: hidden; margin-top: 16px;">
                    <div style="width: {min(confidence, 100.0):.1f}%; height: 100%; border-radius: 999px; background: linear-gradient(90deg, #0284C7, {color_primary});"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"### {t('top3_title')}")
            for rank, (cls_name, prob) in enumerate(top_k, 1):
                cls_disp = SPECIES_DOSSIER.get(cls_name, {}).get(f"name_{lang_key}", cls_name)
                st.markdown(f"""
                <div class="rank-capsule">
                    <div style="display: flex; align-items: center;">
                        <span class="rank-badge">{rank}</span>
                        <span class="rank-title">{cls_disp}</span>
                    </div>
                    <span class="rank-score">{prob:.2f}%</span>
                </div>
                """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="xai-callout">
                <div class="xai-hdr">{t('xai_box_title')}</div>
                {insight_text}
            </div>
            """, unsafe_allow_html=True)

        with col_right:
            st.markdown(f"## {t('xai_heading')}")
            
            ctrl_c1, ctrl_c2, ctrl_c3 = st.columns([1.1, 1.1, 1.0])
            with ctrl_c1:
                blend_alpha = st.slider(
                    t("slider_label"),
                    min_value=0.0,
                    max_value=1.0,
                    value=0.55,
                    step=0.05,
                    key="gradcam_alpha_slider",
                    help=t("slider_help")
                )
            with ctrl_c2:
                focus_threshold = st.slider(
                    "Hotspot Focus Filter" if st.session_state.app_lang == "EN" else "🎯 ফোকাস ফিল্টার",
                    min_value=0.0,
                    max_value=0.85,
                    value=0.0,
                    step=0.05,
                    key="gradcam_focus_slider"
                )
            with ctrl_c3:
                cmap_choice = st.selectbox(
                    "Thermal Colormap" if st.session_state.app_lang == "EN" else "🎨 রঙের প্যালেট (Colormap)",
                    options=[
                        "🔥 Jet (Classic)",
                        "⚡ Turbo (Dynamic)",
                        "🌋 Inferno (Contrast)",
                        "🌿 Viridis (Scientific)",
                        "🔮 Plasma (Neon)",
                        "🌌 Magma (Obsidian)"
                    ],
                    index=0,
                    key="gradcam_cmap_select"
                )

            cmap_dict = {
                "🔥 Jet (Classic)": cv2.COLORMAP_JET,
                "⚡ Turbo (Dynamic)": cv2.COLORMAP_TURBO,
                "🌋 Inferno (Contrast)": cv2.COLORMAP_INFERNO,
                "🌿 Viridis (Scientific)": cv2.COLORMAP_VIRIDIS,
                "🔮 Plasma (Neon)": cv2.COLORMAP_PLASMA,
                "🌌 Magma (Obsidian)": cv2.COLORMAP_MAGMA,
            }
            selected_colormap_code = cmap_dict.get(cmap_choice, cv2.COLORMAP_JET)

            if focus_threshold > 0.0:
                mod_map = np.where(raw_heatmap >= focus_threshold, (raw_heatmap - focus_threshold) / (1.0 - focus_threshold + 1e-8), 0.0)
            else:
                mod_map = raw_heatmap.copy()

            gamma_exponent = 1.0 + (0.55 - blend_alpha) * 0.8
            mod_map = np.clip(np.power(mod_map, max(gamma_exponent, 0.2)), 0.0, 1.0)

            with GradCAM(predictor.model) as gradcam_engine:
                heatmap_img, overlay_img = gradcam_engine.overlay_heatmap(
                    mod_map,
                    display_img,
                    alpha=blend_alpha,
                    colormap=selected_colormap_code
                )

            cam_tab1, cam_tab2, cam_tab3, cam_tab4 = st.tabs([
                t("ch_overlay"),
                t("ch_cam"),
                t("ch_original"),
                t("ch_compare")
            ])

            with cam_tab1:
                st.image(overlay_img, caption=f"Grad-CAM Diagnostic Overlay (α = {blend_alpha:.2f})", use_container_width=True)
            with cam_tab2:
                st.image(heatmap_img, caption=f"Thermal Heatmap Activation Map (α = {blend_alpha:.2f})", use_container_width=True)
            with cam_tab3:
                st.image(display_img, caption=f"Input Specimen ({display_img.width} × {display_img.height} px)", use_container_width=True)
            with cam_tab4:
                comp_c1, comp_c2 = st.columns(2, gap="medium")
                orig_caption = "📸 " + ("মূল ছবি" if st.session_state.app_lang == "BN" else "Original Specimen")
                overlay_caption = "✨ " + ("এআই ওভারলে" if st.session_state.app_lang == "BN" else "AI Overlay") + f" (α = {blend_alpha:.2f})"
                with comp_c1:
                    st.image(display_img, caption=orig_caption, use_container_width=True)
                with comp_c2:
                    st.image(overlay_img, caption=overlay_caption, use_container_width=True)

            st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
            if st.button(t("btn_inspect_report"), type="primary", use_container_width=True):
                st.session_state.show_report_modal = True

        if st.session_state.get("show_report_modal", False):
            st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)
            
            report_meta = SPECIES_METADATA.get(pred_class, {
                "scientific_name": "Unknown species",
                "family": "Insecta • Lepidoptera",
                "appearance": "Distinct biological visual wing markings.",
                "distribution": "Global biodiversity habitat.",
                "key_features": "Diagnostic taxonomic wing venation pattern.",
                "xai_insight": "Model neural attention concentrated on discriminative visual wing patterns."
            })
            
            cert_hash = hashlib.md5(f"{pred_class}_{confidence}".encode()).hexdigest()[:8].upper()
            
            report_bytes = generate_report_card(
                original_image=display_img,
                overlay_image=overlay_img,
                pred_class=pred_class,
                confidence=confidence,
                top_k=top_k
            )

            # Crystal Clear Interactive High-Contrast Native Web Report Dossier
            report_html = f"""<div style="background: #FFFFFF; border: 2.5px solid #0284C7; border-radius: 24px; padding: 28px; box-shadow: 0 16px 40px rgba(2, 132, 199, 0.16); margin-bottom: 20px;">
<div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #E2E8F0; padding-bottom: 18px; margin-bottom: 22px; flex-wrap: wrap; gap: 12px;">
<div>
<div style="display: inline-flex; align-items: center; gap: 6px; background: #E0F2FE; color: #0284C7; font-size: 0.82rem; font-weight: 900; padding: 4px 12px; border-radius: 999px; text-transform: uppercase; margin-bottom: 8px;">
AI BUTTERFLY VISION • BIO-INTELLIGENCE LAB
</div>
<div style="font-size: 1.85rem; font-weight: 900; color: #0F172A; line-height: 1.2;">
{t('cert_preview_title')}
</div>
<div style="font-size: 0.90rem; font-weight: 700; color: #64748B; margin-top: 4px;">
CERTIFICATE-ID: BIO-2026-XAI-{cert_hash} • ResNet-18 + Grad-CAM XAI
</div>
</div>
<div style="background: #ECFDF5; border: 2px solid #10B981; border-radius: 16px; padding: 10px 18px; text-align: right;">
<div style="font-size: 0.78rem; font-weight: 900; color: #059669; text-transform: uppercase;">VERIFIED AI DECISION</div>
<div style="font-size: 1.55rem; font-weight: 900; color: #065F46;">{confidence:.1f}% MATCH</div>
</div>
</div>
<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 18px; margin-top: 20px;">
<div style="background: #F8FAFC; border: 2px solid #0284C7; border-radius: 18px; padding: 20px;">
<div style="font-size: 1.05rem; font-weight: 900; color: #0284C7; margin-bottom: 12px; display: flex; align-items: center; gap: 6px;">
🔬 NEURO-VISUAL ATTENTION (GRAD-CAM)
</div>
<div style="font-size: 0.96rem; color: #0F172A; font-weight: 800; margin-bottom: 8px;">
Gradient Hotspot Diagnostic:
</div>
<div style="font-size: 0.92rem; color: #334155; line-height: 1.55; margin-bottom: 14px; font-weight: 600;">
{report_meta['xai_insight']}
</div>
<div style="font-size: 0.96rem; color: #0F172A; font-weight: 800; margin-bottom: 6px;">
Diagnostic Wing Markers:
</div>
<div style="font-size: 0.92rem; color: #334155; line-height: 1.55; font-weight: 600;">
{report_meta['appearance']}
</div>
</div>
<div style="background: #F0FDF4; border: 2px solid #10B981; border-radius: 18px; padding: 20px;">
<div style="font-size: 1.05rem; font-weight: 900; color: #065F46; margin-bottom: 12px; display: flex; align-items: center; gap: 6px;">
🌿 TAXONOMIC & BIOGEOGRAPHIC PROFILE
</div>
<div style="font-size: 0.96rem; color: #065F46; font-weight: 800; margin-bottom: 8px;">
Geographic Distribution:
</div>
<div style="font-size: 0.92rem; color: #1E293B; line-height: 1.55; margin-bottom: 14px; font-weight: 600;">
{report_meta['distribution']}
</div>
<div style="font-size: 0.96rem; color: #065F46; font-weight: 800; margin-bottom: 6px;">
Key Biological Adaptation:
</div>
<div style="font-size: 0.92rem; color: #1E293B; line-height: 1.55; font-weight: 600;">
{report_meta['key_features']}
</div>
</div>
</div>
<div style="display: flex; justify-content: space-between; align-items: center; border-top: 1.5px solid #E2E8F0; padding-top: 16px; margin-top: 20px; flex-wrap: wrap; gap: 8px;">
<div style="font-size: 0.86rem; font-weight: 700; color: #64748B;">
AI Butterfly Vision • PyTorch ResNet-18 Deep Transfer Learning Architecture
</div>
<div style="font-size: 0.92rem; font-weight: 900; color: #0284C7;">
Lead AI Architect & Engineer: Ohi
</div>
</div>
</div>"""
            st.markdown(report_html, unsafe_allow_html=True)

            action_c1, action_c2 = st.columns([1.5, 1])
            with action_c1:
                st.download_button(
                    label="📥 " + ("অফিশিয়াল ৪K সার্টিফিকেট ডাউনলোড করুন (High-Res PNG)" if st.session_state.app_lang == "BN" else "Download Official 4K Certificate (High-Res PNG)"),
                    data=report_bytes,
                    file_name=f"Butterfly_AI_Report_{pred_class.replace(' ', '_')}.png",
                    mime="image/png",
                    use_container_width=True
                )
            with action_c2:
                if st.button("❌ " + ("রিপোর্ট ভিউ বন্ধ করুন" if st.session_state.app_lang == "BN" else "Close Report View"), use_container_width=True):
                    st.session_state.show_report_modal = False
                    st.rerun()

    else:
        st.info(t("msg_run_prompt"))

else:
    st.info(t("msg_choose_specimen"))

# -----------------------------------------------------------------------------
# 7. Species Bio-Dossier Modal Dialog
# -----------------------------------------------------------------------------
@st.dialog("🦋 " + ("প্রজাতি ডসিয়ার প্রোফাইল" if st.session_state.get("app_lang", "EN") == "BN" else "Species Bio-Dossier Profile"), width="large")
def show_species_dossier_modal(cls_name):
    lang = st.session_state.get("app_lang", "EN")
    dossier = SPECIES_DOSSIER.get(cls_name, {})
    disp_name = dossier.get(f"name_{lang.lower()}", cls_name)
    sci_name = dossier.get("scientific_name", "Unknown")
    fam_name = dossier.get(f"family_{lang.lower()}", dossier.get("family_en", "Lepidoptera"))
    color_p = dossier.get("color_primary", "#0284C7")
    b64_img = species_b64_thumbnails.get(cls_name, "")
    
    habitat_val = dossier.get(f'habitat_{lang.lower()}', '')
    appearance_val = dossier.get(f'appearance_{lang.lower()}', '')
    superpower_val = dossier.get(f'superpower_{lang.lower()}', '')
    ai_attention_val = dossier.get(f'ai_attention_{lang.lower()}', '')
    wingspan_val = dossier.get('wingspan', 'N/A')
    lifespan_val = dossier.get(f'lifespan_{lang.lower()}', 'N/A')
    precision_val = dossier.get('model_precision', '95%')

    dossier_html = f"""<div style="background:#FFFFFF;border-top:5px solid {color_p};border-radius:16px;padding:14px 16px;box-shadow:0 8px 24px rgba(15,23,42,0.06);margin-bottom:12px;">
<div style="display:flex;gap:14px;align-items:center;margin-bottom:10px;">
<div style="width:68px;height:68px;border-radius:50%;overflow:hidden;border:3px solid {color_p};flex-shrink:0;box-shadow:0 4px 12px rgba(0,0,0,0.10);">
<img src="data:image/jpeg;base64,{b64_img}" style="width:100%;height:100%;object-fit:cover;" alt="{disp_name}" />
</div>
<div style="flex:1;min-width:200px;">
<div style="font-size:0.72rem;font-weight:900;color:{color_p};text-transform:uppercase;letter-spacing:0.08em;margin-bottom:1px;">
{t("dossier_modal_tag")}
</div>
<div style="font-size:1.40rem;font-weight:900;color:#0F172A;line-height:1.15;margin-bottom:2px;">
{disp_name}
</div>
<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
<span style="font-size:0.92rem;font-style:italic;color:#0284C7;font-weight:700;">{sci_name}</span>
<span style="background:#F1F5F9;color:#334155;font-size:0.74rem;font-weight:800;padding:2px 7px;border-radius:6px;border:1px solid #CBD5E1;">🌿 {fam_name}</span>
</div>
</div>
</div>

<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;padding:8px 0;border-top:1.5px dashed #E2E8F0;border-bottom:1.5px dashed #E2E8F0;margin-bottom:10px;">
<div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;padding:5px 6px;text-align:center;">
<div style="font-size:0.66rem;font-weight:800;color:#64748B;text-transform:uppercase;">{t("dossier_stat_wingspan")}</div>
<div style="font-size:0.86rem;font-weight:900;color:#0F172A;margin-top:1px;">{wingspan_val}</div>
</div>
<div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;padding:5px 6px;text-align:center;">
<div style="font-size:0.66rem;font-weight:800;color:#64748B;text-transform:uppercase;">{t("dossier_stat_lifespan")}</div>
<div style="font-size:0.86rem;font-weight:900;color:#0F172A;margin-top:1px;">{lifespan_val}</div>
</div>
<div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;padding:5px 6px;text-align:center;">
<div style="font-size:0.66rem;font-weight:800;color:#64748B;text-transform:uppercase;">{t("dossier_stat_precision")}</div>
<div style="font-size:0.86rem;font-weight:900;color:#059669;margin-top:1px;">⚡ {precision_val}</div>
</div>
</div>

<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:8px;">
<div style="background:#F8FAFC;border:1.5px solid #CBD5E1;border-radius:10px;padding:8px 10px;">
<div style="font-size:0.78rem;font-weight:900;color:#0F172A;margin-bottom:2px;display:flex;align-items:center;gap:4px;">
{t('dossier_habitat')}
</div>
<div style="font-size:0.78rem;line-height:1.40;color:#334155;font-weight:600;">
{habitat_val}
</div>
</div>

<div style="background:#F8FAFC;border:1.5px solid #CBD5E1;border-radius:10px;padding:8px 10px;">
<div style="font-size:0.78rem;font-weight:900;color:#0F172A;margin-bottom:2px;display:flex;align-items:center;gap:4px;">
{t('dossier_appearance')}
</div>
<div style="font-size:0.78rem;line-height:1.40;color:#334155;font-weight:600;">
{appearance_val}
</div>
</div>

<div style="background:#FFFBEB;border:1.5px solid #FCD34D;border-left:4px solid #F59E0B;border-radius:10px;padding:8px 10px;">
<div style="font-size:0.78rem;font-weight:900;color:#92400E;margin-bottom:2px;display:flex;align-items:center;gap:4px;">
{t('dossier_superpower')}
</div>
<div style="font-size:0.78rem;line-height:1.40;color:#78350F;font-weight:700;">
{superpower_val}
</div>
</div>

<div style="background:#F0F9FF;border:1.5px solid #BAE6FD;border-left:4px solid #0284C7;border-radius:10px;padding:8px 10px;">
<div style="font-size:0.78rem;font-weight:900;color:#075985;margin-bottom:2px;display:flex;align-items:center;gap:4px;">
{t('dossier_xai')}
</div>
<div style="font-size:0.78rem;line-height:1.40;color:#0369A1;font-weight:700;">
{ai_attention_val}
</div>
</div>
</div>
</div>"""

    st.markdown(dossier_html, unsafe_allow_html=True)
    
    action_c1, action_c2 = st.columns([1.6, 1])
    with action_c1:
        if st.button(t("btn_test_species"), type="primary", use_container_width=True):
            fpath = benchmark_samples.get(cls_name)
            if fpath and os.path.exists(fpath):
                st.session_state.selected_image = Image.open(fpath).convert("RGB")
                st.session_state.selected_filename = f"{cls_name} ({os.path.basename(fpath)})"
                st.session_state.active_source_id = f"dossier_{cls_name}_{os.path.basename(fpath)}"
                st.session_state.analysis_cache = None
                st.session_state.show_report_modal = False
                st.rerun()
    with action_c2:
        if st.button(t("btn_close_cert"), use_container_width=True):
            st.rerun()

# -----------------------------------------------------------------------------
# 8. Supported Butterfly Species Showcase (8 Interactive Cards with Dossier)
# -----------------------------------------------------------------------------
st.markdown("<div style='height: 45px;'></div>", unsafe_allow_html=True)
st.markdown(f"## {t('roster_heading')}")
st.markdown(f"<p style='font-size: 1.15rem; font-weight: 800; color: #0F172A;'>{t('roster_sub')}</p>", unsafe_allow_html=True)

species_items = list(SPECIES_DOSSIER.items())
lang_key = st.session_state.app_lang.lower()

# Row 1: Classes 1 to 4
row1_cols = st.columns(4)
for i in range(4):
    name, d_meta = species_items[i]
    b64_img = species_b64_thumbnails.get(name, "")
    disp_name = d_meta.get(f"name_{lang_key}", name)
    fam_disp = d_meta.get(f"family_{lang_key}", d_meta.get("family_en", "")).split('(')[0].strip()
    
    with row1_cols[i]:
        st.markdown(f"""
        <div class="species-card-v2" style="border-top: 5px solid {d_meta['color_primary']} !important;">
            <div class="species-num-badge">#{i+1}</div>
            <div class="species-thumb-container" style="border: 2.5px solid {d_meta['color_primary']};">
                <img src="data:image/jpeg;base64,{b64_img}" class="species-thumb-img" alt="{disp_name}" />
            </div>
            <div class="card-title-name">{disp_name}</div>
            <div class="card-sci-name">{d_meta['scientific_name']}</div>
            <div class="card-fam-tag">🦋 {fam_disp}</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button(t("btn_explore_dossier"), key=f"dossier_btn_{name}", use_container_width=True):
            show_species_dossier_modal(name)

st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

# Row 2: Classes 5 to 8
row2_cols = st.columns(4)
for i in range(4, 8):
    name, d_meta = species_items[i]
    b64_img = species_b64_thumbnails.get(name, "")
    disp_name = d_meta.get(f"name_{lang_key}", name)
    fam_disp = d_meta.get(f"family_{lang_key}", d_meta.get("family_en", "")).split('(')[0].strip()
    
    with row2_cols[i - 4]:
        st.markdown(f"""
        <div class="species-card-v2" style="border-top: 5px solid {d_meta['color_primary']} !important;">
            <div class="species-num-badge">#{i+1}</div>
            <div class="species-thumb-container" style="border: 2.5px solid {d_meta['color_primary']};">
                <img src="data:image/jpeg;base64,{b64_img}" class="species-thumb-img" alt="{disp_name}" />
            </div>
            <div class="card-title-name">{disp_name}</div>
            <div class="card-sci-name">{d_meta['scientific_name']}</div>
            <div class="card-fam-tag">🦋 {fam_disp}</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button(t("btn_explore_dossier"), key=f"dossier_btn_{name}", use_container_width=True):
            show_species_dossier_modal(name)

# -----------------------------------------------------------------------------
# 9. Footer & Credits
# -----------------------------------------------------------------------------
st.markdown(f"""
<div class="app-footer">
    <div>{t('footer_designed_by')} <span class="footer-author">Ohi</span></div>
    <div style="margin-top: 8px; font-size: 1.05rem; color: #1E293B; font-weight: 700;">
        {t('footer_tech')}
    </div>
</div>
""", unsafe_allow_html=True)
