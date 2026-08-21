"""
AI Butterfly Vision: Streamlit Web Application
An Ultra-High-Contrast, Crystal-Clear Typography & Eye-Comfort Interface for Butterfly Species Classification & Grad-CAM Explainable AI.
Designed & Developed by Ohi.
"""

import os
import sys
import base64
from io import BytesIO
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
# 1. Page Configuration & Masterpiece Ultra-High Contrast CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Butterfly Vision • Explainable AI",
    page_icon="🦋",
    layout="wide",
    initial_sidebar_state="collapsed"
)

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
    /* BULLETPROOF UNIVERSAL TABS (COVERS EVERY SINGLE TAB IN STREAMLIT)         */
    /* ========================================================================= */
    
    /* Tab Container Bar */
    div[role="tablist"],
    [data-baseweb="tab-list"],
    div[data-testid="stTabs"] > div:first-child {
        display: flex !important;
        gap: 12px !important;
        background: #CBD5E1 !important;
        padding: 8px !important;
        border-radius: 16px !important;
        border: 2px solid #94A3B8 !important;
    }
    
    /* All Individual Tab Items */
    button[role="tab"],
    div[role="tab"],
    [data-baseweb="tab"],
    .stTabs [data-baseweb="tab"] {
        border-radius: 12px !important;
        padding: 10px 22px !important;
        font-size: 1.1rem !important;
        font-weight: 900 !important;
        transition: all 0.2s ease !important;
        border: none !important;
    }
    
    /* UNSELECTED TABS */
    button[role="tab"][aria-selected="false"],
    div[role="tab"][aria-selected="false"],
    [data-baseweb="tab"][aria-selected="false"] {
        background-color: #FFFFFF !important;
        border: 2px solid #64748B !important;
        box-shadow: 0 3px 8px rgba(0, 0, 0, 0.08) !important;
    }
    
    button[role="tab"][aria-selected="false"] *,
    div[role="tab"][aria-selected="false"] *,
    [data-baseweb="tab"][aria-selected="false"] *,
    button[role="tab"][aria-selected="false"] p,
    div[role="tab"][aria-selected="false"] p,
    [data-baseweb="tab"][aria-selected="false"] p {
        color: #0F172A !important;
        -webkit-text-fill-color: #0F172A !important;
        font-weight: 900 !important;
        font-size: 1.05rem !important;
    }
    
    /* SELECTED TAB */
    button[role="tab"][aria-selected="true"],
    div[role="tab"][aria-selected="true"],
    [data-baseweb="tab"][aria-selected="true"] {
        background-color: #0284C7 !important;
        border: 2px solid #0369A1 !important;
        box-shadow: 0 4px 14px rgba(2, 132, 199, 0.4) !important;
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
        font-size: 1.05rem !important;
    }
    
    /* REMOVE ALL STREAMLIT RED HIGHLIGHT LINES ON TABS */
    [data-baseweb="tab-highlight"],
    [data-baseweb="tab-border"],
    div[data-testid="stTabs"] hr,
    div[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
        background-color: transparent !important;
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
    /* MASTERPIECE SPLIT HERO & 3D ANIMATED BUTTERFLY SANCTUARY                   */
    /* ========================================================================= */
    .hero-split-wrapper {
        display: grid;
        grid-template-columns: 1.35fr 1.0fr;
        gap: 28px;
        align-items: center;
        padding: 2.2rem 1rem 1.2rem 1rem;
        position: relative;
    }
    
    @media (max-width: 992px) {
        .hero-split-wrapper {
            grid-template-columns: 1fr;
            text-align: center;
        }
    }
    
    .hero-left-pane {
        display: flex;
        flex-direction: column;
        align-items: flex-start;
    }
    
    @media (max-width: 992px) {
        .hero-left-pane {
            align-items: center;
        }
    }
    
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: #E0F2FE;
        border: 2px solid #0284C7;
        color: #0369A1;
        font-size: 0.96rem;
        font-weight: 900;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        padding: 7px 20px;
        border-radius: 999px;
        margin-bottom: 1.1rem;
        box-shadow: 0 4px 14px rgba(2, 132, 199, 0.15);
    }
    .hero-title {
        font-size: 3.8rem;
        font-weight: 900;
        letter-spacing: -0.03em;
        line-height: 1.1;
        color: #0F172A;
        margin-bottom: 0.8rem;
    }
    .hero-subtitle {
        font-size: 1.32rem;
        color: #1E293B;
        font-weight: 700;
        line-height: 1.55;
        margin-bottom: 1.4rem;
    }
    
    .hero-tag-strip {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 0.4rem;
    }
    .hero-tag {
        display: inline-flex;
        align-items: center;
        background: #FFFFFF;
        border: 1.5px solid #CBD5E1;
        color: #0F172A;
        font-size: 0.94rem;
        font-weight: 800;
        padding: 6px 16px;
        border-radius: 999px;
        box-shadow: 0 3px 8px rgba(0,0,0,0.04);
        transition: all 0.2s ease;
    }
    .hero-tag:hover {
        border-color: #0284C7;
        color: #0284C7;
        transform: translateY(-2px);
    }

    /* Luminous Glass Butterfly Sanctuary Card */
    .sanctuary-terrarium {
        background: linear-gradient(135deg, #FFFFFF 0%, #F0F9FF 60%, #E0F2FE 100%);
        border: 2.5px solid #94A3B8;
        border-radius: 28px;
        padding: 24px;
        height: 250px;
        position: relative;
        overflow: hidden;
        box-shadow: 0 16px 36px rgba(2, 132, 199, 0.12), inset 0 0 20px rgba(255, 255, 255, 0.8);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }

    /* Ambient Bioluminescent Sparkles */
    .sanctuary-glow {
        position: absolute;
        top: -30px;
        right: -30px;
        width: 140px;
        height: 140px;
        background: radial-gradient(circle, rgba(2, 132, 199, 0.25) 0%, rgba(2, 132, 199, 0) 70%);
        border-radius: 50%;
        animation: glow-pulse 4s ease-in-out infinite alternate;
        pointer-events: none;
    }
    
    @keyframes glow-pulse {
        0% { transform: scale(0.9); opacity: 0.5; }
        100% { transform: scale(1.3); opacity: 0.9; }
    }

    /* 3D Butterfly Realistic Wing Flutter Mechanics */
    .butterfly-stage {
        position: relative;
        width: 100%;
        height: 140px;
        perspective: 600px;
    }

    /* Monarch Natural Flight */
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

    .butterfly-monarch {
        display: flex;
        align-items: center;
        transform-style: preserve-3d;
    }

    .monarch-wing-left {
        width: 44px;
        height: 48px;
        background: radial-gradient(circle at 100% 50%, #EA580C 20%, #C2410C 70%, #0F172A 95%);
        border-radius: 80% 20% 60% 40%;
        border: 2px solid #0F172A;
        transform-origin: right center;
        animation: wing-flap-left 0.28s ease-in-out infinite alternate;
        box-shadow: 0 4px 10px rgba(234, 88, 12, 0.4);
    }

    .monarch-wing-right {
        width: 44px;
        height: 48px;
        background: radial-gradient(circle at 0% 50%, #EA580C 20%, #C2410C 70%, #0F172A 95%);
        border-radius: 20% 80% 40% 60%;
        border: 2px solid #0F172A;
        transform-origin: left center;
        animation: wing-flap-right 0.28s ease-in-out infinite alternate;
        box-shadow: 0 4px 10px rgba(234, 88, 12, 0.4);
    }

    .butterfly-body {
        width: 6px;
        height: 38px;
        background: #0F172A;
        border-radius: 999px;
        z-index: 5;
        position: relative;
    }
    .butterfly-body::before {
        content: '';
        position: absolute;
        top: -6px;
        left: -4px;
        width: 14px;
        height: 6px;
        border-top: 2px solid #0F172A;
        border-radius: 50%;
    }

    /* Adonis Blue Playful Flutter */
    .adonis-carrier {
        position: absolute;
        top: 60px;
        right: 18%;
        animation: adonis-flight 5.2s ease-in-out infinite alternate;
        transform-style: preserve-3d;
    }

    @keyframes adonis-flight {
        0% { transform: translate3d(0, 0, 0) rotate(-5deg); }
        40% { transform: translate3d(-30px, -22px, 12px) rotate(8deg); }
        80% { transform: translate3d(12px, 16px, -8px) rotate(-4deg); }
        100% { transform: translate3d(-18px, -12px, 5px) rotate(3deg); }
    }

    .butterfly-adonis {
        display: flex;
        align-items: center;
        transform-style: preserve-3d;
    }

    .adonis-wing-left {
        width: 32px;
        height: 36px;
        background: radial-gradient(circle at 100% 50%, #38BDF8 20%, #0284C7 70%, #0369A1 95%);
        border-radius: 80% 20% 60% 40%;
        border: 1.5px solid #0369A1;
        transform-origin: right center;
        animation: wing-flap-left 0.24s ease-in-out infinite alternate;
        box-shadow: 0 4px 12px rgba(2, 132, 199, 0.5);
    }

    .adonis-wing-right {
        width: 32px;
        height: 36px;
        background: radial-gradient(circle at 0% 50%, #38BDF8 20%, #0284C7 70%, #0369A1 95%);
        border-radius: 20% 80% 40% 60%;
        border: 1.5px solid #0369A1;
        transform-origin: left center;
        animation: wing-flap-right 0.24s ease-in-out infinite alternate;
        box-shadow: 0 4px 12px rgba(2, 132, 199, 0.5);
    }

    /* Emerald Cattleheart Flight */
    .emerald-carrier {
        position: absolute;
        top: 15px;
        left: 8%;
        animation: emerald-flight 7.0s ease-in-out infinite alternate;
        transform-style: preserve-3d;
    }
    @keyframes emerald-flight {
        0% { transform: translate3d(0, 0, 0) rotate(8deg); }
        50% { transform: translate3d(20px, 24px, 8px) rotate(-10deg); }
        100% { transform: translate3d(-10px, 8px, -5px) rotate(4deg); }
    }
    .butterfly-emerald {
        display: flex;
        align-items: center;
        transform-style: preserve-3d;
    }
    .emerald-wing-left {
        width: 36px;
        height: 40px;
        background: radial-gradient(circle at 100% 50%, #10B981 25%, #047857 70%, #0F172A 95%);
        border-radius: 80% 20% 60% 40%;
        border: 1.5px solid #065F46;
        transform-origin: right center;
        animation: wing-flap-left 0.25s ease-in-out infinite alternate;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.45);
    }
    .emerald-wing-right {
        width: 36px;
        height: 40px;
        background: radial-gradient(circle at 0% 50%, #10B981 25%, #047857 70%, #0F172A 95%);
        border-radius: 20% 80% 40% 60%;
        border: 1.5px solid #065F46;
        transform-origin: left center;
        animation: wing-flap-right 0.25s ease-in-out infinite alternate;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.45);
    }
    .emerald-body {
        width: 4px;
        height: 32px;
        background: #0F172A;
        border-radius: 999px;
        z-index: 5;
    }

    /* Red Postman Flight */
    .postman-carrier {
        position: absolute;
        top: 75px;
        left: 42%;
        animation: postman-flight 5.8s ease-in-out infinite alternate;
        transform-style: preserve-3d;
    }
    @keyframes postman-flight {
        0% { transform: translate3d(0, 0, 0) rotate(-4deg); }
        45% { transform: translate3d(-18px, -16px, 10px) rotate(6deg); }
        100% { transform: translate3d(22px, 12px, -8px) rotate(-3deg); }
    }
    .butterfly-postman {
        display: flex;
        align-items: center;
        transform-style: preserve-3d;
    }
    .postman-wing-left {
        width: 38px;
        height: 42px;
        background: radial-gradient(circle at 100% 50%, #EF4444 30%, #991B1B 75%, #0F172A 95%);
        border-radius: 80% 20% 60% 40%;
        border: 1.5px solid #7F1D1D;
        transform-origin: right center;
        animation: wing-flap-left 0.27s ease-in-out infinite alternate;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.45);
    }
    .postman-wing-right {
        width: 38px;
        height: 42px;
        background: radial-gradient(circle at 0% 50%, #EF4444 30%, #991B1B 75%, #0F172A 95%);
        border-radius: 20% 80% 40% 60%;
        border: 1.5px solid #7F1D1D;
        transform-origin: left center;
        animation: wing-flap-right 0.27s ease-in-out infinite alternate;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.45);
    }
    .postman-body {
        width: 5px;
        height: 34px;
        background: #0F172A;
        border-radius: 999px;
        z-index: 5;
    }

    /* Southern Dogface / Golden Swallowtail Flight */
    .dogface-carrier {
        position: absolute;
        top: 20px;
        right: 6%;
        animation: dogface-flight 6.2s ease-in-out infinite alternate;
        transform-style: preserve-3d;
    }
    @keyframes dogface-flight {
        0% { transform: translate3d(0, 0, 0) rotate(6deg); }
        55% { transform: translate3d(-15px, 20px, 12px) rotate(-8deg); }
        100% { transform: translate3d(10px, -12px, -6px) rotate(4deg); }
    }
    .butterfly-dogface {
        display: flex;
        align-items: center;
        transform-style: preserve-3d;
    }
    .dogface-wing-left {
        width: 34px;
        height: 38px;
        background: radial-gradient(circle at 100% 50%, #FBBF24 25%, #D97706 70%, #0F172A 95%);
        border-radius: 80% 20% 60% 40%;
        border: 1.5px solid #B45309;
        transform-origin: right center;
        animation: wing-flap-left 0.29s ease-in-out infinite alternate;
        box-shadow: 0 4px 12px rgba(245, 158, 11, 0.45);
    }
    .dogface-wing-right {
        width: 34px;
        height: 38px;
        background: radial-gradient(circle at 0% 50%, #FBBF24 25%, #D97706 70%, #0F172A 95%);
        border-radius: 20% 80% 40% 60%;
        border: 1.5px solid #B45309;
        transform-origin: left center;
        animation: wing-flap-right 0.29s ease-in-out infinite alternate;
        box-shadow: 0 4px 12px rgba(245, 158, 11, 0.45);
    }
    .dogface-body {
        width: 4px;
        height: 30px;
        background: #0F172A;
        border-radius: 999px;
        z-index: 5;
    }

    @keyframes wing-flap-left {
        0% { transform: rotateY(0deg); }
        100% { transform: rotateY(-68deg); }
    }

    @keyframes wing-flap-right {
        0% { transform: rotateY(0deg); }
        100% { transform: rotateY(68deg); }
    }

    /* Terrarium Bottom Badge */
    .terrarium-footer {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(8px);
        border: 1.5px solid #CBD5E1;
        border-radius: 14px;
        padding: 8px 16px;
        z-index: 10;
    }
    .terrarium-title {
        font-size: 0.92rem;
        font-weight: 900;
        color: #0F172A;
        letter-spacing: 0.04em;
    }
    .terrarium-live {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.84rem;
        font-weight: 900;
        color: #059669;
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
        gap: 18px;
        margin: 1.2rem 0 2.5rem 0;
    }
    .metric-card {
        background: #FFFFFF;
        border: 2px solid #94A3B8;
        border-radius: 18px;
        padding: 22px 18px;
        text-align: center;
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.08);
        transition: all 0.25s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        border-color: #0284C7;
        box-shadow: 0 12px 28px rgba(2, 132, 199, 0.2);
    }
    .metric-val {
        font-size: 2.3rem;
        font-weight: 900;
        color: #0284C7;
        margin-bottom: 4px;
    }
    .metric-lbl {
        font-size: 1.0rem;
        font-weight: 900;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #0F172A;
    }

    /* Primary Action Button (Big & Bold) */
    button[kind="primary"] {
        background: #0284C7 !important;
        color: #FFFFFF !important;
        font-weight: 900 !important;
        border: none !important;
        border-radius: 14px !important;
        box-shadow: 0 6px 20px rgba(2, 132, 199, 0.35) !important;
        transition: all 0.25s ease !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-size: 1.15rem !important;
        padding: 0.9rem 2.0rem !important;
    }
    button[kind="primary"]:hover {
        background: #0369A1 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 28px rgba(2, 132, 199, 0.45) !important;
    }

    /* Secondary Gallery Buttons */
    button[kind="secondary"] {
        background: #FFFFFF !important;
        color: #0F172A !important;
        border: 2px solid #64748B !important;
        border-radius: 14px !important;
        font-weight: 900 !important;
        font-size: 1.05rem !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.08) !important;
        padding: 0.9rem 1.1rem !important;
    }
    button[kind="secondary"] * {
        color: #0F172A !important;
        -webkit-text-fill-color: #0F172A !important;
        font-weight: 900 !important;
    }
    button[kind="secondary"]:hover {
        border-color: #0284C7 !important;
        color: #0284C7 !important;
        background: #E0F2FE !important;
        box-shadow: 0 6px 18px rgba(2, 132, 199, 0.25) !important;
        transform: translateY(-2px) !important;
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
        font-size: 1.15rem !important;
        padding: 0.9rem 1.8rem !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
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
        padding: 28px;
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
        font-size: 2.9rem;
        font-weight: 900;
        color: #0F172A;
        letter-spacing: -0.03em;
        line-height: 1.05;
        margin-bottom: 6px;
    }
    .result-meta {
        font-size: 1.3rem;
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
    /* MASTERPIECE PHOTOGRAPHIC SPECIES CARDS (MUSEUM GRADE)                     */
    /* ========================================================================= */
    .species-card-v2 {
        background: #FFFFFF !important;
        border: 2px solid #CBD5E1 !important;
        border-radius: 20px !important;
        padding: 22px 14px !important;
        text-align: center !important;
        height: 100% !important;
        box-shadow: 0 6px 20px rgba(15, 23, 42, 0.05) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        position: relative !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
    }
    .species-card-v2:hover {
        transform: translateY(-6px) !important;
        box-shadow: 0 16px 32px rgba(2, 132, 199, 0.18) !important;
        border-color: #0284C7 !important;
    }
    .species-num-badge {
        position: absolute;
        top: 12px;
        right: 14px;
        background: #F1F5F9;
        color: #0369A1;
        font-weight: 900;
        font-size: 0.85rem;
        padding: 3px 9px;
        border-radius: 8px;
        border: 1px solid #CBD5E1;
    }
    .species-thumb-container {
        width: 106px;
        height: 106px;
        border-radius: 50%;
        padding: 4px;
        background: #FFFFFF;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.1);
        margin: 6px auto 14px auto;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: transform 0.3s ease;
    }
    .species-card-v2:hover .species-thumb-container {
        transform: scale(1.08);
    }
    .species-thumb-img {
        width: 98px;
        height: 98px;
        border-radius: 50%;
        object-fit: cover;
    }
    .card-title-name {
        font-size: 1.18rem !important;
        font-weight: 900 !important;
        color: #0F172A !important;
        margin-bottom: 4px !important;
        line-height: 1.2 !important;
    }
    .card-sci-name {
        font-size: 0.98rem !important;
        font-weight: 700 !important;
        font-style: italic !important;
        color: #0284C7 !important;
        margin-bottom: 10px !important;
    }
    .card-fam-tag {
        display: inline-block !important;
        background: #F8FAFC !important;
        border: 1.5px solid #E2E8F0 !important;
        color: #475569 !important;
        font-size: 0.84rem !important;
        font-weight: 800 !important;
        padding: 4px 14px !important;
        border-radius: 999px !important;
        letter-spacing: 0.02em !important;
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
# 3. Hero Header (Masterpiece Split Layout & 3D Butterfly Sanctuary)
# -----------------------------------------------------------------------------
st.markdown("""
<div class="hero-split-wrapper">
    <div class="hero-left-pane">
        <div class="hero-badge">🦋 ResNet-18 • Deep Transfer Learning • Grad-CAM XAI</div>
        <div class="hero-title">AI Butterfly Vision</div>
        <div class="hero-subtitle">High-Precision Butterfly Species Classification with Real-Time Explainable AI (XAI) Diagnostics</div>
        <div class="hero-tag-strip">
            <span class="hero-tag">🌿 8 Tropical Species</span>
            <span class="hero-tag">🔬 Real-Time Grad-CAM XAI</span>
            <span class="hero-tag">⚡ 97.22% Precision</span>
            <span class="hero-tag">📄 4K Report Generator</span>
        </div>
    </div>
    <div class="hero-right-pane">
        <div class="sanctuary-terrarium">
            <div class="sanctuary-glow"></div>
            <div class="butterfly-stage">
                <!-- 1. 3D Fluttering Emerald Cattleheart -->
                <div class="emerald-carrier">
                    <div class="butterfly-emerald">
                        <div class="emerald-wing-left"></div>
                        <div class="emerald-body"></div>
                        <div class="emerald-wing-right"></div>
                    </div>
                </div>
                <!-- 2. 3D Fluttering Monarch -->
                <div class="monarch-carrier">
                    <div class="butterfly-monarch">
                        <div class="monarch-wing-left"></div>
                        <div class="butterfly-body"></div>
                        <div class="monarch-wing-right"></div>
                    </div>
                </div>
                <!-- 3. 3D Fluttering Red Postman -->
                <div class="postman-carrier">
                    <div class="butterfly-postman">
                        <div class="postman-wing-left"></div>
                        <div class="postman-body"></div>
                        <div class="postman-wing-right"></div>
                    </div>
                </div>
                <!-- 4. 3D Fluttering Adonis Blue -->
                <div class="adonis-carrier">
                    <div class="butterfly-adonis">
                        <div class="adonis-wing-left"></div>
                        <div class="adonis-body"></div>
                        <div class="adonis-wing-right"></div>
                    </div>
                </div>
                <!-- 5. 3D Fluttering Southern Dogface -->
                <div class="dogface-carrier">
                    <div class="butterfly-dogface">
                        <div class="dogface-wing-left"></div>
                        <div class="dogface-body"></div>
                        <div class="dogface-wing-right"></div>
                    </div>
                </div>
            </div>
            <div class="terrarium-footer">
                <span class="terrarium-title">🌸 AI BIO-VISION TERRARIUM</span>
                <span class="terrarium-live"><span class="live-dot"></span> NEURAL ENGINE LIVE</span>
            </div>
        </div>
    </div>
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
st.markdown("## 📸 Select or Capture Butterfly Specimen")

if "selected_image" not in st.session_state:
    st.session_state.selected_image = None
if "selected_filename" not in st.session_state:
    st.session_state.selected_filename = None
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
                # Generate high-res base64 thumbnail for visual cards
                try:
                    t_img = Image.open(fpath).convert("RGB").resize((140, 140), Image.Resampling.LANCZOS)
                    t_buf = BytesIO()
                    t_img.save(t_buf, format="JPEG", quality=92)
                    species_b64_thumbnails[cls] = base64.b64encode(t_buf.getvalue()).decode()
                except Exception:
                    species_b64_thumbnails[cls] = ""

input_tab1, input_tab2, input_tab3 = st.tabs([
    "⚡ 1-Click Benchmark Gallery",
    "📤 Upload Butterfly Image",
    "📷 Live Camera Capture"
])

with input_tab1:
    st.markdown("<p style='font-size: 1.15rem; font-weight: 800; color: #0F172A;'>Click any butterfly card below to select specimen:</p>", unsafe_allow_html=True)
    sample_cols = st.columns(4)
    for idx, (cls_name, fpath) in enumerate(benchmark_samples.items()):
        col_target = sample_cols[idx % 4]
        with col_target:
            if st.button(f"🦋 {cls_name}", key=f"btn_sample_{idx}", use_container_width=True):
                st.session_state.selected_image = Image.open(fpath).convert("RGB")
                st.session_state.selected_filename = f"{cls_name} ({os.path.basename(fpath)})"
                st.session_state.analysis_cache = None
                st.session_state.show_report_modal = False

with input_tab2:
    st.markdown("<p style='font-size: 1.15rem; font-weight: 800; color: #0F172A;'>Drop or browse an image from your computer:</p>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload Image File",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )
    if uploaded_file is not None:
        try:
            st.session_state.selected_image = Image.open(uploaded_file).convert("RGB")
            st.session_state.selected_filename = uploaded_file.name
            st.session_state.analysis_cache = None
            st.session_state.show_report_modal = False
        except Exception as e:
            st.error(f"Error reading image: {e}")

with input_tab3:
    st.markdown("<p style='font-size: 1.15rem; font-weight: 800; color: #0F172A; text-align: center;'>Point your webcam or mobile camera directly at a butterfly specimen:</p>", unsafe_allow_html=True)
    
    # Centered Compact Camera Frame
    cam_col1, cam_col2, cam_col3 = st.columns([1, 2, 1])
    with cam_col2:
        camera_photo = st.camera_input("Take a butterfly snapshot", label_visibility="collapsed")
        if camera_photo is not None:
            try:
                st.session_state.selected_image = Image.open(camera_photo).convert("RGB")
                st.session_state.selected_filename = "Live_Camera_Capture.jpg"
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
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    
    # Image Command Bar
    cmd_col1, cmd_col2 = st.columns([1, 2.0])
    with cmd_col1:
        st.image(active_image, caption=f"Active Specimen: {active_filename}", use_container_width=True)
    
    with cmd_col2:
        st.markdown("### 🔬 Neural Diagnostics Ready")
        st.markdown("<p style='font-size: 1.15rem; font-weight: 700; color: #1E293B;'>Extract deep features with ResNet-18 and compute spatial backpropagation gradients with Grad-CAM.</p>", unsafe_allow_html=True)
        
        btn_c1, btn_c2 = st.columns([2.2, 1.2])
        with btn_c1:
            run_analysis = st.button("✨ Run Neural Analysis", type="primary", use_container_width=True)
        with btn_c2:
            if st.button("🔄 Reset / Clear", use_container_width=True, help="Clear active specimen and start fresh"):
                st.session_state.selected_image = None
                st.session_state.selected_filename = None
                st.session_state.analysis_cache = None
                st.session_state.show_report_modal = False
                st.rerun()

    # Trigger computation on button click and persist into session cache
    if run_analysis:
        with st.spinner("Executing neural feature mapping and gradient backpropagation..."):
            try:
                # 1. Inference
                pred_res = predictor.predict(active_image)
                # 2. Grad-CAM
                with GradCAM(predictor.model) as gradcam_engine:
                    raw_heatmap, _, _ = gradcam_engine.generate(
                        pred_res['input_tensor'],
                        pred_res['predicted_idx']
                    )
                # Persist full analysis in session cache
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

        # Two-Column Results Suite
        col_left, col_right = st.columns([1.15, 1.35], gap="large")

        # --- LEFT COLUMN: Prediction Cockpit & Biological Facts ---
        with col_left:
            st.markdown("## 🎯 Classification Result")
            
            meta = SPECIES_METADATA.get(pred_class, {
                "scientific_name": "Unknown",
                "family": "Insecta",
                "color_primary": "#0284C7",
                "xai_insight": "Model focused on discriminative visual wing patterns."
            })

            gauge_color = "#10B981" if confidence >= 80.0 else "#F59E0B"
            gauge_lbl = "HIGH" if confidence >= 80.0 else "MODERATE"

            st.markdown(f"""
            <div class="result-capsule" style="border-top: 7px solid {meta['color_primary']};">
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
                <div style="background: #CBD5E1; height: 12px; border-radius: 999px; overflow: hidden; margin-top: 16px;">
                    <div style="width: {min(confidence, 100.0):.1f}%; height: 100%; border-radius: 999px; background: linear-gradient(90deg, #0284C7, {meta['color_primary']});"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Top-3 Probabilities
            st.markdown("### 🏆 Top-3 Ranked Probabilities")
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
                <div class="xai-hdr">🔬 Neural Attention Diagnostic (XAI)</div>
                {meta['xai_insight']}
            </div>
            """, unsafe_allow_html=True)

            # Biological Metadata Expander
            with st.expander("📖 View Biological Taxonomy & Ecological Habitat", expanded=False):
                st.markdown(f"""
                - **Visual Markers**: {meta['appearance']}
                - **Geographic Range**: {meta['distribution']}
                - **Ecological Feature**: {meta['key_features']}
                """)

        # --- RIGHT COLUMN: Explainable AI (Grad-CAM) Studio ---
        with col_right:
            st.markdown("## 🔬 Explainable AI Studio (Grad-CAM)")
            st.markdown("<p style='font-size: 1.15rem; font-weight: 800; color: #0F172A;'>The highlighted heat regions indicate image areas that strongly influenced the model's prediction:</p>", unsafe_allow_html=True)

            # Smooth Real-Time Heatmap Blend & Intensity Slider
            blend_alpha = st.slider(
                "Heatmap Blend Intensity & Focus (α)",
                min_value=0.20,
                max_value=0.85,
                value=0.55,
                step=0.05,
                key="gradcam_alpha_slider",
                help="Controls transparency overlay AND adjusts thermal hotspot focus sensitivity"
            )

            # DYNAMIC SENSITIVITY TRANSFORMATION
            gamma_exponent = 1.0 + (0.55 - blend_alpha) * 1.4
            modulated_heatmap = np.clip(np.power(raw_heatmap, gamma_exponent), 0.0, 1.0)

            with GradCAM(predictor.model) as gradcam_engine:
                heatmap_img, overlay_img = gradcam_engine.overlay_heatmap(
                    modulated_heatmap,
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
                    caption=f"Grad-CAM Attention Overlay on {pred_class} (α = {blend_alpha:.2f})",
                    use_container_width=True
                )
            with cam_tab2:
                st.image(
                    heatmap_img,
                    caption=f"Dynamic Thermal Activation Heatmap (Jet Colormap, Sensitivity α = {blend_alpha:.2f})",
                    use_container_width=True
                )
            with cam_tab3:
                st.image(
                    display_img,
                    caption="Original Specimen Input",
                    use_container_width=True
                )
            with cam_tab4:
                comp_col1, comp_col2 = st.columns(2)
                with comp_col1:
                    st.image(display_img, caption="Original Input", use_container_width=True)
                with comp_col2:
                    st.image(overlay_img, caption=f"Dynamic Grad-CAM Overlay (α = {blend_alpha:.2f})", use_container_width=True)

            # -------------------------------------------------------------
            # ON-DEMAND REPORT CARD INSPECTOR BUTTON
            # -------------------------------------------------------------
            st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
            
            if st.button("📄 Generate & Inspect AI Report Certificate", type="primary", use_container_width=True):
                st.session_state.show_report_modal = True

        # -----------------------------------------------------------------
        # MODAL / EXPANDED REPORT CERTIFICATE INSPECTION SUITE
        # -----------------------------------------------------------------
        if st.session_state.get("show_report_modal", False):
            st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)
            
            # Generate Ultra-HD 2000x1250 Report Graphic
            report_bytes = generate_report_card(
                original_image=display_img,
                overlay_image=overlay_img,
                pred_class=pred_class,
                confidence=confidence,
                top_k=top_k
            )

            # Dedicated Certificate Inspection Card
            st.markdown("""
            <div style="background: #FFFFFF; border: 2.5px solid #0284C7; border-radius: 20px; padding: 24px; box-shadow: 0 15px 35px rgba(2, 132, 199, 0.15); margin-bottom: 25px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                    <div>
                        <div style="font-size: 0.95rem; font-weight: 900; color: #0284C7; text-transform: uppercase; letter-spacing: 0.08em;">Official Specimen Inspection</div>
                        <div style="font-size: 1.85rem; font-weight: 900; color: #0F172A;">AI Inspection Certificate Preview</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.image(
                report_bytes,
                caption=f"Ultra-HD Inspection Certificate • Specimen: {pred_class} • Verified by ResNet-18 & Grad-CAM",
                use_container_width=True
            )

            st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

            # Action Buttons Row: Download & Close/Exit
            action_c1, action_c2 = st.columns([1.5, 1])
            with action_c1:
                st.download_button(
                    label="📥 Download Official Report Certificate (High-Res PNG)",
                    data=report_bytes,
                    file_name=f"Butterfly_AI_Report_{pred_class.replace(' ', '_')}.png",
                    mime="image/png",
                    use_container_width=True
                )
            with action_c2:
                if st.button("❌ Close / Exit Report View", use_container_width=True):
                    st.session_state.show_report_modal = False
                    st.rerun()

    else:
        st.info("👆 Click **✨ Run Neural Analysis** above to identify species and generate Grad-CAM explanation.")

else:
    st.info("💡 Choose a butterfly specimen above via 1-Click Gallery, Image Upload, or Live Camera to start AI analysis.")

# -----------------------------------------------------------------------------
# 7. Supported Butterfly Species Showcase (Breathtaking Photographic Cards)
# -----------------------------------------------------------------------------
st.markdown("<div style='height: 45px;'></div>", unsafe_allow_html=True)
st.markdown("## 🌿 Supported Butterfly Species Taxonomy (8 Classes)")
st.markdown("<p style='font-size: 1.15rem; font-weight: 800; color: #0F172A;'>The neural network is specialized to distinguish the following 8 butterfly species with deep feature attention:</p>", unsafe_allow_html=True)

species_items = list(SPECIES_METADATA.items())

# Row 1: Classes 1 to 4
row1_cols = st.columns(4)
for i in range(4):
    name, s_meta = species_items[i]
    b64_img = species_b64_thumbnails.get(name, "")
    with row1_cols[i]:
        st.markdown(f"""
        <div class="species-card-v2" style="border-top: 5px solid {s_meta['color_primary']} !important;">
            <div class="species-num-badge">#{i+1}</div>
            <div class="species-thumb-container" style="border: 2.5px solid {s_meta['color_primary']};">
                <img src="data:image/jpeg;base64,{b64_img}" class="species-thumb-img" alt="{name}" />
            </div>
            <div class="card-title-name">{name}</div>
            <div class="card-sci-name">{s_meta['scientific_name']}</div>
            <div class="card-fam-tag">🦋 {s_meta['family'].split('(')[0].strip()}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

# Row 2: Classes 5 to 8
row2_cols = st.columns(4)
for i in range(4, 8):
    name, s_meta = species_items[i]
    b64_img = species_b64_thumbnails.get(name, "")
    with row2_cols[i - 4]:
        st.markdown(f"""
        <div class="species-card-v2" style="border-top: 5px solid {s_meta['color_primary']} !important;">
            <div class="species-num-badge">#{i+1}</div>
            <div class="species-thumb-container" style="border: 2.5px solid {s_meta['color_primary']};">
                <img src="data:image/jpeg;base64,{b64_img}" class="species-thumb-img" alt="{name}" />
            </div>
            <div class="card-title-name">{name}</div>
            <div class="card-sci-name">{s_meta['scientific_name']}</div>
            <div class="card-fam-tag">🦋 {s_meta['family'].split('(')[0].strip()}</div>
        </div>
        """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 8. Footer & Credits
# -----------------------------------------------------------------------------
st.markdown("""
<div class="app-footer">
    <div>Designed & Developed by <span class="footer-author">Ohi</span></div>
    <div style="margin-top: 8px; font-size: 1.05rem; color: #1E293B; font-weight: 700;">
        Deep Transfer Learning (ResNet-18) • Native PyTorch Grad-CAM • TorchScript Mobile & Edge Architecture
    </div>
</div>
""", unsafe_allow_html=True)
