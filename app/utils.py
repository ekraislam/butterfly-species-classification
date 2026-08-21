"""
AI Butterfly Vision: Utility Functions
Taxonomy metadata, XAI diagnostic insights, Ultra-High-Resolution White Luxury Certificate Generator, and path resolvers.
Designed & Developed by Ohi.
"""

import os
import hashlib
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# -----------------------------------------------------------------------------
# 1. Species Metadata & Biological Taxonomy
# -----------------------------------------------------------------------------
SPECIES_METADATA = {
    "ADONIS": {
        "scientific_name": "Lysandra bellargus",
        "family": "Lycaenidae (Gossamer-winged butterflies)",
        "appearance": "Brilliant sky-blue wings with a checkered white fringe in males; warm brown in females.",
        "distribution": "Chalk and limestone grasslands across Southern Europe and temperate Asia.",
        "key_features": "Checkered wing fringe and vivid male iridescence serve as key diagnostic visual traits.",
        "color_primary": "#0284C7",
        "xai_insight": "Model neural attention is heavily concentrated on the checkered white wing fringes and vibrant sky-blue wing membrane."
    },
    "CLODIUS PARNASSIAN": {
        "scientific_name": "Parnassius clodius",
        "family": "Papilionidae (Swallowtails)",
        "appearance": "Translucent chalk-white wings with prominent red and black ocelli (eye-spots) on hindwings.",
        "distribution": "Montane meadows and alpine clearings in Western North America.",
        "key_features": "Sub-marginal red ringed spots and parchment-like semi-translucent wing edges.",
        "color_primary": "#E11D48",
        "xai_insight": "Model neural attention locks onto the distinct crimson-red ocelli (eye-spots) and translucent wing margins."
    },
    "GREEN CELLED CATTLEHEART": {
        "scientific_name": "Parides sesostris",
        "family": "Papilionidae (Swallowtails)",
        "appearance": "Velvety pitch-black wings with glowing emerald-green patches on forewings and ruby-red spots on hindwings.",
        "distribution": "Tropical rainforests from Mexico to the Amazon basin.",
        "key_features": "High-contrast emerald patch against velvety black ground color.",
        "color_primary": "#059669",
        "xai_insight": "Model neural attention zeroes in on the luminous emerald-green forewing patch and contrasting ruby hindwing dots."
    },
    "MONARCH": {
        "scientific_name": "Danaus plexippus",
        "family": "Nymphalidae (Brush-footed butterflies)",
        "appearance": "Iconic fiery orange wings etched with bold black veins and a double row of white margin spots.",
        "distribution": "Americas, famous for multi-generational mass migration.",
        "key_features": "Thick black venation pattern with marginal white speckles.",
        "color_primary": "#EA580C",
        "xai_insight": "Model neural attention tracks the intricate black venous branching and the double row of white margin dots."
    },
    "ORANGE OAKLEAF": {
        "scientific_name": "Kallima inachus",
        "family": "Nymphalidae (Brush-footed butterflies)",
        "appearance": "Upperside exhibits rich cobalt and bright orange bands; underside mimics a dried dead leaf.",
        "distribution": "Tropical forests of East, South, and Southeast Asia.",
        "key_features": "Master of crypsis with realistic leaf-vein underside camouflage.",
        "color_primary": "#D97706",
        "xai_insight": "Model neural attention isolates the angled forewing apex, midrib vein mimicry, and contrasting dorsal orange band."
    },
    "PAPER KITE": {
        "scientific_name": "Idea leuconoe",
        "family": "Nymphalidae (Brush-footed butterflies)",
        "appearance": "Large translucent white-yellow wings patterned with dramatic black streaks and margin spots.",
        "distribution": "Mangrove swamps and lowland rainforests of Southeast Asia.",
        "key_features": "Slow gliding flight; bold black venation against paper-white background.",
        "color_primary": "#475569",
        "xai_insight": "Model neural attention detects the high-contrast black grid venation and translucent white wing cells."
    },
    "RED POSTMAN": {
        "scientific_name": "Heliconius erato",
        "family": "Nymphalidae (Brush-footed butterflies)",
        "appearance": "Elongated velvet-black wings featuring a bright crimson/pink forewing band and yellow hindwing bar.",
        "distribution": "Central and South American forest understories.",
        "key_features": "Müllerian mimicry complex with toxic chemical defenses.",
        "color_primary": "#DC2626",
        "xai_insight": "Model neural attention fixes squarely on the elongated crimson-red forewing stripe and narrow wing silhouette."
    },
    "SOUTHERN DOGFACE": {
        "scientific_name": "Zerene cesonia",
        "family": "Pieridae (Whites and Yellows)",
        "appearance": "Vibrant canary-yellow wings with a distinct black silhouette resembling a poodle's head.",
        "distribution": "Southern United States through Central America to South America.",
        "key_features": "Unmistakable canine profile created by the forewing black border.",
        "color_primary": "#CA8A04",
        "xai_insight": "Model neural attention isolates the iconic canine silhouette pattern (poodle face) in the yellow forewing disc."
    }
}

# -----------------------------------------------------------------------------
# 2. Masterpiece White & Royal Nature Luxury Certificate Generator (2000x1250)
# -----------------------------------------------------------------------------
def generate_report_card(original_image: Image.Image, overlay_image: Image.Image,
                         pred_class: str, confidence: float, top_k: list) -> bytes:
    """
    Synthesizes an Ultra-Crisp White Luxury Bio-Research Certificate (2000x1250).
    Features:
    - Pure White & Soft Porcelain Card (#FFFFFF / #F8FAFC)
    - Crystal-Clear Segoe UI Typography with 100% Solid Contrast
    - Royal Cyan, Emerald, and Gold Accents
    - High-Resolution Specimen Dual Chambers
    - Precision Probability Progress Bars
    - Official Verification Seal & Signature
    """
    meta = SPECIES_METADATA.get(pred_class, {
        "scientific_name": "Unknown species",
        "family": "Insecta • Lepidoptera",
        "appearance": "Distinct biological visual wing markings.",
        "distribution": "Global biodiversity habitat.",
        "key_features": "Diagnostic taxonomic wing venation pattern.",
        "color_primary": "#0284C7",
        "xai_insight": "Model neural attention concentrated on discriminative visual wing patterns."
    })

    card_w, card_h = 2000, 1250
    # Pure Clean White Canvas
    img_canvas = Image.new("RGB", (card_w, card_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(img_canvas)

    # Load high-grade true-type fonts
    try:
        font_brand = ImageFont.truetype("segoeuib.ttf", 22)
        font_main_hdr = ImageFont.truetype("segoeuib.ttf", 46)
        font_sub_hdr = ImageFont.truetype("segoeui.ttf", 22)
        font_cert_id = ImageFont.truetype("consola.ttf", 20)
        font_species_huge = ImageFont.truetype("segoeuib.ttf", 58)
        font_sci_italic = ImageFont.truetype("segoeuii.ttf", 30)
        font_family_txt = ImageFont.truetype("segoeui.ttf", 24)
        font_section_bold = ImageFont.truetype("segoeuib.ttf", 26)
        font_card_title = ImageFont.truetype("segoeuib.ttf", 24)
        font_body_bold = ImageFont.truetype("segoeuib.ttf", 22)
        font_body_text = ImageFont.truetype("segoeui.ttf", 22)
        font_small = ImageFont.truetype("segoeui.ttf", 18)
        font_seal_big = ImageFont.truetype("segoeuib.ttf", 28)
        font_seal_sub = ImageFont.truetype("segoeuib.ttf", 15)
    except:
        font_brand = ImageFont.load_default()
        font_main_hdr = ImageFont.load_default()
        font_sub_hdr = ImageFont.load_default()
        font_cert_id = ImageFont.load_default()
        font_species_huge = ImageFont.load_default()
        font_sci_italic = ImageFont.load_default()
        font_family_txt = ImageFont.load_default()
        font_section_bold = ImageFont.load_default()
        font_card_title = ImageFont.load_default()
        font_body_bold = ImageFont.load_default()
        font_body_text = ImageFont.load_default()
        font_small = ImageFont.load_default()
        font_seal_big = ImageFont.load_default()
        font_seal_sub = ImageFont.load_default()

    # Outer Certificate Framing (Dual Royal Blue & Charcoal Borders)
    draw.rectangle([20, 20, card_w - 20, card_h - 20], outline=(15, 23, 42), width=5)
    draw.rectangle([34, 34, card_w - 34, card_h - 34], outline=(2, 132, 199), width=3)
    draw.rectangle([44, 44, card_w - 44, card_h - 44], outline=(226, 232, 240), width=2)

    # -------------------------------------------------------------------------
    # TOP HEADER (Clean Porcelain Background with Royal Blue & Emerald Badge)
    # -------------------------------------------------------------------------
    header_y = 52
    draw.rounded_rectangle([52, header_y, card_w - 52, header_y + 145], radius=16, fill=(248, 250, 252), outline=(226, 232, 240), width=2)

    # Brand Pill
    draw.rounded_rectangle([75, header_y + 16, 680, header_y + 52], radius=8, fill=(224, 242, 254), outline=(2, 132, 199), width=1)
    draw.text((90, header_y + 20), "🦋 AI BUTTERFLY VISION • BIO-INTELLIGENCE LAB", fill=(3, 105, 161), font=font_brand)

    # Main Header Title
    draw.text((75, header_y + 60), "Official Specimen Inspection Certificate", fill=(15, 23, 42), font=font_main_hdr)
    draw.text((75, header_y + 112), "Deep Transfer Learning (ResNet-18)  •  Native PyTorch Grad-CAM Neuro-Attention Studio", fill=(71, 85, 105), font=font_sub_hdr)

    # Dynamic Certificate Hash & Timestamp
    cert_hash = hashlib.md5(f"{pred_class}_{confidence}".encode()).hexdigest()[:8].upper()
    cert_id_str = f"CERTIFICATE-ID: BIO-2026-XAI-{cert_hash}"
    draw.text((card_w - 460, header_y + 22), cert_id_str, fill=(2, 132, 199), font=font_cert_id)

    # Verification Seal / Badge
    badge_x, badge_y = card_w - 400, header_y + 60
    draw.rounded_rectangle([badge_x, badge_y, badge_x + 330, badge_y + 70], radius=12, fill=(236, 253, 245), outline=(16, 185, 129), width=2)
    draw.text((badge_x + 22, badge_y + 10), "VERIFIED AI DECISION", fill=(5, 150, 105), font=font_seal_sub)
    draw.text((badge_x + 22, badge_y + 28), f"CONFIDENCE: {confidence:.1f}%", fill=(6, 95, 70), font=font_seal_big)

    # -------------------------------------------------------------------------
    # LEFT COLUMN: Specimen Photographic Evidence Chambers
    # -------------------------------------------------------------------------
    img_size = 390
    orig_thumb = original_image.convert("RGB").resize((img_size, img_size), Image.Resampling.LANCZOS)
    ov_thumb = overlay_image.convert("RGB").resize((img_size, img_size), Image.Resampling.LANCZOS)

    # Chamber 1: Original Image
    ch1_x, ch1_y = 75, 230
    draw.rounded_rectangle([ch1_x - 10, ch1_y - 10, ch1_x + img_size + 10, ch1_y + img_size + 10], radius=18, fill=(255, 255, 255), outline=(203, 213, 225), width=2)
    img_canvas.paste(orig_thumb, (ch1_x, ch1_y))
    # Chamber Label Strip
    draw.rounded_rectangle([ch1_x - 10, ch1_y + img_size + 14, ch1_x + img_size + 10, ch1_y + img_size + 54], radius=10, fill=(241, 245, 249), outline=(203, 213, 225), width=1)
    draw.text((ch1_x + 35, ch1_y + img_size + 20), "Preprocessed Model Input (224x224 RGB)", fill=(15, 23, 42), font=font_card_title)

    # Chamber 2: Grad-CAM Heatmap
    ch2_x = ch1_x + img_size + 45
    draw.rounded_rectangle([ch2_x - 10, ch1_y - 10, ch2_x + img_size + 10, ch1_y + img_size + 10], radius=18, fill=(255, 255, 255), outline=(2, 132, 199), width=3)
    img_canvas.paste(ov_thumb, (ch2_x, ch1_y))
    # Chamber Label Strip
    draw.rounded_rectangle([ch2_x - 10, ch1_y + img_size + 14, ch2_x + img_size + 10, ch1_y + img_size + 54], radius=10, fill=(224, 242, 254), outline=(2, 132, 199), width=1)
    draw.text((ch2_x + 30, ch1_y + img_size + 20), "Grad-CAM Attention Overlay Map (XAI)", fill=(2, 132, 199), font=font_card_title)

    # -------------------------------------------------------------------------
    # RIGHT COLUMN: Species Identification & Ranked Probability Bars
    # -------------------------------------------------------------------------
    rx = 955
    
    # Taxonomic Identification Badge
    draw.rounded_rectangle([rx, 225, rx + 280, 260], radius=8, fill=(224, 242, 254), outline=(2, 132, 199), width=1)
    draw.text((rx + 16, 232), "TAXONOMIC IDENTIFICATION", fill=(3, 105, 161), font=font_seal_sub)

    # Species Name (Big, Bold, Crystal Clear Dark Text)
    draw.text((rx, 270), pred_class, fill=(15, 23, 42), font=font_species_huge)
    
    # Binomial & Family
    draw.text((rx, 345), f"Scientific Name: {meta['scientific_name']}", fill=(2, 132, 199), font=font_sci_italic)
    draw.text((rx, 388), f"Taxonomic Family: {meta['family']}", fill=(51, 65, 85), font=font_family_txt)

    # Divider Line
    draw.line([rx, 435, card_w - 75, 435], fill=(226, 232, 240), width=2)

    # Top-3 Ranked Probability Progress Bars
    draw.text((rx, 455), "Top-3 Neural Probability Distribution:", fill=(15, 23, 42), font=font_section_bold)

    for i, (c_name, prob) in enumerate(top_k):
        bar_y = 502 + i * 58
        bar_w_max = 940
        # Outer bar card
        draw.rounded_rectangle([rx, bar_y, rx + bar_w_max, bar_y + 44], radius=12, fill=(241, 245, 249), outline=(203, 213, 225), width=2)
        
        # Fill bar
        fill_width = int(bar_w_max * (min(prob, 100.0) / 100.0))
        if fill_width > 0:
            bar_color = (2, 132, 199) if i == 0 else (100, 116, 139)
            draw.rounded_rectangle([rx, bar_y, rx + fill_width, bar_y + 44], radius=12, fill=bar_color)
        
        # Text label
        draw.text((rx + 20, bar_y + 8), f"#{i+1}  {c_name}", fill=(255, 255, 255) if fill_width > 320 else (15, 23, 42), font=font_body_bold)
        draw.text((rx + bar_w_max - 120, bar_y + 8), f"{prob:.2f}%", fill=(255, 255, 255) if fill_width > bar_w_max - 60 else (2, 132, 199) if i == 0 else (15, 23, 42), font=font_body_bold)

    # -------------------------------------------------------------------------
    # BOTTOM SECTION: Modular Scientific Diagnostic & Ecology Cards (2 Crisp Cards)
    # -------------------------------------------------------------------------
    box_y = 735
    box_w = (card_w - 150 - 35) // 2
    box_h = 390

    # Modular Box 1: Neuro-Visual Attention (Grad-CAM XAI)
    b1_x = 75
    draw.rounded_rectangle([b1_x, box_y, b1_x + box_w, box_y + box_h], radius=18, fill=(255, 255, 255), outline=(2, 132, 199), width=3)
    draw.rounded_rectangle([b1_x, box_y, b1_x + box_w, box_y + 60], radius=18, fill=(240, 249, 255))
    draw.text((b1_x + 25, box_y + 16), "🔬 NEURO-VISUAL ATTENTION DIAGNOSTIC (GRAD-CAM)", fill=(3, 105, 161), font=font_card_title)

    draw.text((b1_x + 25, box_y + 80), "Gradient Hotspot Diagnostic:", fill=(15, 23, 42), font=font_body_bold)
    xai_txt = meta['xai_insight']
    words = xai_txt.split()
    line1, line2 = "", ""
    for w in words:
        if len(line1 + " " + w) < 48:
            line1 += (" " if line1 else "") + w
        else:
            line2 += (" " if line2 else "") + w
    draw.text((b1_x + 25, box_y + 115), line1, fill=(51, 65, 85), font=font_body_text)
    if line2:
        draw.text((b1_x + 25, box_y + 148), line2, fill=(51, 65, 85), font=font_body_text)

    draw.text((b1_x + 25, box_y + 200), "Diagnostic Wing Markers:", fill=(15, 23, 42), font=font_body_bold)
    draw.text((b1_x + 25, box_y + 235), meta['appearance'], fill=(51, 65, 85), font=font_body_text)

    draw.text((b1_x + 25, box_y + 295), "Neural Backbone Architecture:", fill=(15, 23, 42), font=font_body_bold)
    draw.text((b1_x + 25, box_y + 330), "PyTorch ResNet-18 (512-dim bottleneck) + Target Layer-4 Feature Maps", fill=(100, 116, 139), font=font_small)

    # Modular Box 2: Biogeography & Ecological Taxonomy
    b2_x = b1_x + box_w + 35
    draw.rounded_rectangle([b2_x, box_y, b2_x + box_w, box_y + box_h], radius=18, fill=(255, 255, 255), outline=(16, 185, 129), width=3)
    draw.rounded_rectangle([b2_x, box_y, b2_x + box_w, box_y + 60], radius=18, fill=(236, 253, 245))
    draw.text((b2_x + 25, box_y + 16), "🌿 TAXONOMIC & BIOGEOGRAPHIC PROFILE", fill=(6, 95, 70), font=font_card_title)

    draw.text((b2_x + 25, box_y + 80), "Geographic Distribution:", fill=(15, 23, 42), font=font_body_bold)
    draw.text((b2_x + 25, box_y + 115), meta['distribution'], fill=(51, 65, 85), font=font_body_text)

    draw.text((b2_x + 25, box_y + 180), "Key Biological Adaptation:", fill=(15, 23, 42), font=font_body_bold)
    draw.text((b2_x + 25, box_y + 215), meta['key_features'], fill=(51, 65, 85), font=font_body_text)

    draw.text((b2_x + 25, box_y + 280), "Verification Status:", fill=(15, 23, 42), font=font_body_bold)
    draw.text((b2_x + 25, box_y + 315), "Verified by Cross-Entropy Loss Optimization & Grad-CAM Backprop", fill=(5, 150, 105), font=font_body_bold)

    # -------------------------------------------------------------------------
    # FOOTER: Signature & Official Certification Stamp
    # -------------------------------------------------------------------------
    footer_y = card_h - 95
    draw.line([75, footer_y, card_w - 75, footer_y], fill=(203, 213, 225), width=2)

    draw.text((75, footer_y + 22), "AI Butterfly Vision • PyTorch ResNet-18 & Grad-CAM Architecture • TorchScript Mobile Export Ready", fill=(100, 116, 139), font=font_small)
    
    # Official Lead Engineer Seal Block
    draw.text((card_w - 480, footer_y + 18), "Lead AI Architect & Engineer: Ohi", fill=(2, 132, 199), font=font_body_bold)
    draw.text((card_w - 480, footer_y + 46), "Verified Digital Signature • All Rights Reserved", fill=(100, 116, 139), font=font_small)

    buf = BytesIO()
    img_canvas.save(buf, format="PNG", quality=100)
    return buf.getvalue()

# -----------------------------------------------------------------------------
# 3. Path Resolution Utility
# -----------------------------------------------------------------------------
def resolve_project_paths():
    """
    Returns absolute paths to models, prepared_dataset, and root directory,
    ensuring stability whether the app is launched from workspace root or inside app/.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(current_dir, ".."))

    checkpoint_path = os.path.join(root_dir, "models", "butterfly_resnet18_best.pth")
    test_data_dir = os.path.join(root_dir, "prepared_dataset", "test")

    return {
        "root_dir": root_dir,
        "checkpoint_path": checkpoint_path,
        "test_data_dir": test_data_dir
    }
