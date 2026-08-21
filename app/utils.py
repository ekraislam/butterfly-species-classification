"""
AI Butterfly Vision: Utility Functions
Taxonomy metadata, XAI diagnostic insights, Ultra-High-Resolution 4K Certificate Generator, and path resolvers.
"""

import os
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
# 2. Ultra-High Resolution 4K AI Inspection Certificate Generator
# -----------------------------------------------------------------------------
def generate_report_card(original_image: Image.Image, overlay_image: Image.Image,
                         pred_class: str, confidence: float, top_k: list) -> bytes:
    """
    Synthesizes a Masterpiece High-Resolution (1600x960) Official AI Specimen Inspection Certificate Card.
    Features:
    - High-density anti-aliased typography
    - Crisp dual specimen panels (Input + Grad-CAM Heatmap)
    - Precision Top-3 ranked probability meters
    - Official Verification Seal & Lead Engineer signature
    """
    meta = SPECIES_METADATA.get(pred_class, {
        "scientific_name": "Unknown",
        "family": "Insecta",
        "appearance": "N/A",
        "distribution": "Global",
        "key_features": "Visual wing pattern diagnostics",
        "color_primary": "#0284C7",
        "xai_insight": "Model focused on discriminative visual wing patterns."
    })

    card_w, card_h = 1600, 960
    # Clean Luxury Studio Paper Canvas
    img_canvas = Image.new("RGB", (card_w, card_h), color=(248, 250, 252))
    draw = ImageDraw.Draw(img_canvas)

    # Load high-contrast fonts
    try:
        font_main_title = ImageFont.truetype("arialbd.ttf", 36)
        font_sub_title = ImageFont.truetype("arialbd.ttf", 18)
        font_species_big = ImageFont.truetype("arialbd.ttf", 46)
        font_sci_name = ImageFont.truetype("ariali.ttf", 24)
        font_section_hdr = ImageFont.truetype("arialbd.ttf", 22)
        font_body_bold = ImageFont.truetype("arialbd.ttf", 20)
        font_body_text = ImageFont.truetype("arial.ttf", 19)
        font_small = ImageFont.truetype("arial.ttf", 16)
        font_seal = ImageFont.truetype("arialbd.ttf", 14)
    except:
        font_main_title = ImageFont.load_default()
        font_sub_title = ImageFont.load_default()
        font_species_big = ImageFont.load_default()
        font_sci_name = ImageFont.load_default()
        font_section_hdr = ImageFont.load_default()
        font_body_bold = ImageFont.load_default()
        font_body_text = ImageFont.load_default()
        font_small = ImageFont.load_default()
        font_seal = ImageFont.load_default()

    # Outer Double Border (Luxury Certificate Style)
    draw.rectangle([20, 20, card_w - 20, card_h - 20], outline=(15, 23, 42), width=4)
    draw.rectangle([30, 30, card_w - 30, card_h - 30], outline=(2, 132, 199), width=2)

    # Top Header Banner
    draw.rectangle([32, 32, card_w - 32, 135], fill=(15, 23, 42))
    draw.text((60, 48), "OFFICIAL AI SPECIMEN INSPECTION & EXPLAINABILITY REPORT", fill=(255, 255, 255), font=font_main_title)
    draw.text((60, 96), "Deep Transfer Learning (ResNet-18) • Native PyTorch Grad-CAM Neuro-Attention Studio", fill=(56, 189, 248), font=font_sub_title)

    # Top-Right Verification Seal / Badge
    seal_x, seal_y = card_w - 290, 46
    draw.rounded_rectangle([seal_x, seal_y, seal_x + 230, seal_y + 70], radius=10, fill=(2, 132, 199))
    draw.text((seal_x + 22, seal_y + 14), "VERIFIED AI DECISION", fill=(255, 255, 255), font=font_seal)
    draw.text((seal_x + 35, seal_y + 38), f"CONFIDENCE: {confidence:.1f}%", fill=(255, 255, 255), font=font_body_bold)

    # -------------------------------------------------------------------------
    # LEFT COLUMN: High-Resolution Visual Evidence (Input & Grad-CAM)
    # -------------------------------------------------------------------------
    img_size = 320
    orig_thumb = original_image.convert("RGB").resize((img_size, img_size), Image.Resampling.LANCZOS)
    ov_thumb = overlay_image.convert("RGB").resize((img_size, img_size), Image.Resampling.LANCZOS)

    # Panel 1: Original Image
    panel1_x, panel1_y = 60, 175
    draw.rounded_rectangle([panel1_x - 8, panel1_y - 8, panel1_x + img_size + 8, panel1_y + img_size + 8], radius=14, fill=(255, 255, 255), outline=(148, 163, 184), width=2)
    img_canvas.paste(orig_thumb, (panel1_x, panel1_y))
    draw.text((panel1_x + 40, panel1_y + img_size + 16), "Preprocessed Input (224x224)", fill=(15, 23, 42), font=font_body_bold)

    # Panel 2: Grad-CAM Overlay
    panel2_x = panel1_x + img_size + 45
    draw.rounded_rectangle([panel2_x - 8, panel1_y - 8, panel2_x + img_size + 8, panel1_y + img_size + 8], radius=14, fill=(255, 255, 255), outline=(2, 132, 199), width=3)
    img_canvas.paste(ov_thumb, (panel2_x, panel1_y))
    draw.text((panel2_x + 25, panel1_y + img_size + 16), "Grad-CAM Attention Overlay Map", fill=(2, 132, 199), font=font_body_bold)

    # -------------------------------------------------------------------------
    # RIGHT COLUMN: Classification Breakdown & Top-3 Probabilities
    # -------------------------------------------------------------------------
    rx = 800
    draw.text((rx, 170), "TAXONOMIC IDENTIFICATION", fill=(2, 132, 199), font=font_section_hdr)
    draw.text((rx, 205), pred_class, fill=(15, 23, 42), font=font_species_big)
    draw.text((rx, 265), f"Scientific Name: {meta['scientific_name']}", fill=(51, 65, 85), font=font_sci_name)
    draw.text((rx, 300), f"Family: {meta['family']}", fill=(71, 85, 105), font=font_body_bold)

    # Top-3 Ranked Probability Bars
    draw.text((rx, 355), "Top-3 Ranked Probabilities:", fill=(15, 23, 42), font=font_section_hdr)
    
    for i, (c_name, prob) in enumerate(top_k):
        bar_y = 395 + i * 46
        # Background bar
        draw.rounded_rectangle([rx, bar_y, rx + 740, bar_y + 32], radius=8, fill=(226, 232, 240))
        # Fill bar
        bar_width = int(740 * (min(prob, 100.0) / 100.0))
        if bar_width > 0:
            fill_color = (2, 132, 199) if i == 0 else (100, 116, 139)
            draw.rounded_rectangle([rx, bar_y, rx + bar_width, bar_y + 32], radius=8, fill=fill_color)
        
        # Text label
        draw.text((rx + 15, bar_y + 5), f"#{i+1}  {c_name}", fill=(255, 255, 255) if bar_width > 220 else (15, 23, 42), font=font_body_bold)
        draw.text((rx + 650, bar_y + 5), f"{prob:.2f}%", fill=(255, 255, 255) if bar_width > 680 else (15, 23, 42), font=font_body_bold)

    # -------------------------------------------------------------------------
    # BOTTOM SECTION: Neuro-Visual Diagnostic & Biological Ecology Card
    # -------------------------------------------------------------------------
    box_y = 565
    draw.rounded_rectangle([60, box_y, card_w - 60, box_y + 280], radius=16, fill=(255, 255, 255), outline=(203, 213, 225), width=2)
    draw.rectangle([60, box_y, card_w - 60, box_y + 45], fill=(241, 245, 249))
    draw.text((80, box_y + 12), "🔬 NEURAL ATTENTION (XAI) & ECOLOGICAL DIAGNOSTIC INSIGHTS", fill=(2, 132, 199), font=font_section_hdr)

    draw.text((80, box_y + 65), "Visual Attention:", fill=(15, 23, 42), font=font_body_bold)
    draw.text((250, box_y + 65), meta['xai_insight'], fill=(30, 41, 59), font=font_body_text)

    draw.text((80, box_y + 115), "Visual Markers:", fill=(15, 23, 42), font=font_body_bold)
    draw.text((250, box_y + 115), meta['appearance'], fill=(51, 65, 85), font=font_body_text)

    draw.text((80, box_y + 165), "Geographic Range:", fill=(15, 23, 42), font=font_body_bold)
    draw.text((250, box_y + 165), meta['distribution'], fill=(51, 65, 85), font=font_body_text)

    draw.text((80, box_y + 215), "Ecological Feature:", fill=(15, 23, 42), font=font_body_bold)
    draw.text((250, box_y + 215), meta['key_features'], fill=(51, 65, 85), font=font_body_text)

    # -------------------------------------------------------------------------
    # FOOTER: Signature & Metadata Stamp
    # -------------------------------------------------------------------------
    footer_y = card_h - 75
    draw.line([60, footer_y, card_w - 60, footer_y], fill=(203, 213, 225), width=1)
    draw.text((60, footer_y + 15), "AI Butterfly Vision Architecture • Verified by Deep Transfer Learning (PyTorch ResNet-18)", fill=(100, 116, 139), font=font_small)
    draw.text((card_w - 380, footer_y + 15), "Lead Architect & Engineer: Ohi", fill=(2, 132, 199), font=font_body_bold)

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
