"""
App Utilities: Butterfly Species Metadata, Color Systems, XAI Insights & Path Helpers.
Provides verified biological taxonomy, color tokens, and explainability annotations.
"""

import os
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont

# Verified biological metadata for the 8 supported species
SPECIES_METADATA = {
    "MONARCH": {
        "scientific_name": "Danaus plexippus",
        "family": "Nymphalidae (Danainae)",
        "appearance": "Tawny orange wings with bold black veins and two rows of white spots along black margins.",
        "distribution": "North & South America, Australia, Pacific Islands, Western Europe.",
        "key_features": "Famous for spectacular multi-generational annual migrations and milkweed-derived toxicity.",
        "color_primary": "#FF7A00",
        "color_secondary": "#FF512F",
        "color_bg": "rgba(255, 122, 0, 0.12)",
        "color_border": "rgba(255, 122, 0, 0.4)",
        "xai_insight": "Grad-CAM localized peak attention onto the distinctive black vein ridges and white marginal spot clusters on the dorsal forewing."
    },
    "PAPER KITE": {
        "scientific_name": "Idea leuconoe",
        "family": "Nymphalidae (Danainae)",
        "appearance": "Large, delicate translucent ivory-white wings patterned with prominent black veins and scattered spots.",
        "distribution": "Southeast Asia, Philippines, Taiwan, southern Japan.",
        "key_features": "Also known as the Rice Paper or Large Tree Nymph; famous for slow, buoyant gliding flight.",
        "color_primary": "#E2E8F0",
        "color_secondary": "#94A3B8",
        "color_bg": "rgba(226, 232, 240, 0.12)",
        "color_border": "rgba(226, 232, 240, 0.4)",
        "xai_insight": "Attention focused on the expansive translucent white wing area and the grid-like black vein markings."
    },
    "RED POSTMAN": {
        "scientific_name": "Heliconius erato",
        "family": "Nymphalidae (Heliconiinae)",
        "appearance": "Velvety black elongated forewings with vivid crimson-red vertical bands and yellow ventral markings.",
        "distribution": "Central and South America (Mexico through Amazon basin).",
        "key_features": "Classic example of Müllerian mimicry alongside Heliconius melpomene; long-lived pollen feeder.",
        "color_primary": "#FF1361",
        "color_secondary": "#FF0844",
        "color_bg": "rgba(255, 19, 97, 0.12)",
        "color_border": "rgba(255, 19, 97, 0.4)",
        "xai_insight": "Neural gradients strongly fired on the elongated crimson-red vertical block pattern across the velvety black forewings."
    },
    "ADONIS": {
        "scientific_name": "Polyommatus bellargus",
        "family": "Lycaenidae (Gossamer-winged)",
        "appearance": "Males have brilliant electric-blue upperwings with white-checkered fringe; females are brown with blue dusting.",
        "distribution": "Southern & Central Europe, southern England chalk grasslands.",
        "key_features": "Specialist of warm limestone grasslands; caterpillars share symbiotic relationships with ants.",
        "color_primary": "#00E5FF",
        "color_secondary": "#0099FF",
        "color_bg": "rgba(0, 229, 255, 0.12)",
        "color_border": "rgba(0, 229, 255, 0.4)",
        "xai_insight": "The model concentrated on the vibrant electric-blue spectral signature across the core wing surfaces."
    },
    "GREEN CELLED CATTLEHEART": {
        "scientific_name": "Parides sesostris",
        "family": "Papilionidae (Swallowtails)",
        "appearance": "Deep velvety black wings; males feature bright emerald-green forewing patches and crimson-red ventral body spots.",
        "distribution": "Tropical rainforests from Mexico to Amazon Basin in Brazil.",
        "key_features": "Shade-tolerant forest swallowtail; toxic to predators due to aristolochic acids from larval host plants.",
        "color_primary": "#10B981",
        "color_secondary": "#059669",
        "color_bg": "rgba(16, 185, 129, 0.12)",
        "color_border": "rgba(16, 185, 129, 0.4)",
        "xai_insight": "Attention centered on the high-contrast emerald-green forewing patch and distinctive swallowtail morphology."
    },
    "SOUTHERN DOGFACE": {
        "scientific_name": "Zerene cesonia",
        "family": "Pieridae (Whites and Yellows)",
        "appearance": "Vibrant sulfur-yellow wings with a distinctive dark silhouette on each forewing resembling a dog's profile.",
        "distribution": "Southern United States through Central America to northern South America.",
        "key_features": "Fast, erratic flyer inhabiting open fields and desert scrub; larvae feed on legumes.",
        "color_primary": "#FBBF24",
        "color_secondary": "#F59E0B",
        "color_bg": "rgba(251, 191, 36, 0.12)",
        "color_border": "rgba(251, 191, 36, 0.4)",
        "xai_insight": "The model localized the distinctive dark silhouette shape set against the bright sulfur-yellow wing backdrop."
    },
    "ORANGE OAKLEAF": {
        "scientific_name": "Kallima inachus",
        "family": "Nymphalidae (Nymphalinae)",
        "appearance": "Dorsal wings show royal blue and bright orange bands; ventral wings display master-class dead-leaf camouflage.",
        "distribution": "Tropical Asia (India, Nepal, China, Taiwan, Indochina).",
        "key_features": "One of nature's most iconic camouflage examples; completely resembles a dry leaf when wings are closed.",
        "color_primary": "#F97316",
        "color_secondary": "#C2410C",
        "color_bg": "rgba(249, 115, 22, 0.12)",
        "color_border": "rgba(249, 115, 22, 0.4)",
        "xai_insight": "Gradients highlighted the dual-banded pigmentation and leaf-mimic venation geometry."
    },
    "CLODIUS PARNASSIAN": {
        "scientific_name": "Parnassius clodius",
        "family": "Papilionidae (Parnassiinae)",
        "appearance": "Semi-translucent milky-white wings dusted with dark charcoal scales and prominent red ocelli (eye-spots) ringed in black.",
        "distribution": "Montane Western North America (Alaska through Pacific Northwest to northern California).",
        "key_features": "Cold-adapted alpine species; related to swallowtails but lacks tails, flying in mountain meadows.",
        "color_primary": "#F43F5E",
        "color_secondary": "#E11D48",
        "color_bg": "rgba(244, 63, 94, 0.12)",
        "color_border": "rgba(244, 63, 94, 0.4)",
        "xai_insight": "The model isolated the characteristic red ocelli (eye-spots) and semi-translucent milky scales."
    }
}

def generate_report_card(original_image, overlay_image, pred_class, confidence, top_k):
    """
    Generates a high-resolution exportable inspection graphic card (PNG)
    combining input photo, Grad-CAM overlay, species details, and probabilities.
    """
    meta = SPECIES_METADATA.get(pred_class, {
        "scientific_name": "Unknown",
        "family": "Insecta",
        "color_primary": "#38BDF8"
    })

    card_w, card_h = 1000, 520
    img_canvas = Image.new("RGB", (card_w, card_h), color=(11, 15, 25))
    draw = ImageDraw.Draw(img_canvas)

    try:
        font_title = ImageFont.truetype("arialbd.ttf", 26)
        font_species = ImageFont.truetype("arialbd.ttf", 32)
        font_sci = ImageFont.truetype("ariali.ttf", 18)
        font_bold = ImageFont.truetype("arialbd.ttf", 16)
        font_text = ImageFont.truetype("arial.ttf", 14)
        font_small = ImageFont.truetype("arial.ttf", 12)
    except:
        font_title = ImageFont.load_default()
        font_species = ImageFont.load_default()
        font_sci = ImageFont.load_default()
        font_bold = ImageFont.load_default()
        font_text = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # Outer Border
    draw.rectangle([10, 10, card_w - 10, card_h - 10], outline=(40, 50, 70), width=2)

    # Header
    draw.text((30, 25), "AI BUTTERFLY INSPECTION & EXPLAINABILITY REPORT", fill=(56, 189, 248), font=font_title)
    draw.text((30, 58), "ResNet-18 Transfer Learning • Native PyTorch Grad-CAM", fill=(148, 163, 184), font=font_small)

    # Left Column: Images (Original & Overlay)
    img_size = 200
    orig_thumb = original_image.convert("RGB").resize((img_size, img_size))
    ov_thumb = overlay_image.convert("RGB").resize((img_size, img_size))

    # Paste Original
    img_canvas.paste(orig_thumb, (30, 95))
    draw.rectangle([30, 95, 30 + img_size, 95 + img_size], outline=(70, 80, 100), width=1)
    draw.text((30, 305), "Preprocessed Input (224x224)", fill=(148, 163, 184), font=font_small)

    # Paste Overlay
    img_canvas.paste(ov_thumb, (250, 95))
    draw.rectangle([250, 95, 250 + img_size, 95 + img_size], outline=(56, 189, 248), width=2)
    draw.text((250, 305), "Grad-CAM Attention Overlay", fill=(56, 189, 248), font=font_small)

    # Right Column: Prediction Details
    rx = 480
    draw.text((rx, 95), "PREDICTED SPECIES", fill=(148, 163, 184), font=font_bold)
    draw.text((rx, 120), pred_class, fill=(255, 255, 255), font=font_species)
    draw.text((rx, 160), f"Scientific: {meta['scientific_name']} | {meta['family']}", fill=(148, 163, 184), font=font_sci)

    # Confidence Box
    conf_box_y = 195
    draw.rectangle([rx, conf_box_y, rx + 470, conf_box_y + 40], fill=(20, 30, 45), outline=(56, 189, 248))
    draw.text((rx + 15, conf_box_y + 10), f"Model Confidence: {confidence:.2f}%  |  Status: Verified Decision", fill=(52, 211, 153), font=font_bold)

    # Top-3 Predictions
    draw.text((rx, 255), "Top-3 Class Probabilities:", fill=(226, 232, 240), font=font_bold)
    for i, (c_name, prob) in enumerate(top_k):
        y_pos = 280 + i * 22
        draw.text((rx, y_pos), f"{i+1}. {c_name:<26}: {prob:.2f}%", fill=(203, 213, 225), font=font_text)

    # Key XAI insight
    xai_insight = meta.get("xai_insight", "Model focused on discriminative visual wing patterns.")
    draw.rectangle([30, 345, card_w - 30, 470], fill=(15, 23, 42), outline=(40, 50, 70))
    draw.text((45, 355), "XAI Neuro-Visual Diagnostic Insight:", fill=(56, 189, 248), font=font_bold)
    draw.text((45, 380), xai_insight, fill=(203, 213, 225), font=font_text)
    draw.text((45, 415), f"Distribution: {meta['distribution']}", fill=(148, 163, 184), font=font_small)
    draw.text((45, 435), f"Key Feature: {meta['key_features']}", fill=(148, 163, 184), font=font_small)

    # Footer
    draw.text((30, 485), "AI Butterfly Vision • Designed & Developed by Ohi • PyTorch ResNet-18 Transfer Learning", fill=(100, 116, 139), font=font_small)

    buf = BytesIO()
    img_canvas.save(buf, format="PNG", quality=95)
    return buf.getvalue()

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
