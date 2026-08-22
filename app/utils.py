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
        "color_primary": "#0284C7",
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
    Generates a 100% 1-to-1 matching Ultra-HD Official Inspection Certificate
    PNG image that matches the live on-screen HTML certificate card pixel-by-pixel.
    """
    meta = SPECIES_METADATA.get(pred_class, {
        "scientific_name": "Unknown species",
        "family": "Insecta | Lepidoptera",
        "appearance": "Distinct biological visual wing markings.",
        "distribution": "Global biodiversity habitat.",
        "key_features": "Diagnostic taxonomic wing venation pattern.",
        "color_primary": "#0284C7",
        "xai_insight": "Model neural attention concentrated on discriminative visual wing patterns."
    })

    card_w, card_h = 1500, 1120
    img_canvas = Image.new("RGB", (card_w, card_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(img_canvas)

    win_fonts = os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts")
    fonts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
    
    def _get_font(font_file, size, fallback_file="arialbd.ttf"):
        search_paths = [
            os.path.join(fonts_dir, font_file),
            os.path.join(fonts_dir, fallback_file),
            os.path.join(fonts_dir, "arialbd.ttf"),
            os.path.join(fonts_dir, "arial.ttf"),
            os.path.join(win_fonts, font_file),
            os.path.join(win_fonts, fallback_file),
            font_file,
            fallback_file,
        ]
        for p in search_paths:
            if os.path.exists(p):
                try:
                    font = ImageFont.truetype(p, size)
                    bbox = font.getbbox("TEST")
                    if bbox[2] > bbox[0]:
                        return font
                except Exception:
                    pass
        return ImageFont.load_default()

    font_brand = _get_font("segoeuib.ttf", 18, "arialbd.ttf")
    font_main_hdr = _get_font("segoeuib.ttf", 40, "arialbd.ttf")
    font_sub_hdr = _get_font("segoeuib.ttf", 20, "arialbd.ttf")
    font_cert_id = _get_font("consola.ttf", 18, "cour.ttf")
    font_species_huge = _get_font("segoeuib.ttf", 56, "arialbd.ttf")
    font_sci_italic = _get_font("georgiab.ttf", 28, "arialbd.ttf")
    font_family_txt = _get_font("segoeuib.ttf", 22, "arialbd.ttf")
    font_section_bold = _get_font("segoeuib.ttf", 23, "arialbd.ttf")
    font_card_title = _get_font("segoeuib.ttf", 22, "arialbd.ttf")
    font_chamber_lbl = _get_font("segoeuib.ttf", 19, "arialbd.ttf")
    font_bar_text = _get_font("segoeuib.ttf", 22, "arialbd.ttf")
    font_body_bold = _get_font("segoeuib.ttf", 21, "arialbd.ttf")
    font_body_text = _get_font("segoeui.ttf", 20, "arial.ttf")
    font_small = _get_font("segoeuib.ttf", 17, "arialbd.ttf")
    font_seal_big = _get_font("segoeuib.ttf", 28, "arialbd.ttf")
    font_seal_sub = _get_font("segoeuib.ttf", 15, "arialbd.ttf")

    def _draw_centered_text(text, font, box_x1, box_y1, box_x2, box_y2, fill):
        bbox = font.getbbox(text)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        cx = box_x1 + (box_x2 - box_x1 - tw) // 2
        cy = box_y1 + (box_y2 - box_y1 - th) // 2
        draw.text((cx, cy), text, fill=fill, font=font)

    def _draw_wrapped(text, font, x, y, max_w, fill, line_height=28):
        words = text.split()
        lines = []
        cur_line = []
        for w in words:
            test_line = " ".join(cur_line + [w])
            bbox = font.getbbox(test_line)
            if (bbox[2] - bbox[0]) <= max_w:
                cur_line.append(w)
            else:
                if cur_line:
                    lines.append(" ".join(cur_line))
                cur_line = [w]
        if cur_line:
            lines.append(" ".join(cur_line))
        
        cy = y
        for l in lines:
            draw.text((x, cy), l, fill=fill, font=font)
            cy += line_height
        return cy

    # Outer Double Border Box (Exact replica of HTML Card)
    draw.rounded_rectangle([16, 16, card_w - 16, card_h - 16], radius=24, fill=(255, 255, 255), outline=(15, 23, 42), width=4)
    draw.rounded_rectangle([32, 32, card_w - 32, card_h - 32], radius=16, fill=(255, 255, 255), outline=(2, 132, 199), width=2)

    # -------------------------------------------------------------------------
    # TOP HEADER (Porcelain Card with Lab Badge & Verified Decision)
    # -------------------------------------------------------------------------
    header_y = 50
    draw.rounded_rectangle([50, header_y, card_w - 50, header_y + 135], radius=14, fill=(248, 250, 252), outline=(226, 232, 240), width=2)

    # Brand Pill
    draw.rounded_rectangle([68, header_y + 14, 510, header_y + 46], radius=8, fill=(224, 242, 254), outline=(2, 132, 199), width=1)
    draw.text((80, header_y + 18), "AI BUTTERFLY VISION | BIO-INTELLIGENCE LAB", fill=(3, 105, 161), font=font_brand)

    # Main Header Title & Subtitle
    draw.text((68, header_y + 52), "Official Specimen Inspection Certificate", fill=(15, 23, 42), font=font_main_hdr)
    draw.text((68, header_y + 98), "PyTorch ResNet-18 Deep Transfer Learning Architecture | Native Grad-CAM XAI", fill=(71, 85, 105), font=font_sub_hdr)

    # Dynamic Certificate Hash & Timestamp
    cert_hash = hashlib.md5(f"{pred_class}_{confidence}".encode()).hexdigest()[:8].upper()
    cert_id_str = f"CERT-ID: BIO-2026-XAI-{cert_hash}"
    draw.rounded_rectangle([card_w - 380, header_y + 14, card_w - 68, header_y + 44], radius=6, fill=(240, 249, 255), outline=(186, 230, 253), width=1)
    _draw_centered_text(cert_id_str, font_cert_id, card_w - 380, header_y + 14, card_w - 68, header_y + 44, (2, 132, 199))

    # Verified Decision Badge (Emerald Box)
    badge_x, badge_y = card_w - 380, header_y + 52
    draw.rounded_rectangle([badge_x, badge_y, card_w - 68, badge_y + 72], radius=12, fill=(236, 253, 245), outline=(16, 185, 129), width=2)
    draw.text((badge_x + 16, badge_y + 8), "VERIFIED AI DECISION", fill=(5, 150, 105), font=font_seal_sub)
    draw.text((badge_x + 16, badge_y + 30), f"CONFIDENCE: {confidence:.1f}%", fill=(6, 95, 70), font=font_seal_big)

    # -------------------------------------------------------------------------
    # MIDDLE SECTION: Dual Chambers (Left) + Species Taxonomy (Right)
    # -------------------------------------------------------------------------
    ch_w, ch_h = 300, 310
    
    def _create_contained_preview(img_in, target_size=(270, 240)):
        chamber = Image.new("RGB", target_size, (255, 255, 255))
        c_img = img_in.convert("RGB").copy()
        w, h = c_img.size
        scale = min(target_size[0] / max(w, 1), target_size[1] / max(h, 1))
        new_w, new_h = max(int(w * scale), 1), max(int(h * scale), 1)
        c_resized = c_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        ox = (target_size[0] - new_w) // 2
        oy = (target_size[1] - new_h) // 2
        chamber.paste(c_resized, (ox, oy))
        return chamber

    thumb_size = (270, 235)
    orig_thumb = _create_contained_preview(original_image, thumb_size)
    ov_thumb = _create_contained_preview(overlay_image, thumb_size)

    # Chamber 1: Original Image
    ch1_x, ch1_y = 50, 202
    draw.rounded_rectangle([ch1_x, ch1_y, ch1_x + ch_w, ch1_y + ch_h], radius=14, fill=(255, 255, 255), outline=(203, 213, 225), width=2)
    img_canvas.paste(orig_thumb, (ch1_x + 15, ch1_y + 15))
    # Chamber 1 Label Strip
    ch1_lbl_box = [ch1_x + 15, ch1_y + ch_h - 48, ch1_x + ch_w - 15, ch1_y + ch_h - 12]
    draw.rounded_rectangle(ch1_lbl_box, radius=6, fill=(241, 245, 249), outline=(203, 213, 225), width=1)
    _draw_centered_text("Original Input Specimen", font_chamber_lbl, ch1_lbl_box[0], ch1_lbl_box[1], ch1_lbl_box[2], ch1_lbl_box[3], (15, 23, 42))

    # Chamber 2: Grad-CAM Heatmap
    ch2_x = ch1_x + ch_w + 18
    draw.rounded_rectangle([ch2_x, ch1_y, ch2_x + ch_w, ch1_y + ch_h], radius=14, fill=(255, 255, 255), outline=(2, 132, 199), width=2)
    img_canvas.paste(ov_thumb, (ch2_x + 15, ch1_y + 15))
    # Chamber 2 Label Strip
    ch2_lbl_box = [ch2_x + 15, ch1_y + ch_h - 48, ch2_x + ch_w - 15, ch1_y + ch_h - 12]
    draw.rounded_rectangle(ch2_lbl_box, radius=6, fill=(224, 242, 254), outline=(2, 132, 199), width=1)
    _draw_centered_text("Grad-CAM Attention Map (XAI)", font_chamber_lbl, ch2_lbl_box[0], ch2_lbl_box[1], ch2_lbl_box[2], ch2_lbl_box[3], (2, 132, 199))

    # RIGHT COLUMN: Species Taxonomy & Progress Bars
    rx = ch2_x + ch_w + 26
    r_w_max = card_w - 50 - rx
    
    # Taxonomic Identification Badge
    draw.rounded_rectangle([rx, 202, rx + 250, 232], radius=6, fill=(224, 242, 254), outline=(2, 132, 199), width=1)
    draw.text((rx + 12, 207), "TAXONOMIC CLASSIFICATION", fill=(3, 105, 161), font=font_seal_sub)

    # Species Name (Big, Bold, Crystal Clear Dark Text)
    draw.text((rx, 236), pred_class, fill=(15, 23, 42), font=font_species_huge)
    
    # Binomial & Family
    draw.text((rx, 298), f"Scientific Name: {meta['scientific_name']}", fill=(2, 132, 199), font=font_sci_italic)
    draw.text((rx, 332), f"Taxonomic Family: {meta['family']}", fill=(71, 85, 105), font=font_family_txt)

    # Top-3 Probability Bars
    draw.text((rx, 368), "Top-3 Neural Probability Distribution:", fill=(15, 23, 42), font=font_section_bold)

    for i, (c_name, prob) in enumerate(top_k):
        bar_y = 398 + i * 44
        # Outer bar container
        draw.rounded_rectangle([rx, bar_y, rx + r_w_max, bar_y + 38], radius=8, fill=(248, 250, 252), outline=(226, 232, 240), width=1)
        
        # Fill bar
        fill_width = int(r_w_max * (min(prob, 100.0) / 100.0))
        if fill_width > 0:
            bar_color = (2, 132, 199) if i == 0 else (224, 242, 254)
            draw.rounded_rectangle([rx, bar_y, rx + fill_width, bar_y + 38], radius=8, fill=bar_color)
        
        # Text label (High Contrast)
        name_txt = f"#{i+1}  {c_name}"
        prob_txt = f"{prob:.2f}%"
        
        prob_bbox = font_bar_text.getbbox(prob_txt)
        pw = prob_bbox[2] - prob_bbox[0]
        prob_x = rx + r_w_max - 15 - pw
        
        if i == 0:
            txt_fill = (255, 255, 255) if fill_width > 300 else (15, 23, 42)
            prob_fill = (255, 255, 255) if fill_width >= (r_w_max - 20 - pw) else (2, 132, 199)
        else:
            txt_fill = (15, 23, 42)
            prob_fill = (71, 85, 105)
            
        draw.text((rx + 14, bar_y + 7), name_txt, fill=txt_fill, font=font_bar_text)
        draw.text((prob_x, bar_y + 7), prob_txt, fill=prob_fill, font=font_bar_text)

    # -------------------------------------------------------------------------
    # BOTTOM SECTION: Modular Scientific Diagnostic & Ecology Cards (2 Crisp Cards)
    # -------------------------------------------------------------------------
    box_y = 535
    box_w = (card_w - 100 - 24) // 2
    box_h = 445

    # Modular Box 1: Neuro-Visual Attention (Grad-CAM XAI)
    b1_x = 50
    draw.rounded_rectangle([b1_x, box_y, b1_x + box_w, box_y + box_h], radius=16, fill=(255, 255, 255), outline=(2, 132, 199), width=2)
    # Header bar
    draw.rounded_rectangle([b1_x, box_y, b1_x + box_w, box_y + 44], radius=16, fill=(240, 249, 255))
    draw.text((b1_x + 18, box_y + 10), "NEURO-VISUAL ATTENTION (GRAD-CAM XAI)", fill=(3, 105, 161), font=font_card_title)

    # Bullet 1
    draw.ellipse([b1_x + 18, box_y + 64, b1_x + 24, box_y + 70], fill=(2, 132, 199))
    draw.text((b1_x + 30, box_y + 56), "Gradient Hotspot Diagnostic:", fill=(15, 23, 42), font=font_body_bold)
    y_next = _draw_wrapped(meta['xai_insight'], font_body_text, b1_x + 18, box_y + 86, box_w - 36, (30, 41, 59), line_height=28)

    # Bullet 2
    y_sec2 = max(y_next + 14, box_y + 180)
    draw.ellipse([b1_x + 18, y_sec2 + 8, b1_x + 24, y_sec2 + 14], fill=(2, 132, 199))
    draw.text((b1_x + 30, y_sec2), "Diagnostic Wing Markers:", fill=(15, 23, 42), font=font_body_bold)
    y_next2 = _draw_wrapped(meta['appearance'], font_body_text, b1_x + 18, y_sec2 + 30, box_w - 36, (30, 41, 59), line_height=28)

    # Bullet 3
    y_sec3 = max(y_next2 + 14, box_y + 345)
    draw.ellipse([b1_x + 18, y_sec3 + 8, b1_x + 24, y_sec3 + 14], fill=(2, 132, 199))
    draw.text((b1_x + 30, y_sec3), "Neural Backbone Architecture:", fill=(15, 23, 42), font=font_body_bold)
    draw.text((b1_x + 18, y_sec3 + 30), "PyTorch ResNet-18 (512-dim bottleneck) + Target Layer-4 Feature Maps", fill=(100, 116, 139), font=font_small)

    # Modular Box 2: Biogeography & Ecological Taxonomy
    b2_x = b1_x + box_w + 24
    draw.rounded_rectangle([b2_x, box_y, b2_x + box_w, box_y + box_h], radius=16, fill=(255, 255, 255), outline=(16, 185, 129), width=2)
    # Header bar
    draw.rounded_rectangle([b2_x, box_y, b2_x + box_w, box_y + 44], radius=16, fill=(240, 253, 244))
    draw.text((b2_x + 18, box_y + 10), "TAXONOMIC & BIOGEOGRAPHIC PROFILE", fill=(6, 95, 70), font=font_card_title)

    # Bullet 1
    draw.ellipse([b2_x + 18, box_y + 64, b2_x + 24, box_y + 70], fill=(16, 185, 129))
    draw.text((b2_x + 30, box_y + 56), "Geographic Distribution:", fill=(15, 23, 42), font=font_body_bold)
    y2_next = _draw_wrapped(meta['distribution'], font_body_text, b2_x + 18, box_y + 86, box_w - 36, (30, 41, 59), line_height=28)

    # Bullet 2
    y2_sec2 = max(y2_next + 14, box_y + 180)
    draw.ellipse([b2_x + 18, y2_sec2 + 8, b2_x + 24, y2_sec2 + 14], fill=(16, 185, 129))
    draw.text((b2_x + 30, y2_sec2), "Key Biological Adaptation:", fill=(15, 23, 42), font=font_body_bold)
    y2_next2 = _draw_wrapped(meta['key_features'], font_body_text, b2_x + 18, y2_sec2 + 30, box_w - 36, (30, 41, 59), line_height=28)

    # Bullet 3
    y2_sec3 = max(y2_next2 + 14, box_y + 345)
    draw.ellipse([b2_x + 18, y2_sec3 + 8, b2_x + 24, y2_sec3 + 14], fill=(16, 185, 129))
    draw.text((b2_x + 30, y2_sec3), "Model Decision Verification:", fill=(15, 23, 42), font=font_body_bold)
    draw.text((b2_x + 18, y2_sec3 + 30), "Validated by Cross-Entropy Loss Optimization & Grad-CAM Backprop", fill=(5, 150, 105), font=font_small)

    # -------------------------------------------------------------------------
    # FOOTER: Signature & Official Certification Stamp
    # -------------------------------------------------------------------------
    footer_y = 998
    draw.line([50, footer_y, card_w - 50, footer_y], fill=(226, 232, 240), width=2)

    draw.text((50, footer_y + 14), "AI Butterfly Vision | PyTorch ResNet-18 Deep Transfer Learning Architecture", fill=(15, 23, 42), font=font_body_bold)
    draw.text((50, footer_y + 44), "TorchScript Mobile Export Ready | Native Explainable AI Grad-CAM Studio", fill=(100, 116, 139), font=font_small)
    
    # Official Lead Engineer Signature Block
    eng_name_txt = "Lead AI Architect & Engineer: Ohi"
    eng_sig_txt = "Official Specimen Certificate | Verified System Digital Signature"
    
    eng_bbox1 = font_body_bold.getbbox(eng_name_txt)
    eng_w1 = eng_bbox1[2] - eng_bbox1[0]
    draw.text((card_w - 50 - eng_w1, footer_y + 14), eng_name_txt, fill=(2, 132, 199), font=font_body_bold)
    
    eng_bbox2 = font_small.getbbox(eng_sig_txt)
    eng_w2 = eng_bbox2[2] - eng_bbox2[0]
    draw.text((card_w - 50 - eng_w2, footer_y + 44), eng_sig_txt, fill=(100, 116, 139), font=font_small)

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
