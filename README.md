# 🦋 AI-Based Butterfly Species Classification & Visual Explanation System

An end-to-end Computer Vision and Explainable AI (XAI) system for classifying 8 distinct butterfly species using deep transfer learning (ResNet-18) and visualizing model decision regions in real time using native PyTorch Grad-CAM.

---

## 📌 Project Overview

- **Core Goal**: Accurate classification of butterfly species with interpretable visual explanations for biological and educational applications.
- **Model Architecture**: ResNet-18 with ImageNet-1k pretrained backbone and a customized 8-class classification head.
- **Explainability Engine**: Gradient-weighted Class Activation Mapping (Grad-CAM) targeting the final convolutional block (`model.layer4[-1]`) using native PyTorch hooks.
- **Frontend**: Interactive Streamlit web application optimized for lightweight CPU execution and Streamlit Community Cloud deployment.

---

## 🦋 Supported Butterfly Species (8 Classes)

| # | Common Name | Scientific Name | Family | Key Visual Marker |
|---|---|---|---|---|
| 1 | **MONARCH** | *Danaus plexippus* | Nymphalidae | Tawny orange wings, bold black veins, white margin spots |
| 2 | **PAPER KITE** | *Idea leuconoe* | Nymphalidae | Large translucent ivory wings, dark vein striping |
| 3 | **RED POSTMAN** | *Heliconius erato* | Nymphalidae | Velvety black elongated wings, vivid crimson vertical band |
| 4 | **ADONIS** | *Polyommatus bellargus* | Lycaenidae | Electric sky-blue dorsal wings with white-checkered fringe |
| 5 | **GREEN CELLED CATTLEHEART** | *Parides sesostris* | Papilionidae | Velvety black wings, bright emerald-green forewing patch |
| 6 | **SOUTHERN DOGFACE** | *Zerene cesonia* | Pieridae | Sulfur-yellow wings with dark dog's head silhouette |
| 7 | **ORANGE OAKLEAF** | *Kallima inachus* | Nymphalidae | Royal blue & orange dorsal bands; dried-leaf camouflage |
| 8 | **CLODIUS PARNASSIAN** | *Parnassius clodius* | Papilionidae | Translucent milky wings with dark dusting, red ocelli |

---

## 📊 Model Performance & Evaluation

Evaluated on an independent, stratified holdout test split (**72 test images**, zero training leakage):

| Metric | Score |
| :--- | :---: |
| **Test Accuracy** | **`97.22%`** (70 / 72 correct) |
| **Macro Precision** | **`97.50%`** |
| **Macro Recall** | **`97.22%`** |
| **Macro F1-Score** | **`97.12%`** |
| **Weighted F1-Score** | **`97.12%`** |
| **Model File Size** | **`44.84 MB`** (GitHub-safe, < 50 MB) |

---

## 🌟 Key Features

- **Real-Time Classification**: Instant top-1 prediction and top-3 ranked probabilities.
- **Visual Explainability (Grad-CAM)**: Generates heatmaps localizing specific wing patterns driving neural activations.
- **Dynamic Heatmap Overlay**: Adjustable transparency slider ($\alpha \in [0.2, 0.8]$) with multi-tab image inspection.
- **Species Knowledge Base**: Factual biological reference cards for all 8 species.
- **Lightweight CPU Deployment**: Zero GPU requirement; fast inference on standard hardware.

---

## 🚀 Running Locally

1. **Clone the repository**:
   ```bash
   git clone https://github.com/<your-username>/Butterfly-Classification.git
   cd Butterfly-Classification
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the Streamlit web app**:
   ```bash
   streamlit run app/app.py
   ```

4. Open your browser at `http://localhost:8501`.

---

## ☁️ Deployment on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Select **New App** $\rightarrow$ choose your repository, branch (`main`), and set the main file path to:
   ```text
   app/app.py
   ```
4. Deploy! The app will automatically install dependencies from `requirements.txt` and load `models/butterfly_resnet18_best.pth`.

---

## ⚠️ Limitations & Disclaimer

- **Class Scope**: The classifier is specialized strictly for the 8 trained butterfly species. Non-butterfly images or unsupported species will be mapped to the closest visual match among the 8 classes.
- **Image Quality**: Extreme blur, partial occlusions, or poor lighting may reduce prediction certainty.
- **Educational / Research Purpose**: Designed as an academic prototype for Explainable AI and transfer learning demonstration.
