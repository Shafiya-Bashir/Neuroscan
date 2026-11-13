# NEUROSCAN-AI 🧠🔬
### Clinical Neuro-Radiology Workstation with Grad-CAM Explainable AI (XAI)

**Developed by:** Dar Rehan Rasool  
**Domain:** Medical Deep Learning, Neuro-Oncology, Computer Vision, Explainable AI (XAI)

---

## 📌 Project Overview
**NEUROSCAN-AI** is an advanced Clinical Neuro-Radiology Workstation designed to solve the notorious "black box" problem of Deep Learning in medicine. It classifies brain MRI scans across four distinct pathological states (**Glioblastoma Multiforme**, **Meningioma**, **Pituitary Adenoma**, and **Healthy Brain**) and overlays **Grad-CAM (Gradient-weighted Class Activation Mapping)** heatmaps that visually prove *why* the neural network made each diagnostic decision.

---

## 🔊 The 3-Tier Audio & Color System

| Alert Tier | Diagnostic Severity | Visual HUD Overlay | Sound & Audio Alert | Spoken Radiology Voice Briefing |
| :---: | :---: | :---: | :---: | :---: |
| 🔴 **Tier 1** | **CRITICAL MALIGNANT (Glioblastoma / Meningioma)** | Pulsing Crimson Border & Red Lesion Contour | **Rapid Clinical Warning Pulse** (880Hz / 988Hz Square Wave: `BEEP-BEEP-BEEP-BEEP!`) | *"Critical alert: High-grade Glioblastoma identified in right frontotemporal region with 98.7% confidence. Midline shift and edema noted."* |
| 🟡 **Tier 2** | **SUSPICIOUS / LOW-GRADE (Pituitary Adenoma)** | Amber Reticle & Sellar Highlighting | **Dual-Tone Caution Ping** (587Hz $\rightarrow$ 440Hz Triangle Wave) | *"Warning: Low-grade Pituitary adenoma detected in sellar region. Clinical contrast enhancement advised."* |
| 🟢 **Tier 3** | **HEALTHY / NEGATIVE (Normal Brain Parenchyma)** | Emerald Green HUD & Symmetrical Ventricles | **Harmonic C-Major Medical Confirmation Chime** (523Hz $\rightarrow$ 659Hz $\rightarrow$ 784Hz) | *"Evaluation complete: Normal brain MRI scan. Intracranial structures and ventricular morphology are completely healthy."* |

---

## 📐 Explainable AI (XAI) Mathematical Formulation

Grad-CAM computes the importance weight $\alpha_k^c$ of feature map $k$ for diagnostic class $c$:

$$\alpha_k^c = \frac{1}{Z} \sum_{i=1}^u \sum_{j=1}^v \frac{\partial y^c}{\partial A_{i,j}^k}$$

Where:
* $y^c$ is the pre-softmax score for class $c$.
* $A^k$ is the $k$-th convolutional feature map of the final layer (`conv5_block3_out` in DenseNet-121).
* $Z = u \times v$ represents the spatial dimensions of the feature map.

The final heat localization map is computed via Rectified Linear Unit (ReLU) filtering:

$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$

*The ReLU activation ensures only features that positively correlate with the malignant class are rendered in fiery red.*

---

## 🖥️ Workstation Features & Controls

1. **Interactive Grad-CAM Wiping Station:**
   * **Opacity Slider (0% to 100%):** Drag back and forth to blend between the raw T1-weighted MRI scan and the underlying neural attention heatmap.
   * **Colormap Selector:** Switch between **Jet** (Classic Blue-to-Red), **Hot** (Thermal), and **Plasma**.
2. **Quantitative Lesion Biometrics:**
   * Calculates tumor surface area in $mm^2$.
   * Assesses midline shift and ventricular mass effect.
3. **PACS Crosshairs:**
   * Real-time $(X, Y)$ voxel coordinates and intensity readouts.
4. **Clinical Report Generator:**
   * Generates a signed, official radiological diagnostic PDF report complete with patient demographics, AI confidence, and lead radiologist sign-off.

---

## 🚀 Quickstart & Deployment

```bash
# Install dependencies
npm install

# Start local server
npm run dev
# Or run production build
npm run build
npm start
```
Open [http://localhost:3000](http://localhost:3000) in your browser.
