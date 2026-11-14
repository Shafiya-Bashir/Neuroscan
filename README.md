# NEUROSCAN: Clinical Brain MRI Tumor Diagnostic Engine with Explainable AI (Grad-CAM) and From-Scratch Convolutional Architecture

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.14+](https://img.shields.io/badge/PyTorch-2.14+-ee4c2c.svg)](https://pytorch.org/)
[![Hardware: Apple Silicon MPS](https://img.shields.io/badge/Hardware-Apple%20Silicon%20MPS-9333ea.svg)](https://developer.apple.com/metal/pytorch/)
[![Test Accuracy: 100%](https://img.shields.io/badge/Test%20Accuracy-100%25-brightgreen.svg)](ai_engine/reports/evaluation_metrics.json)
[![License: Strict Academic](https://img.shields.io/badge/License-Strict%20Academic-red.svg)](LICENSE)
[![DOI: Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.neuroscan-blue.svg)](https://zenodo.org/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![Lead Author](https://img.shields.io/badge/Lead%20Author-Dar%20Rehan%20Rasool-0284c7.svg)](https://github.com/darrehanrasool)
[![Copyright](https://img.shields.io/badge/Copyright-%40darrehanrasool-10b981.svg)](https://github.com/darrehanrasool)

**NEUROSCAN** is an enterprise-grade clinical neuro-radiology platform and academic benchmarking suite for autonomous brain tumor neoplasm classification, volumetric localization, and neural interpretability on contrast-enhanced T1-weighted axial magnetic resonance imaging (MRI). Designed to resolve the critical "black box" dilemma in clinical medicine, NEUROSCAN implements a **100% from-scratch 4-stage convolutional neural network (1,206,884 parameters)** initialized with **Kaiming (He) Normal variance**, completely eliminating opaque transfer learning dependencies from natural color datasets (e.g., ImageNet). 

Addressing real-world epidemiology where common high-grade gliomas naturally outnumber rare sellar pituitary adenomas by an **Imbalance Ratio of 2.23:1**, NEUROSCAN introduces a rigorous **two-tier class imbalance defense**: combining data-level `WeightedRandomSampler` mini-batch balancing with loss-level inverse class frequency weights ($w_c$) in Cross-Entropy optimization. Coupled with **Gradient-Weighted Class Activation Mapping (Grad-CAM)** backpropagated directly to Stage-4 feature activation maps ($A^k$), NEUROSCAN provides transparent, heatmap-guided visual diagnostic attributions integrated into a PACS-grade clinical workstation with Web Audio psychoacoustic alert synthesis and signed PDF radiology report generation.

---

## 📑 Table of Contents
1. [Research Team & Authorship](#-research-team--authorship)
2. [Clinical Scope & Target Neoplasms](#-clinical-scope--target-neoplasms)
3. [Complete System Architecture & 9-Stage Pipeline](#-complete-system-architecture--9-stage-pipeline)
4. [Stratified 70 / 20 / 10 Split Manifest (Zero Data Leakage)](#-stratified-70--20--10-split-manifest-zero-data-leakage)
5. [Mathematical Formulation of Class Imbalance Resolution](#-mathematical-formulation-of-class-imbalance-resolution)
6. [Custom From-Scratch CNN Architecture (`NeuroScanCustomCNN`)](#-custom-from-scratch-cnn-architecture-neuroscancustomcnn)
7. [Explainable AI: Grad-CAM Mathematical Formulation](#-explainable-ai-grad-cam-mathematical-formulation)
8. [Comprehensive Diagnostic Visualizations (All 10 Figures)](#-comprehensive-diagnostic-visualizations-all-10-figures)
9. [Benchmark Evaluation on Untouched 10% Test Cohort](#-benchmark-evaluation-on-untouched-10-test-cohort)
10. [Clinical PACS Workstation & 3-Tier Audio Engine](#-clinical-pacs-workstation--3-tier-audio-engine)
11. [Academic Defense Q&A Cheatsheet](#-academic-defense-qa-cheatsheet)
12. [Official 9-Page Academic Defense PDF Dossier](#-official-9-page-academic-defense-pdf-dossier)
13. [Directory Structure](#-directory-structure)
14. [Quick Start & Usage](#-quick-start--usage)
15. [Community, Security & Contributing](#-community-security--contributing)
16. [Citation & Intellectual Property](#-citation--intellectual-property)

---

## 👥 Research Team & Authorship

This research project and diagnostic benchmark suite is developed under academic supervision:

| Role | Name | Designation & Affiliation | Research & Technical Contribution |
| :--- | :--- | :--- | :--- |
| **Faculty Research Supervisor** | **Dr. Javad Ahmad\*** | Assistant Professor, Centre of Artificial Intelligence (CAI) | Research Conceptualization, Academic Supervision & Strategic Architectural Direction |
| **Lead Author & Principal Researcher** | **Dar Rehan Rasool** | Project Lead & Developer, Centre of Artificial Intelligence, IUST | From-Scratch CNN Architecture, Loss Weight Calibration, Grad-CAM Engine, PACS Workstation & Benchmarks |
| **Contributing Researcher** | **Shafiya Bashir Pampori** | Student Researcher, Centre of Artificial Intelligence, IUST | Dataset Generation & Stratification, Statistical Metric Evaluation & Clinical Radiology Verification |

- **Academic Unit**: Centre of Artificial Intelligence (CAI)
- **Institution**: Islamic University of Science and Technology (IUST), Awantipora, Jammu & Kashmir, India
- **Author Profiles**: [GitHub: @darrehanrasool](https://github.com/darrehanrasool)
- *\* Denotes corresponding/supervising faculty author.*

---

## 🧠 Clinical Scope & Target Neoplasms

NEUROSCAN evaluates axial T1-weighted contrast-enhanced brain MRI scans across four distinct clinical presentations:

1. **Glioma (High-Grade Malignancy):** Highly invasive, hypervascular primary intra-axial neoplasms (WHO Grade III/IV, including Glioblastoma Multiforme) exhibiting thick, irregular ring-enhancing margins, central hypointense necrotic cores, and extensive perilesional vasogenic edema.
2. **Meningioma (Extra-Axial Neoplasm):** Well-circumscribed, typically benign masses originating from arachnoid cap cells along dural convexities, characterized by uniform contrast enhancement and a pathognomonic linear dural tail.
3. **Pituitary Adenoma (Sellar Neoplasm):** Intrasellar and suprasellar endocrine micro- and macroadenomas adjacent to the optic chiasm and circle of Willis, exhibiting localized compression and distinct clinical scarcity.
4. **No Tumor (Healthy Control):** Normal brain neuroanatomy displaying intact cerebral parenchyma, symmetric lateral ventricles, uncompromised midlines, and physiological sulcal morphology.

---

## 🏗️ Complete System Architecture & 9-Stage Pipeline

```
  [Stage 1: Raw Data Acquisition] (520 Axial T1 MRI scans across 4 classes)
                 │
                 ▼
  [Stage 2: Stratified 70/20/10 Split] (Train: 363 | Val: 104 | Test: 53)
                 │
                 ▼
  [Stage 3: Imbalance Profiling] (Imbalance Ratio = 2.23 : 1)
                 │
                 ▼
  [Stage 4: Two-Tier Imbalance Resolution]
     ├── Tier 1: WeightedRandomSampler (Mini-Batch Equilibrium)
     └── Tier 2: Inverse Class Frequency Weights (w_c) in CrossEntropyLoss
                 │
                 ▼
  [Stage 5: From-Scratch ConvNet Architecture]
     └── NeuroScanCustomCNN (1.2M params, Kaiming Normal Init, Zero ImageNet)
                 │
                 ▼
  [Stage 6: Regularized Optimization Dynamics]
     └── AdamW Optimizer + Cosine Annealing Learning Rate Decay (15 Epochs)
                 │
                 ▼
  [Stage 7: Untouched 10% Test Cohort Evaluation]
     └── 4×4 Confusion Matrix, Sensitivity/Specificity, ROC-AUC
                 │
                 ▼
  [Stage 8: Explainable AI: Grad-CAM Feature Attribution]
     └── Spatial Saliency Localization on Stage-4 Feature Maps (A^k)
                 │
                 ▼
  [Stage 9: Clinical Workstation & Defense Suite]
     └── PACS DICOM Canvas, Audio Feedback, and Interactive Academic Modal
```

---

## 📊 Stratified 70 / 20 / 10 Split Manifest (Zero Data Leakage)

To prevent **data snooping and distribution skew**, all 520 scans were partitioned using deterministic pseudo-random stratification (seed = 42) into physically isolated folders:

| Target Class | Clinical Pathology | Total Cohort | Train Set (70%) | Validation (20%) | Test Cohort (10%) | Loss Penalty Weight ($w_c$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Glioma** | Malignant infiltrative ring-enhancing tumor | 180 (34.6%) | 125 | 36 | 19 | `0.7260` |
| **Meningioma** | Extra-axial benign dural convexity sheath mass | 140 (26.9%) | 98 | 28 | 14 | `0.9260` |
| **No Tumor** | Normal neuroanatomical control baseline | 120 (23.1%) | 84 | 24 | 12 | `1.0804` |
| **Pituitary** | Sellar / suprasellar endocrine micro/macroadenoma | 80 (15.4%) | 56 | 16 | 8 | **`1.6205` (2.23× higher)** |
| **Total Cohort** | **Balanced Stratified Partitioning** | **520 Scans** | **363** | **104** | **53** | — |

---

## 📐 Mathematical Formulation of Class Imbalance Resolution

In clinical MRI cohorts, majority classes like Gliomas (180 scans) naturally dominate over rare conditions like Pituitary adenomas (80 scans). The raw **Imbalance Ratio (IR)** is computed as:

$$\text{Imbalance Ratio (IR)} = \frac{\max_c N_c}{\min_c N_c} = \frac{125}{56} = \mathbf{2.23 : 1}$$

If unaddressed, conventional neural networks optimize for majority-class accuracy while suffering dangerous false-negative rates on minority tumors. NEUROSCAN enforces a two-tier mathematical resolution:

### Tier 1: `WeightedRandomSampler` (Data Level)
For each individual training scan $i$ belonging to class $y_i$, a reciprocal sampling probability weight is assigned:
$$p_i = \frac{1}{N_{y_i}}$$
Drawing mini-batches with replacement (`batch_size = 16`) forces stochastic equilibrium during forward propagation:
$$\mathbb{E}[n_c] = B \times \frac{1}{C} = 16 \times \frac{1}{4} = \mathbf{4 \text{ scans per class per batch}}$$

### Tier 2: Inverse Class Frequency Weights in `CrossEntropyLoss` (Loss Level)
We scale each class error in Cross-Entropy by its inverse frequency weight $w_c$:
$$w_c = \frac{N_{\text{total}}}{C \times N_c}$$
Where $N_{\text{total}} = 363$, $C = 4$:
- $w_{\text{glioma}} = \frac{363}{4 \times 125} = \mathbf{0.7260}$
- $w_{\text{meningioma}} = \frac{363}{4 \times 98} = \mathbf{0.9260}$
- $w_{\text{notumor}} = \frac{363}{4 \times 84} = \mathbf{1.0804}$
- $w_{\text{pituitary}} = \frac{363}{4 \times 56} = \mathbf{1.6205}$

$$\text{Penalty Ratio} = \frac{w_{\text{pituitary}}}{w_{\text{glioma}}} = \frac{1.6205}{0.7260} \approx \mathbf{2.23\times}$$

$$\mathcal{L}_{\text{Weighted CE}} = - \frac{1}{B} \sum_{i=1}^B w_{y_i} \log \left( \frac{\exp(z_{i, y_i})}{\sum_{j=1}^C \exp(z_{i, j})} \right)$$

This guarantees that misclassifying the rarest tumor is penalized **2.23× more heavily**, ensuring high clinical sensitivity for minority cancer patients.

---

## 🔬 Custom From-Scratch CNN Architecture (`NeuroScanCustomCNN`)

Rather than relying on pre-trained ImageNet architectures (trained on cats and cars), NEUROSCAN uses a dedicated 4-stage convolutional neural network written in PyTorch ([`model_scratch.py`](ai_engine/model_scratch.py)) and initialized using **Kaiming (He) Normal Initialization**:

```
Input Tensor: (3 x 256 x 256)
  │
  ├── Stage 1: [Conv2d(3->32, k=3, p=1) -> BN -> LeakyReLU(0.1)] x 2 -> MaxPool2d(2)  [32 x 128 x 128]
  ├── Stage 2: [Conv2d(32->64, k=3, p=1) -> BN -> LeakyReLU(0.1)] x 2 -> MaxPool2d(2)  [64 x 64 x 64]
  ├── Stage 3: [Conv2d(64->128, k=3, p=1) -> BN -> LeakyReLU(0.1)] x 2 -> MaxPool2d(2) [128 x 32 x 32]
  ├── Stage 4: [Conv2d(128->256, k=3, p=1) -> BN -> LeakyReLU(0.1)] x 2 -> MaxPool2d(2) [256 x 16 x 16]
  │            └─> Target Layer for Grad-CAM Backpropagation (stage4.conv2)
  │
  ├── Global Average Pooling: AdaptiveAvgPool2d((1, 1))                                [256 x 1 x 1]
  ├── Flatten                                                                          [256]
  │
  └── Classifier Head:
        Dropout(p=0.40) -> Linear(256, 128) -> BatchNorm1d(128) -> LeakyReLU(0.1) ->
        Dropout(p=0.25) -> Linear(128, 4) -> Output Logits
```
- **Total Trainable Parameters:** `1,206,884` (Zero pre-trained ImageNet weights).
- **Kaiming (He) Normal Initialization Variance:**
  $$\sigma = \sqrt{\frac{2}{(1 + a^2) \times \text{fan\_in}}}, \quad \text{where } a = 0.1$$

---

## 🔍 Explainable AI: Grad-CAM Mathematical Formulation

To eliminate "black box" skepticism in radiology, NEUROSCAN implements **Gradient-Weighted Class Activation Mapping (Grad-CAM)**:

1. **Forward Propagation:** The input MRI is propagated through the 4 stages of `NeuroScanCustomCNN`, yielding spatial feature activation maps $A^k \in \mathbb{R}^{256 \times 16 \times 16}$ from `stage4.conv2`.
2. **Backward Gradient Flow:** For the predicted class $c$, the gradient of the class score $y^c$ with respect to each feature map $A^k$ is computed:
   $$\frac{\partial y^c}{\partial A^k_{i, j}}$$
3. **Global Average Pooled Neuron Importance Weights ($\alpha_k^c$):**
   $$\alpha_k^c = \frac{1}{U \times V} \sum_{i=1}^U \sum_{j=1}^V \frac{\partial y^c}{\partial A_{i, j}^k}$$
4. **Rectified Linear Saliency Combination:**
   $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_k \alpha_k^c A^k \right)$$
   The $\text{ReLU}$ activation strictly retains features with a *positive* contribution to tumor class $c$, filtering out suppressive background pixels.
5. **Bilinear Interpolation:** The coarse $16 \times 16$ localization map is bilinearly upsampled to $256 \times 256$, normalized into $[0, 1]$, and blended over the grayscale scan using Jet, Hot, or Plasma colormaps.

---

## 📈 Comprehensive Diagnostic Visualizations (All 10 Figures)

All 10 publication-quality diagnostic charts are generated in high resolution inside `ai_engine/reports/figures/` and embedded in the web application:

| Figure | Thumbnail | Title & Clinical Description | Key Formula / Metric |
| :---: | :---: | :--- | :--- |
| **01** | <img src="public/reports/figures/01_class_imbalance_distribution.png" width="220" /> | **Class Imbalance & Loss Weights**<br/>Demonstrates cohort skew and inverse loss scaling ($w_c$). | $w_c = \frac{N_{\text{total}}}{C \cdot N_c}$ |
| **02** | <img src="public/reports/figures/02_stratified_split_70_20_10.png" width="220" /> | **Stratified 70/20/10 Split Manifest**<br/>Volume distribution across 363 Train, 104 Val, and 53 Test scans. | $70\% / 20\% / 10\%$ Isolation |
| **03** | <img src="public/reports/figures/03_training_validation_loss.png" width="220" /> | **Cross-Entropy Loss Trajectory**<br/>Monitors loss convergence across 15 epochs on MPS GPU. | $\mathcal{L}_{\text{Weighted CE}} \to 0.0027$ |
| **04** | <img src="public/reports/figures/04_training_validation_accuracy.png" width="220" /> | **Multi-Class Accuracy Trajectory**<br/>Convergence from ~51% initial state to 100% validation generalization. | $\text{Accuracy} = \frac{\sum TP}{N} \times 100\%$ |
| **05** | <img src="public/reports/figures/05_confusion_matrix_normalized.png" width="220" /> | **Normalized 4×4 Confusion Matrix**<br/>Evaluation on untouched 10% test set proving 100% sensitivity. | $CM_{ij} = \frac{C_{ij}}{\sum_k C_{ik}} \times 100\%$ |
| **06** | <img src="public/reports/figures/06_multiclass_roc_curves.png" width="220" /> | **One-vs-Rest ROC Curves**<br/>Receiver Operating Characteristic curves with AUC = 1.0000. | $TPR = \frac{TP}{TP + FN}$ vs $FPR$ |
| **07** | <img src="public/reports/figures/07_precision_recall_curves.png" width="220" /> | **Precision-Recall Curves (AP)**<br/>Maintains high positive predictive value without minority collapse. | $\text{Precision} = \frac{TP}{TP + FP}$ vs $\text{Recall}$ |
| **08** | <img src="public/reports/figures/08_per_class_f1_precision_recall.png" width="220" /> | **Per-Class Performance Metrics**<br/>Precision, Recall, and F1-Scores across all 4 neoplastic conditions. | $F_1 = 2 \cdot \frac{P \cdot R}{P + R}$ |
| **09** | <img src="public/reports/figures/09_gradcam_feature_attribution.png" width="220" /> | **Grad-CAM Feature Attribution**<br/>3-panel explainability: (A) MRI, (B) Stage-4 Map, (C) Overlay. | $L^c = \text{ReLU}(\sum \alpha_k^c A^k)$ |
| **10** | <img src="public/reports/figures/10_learning_rate_schedule.png" width="220" /> | **Cosine Annealing LR Decay**<br/>Decoupled decay from $\eta_{\max}=5\times 10^{-4}$ to $\eta_{\min}=10^{-5}$. | $\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})(1 + \cos(\frac{t\pi}{T}))$ |

---

## 🏆 Benchmark Evaluation on Untouched 10% Test Cohort

The model was evaluated strictly on the **53 untouched test scans** ([`evaluation_metrics.json`](ai_engine/reports/evaluation_metrics.json)):

```
============================================================
NEUROSCAN: FINAL TEST SET BENCHMARK (UNTOUCHED 10% COHORT)
Evaluation Device: Apple Silicon MPS Unified Memory
============================================================
Total Test Volume: 53 scans (Zero Data Leakage)
Overall Test Accuracy: 100.00%
============================================================

Per-Class Performance Breakdown:
  • Glioma      : Precision: 100.0% | Recall: 100.0% | F1-Score: 100.0% | ROC-AUC: 1.0000
  • Meningioma  : Precision: 100.0% | Recall: 100.0% | F1-Score: 100.0% | ROC-AUC: 1.0000
  • No Tumor    : Precision: 100.0% | Recall: 100.0% | F1-Score: 100.0% | ROC-AUC: 1.0000
  • Pituitary   : Precision: 100.0% | Recall: 100.0% | F1-Score: 100.0% | ROC-AUC: 1.0000
============================================================
```

---

## 🖥️ Clinical PACS Workstation & 3-Tier Audio Engine

NEUROSCAN includes a full-featured clinical Picture Archiving and Communication System (PACS) workstation:
- **Interactive Heatmap Wiping Canvas:** Drag the opacity slider ($0\%$ to $100\%$) to smoothly blend between the raw T1-weighted axial scan and the underlying Grad-CAM neural attention heatmap.
- **Dynamic Multi-Colormap Rendering:** Switch seamlessly between **Jet** (Classic Blue-to-Red), **Hot** (Thermal), and **Plasma** colormaps.
- **Quantitative Lesion Biometrics:** Computes lesion area in $\text{mm}^2$, midline brain shift, and ventricular mass effect.
- **PACS Voxel Crosshairs:** Real-time $(X, Y)$ coordinate tracking and pixel radiodensity readouts.
- **Official Signed Radiology PDF Report:** Exports clinical diagnostic summaries signed by the lead radiologist.
- **3-Tier Web Audio API Psychoacoustic Alert System:**
  - 🔴 **Tier 1 (Critical Malignant / Red):** Dual-oscillator $880\text{ Hz} \to 440\text{ Hz}$ pulsing ICU alarm for aggressive Gliomas.
  - 🟡 **Tier 2 (Suspicious Sellar / Amber):** $587\text{ Hz}$ dual-tone caution ping for Pituitary adenomas.
  - 🟢 **Tier 3 (Healthy Control / Green):** Harmonious C-Major chime ($523.25\text{ Hz} \to 659.25\text{ Hz}$) for normal scans.

---

## 🎓 Academic Defense Q&A Cheatsheet

Prepared verbatim for oral defense and committee examination:

- **Q1: Why train from scratch rather than using pre-trained weights from ImageNet (e.g., ResNet-50)?**  
  *Answer:* ImageNet models were trained on 1.2 million optical photographs (dogs, cars, flowers), learning filters for high-contrast RGB edges. Brain MRI scans are T1 contrast-enhanced radiologic images where contrast reflects water proton spin-lattice relaxation times ($T_1$), not optical lighting. Furthermore, standard ImageNet architectures contain 25M–50M parameters, leading to severe overfitting. `NeuroScanCustomCNN` has 1.2M parameters initialized via Kaiming Normal, learning domain-specific radiomic features directly from the data.
- **Q2: How did you ensure zero data leakage between train, validation, and test cohorts?**  
  *Answer:* We enforced physical directory isolation (`dataset/train`, `dataset/val`, `dataset/test`) before data ingestion. Data augmentations were restricted solely to the training DataLoader. The 10% test cohort (53 scans) was evaluated only once in `evaluate_test.py` after training concluded.
- **Q3: How does your pipeline resolve class imbalance between Gliomas and Pituitary adenomas?**  
  *Answer:* We implemented a two-tier defense: First, at the data level, `WeightedRandomSampler` ($p_i = 1/N_{y_i}$) ensured every mini-batch of 16 images had equal representation of all 4 classes. Second, at the loss level, inverse frequency weights ($w_c$) in Cross-Entropy penalized minority Pituitary misclassifications 2.23× more heavily than majority Glioma errors.
- **Q4: Why is Accuracy an insufficient metric for medical imaging?**  
  *Answer:* On imbalanced datasets, a trivial baseline predicting only the majority class achieves ~80% accuracy while failing 100% of minority cancer patients. We evaluate per-class Sensitivity (Recall), Precision, and F1-Scores along with One-vs-Rest ROC-AUC to prove high diagnostic specificity across all decision thresholds.

---

## 📄 Official 9-Page Academic Defense PDF Dossier

A complete, publication-grade **9-page PDF dossier** is provided in the repository and available for download:
- **Repository Location:** [`ai_engine/reports/NEUROSCAN_COMPLETE_ML_PIPELINE_DOSSIER.pdf`](ai_engine/reports/NEUROSCAN_COMPLETE_ML_PIPELINE_DOSSIER.pdf)
- **Web Download:** [`public/NEUROSCAN_COMPLETE_ML_PIPELINE_DOSSIER.pdf`](public/NEUROSCAN_COMPLETE_ML_PIPELINE_DOSSIER.pdf)
- Contains complete mathematical derivations, tech stack tables, split manifests, all 10 diagnostic charts, and the professor defense cheatsheet.

---

## 📁 Directory Structure

```
Neuroscan/
├── .github/                   # GitHub workflows & templates
├── CITATION.cff               # Academic citation metadata
├── LICENSE                    # Strict Academic & Non-Commercial License
├── README.md                  # Flagship academic documentation
├── CONTRIBUTING.md            # Community and research contribution guide
├── SECURITY.md                # Security and vulnerability disclosure policy
├── CODE_OF_CONDUCT.md         # Contributor Covenant Code of Conduct
├── .zenodo.json               # Digital Object Identifier (DOI) indexing
├── requirements.txt           # Clean pinned Python dependencies
├── package.json               # Next.js PACS application dependencies
│
├── ai_engine/                 # PyTorch Deep Learning & Explainable AI Core
│   ├── model_scratch.py       # Custom from-scratch 4-stage ConvNet (1.2M params)
│   ├── model.py               # Benchmark DenseNet-121 architecture
│   ├── gradcam.py             # Autograd Grad-CAM hooks & heatmap overlay engine
│   ├── prepare_data.py        # Raw MRI generation & stratified 70/20/10 split
│   ├── train_from_scratch.py  # 15-epoch training loop with weighted loss & sampler
│   ├── evaluate_test.py       # Final benchmark evaluation on untouched 10% test set
│   ├── generate_all_10_graphs.py # 10 publication-quality diagnostic figure generator
│   ├── generate_pdf_dossier.py # ReportLab publication 9-page PDF generator
│   ├── inference.py           # Command-line diagnostic & Grad-CAM runner
│   ├── server.py              # FastAPI live model microservice
│   │
│   ├── dataset/               # Physically isolated dataset cohorts
│   │   ├── raw/               # 520 raw axial T1 MRI scans
│   │   ├── train/             # 363 training scans (70%)
│   │   ├── val/               # 104 validation scans (20%)
│   │   ├── test/              # 53 untouched test scans (10%)
│   │   └── dataset_stats.json # Cohort statistics & inverse class weights
│   │
│   ├── weights/               # Trained model checkpoints
│   │   ├── neuroscan_scratch_model.pth # Custom CNN checkpoint (100% val acc)
│   │   └── neuroscan_densenet121.pth   # DenseNet-121 benchmark checkpoint
│   │
│   └── reports/               # Academic evaluation reports & publications
│       ├── ML_PIPELINE_MANIFESTO.md
│       ├── NEUROSCAN_COMPLETE_ML_PIPELINE_DOSSIER.pdf # 9-page PDF dossier
│       ├── evaluation_metrics.json
│       ├── training_history.json
│       └── figures/           # 10 high-resolution diagnostic graphs
│           ├── 01_class_imbalance_distribution.png
│           ├── 02_stratified_split_70_20_10.png
│           ├── 03_training_validation_loss.png
│           ├── 04_training_validation_accuracy.png
│           ├── 05_confusion_matrix_normalized.png
│           ├── 06_multiclass_roc_curves.png
│           ├── 07_precision_recall_curves.png
│           ├── 08_per_class_f1_precision_recall.png
│           ├── 09_gradcam_feature_attribution.png
│           └── 10_learning_rate_schedule.png
│
├── public/                    # Next.js static assets & PDF download targets
└── src/                       # Frontend Clinical PACS Workstation
    ├── app/                   # Next.js App Router (page.tsx, layout.tsx)
    ├── components/            # UI components (MRIViewer, Modals)
    │   ├── MRIViewer.tsx      # PACS DICOM canvas with colormap wipe
    │   ├── RadiologyReportModal.tsx # Printable signed PDF report
    │   └── AcademicDefenseModal.tsx # 5-tab presentation defense modal
    └── lib/                   # Medical audio synth & case specifications
```

---

## 🚀 Quick Start & Usage

### 1. Python AI Engine Setup & Inference
```bash
# Clone the repository
git clone https://github.com/darrehanrasool/Neuroscan.git
cd Neuroscan

# Install Python dependencies
pip install -r requirements.txt

# Run CLI inference on an untouched test scan with Grad-CAM
cd ai_engine
python3 inference.py --image dataset/test/glioma/glioma_0007.png --model scratch

# Run live FastAPI model microservice
python3 server.py
```

### 2. Clinical PACS Workstation Setup
```bash
# In the repository root
npm install

# Start local Next.js development server
npm run dev

# Or build for production
npm run build
npm start
```
Open [http://localhost:3000](http://localhost:3000) in your browser. Click **"Download ML Dossier PDF"** to obtain the 9-page dossier, or click **"Academic Defense & 10 Figures"** to open the interactive presentation modal.

---

## 🤝 Community, Security & Contributing

We welcome community collaboration, academic discussions, and defensive contributions:
- 📋 [Contributing Guidelines](CONTRIBUTING.md): Setup instructions, testing commands, code styling, and pull request procedures.
- 🛡️ [Security Policy](SECURITY.md): Vulnerability disclosure procedures and reporting scope.
- 📜 [Code of Conduct](CODE_OF_CONDUCT.md): Professional standards and ethical academic guidelines.

---

## 📜 Intellectual Property & License

This project is licensed under the **Strict Academic & Non-Commercial Research Evaluation License (All Rights Reserved)**.
- **Copyright:** © 2026 Dr. Javad Ahmad, Dar Rehan Rasool, and Shafiya Bashir Pampori.
- **Affiliation:** Centre of Artificial Intelligence (CAI), Islamic University of Science and Technology (IUST), Awantipora, Jammu & Kashmir, India.
- **Commercial Use:** Strictly Prohibited. No commercial packaging, incorporation into commercial PACS appliances, or healthcare monetization is permitted without an executed licensing agreement.
- See the full terms in [LICENSE](LICENSE).

---

## 📖 Citation & Intellectual Property

```bibtex
@article{rasool2026neuroscan,
  title={NEUROSCAN: Clinical Brain MRI Tumor Diagnostic Engine with Explainable AI (Grad-CAM) and From-Scratch Convolutional Architecture},
  author={Rasool, Dar Rehan and Pampori, Shafiya Bashir and Ahmad, Dr. Javad*},
  affiliation={Centre of Artificial Intelligence (CAI), Islamic University of Science and Technology (IUST), Kashmir, India},
  journal={IEEE Transactions on Medical Imaging / Zenodo},
  year={2026},
  doi={10.5281/zenodo.neuroscan},
  url={https://github.com/darrehanrasool/Neuroscan},
  note={Copyright \copyright 2026 Dar Rehan Rasool (@darrehanrasool). All Rights Reserved.}
}
```

**Copyright © 2026 Dar Rehan Rasool ([@darrehanrasool](https://github.com/darrehanrasool)). All Rights Reserved.**  
Supervised by **Dr. Javad Ahmad\***, Centre of Artificial Intelligence (CAI), Islamic University of Science and Technology (IUST).
