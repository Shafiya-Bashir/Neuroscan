# NEUROSCAN: Machine Learning Pipeline & Architectural Manifesto
**Project Title:** Clinical Brain MRI Tumor Diagnostic Engine with Explainable AI (Grad-CAM)  
**Faculty Supervisor:** Dr. Syed Wajid Alam  
**Lead Developer / Engineer:** Dar Rehan Rasool  
**Student Contributor:** Shafiya Pampori  
**Institution:** Islamic University of Science and Technology (IUST)  
**Repository:** [github.com/darrehanrasool/Neuroscan](https://github.com/darrehanrasool/Neuroscan)  
**Date:** Academic AI Project Defense (November 2026)

---

## 1. Executive Summary & Problem Formulation

Neoplasms of the central nervous system (CNS) represent a critical diagnostic challenge in neuro-oncology. The four most frequent clinical presentations in contrast-enhanced T1-weighted axial magnetic resonance imaging (MRI) are:
1. **Glioma:** Infiltrative, highly vascularized primary intra-axial malignancies (e.g., Glioblastoma Multiforme, WHO Grade IV) characterized by irregular contrast-enhancing ring margins and extensive perilesional vasogenic edema.
2. **Meningioma:** Typically benign extra-axial masses arising from arachnoid cap cells of the meninges, featuring homogeneous contrast enhancement and a pathognomonic "dural tail".
3. **Pituitary Adenoma:** Sellar and suprasellar endocrine neoplasms adjacent to the optic chiasm, exhibiting distinctive micro- and macroadenoma expansion patterns.
4. **No Tumor (Healthy Control):** Unremarkable neuroanatomical axial slices displaying symmetrical bilateral cerebral hemispheres, normal lateral ventricles, and intact cortical sulci.

The goal of **NEUROSCAN-AI** is to build a clinically interpretable, fully transparent Computer Vision and Deep Learning diagnostic system **trained completely from scratch** (with zero reliance on pre-trained ImageNet weights), with mathematical treatment for real-world **class imbalance**, strict **70 / 20 / 10 stratified cohort splitting**, and **Grad-CAM visual feature attribution**.

---

## 2. Comprehensive Technology Stack

| Domain | Technology / Library | Version / Specification | Architectural Purpose |
| :--- | :--- | :--- | :--- |
| **Deep Learning Framework** | `PyTorch` | `2.14.0` | Custom tensor computation, backpropagation autograd, loss functions |
| **Hardware Acceleration** | Apple Silicon MPS (`torch.backends.mps`) | M-Series GPU Unified Memory | Fast tensor matrix multiplications & forward/backward passes |
| **Computer Vision** | `Torchvision`, `Pillow (PIL)` | `0.19.0`, `11.0.0` | Data transforms, tensor normalization, anatomical MRI synthesis |
| **Statistical & ML Metrics** | `Scikit-Learn`, `SciPy` | `1.5.0`, `1.14.0` | Confusion matrix, ROC-AUC, PR-AUC, classification report |
| **Scientific Visualization** | `Matplotlib`, `Seaborn` | `3.9.0`, `0.13.0` | 10 high-resolution publication-quality diagnostic charts |
| **Web & Backend API** | `FastAPI`, `Uvicorn` | `0.115.0`, `0.32.0` | Asynchronous REST microservice for live DICOM inference |
| **Frontend PACS Workstation** | `Next.js 16 (React 19)`, `Tailwind CSS` | App Router, Lucide Icons | Clinical PACS workstation, multi-colormap viewer, audio alerts |
| **Clinical Audio Synthesis** | Web Audio API (Dual Oscillator Synthesizer) | Native Browser AudioContext | Psychoacoustic sound feedback: Malignant alarm (red), caution ping (amber), healthy chime (green) |
| **Speech Briefing** | Web SpeechSynthesis API | Samantha / Google Voice Engine | Audio-verbal clinical summary readouts for radiologists |

---

## 3. The 9-Stage End-to-End Machine Learning Pipeline

```
  [Stage 1: Raw Data Acquisition]
                 │
                 ▼
  [Stage 2: Stratified 70/20/10 Split] ── (Zero Data Leakage)
                 │
                 ▼
  [Stage 3: Class Imbalance Profiling] ── (Imbalance Ratio = 2.23:1)
                 │
                 ▼
  [Stage 4: Two-Tier Imbalance Resolution]
     ├── Tier 1: WeightedRandomSampler (Mini-Batch Equilibrium)
     └── Tier 2: Inverse Frequency CrossEntropyLoss Weights (w_c)
                 │
                 ▼
  [Stage 5: Custom CNN Architecture From Scratch]
     └── Kaiming (He) Normal Initialization + LeakyReLU
                 │
                 ▼
  [Stage 6: Regularized Optimization Dynamics]
     └── AdamW Optimizer + Cosine Annealing Learning Rate Decay
                 │
                 ▼
  [Stage 7: Untouched 10% Test Evaluation]
     └── Multi-class Confusion Matrix, Sensitivity/Specificity, ROC-AUC
                 │
                 ▼
  [Stage 8: Explainable AI: Grad-CAM Saliency Maps]
     └── Saliency Localization on Stage-4 Conv Feature Maps (A^k)
                 │
                 ▼
  [Stage 9: Clinical Workstation Deployment]
     └── PACS DICOM Canvas, Audio Feedback, PDF Radiology Report
```

---

## 4. Stage-by-Stage Technical Details

### Stage 1: Raw Data Acquisition & Cohort Generation
A total of **520 raw axial T1 contrast-enhanced MRI scans** were generated into `ai_engine/dataset/raw/`, replicating authentic neuropathological characteristics across all 4 clinical target classes:
- `glioma`: 180 scans (34.6%) — High prevalence, ring enhancement, edema.
- `meningioma`: 140 scans (26.9%) — Extra-axial convexity, dural attachment.
- `notumor`: 120 scans (23.1%) — Normal baseline brain anatomy.
- `pituitary`: 80 scans (15.4%) — Intrasellar adenoma, lower clinical incidence.

### Stage 2: Stratified 70 / 20 / 10 Splitting Pipeline
To prevent **data snooping and distribution shift**, the entire cohort is partitioned into three mutually exclusive subsets using deterministic pseudo-random stratification (seed = 42):
- **Training Set (70% = 363 scans):** Used strictly for updating neural network weights via gradient descent.
  - `glioma`: 125 | `meningioma`: 98 | `pituitary`: 56 | `notumor`: 84
- **Validation Set (20% = 104 scans):** Used strictly for hyperparameter monitoring and best checkpoint selection.
  - `glioma`: 36 | `meningioma`: 28 | `pituitary`: 16 | `notumor`: 24
- **Test Set (10% = 53 scans):** Kept completely untouched until final benchmark evaluation.
  - `glioma`: 19 | `meningioma`: 14 | `pituitary`: 8 | `notumor`: 12

### Stage 3 & 4: Class Imbalance Mathematical Treatment
In clinical practice, rare tumors (e.g. Pituitary adenomas) are underrepresented compared to common aggressive gliomas. If unaddressed, standard neural networks develop an optimization bias toward the majority class, yielding high superficial accuracy but dangerously low sensitivity (high false-negative rates) on the minority class.

#### 1. Imbalance Ratio Calculation
$$\text{Imbalance Ratio (IR)} = \frac{\max_c N_c}{\min_c N_c} = \frac{125}{56} = 2.23 : 1$$

#### 2. Tier 1: Weighted Random Sampler
For each individual training scan $i$ belonging to class $y_i$, we assign a sampling probability weight:
$$p_i = \frac{1}{N_{y_i}}$$
When drawing batches with replacement (`replacement=True`), the expected frequency of class $c$ in any mini-batch of size $B=16$ becomes:
$$\mathbb{E}[n_c] = B \times \frac{1}{C} = 16 \times \frac{1}{4} = 4 \text{ scans per batch}$$
This forces mini-batch equilibrium, eliminating majority-class gradient domination during backpropagation.

#### 3. Tier 2: Inverse Frequency Class Weights in Loss Function
We calibrate the standard Cross-Entropy Loss by scaling each class error with its inverse frequency weight $w_c$:
$$w_c = \frac{N_{\text{total}}}{C \times N_c}$$
Where $N_{\text{total}} = 363$, $C = 4$, and $N_c$ is the training count for class $c$:
- $w_{\text{glioma}} = \frac{363}{4 \times 125} = \mathbf{0.7260}$
- $w_{\text{meningioma}} = \frac{363}{4 \times 98} = \mathbf{0.9260}$
- $w_{\text{notumor}} = \frac{363}{4 \times 84} = \mathbf{1.0804}$
- $w_{\text{pituitary}} = \frac{363}{4 \times 56} = \mathbf{1.6205}$

$$\frac{w_{\text{pituitary}}}{w_{\text{glioma}}} = \frac{1.6205}{0.7260} \approx \mathbf{2.23\times}$$
Any misclassification of the rare Pituitary class incurs a **2.23× higher gradient penalty**, guaranteeing that the model optimizes for high recall in the rarest clinical condition.

$$\mathcal{L}_{\text{Weighted CE}} = - \frac{1}{B} \sum_{i=1}^B w_{y_i} \log \left( \frac{\exp(z_{i, y_i})}{\sum_{j=1}^C \exp(z_{i, j})} \right)$$

---

### Stage 5: Custom Architecture (`NeuroScanCustomCNN`) Built From Scratch

Rather than importing an opaque pre-trained model (like ResNet or VGG trained on dogs and cars from ImageNet), **NEUROSCAN-AI** uses a dedicated 4-stage convolutional neural network initialized using **Kaiming (He) Normal Initialization**:

```
Input (3 x 256 x 256)
  │
  ├── Stage 1: [Conv(3->32, k=3, p=1) -> BN -> LeakyReLU(0.1)] x 2 -> MaxPool(2)  [32 x 128 x 128]
  ├── Stage 2: [Conv(32->64, k=3, p=1) -> BN -> LeakyReLU(0.1)] x 2 -> MaxPool(2)  [64 x 64 x 64]
  ├── Stage 3: [Conv(64->128, k=3, p=1) -> BN -> LeakyReLU(0.1)] x 2 -> MaxPool(2) [128 x 32 x 32]
  ├── Stage 4: [Conv(128->256, k=3, p=1) -> BN -> LeakyReLU(0.1)] x 2 -> MaxPool(2) [256 x 16 x 16]
  │            └─> Target Layer for Grad-CAM Backpropagation (stage4.conv2)
  │
  ├── Global Average Pooling: AdaptiveAvgPool2d((1, 1))                            [256 x 1 x 1]
  ├── Flatten                                                                     [256]
  │
  └── Classifier Head:
        Dropout(p=0.40) -> Linear(256, 128) -> BatchNorm1d(128) -> LeakyReLU(0.1) ->
        Dropout(p=0.25) -> Linear(128, 4) -> Logits
```
- **Total Trainable Parameters:** `1,206,884` (1.2M parameters)
- **Kaiming Normal Variance:**
  $$\sigma = \sqrt{\frac{2}{(1 + a^2) \times \text{fan\_in}}}, \quad \text{where } a = 0.1$$
  This prevents vanishing gradients in early layers and exploding activations in late layers.

---

### Stage 6: Optimization Dynamics & Learning Rate Schedule
- **Optimizer:** `AdamW` (L2 decoupled weight decay $\lambda = 10^{-4}$, $\beta_1 = 0.9, \beta_2 = 0.999$).
- **Learning Rate Schedule:** Cosine Annealing decay across 15 epochs:
  $$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min}) \left( 1 + \cos\left(\frac{t \pi}{T_{\max}}\right) \right)$$
  - Initial $\eta_{\max} = 5 \times 10^{-4}$
  - Terminal $\eta_{\min} = 1 \times 10^{-5}$
  - $T_{\max} = 15$ epochs

---

### Stage 7: Evaluation on Untouched 10% Test Cohort
The model is evaluated strictly on the 53 scans in `dataset/test/` that were never seen during training or validation:
- **Confusion Matrix:** 4×4 contingency table tracking True Positives, False Positives, True Negatives, False Negatives.
- **Sensitivity (Recall):** $\text{Recall} = \frac{TP}{TP + FN}$
- **Positive Predictive Value (Precision):** $\text{Precision} = \frac{TP}{TP + FP}$
- **Harmonic Mean (F1-Score):** $F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$
- **Area Under the ROC Curve (AUC):** Measures discriminative capability across all classification thresholds.

---

### Stage 8: Explainable AI with Grad-CAM

To satisfy clinical governance and eliminate "black box" skepticism, we implement **Gradient-Weighted Class Activation Mapping (Grad-CAM)**:

1. **Forward Pass:** The MRI image is propagated through the 4 stages of `NeuroScanCustomCNN`, extracting the final spatial feature activation tensor $A \in \mathbb{R}^{256 \times 16 \times 16}$ from `stage4.conv2`.
2. **Backward Gradient Flow:** For the predicted class $c$, we compute the gradient of the class score $y^c$ with respect to each feature map $A^k$:
   $$\frac{\partial y^c}{\partial A^k_{i, j}}$$
3. **Global Average Pooled Importance Weights ($\alpha_k^c$):**
   $$\alpha_k^c = \frac{1}{U \times V} \sum_{i=1}^U \sum_{j=1}^V \frac{\partial y^c}{\partial A_{i, j}^k}$$
4. **Rectified Linear Combination (Heatmap Localization):**
   $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_k \alpha_k^c A^k \right)$$
   Applying the $\text{ReLU}$ function ensures that only features that have a *positive* contribution to the target class are highlighted, while suppressive features are zeroed out.
5. **Bilinear Upsampling & Colormap Overlay:** The resulting $16 \times 16$ coarse localization map is bilinearly interpolated back to $256 \times 256$, normalized into $[0, 1]$, and blended over the grayscale MRI using Jet, Hot, or Plasma colormaps.

---

## 5. Defense Q&A: Key Questions the Professor May Ask

### Question 1: "Why build a model from scratch instead of using pre-trained weights from ImageNet (e.g., ResNet50)?"
**Answer:**
> "ImageNet models were pre-trained on 1.2 million natural color photographs (dogs, cars, flowers), which learn filters for high-contrast edges and vibrant RGB textures. Brain MRI scans, however, are T1 contrast-enhanced radiologic images where contrast reflects water proton relaxation times ($T_1$), not natural lighting. By designing `NeuroScanCustomCNN` from scratch and initializing with Kaiming Normal weights tailored for LeakyReLU, the network learns domain-specific radiomic features directly from the data. Furthermore, pre-trained ImageNet architectures often contain 25M–50M parameters, resulting in massive overparameterization and memorization on medical cohorts. Our 1.2M-parameter architecture achieves optimal regularization without black-box transfer learning."

### Question 2: "How did you ensure there is no data leakage between train, validation, and test sets?"
**Answer:**
> "We enforced strict physical directory isolation (`dataset/train`, `dataset/val`, `dataset/test`) at the very beginning of the pipeline prior to any transformation. Data augmentations (random rotations, color jitter, horizontal flips) are applied strictly within the training DataLoader. The validation and test pipelines execute only deterministic resizing and ImageNet normalization. Most importantly, the 10% test cohort was never seen by the model during hyperparameter tuning, loss computation, or weight updates; it was loaded only once in `evaluate_test.py` for final reporting."

### Question 3: "How does your pipeline handle the fact that Pituitary adenomas are far less common than Gliomas?"
**Answer:**
> "We implemented a two-tier defense against class imbalance:
> 1. **At the Data Level:** We used PyTorch's `WeightedRandomSampler`, which calculates the reciprocal of each class frequency ($1/N_c$) and samples images with replacement so that every mini-batch of 16 images contains an equal proportion of each class.
> 2. **At the Loss Level:** In `nn.CrossEntropyLoss(weight=class_weights)`, we applied inverse class frequency weights $w_c = \frac{N_{\text{total}}}{C \cdot N_c}$. The minority Pituitary class has a weight of $1.62$, while the majority Glioma class has a weight of $0.73$. This means the network is penalized **2.23 times more heavily** for misclassifying a Pituitary tumor, preventing the model from ignoring the minority class."

### Question 4: "Why did you use Grad-CAM instead of plain CAM or vanilla saliency maps?"
**Answer:**
> "Vanilla saliency maps (raw input gradients) are notoriously noisy and often reflect simple high-frequency edge textures rather than semantic lesion concepts. Plain CAM requires removing the dense classification head and replacing it with a Global Average Pooling layer directly preceding the softmax, which restricts architecture design. Grad-CAM, by contrast, operates on any convolutional layer by taking the gradients of the target class score with respect to the feature activation maps ($A^k$). This provides coarse, semantically rich localization of exactly where the tumor boundary influenced the model's diagnostic classification."
