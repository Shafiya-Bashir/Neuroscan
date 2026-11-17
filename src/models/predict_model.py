"""
NEUROSCAN-AI: Step 5 - Production CLI Inference, Grad-CAM & Difference Map Generator
Author: Dar Rehan Rasool
Usage:
  python src/models/predict_model.py
  python src/models/predict_model.py --image path/to/mri_scan.png
"""

import os
import argparse
import json
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw
from torchvision import transforms
from model import NeuroScanCustomCNN, GradCAMExplainer

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
CLASS_DETAILS = {
    "glioma": {
        "name": "High-Grade Glioblastoma / Malignant Glioma",
        "severity": "CRITICAL MALIGNANT",
        "description": "Infiltrative parenchymal neoplasm with peripheral ring contrast enhancement and surrounding vasogenic edema.",
        "action": "Immediate neurosurgical oncology referral, surgical resection planning, and stereotactic biopsy."
    },
    "meningioma": {
        "name": "Dural Convexity / Parasagittal Meningioma",
        "severity": "CRITICAL EXTRA-AXIAL",
        "description": "Circumscribed extra-axial lesion displaying vivid homogeneous enhancement and dural tail sign.",
        "action": "Neurosurgical consultation for surgical excision / Simpson Grade evaluation and regular MRI surveillance."
    },
    "pituitary": {
        "name": "Pituitary Macroadenoma (Sellar Neoplasm)",
        "severity": "ELEVATED NEUROENDOCRINE CAUTION",
        "description": "Sellar and suprasellar expanding mass with remodeling of the sella turcica and optic chiasm proximity.",
        "action": "Visual field Humphrey perimetry, endocrine pituitary panel, and transsphenoidal endoscopic evaluation."
    },
    "notumor": {
        "name": "Normal Brain Parenchyma (Healthy Control)",
        "severity": "NORMAL / UNREMARKABLE",
        "description": "Symmetrical cerebral hemispheres, normal ventricular morphology, no focal mass lesion or pathological contrast enhancement.",
        "action": "Routine neurological follow-up as clinically indicated; no neurosurgical intervention required."
    }
}

def analyze_mri(image_path: str, model_path: str = None, norm_params_path: str = None, output_dir: str = None):
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if model_path is None:
        model_path = os.path.join(base_dir, "models", "neuroscan_best_model.pth")
    if norm_params_path is None:
        norm_params_path = os.path.join(base_dir, "data", "processed", "norm_params.json")
    if output_dir is None:
        output_dir = os.path.join(base_dir, "reports")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Load Normalization
    with open(norm_params_path, "r") as f:
        norm_data = json.load(f)
    mean = norm_data["train_fitted_mean"]
    std = norm_data["train_fitted_std"]

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    raw_img = Image.open(image_path).convert("RGB").resize((256, 256))
    input_tensor = transform(raw_img).unsqueeze(0).to(device)

    # Load Model Checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model = NeuroScanCustomCNN(num_classes=4).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Inference
    logits = model(input_tensor)
    probs = F.softmax(logits, dim=1).squeeze().detach().cpu().numpy()
    pred_idx = int(np.argmax(probs))
    pred_class = CLASSES[pred_idx]
    confidence = float(probs[pred_idx]) * 100.0

    # Grad-CAM
    explainer = GradCAMExplainer(model, model.get_gradcam_target_layer())
    cam_heatmap = explainer.generate(input_tensor, target_class=pred_idx)

    # Difference / Lesion Isolation Mask
    # Threshold Grad-CAM above 50% activation to delineate tumor portion
    tumor_mask = (cam_heatmap >= 0.50).astype(np.float32)
    has_tumor = pred_class != "notumor" and np.sum(tumor_mask) > 15

    # Compute detected centroid & bounding box
    if has_tumor:
        y_indices, x_indices = np.where(tumor_mask > 0)
        centroid_x = int(np.mean(x_indices))
        centroid_y = int(np.mean(y_indices))
        bbox = [int(np.min(x_indices)), int(np.min(y_indices)), int(np.max(x_indices)), int(np.max(y_indices))]
        area_mm2 = int(len(x_indices) * 1.45) # pixel-to-mm2 factor
    else:
        centroid_x, centroid_y = 128, 128
        bbox = [0, 0, 0, 0]
        area_mm2 = 0

    info = CLASS_DETAILS[pred_class]

    # Clinical Console Report
    print("=" * 70)
    print("                 NEUROSCAN-AI: CLINICAL RADIOLOGY REPORT               ")
    print("=" * 70)
    print(f"Target Scan File  : {os.path.basename(image_path)}")
    print(f"Primary Diagnosis : {info['name']}")
    print(f"Diagnostic Triage : [{info['severity']}]")
    print(f"Model Confidence  : {confidence:.2f}%")
    print("\nSoftmax Probability Distribution:")
    for i, c in enumerate(CLASSES):
        bar = "█" * int(probs[i] * 30)
        print(f"  • {c:<12}: {probs[i]*100:6.2f}% | {bar}")

    print("\nQuantitative Tumor Localization & Portion Biometrics:")
    if has_tumor:
        print(f"  • Detected Region Centroid : (X: {centroid_x}, Y: {centroid_y})")
        print(f"  • Bounding Reticle Box     : [X1: {bbox[0]}, Y1: {bbox[1]}, X2: {bbox[2]}, Y2: {bbox[3]}]")
        print(f"  • Estimated Cross-Section  : {area_mm2} mm²")
    else:
        print("  • No focal neoplastic lesion detected in intracranial compartments.")

    print(f"\nRadiological Finding:\n  {info['description']}")
    print(f"\nRecommended Clinical Action:\n  {info['action']}")
    print("=" * 70)

    # Generate Triad Diagnostic Output Image
    # 1. Raw MRI
    raw_np = np.array(raw_img)

    # 2. Jet Colormap for Grad-CAM
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    jet = mpl.colormaps.get_cmap("jet")
    cam_colored = (jet(cam_heatmap)[:, :, :3] * 255).astype(np.uint8)
    blend_np = (0.55 * raw_np + 0.45 * cam_colored).astype(np.uint8)

    # 3. Difference / Isolation Map (Neon Green Highlight on detected portion)
    diff_np = raw_np.copy()
    if has_tumor:
        diff_np[tumor_mask > 0, 1] = np.clip(diff_np[tumor_mask > 0, 1] + 130, 0, 255)
        diff_np[tumor_mask > 0, 0] = np.clip(diff_np[tumor_mask > 0, 0] - 40, 0, 255)
        diff_np[tumor_mask > 0, 2] = np.clip(diff_np[tumor_mask > 0, 2] - 40, 0, 255)

    triad = np.hstack([raw_np, blend_np, diff_np])
    triad_img = Image.fromarray(triad)
    draw_t = ImageDraw.Draw(triad_img)
    draw_t.text((10, 10), "1. Raw T1-CE MRI", fill=(255, 255, 255))
    draw_t.text((266, 10), "2. Grad-CAM Activation", fill=(255, 255, 255))
    draw_t.text((522, 10), "3. Detected Portion Difference", fill=(100, 255, 100))

    out_file = os.path.join(output_dir, "sample_prediction_output.png")
    os.makedirs(output_dir, exist_ok=True)
    triad_img.save(out_file)
    print(f"✓ Diagnostic visual triad saved to: {out_file}\n")

    return {
        "diagnosis": info["name"],
        "class_slug": pred_class,
        "confidence": confidence,
        "severity": info["severity"],
        "probabilities": {c: round(float(probs[i]) * 100, 2) for i, c in enumerate(CLASSES)},
        "centroid": {"x": centroid_x, "y": centroid_y},
        "bbox": bbox,
        "area_mm2": area_mm2
    }

def main():
    parser = argparse.ArgumentParser(description="NeuroScan-AI MRI Inference & Localization")
    parser.add_argument("--image", type=str, default=None, help="Path to input MRI scan image")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if args.image is None:
        # Default to a test glioma scan
        test_dir = os.path.join(base_dir, "data", "processed", "test", "glioma")
        if os.path.exists(test_dir) and len(os.listdir(test_dir)) > 0:
            sample_file = sorted(os.listdir(test_dir))[0]
            args.image = os.path.join(test_dir, sample_file)
        else:
            raise FileNotFoundError("No input image specified and test folder empty.")

    analyze_mri(args.image)

if __name__ == "__main__":
    main()
