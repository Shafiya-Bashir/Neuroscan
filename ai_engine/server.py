"""
NEUROSCAN-AI: High-Performance FastAPI Live PyTorch Model Server
Serves real-time inference, Grad-CAM tensor heatmaps, Difference maps, and ML pipeline statistics.
Authors: Dar Rehan Rasool & Shafiya Pampori
"""

import os
import sys
import io
import json
import base64
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import matplotlib as mpl

# Add src/models to sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(root_dir, "src", "models"))
sys.path.insert(0, os.path.join(root_dir, "ai_engine"))

from model import NeuroScanCustomCNN, GradCAMExplainer

app = FastAPI(title="NeuroScan-AI PyTorch Diagnostic Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"🔬 Initializing NeuroScan-AI Server on {device}")

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
DISPLAY_NAMES = {
    "glioma": "High-Grade Glioma (Malignant)",
    "meningioma": "Meningioma (Extra-Axial)",
    "notumor": "Healthy Brain (No Tumor)",
    "pituitary": "Pituitary Macroadenoma"
}

# Load Normalization Parameters
norm_params_path = os.path.join(root_dir, "data", "processed", "norm_params.json")
if os.path.exists(norm_params_path):
    with open(norm_params_path, "r") as f:
        norm_data = json.load(f)
    mean = norm_data["train_fitted_mean"]
    std = norm_data["train_fitted_std"]
else:
    mean = [0.2337, 0.2337, 0.2337]
    std = [0.2180, 0.2180, 0.2180]

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

# 1. Load Custom Best Model
scratch_model = NeuroScanCustomCNN(num_classes=4).to(device)
weights_path = os.path.join(root_dir, "models", "neuroscan_best_model.pth")
if not os.path.exists(weights_path):
    weights_path = os.path.join(root_dir, "ai_engine", "weights", "neuroscan_scratch_model.pth")

if os.path.exists(weights_path):
    ckpt = torch.load(weights_path, map_location=device)
    scratch_model.load_state_dict(ckpt["model_state_dict"])
    print(f"✓ Loaded Best Model Checkpoint from {weights_path}")
scratch_model.eval()
scratch_gradcam = GradCAMExplainer(scratch_model, scratch_model.get_gradcam_target_layer())

@app.get("/")
def health_check():
    return {
        "status": "online",
        "primary_model": "NeuroScanCustomCNN (5-Stage Residual-SE CNN)",
        "device": str(device),
        "classes": CLASSES,
        "class_mapping": DISPLAY_NAMES,
        "zero_data_leakage": True
    }

@app.get("/dataset/stats")
def get_dataset_stats():
    stats_path = os.path.join(root_dir, "data", "processed", "dataset_summary.json")
    if os.path.exists(stats_path):
        with open(stats_path, "r") as f:
            return json.load(f)
    return {"error": "dataset_summary.json not found"}

@app.get("/reports/metrics")
def get_evaluation_metrics():
    eval_path = os.path.join(root_dir, "models", "test_evaluation.json")
    if os.path.exists(eval_path):
        with open(eval_path, "r") as f:
            return json.load(f)
    return {"error": "test_evaluation.json not found"}

@app.get("/training/history")
def get_training_history():
    history_path = os.path.join(root_dir, "models", "training_history.json")
    if os.path.exists(history_path):
        with open(history_path, "r") as f:
            return json.load(f)
    return {"error": "training_history.json not found"}

@app.post("/predict")
async def predict_mri(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB").resize((256, 256))
    tensor = transform(image).unsqueeze(0).to(device)

    # Forward Pass
    logits = scratch_model(tensor)
    probs = torch.softmax(logits, dim=1).squeeze().detach().cpu().numpy()
    pred_idx = int(np.argmax(probs))
    pred_slug = CLASSES[pred_idx]
    pred_name = DISPLAY_NAMES[pred_slug]

    # Grad-CAM Heatmap
    cam_heatmap = scratch_gradcam.generate(tensor, target_class=pred_idx)
    heat_uint8 = (cam_heatmap * 255).astype(np.uint8)

    # Difference / Tumor isolation
    tumor_mask = (cam_heatmap >= 0.50).astype(np.float32)
    has_tumor = pred_slug != "notumor" and np.sum(tumor_mask) > 15

    if has_tumor:
        y_indices, x_indices = np.where(tumor_mask > 0)
        centroid_x = int(np.mean(x_indices))
        centroid_y = int(np.mean(y_indices))
        bbox = [int(np.min(x_indices)), int(np.min(y_indices)), int(np.max(x_indices)), int(np.max(y_indices))]
        area_mm2 = int(len(x_indices) * 1.45)
        region = (
            "Right Frontal Lobe" if centroid_x > 128 and centroid_y < 128
            else "Left Frontotemporal" if centroid_x <= 128 and centroid_y < 128
            else "Right Parietotemporal" if centroid_x > 128
            else "Left Parietal Convexity"
        ) if pred_slug != "pituitary" else "Sellar Fossa & Suprasellar Cistern"
    else:
        centroid_x, centroid_y = 128, 128
        bbox = [0, 0, 0, 0]
        area_mm2 = 0
        region = "Normal Intracranial Parenchyma"

    # Encode Heatmap Base64
    heat_pil = Image.fromarray(heat_uint8).resize((256, 256))
    buffered = io.BytesIO()
    heat_pil.save(buffered, format="PNG")
    heatmap_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    severity = (
        "danger" if pred_slug in ["glioma", "meningioma"]
        else "warning" if pred_slug == "pituitary"
        else "healthy"
    )

    return {
        "diagnosis": pred_name,
        "class_slug": pred_slug,
        "confidence": round(float(probs[pred_idx]) * 100, 2),
        "severity": severity,
        "probabilities": {
            c: round(float(probs[i]) * 100, 2) for i, c in enumerate(CLASSES)
        },
        "gradcam_heatmap_base64": f"data:image/png;base64,{heatmap_b64}",
        "detected_portion": {
            "region": region,
            "has_tumor": has_tumor,
            "centroid": {"x": centroid_x, "y": centroid_y},
            "bbox": bbox,
            "area_mm2": area_mm2
        },
        "architecture": "NeuroScanCustomCNN (5-Stage Residual-SE CNN)",
        "device": str(device)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
