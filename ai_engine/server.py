"""
NEUROSCAN-AI: High-Performance FastAPI Live PyTorch Model Server
Serves real-time inference, Grad-CAM tensor heatmaps, and ML pipeline statistics.
Authors: Dar Rehan Rasool & Shafiya Pampori
"""

import os
import io
import json
import base64
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from model_scratch import NeuroScanCustomCNN
from model import NeuroScanDenseNet
from gradcam import GradCAM

app = FastAPI(title="NeuroScan-AI PyTorch Diagnostic Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

base_dir = os.path.dirname(os.path.abspath(__file__))
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"🔬 Initializing NeuroScan-AI Server on {device}")

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
DISPLAY_NAMES = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "notumor": "Healthy (No Tumor)",
    "pituitary": "Pituitary Adenoma"
}

# 1. Load Custom From-Scratch Model
scratch_model = NeuroScanCustomCNN(num_classes=4).to(device)
scratch_weights_path = os.path.join(base_dir, "weights", "neuroscan_scratch_model.pth")
if os.path.exists(scratch_weights_path):
    ckpt = torch.load(scratch_weights_path, map_location="cpu")
    scratch_model.load_state_dict(ckpt["model_state_dict"])
    print(f"✓ Loaded Custom From-Scratch Model from {scratch_weights_path}")
scratch_model.eval()
scratch_gradcam = GradCAM(scratch_model, scratch_model.get_gradcam_target_layer())

# Transform for Scratch Model (256x256)
scratch_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.get("/")
def health_check():
    return {
        "status": "online",
        "primary_model": "NeuroScanCustomCNN (Trained From Scratch)",
        "device": str(device),
        "classes": CLASSES,
        "class_mapping": DISPLAY_NAMES
    }

@app.get("/dataset/stats")
def get_dataset_stats():
    stats_path = os.path.join(base_dir, "dataset", "dataset_stats.json")
    if os.path.exists(stats_path):
        with open(stats_path, "r") as f:
            return json.load(f)
    return {"error": "dataset_stats.json not found"}

@app.get("/reports/metrics")
def get_evaluation_metrics():
    eval_path = os.path.join(base_dir, "reports", "evaluation_metrics.json")
    if os.path.exists(eval_path):
        with open(eval_path, "r") as f:
            return json.load(f)
    return {"error": "evaluation_metrics.json not found"}

@app.get("/training/history")
def get_training_history():
    history_path = os.path.join(base_dir, "reports", "training_history.json")
    if os.path.exists(history_path):
        with open(history_path, "r") as f:
            return json.load(f)
    return {"error": "training_history.json not found"}

@app.post("/predict")
async def predict_mri(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    tensor = scratch_transform(image).unsqueeze(0).to(device)

    # Forward Pass
    logits = scratch_model(tensor)
    probs = torch.softmax(logits, dim=1).squeeze().detach().cpu().numpy()
    pred_idx = int(np.argmax(probs))
    pred_slug = CLASSES[pred_idx]
    pred_name = DISPLAY_NAMES[pred_slug]

    # Grad-CAM Heatmap
    cam_heatmap = scratch_gradcam.generate(tensor, target_class=pred_idx)
    heat_uint8 = (cam_heatmap * 255).astype(np.uint8)
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
        "architecture": "NeuroScanCustomCNN (4-Stage ConvNet From Scratch)",
        "device": str(device)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
