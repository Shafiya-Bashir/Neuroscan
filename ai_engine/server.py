"""
NEUROSCAN-AI: FastAPI Live PyTorch Model Server
Serves real-time inference and Grad-CAM tensor heatmaps to the web dashboard.
"""

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
import numpy as np
from PIL import Image
import io
import base64
import torchvision.transforms as transforms
from model import NeuroScanDenseNet, CLASS_NAMES
from gradcam import GradCAM

app = FastAPI(title="NeuroScan-AI PyTorch Inference Server")

# Enable CORS for Next.js web application
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model & device setup
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Loading NeuroScan DenseNet-121 on device: {device}")

model = NeuroScanDenseNet(num_classes=4, pretrained=False)
weights_path = "weights/neuroscan_densenet121.pth"
try:
    ckpt = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"Loaded weights from {weights_path}")
except Exception as e:
    print(f"Warning: using uninitialized weights ({e})")

model.to(device)
model.eval()

# Preprocessing transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

gradcam_engine = GradCAM(model, model.get_target_layer())

@app.get("/")
def health_check():
    return {
        "status": "online",
        "model": "DenseNet-121",
        "device": str(device),
        "classes": CLASS_NAMES
    }

@app.post("/predict")
async def predict_mri(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    
    # Forward Pass
    logits = model(tensor)
    probs = torch.softmax(logits, dim=1).squeeze().detach().cpu().numpy()
    pred_idx = int(np.argmax(probs))
    pred_class = CLASS_NAMES[pred_idx]
    
    # Grad-CAM Backprop
    cam_heatmap = gradcam_engine.generate(tensor, target_class=pred_idx)
    
    # Encode heatmap to base64
    heat_uint8 = (cam_heatmap * 255).astype(np.uint8)
    heat_pil = Image.fromarray(heat_uint8).resize((256, 256))
    buffered = io.BytesIO()
    heat_pil.save(buffered, format="PNG")
    heatmap_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    return {
        "diagnosis": pred_class,
        "confidence": round(float(probs[pred_idx]) * 100, 2),
        "severity": "danger" if pred_class in ["Glioma", "Meningioma"] else "warning" if pred_class == "Pituitary" else "healthy",
        "probabilities": {
            "glioma": round(float(probs[0]) * 100, 2),
            "meningioma": round(float(probs[1]) * 100, 2),
            "pituitary": round(float(probs[2]) * 100, 2),
            "healthy": round(float(probs[3]) * 100, 2),
        },
        "gradcam_heatmap_base64": f"data:image/png;base64,{heatmap_b64}",
        "device": str(device)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
