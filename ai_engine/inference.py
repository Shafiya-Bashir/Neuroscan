"""
NEUROSCAN-AI: PyTorch Inference & Grad-CAM Visualization CLI
Author: Dar Rehan Rasool
Usage:
    python3 inference.py --image sample_mri/glioma_sample.png
"""

import argparse
import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from model import NeuroScanDenseNet, CLASS_NAMES
from gradcam import GradCAM

def run_inference(image_path: str, weights_path: str = "weights/neuroscan_densenet121.pth"):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f"🔬 NEUROSCAN-AI Clinical Neural Inference Engine")
    print(f"Inference Device: {device}")
    print(f"Image Input: {image_path}")
    print(f"=======================================================")
    
    if not os.path.exists(image_path):
        print(f"Error: Image '{image_path}' not found.")
        return
        
    # 1. Load Pretrained PyTorch Architecture
    model = NeuroScanDenseNet(num_classes=4, pretrained=False)
    if os.path.exists(weights_path):
        ckpt = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"✓ Loaded Deep Learning Weights from: {weights_path}")
    model.to(device)
    model.eval()
    
    # 2. Image Preprocessing Transformation Pipeline
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    raw_img = Image.open(image_path).convert("RGB")
    tensor_img = transform(raw_img).unsqueeze(0).to(device)
    
    # 3. Deep Learning Forward Pass
    logits = model(tensor_img)
    probs = torch.softmax(logits, dim=1).squeeze().detach().cpu().numpy()
    pred_idx = np.argmax(probs)
    pred_class = CLASS_NAMES[pred_idx]
    confidence = probs[pred_idx] * 100.0
    
    print(f"\n🧠 DIAGNOSTIC CLASSIFICATION RESULT:")
    print(f"-------------------------------------------------------")
    print(f"► Primary Diagnosis : {pred_class}")
    print(f"► Prediction Confidence: {confidence:.2f}%")
    print(f"► Severity Status      : {'CRITICAL MALIGNANT (RED)' if pred_class in ['Glioma', 'Meningioma'] else 'HEALTHY (GREEN)'}")
    print(f"\n📊 Softmax Class Probability Distribution:")
    for idx, name in enumerate(CLASS_NAMES):
        bar_len = int(probs[idx] * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        print(f"  {name:12}: {probs[idx]*100:5.2f}% |{bar}|")
        
    # 4. Explainable AI (Grad-CAM) Backpropagation
    print(f"\n🔍 Executing Grad-CAM Backpropagation on target layer...")
    gradcam_engine = GradCAM(model, model.get_target_layer())
    cam_heatmap = gradcam_engine.generate(tensor_img, target_class=pred_idx)
    
    # Save Heatmap Output Image
    out_cam_path = "output_gradcam.png"
    # Convert normalized heatmap to RGB
    heat_uint8 = (cam_heatmap * 255).astype(np.uint8)
    cam_pil = Image.fromarray(heat_uint8).resize(raw_img.size)
    cam_pil.save(out_cam_path)
    
    print(f"✓ Grad-CAM Neural Attention Map successfully saved to: {out_cam_path}")
    print(f"=======================================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NeuroScan-AI CLI Inference")
    parser.add_argument("--image", type=str, default="sample_mri/glioma_sample.png", help="Path to Brain MRI scan")
    parser.add_argument("--weights", type=str, default="weights/neuroscan_densenet121.pth", help="Path to model weights")
    args = parser.parse_args()
    
    run_inference(args.image, args.weights)
