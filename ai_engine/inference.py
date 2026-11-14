"""
NEUROSCAN-AI: PyTorch Inference & Grad-CAM Visualization CLI
Author: Dar Rehan Rasool
Supports:
  1. From-Scratch Custom CNN (NeuroScanCustomCNN - Zero Pre-trained Weights)
  2. DenseNet-121 Benchmark Model
Usage:
  python3 inference.py --image dataset/test/glioma/glioma_0001.png --model scratch
"""

import argparse
import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from model_scratch import NeuroScanCustomCNN
from model import NeuroScanDenseNet, CLASS_NAMES
from gradcam import GradCAM

CLASS_NAMES_LOWER = ["glioma", "meningioma", "notumor", "pituitary"]

def run_inference(image_path: str, model_type: str = "scratch"):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("\n" + "=" * 60)
    print("🔬 NEUROSCAN-AI Clinical Neural Inference Engine")
    print(f"Target Architecture : {'Custom CNN (From Scratch)' if model_type == 'scratch' else 'DenseNet-121'}")
    print(f"Inference Device    : {device}")
    print(f"Target MRI Input    : {image_path}")
    print("=" * 60)

    if not os.path.exists(image_path):
        print(f"Error: Image '{image_path}' not found.")
        return

    # 1. Initialize Architecture & Load Weights
    if model_type == "scratch":
        weights_path = os.path.join(base_dir, "weights", "neuroscan_scratch_model.pth")
        model = NeuroScanCustomCNN(num_classes=4)
        img_size = 256
        target_layer = model.get_gradcam_target_layer()
    else:
        weights_path = os.path.join(base_dir, "weights", "neuroscan_densenet121.pth")
        model = NeuroScanDenseNet(num_classes=4, pretrained=False)
        img_size = 224
        target_layer = model.get_target_layer()

    if os.path.exists(weights_path):
        ckpt = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"✓ Loaded Verified Checkpoint: {weights_path}")
    else:
        print(f"Warning: Checkpoint not found at {weights_path}, using randomized weights.")

    model.to(device)
    model.eval()

    # 2. Image Preprocessing Transformation Pipeline
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    raw_img = Image.open(image_path).convert("RGB")
    tensor_img = transform(raw_img).unsqueeze(0).to(device)

    # 3. Deep Learning Forward Pass
    logits = model(tensor_img)
    probs = torch.softmax(logits, dim=1).squeeze().detach().cpu().numpy()
    pred_idx = np.argmax(probs)
    
    names = CLASS_NAMES if model_type != "scratch" else [c.capitalize() for c in CLASS_NAMES_LOWER]
    pred_class = names[pred_idx]
    confidence = probs[pred_idx] * 100.0

    print("\n🧠 DIAGNOSTIC CLASSIFICATION RESULT:")
    print("-" * 60)
    print(f"► Primary Diagnosis   : {pred_class}")
    print(f"► Prediction Confidence: {confidence:.2f}%")
    print(f"► Severity Triage Status: {'CRITICAL MALIGNANT (RED ALERT)' if pred_class.lower() in ['glioma', 'meningioma'] else 'HEALTHY (GREEN CHIME)' if 'no' in pred_class.lower() else 'CAUTION / SELLAR (AMBER PING)'}")
    print("\n📊 Softmax Class Probability Distribution:")
    for i, c in enumerate(names):
        bar_len = int(probs[i] * 35)
        bar = "█" * bar_len + "░" * (35 - bar_len)
        print(f"  • {c:<12}: {probs[i]*100:6.2f}% |{bar}|")

    # 4. Grad-CAM Feature Attribution Heatmap
    gradcam = GradCAM(model, target_layer)
    cam = gradcam.generate(tensor_img, target_class=pred_idx)
    out_cam_path = os.path.join(base_dir, "output_gradcam.png")
    gradcam.overlay_heatmap(raw_img, cam, save_path=out_cam_path, colormap="jet", alpha=0.55)
    print(f"\n✓ Explainable AI: Grad-CAM Saliency Map Saved to -> {out_cam_path}")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NeuroScan-AI Model Inference CLI")
    parser.add_argument("--image", type=str, default="sample_mri/glioma_sample.png", help="Path to MRI image")
    parser.add_argument("--model", type=str, choices=["scratch", "densenet"], default="scratch", help="Architecture to run")
    args = parser.parse_args()

    run_inference(args.image, args.model)
