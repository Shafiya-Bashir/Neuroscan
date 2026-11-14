"""
NEUROSCAN-AI: Raw Dataset Generator & Stratified 70/20/10 Split Pipeline
Author: Dar Rehan Rasool
"""

import os
import shutil
import json
import random
import numpy as np
from PIL import Image, ImageDraw

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]

# Clinically realistic imbalanced distribution
RAW_CLASS_COUNTS = {
    "glioma": 180,     # Highest prevalence (Malignant primary brain tumor)
    "meningioma": 140, # Dural extra-axial
    "notumor": 120,    # Normal healthy control scans
    "pituitary": 80    # Sellar adenoma (Rarest class in cohort)
}

def generate_anatomical_mri(label: str, seed: int) -> Image.Image:
    """Generates synthetic high-resolution axial brain MRI with authentic anatomical features."""
    random.seed(seed)
    np.random.seed(seed)
    
    img = Image.new("RGB", (256, 256), color=(4, 6, 12))
    draw = ImageDraw.Draw(img)
    
    # 1. Outer Cranium / Skull Bone
    draw.ellipse([24, 16, 232, 240], fill=(22, 28, 40), outline=(50, 60, 80), width=3)
    # Diploic space marrow
    draw.ellipse([28, 20, 228, 236], outline=(38, 46, 62), width=2)
    # Cerebral Parenchyma (Gray/White matter)
    draw.ellipse([34, 26, 222, 230], fill=(45, 55, 70))
    
    # 2. Lateral Ventricles (CSF hypointensity)
    cx, cy = 128, 128
    draw.ellipse([cx - 16, cy - 35, cx - 4, cy + 30], fill=(8, 12, 18))
    draw.ellipse([cx + 4, cy - 35, cx + 16, cy + 30], fill=(8, 12, 18))
    # Interhemispheric fissure
    draw.line([cx, 26, cx, 230], fill=(18, 24, 34), width=2)
    
    # 3. Pathological Lesions
    if label == "glioma":
        # Infiltrative necrotic ring in right frontotemporal region
        tx, ty = random.randint(145, 175), random.randint(80, 115)
        r = random.randint(22, 32)
        # Vasogenic edema halo
        draw.ellipse([tx - r*1.5, ty - r*1.5, tx + r*1.5, ty + r*1.5], fill=(95, 115, 145))
        # Hyperintense contrast ring
        draw.ellipse([tx - r, ty - r, tx + r, ty + r], fill=(210, 225, 250))
        # Central hypointense necrotic core
        draw.ellipse([tx - r*0.45, ty - r*0.45, tx + r*0.45, ty + r*0.45], fill=(25, 32, 45))
        
    elif label == "meningioma":
        # Dural convexity extra-axial mass with distinct dural tail
        tx, ty = random.randint(70, 95), random.randint(70, 105)
        r = random.randint(18, 26)
        # Well-circumscribed hyperintense mass
        draw.ellipse([tx - r, ty - r, tx + r, ty + r], fill=(225, 235, 255))
        # Dural attachment tail
        draw.line([tx - r, ty, tx - r - 16, ty - 14], fill=(200, 220, 245), width=3)
        
    elif label == "pituitary":
        # Sellar / suprasellar mass at skull base / optic chiasm
        tx, ty = cx + random.randint(-4, 4), cy + random.randint(12, 22)
        r = random.randint(12, 18)
        draw.ellipse([tx - r*1.2, ty - r*1.2, tx + r*1.2, ty + r*1.2], fill=(130, 150, 180))
        draw.ellipse([tx - r, ty - r, tx + r, ty + r], fill=(210, 225, 250))
        
    elif label == "notumor":
        # Perfectly symmetrical healthy ventricles and gray matter
        pass
        
    # Subtle gaussian noise to simulate MRI RF coil texture
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, 3.5, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    
    return Image.fromarray(arr)

def build_dataset_and_split():
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
    raw_dir = os.path.join(base_dir, "raw")
    train_dir = os.path.join(base_dir, "train")
    val_dir = os.path.join(base_dir, "val")
    test_dir = os.path.join(base_dir, "test")
    
    # Clean previous splits if any
    for d in [train_dir, val_dir, test_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)
        for c in CLASSES:
            os.makedirs(os.path.join(d, c), exist_ok=True)
            
    for c in CLASSES:
        os.makedirs(os.path.join(raw_dir, c), exist_ok=True)
        
    print(f"==================================================")
    print(f"1. Generating Raw Imbalanced Brain MRI Scans...")
    print(f"==================================================")
    
    all_files_by_class = {}
    
    for c in CLASSES:
        count = RAW_CLASS_COUNTS[c]
        all_files_by_class[c] = []
        for i in range(count):
            filename = f"{c}_{i+1:04d}.png"
            filepath = os.path.join(raw_dir, c, filename)
            
            # Generate and save raw image
            img = generate_anatomical_mri(c, seed=(hash(c) + i) % 100000)
            img.save(filepath)
            all_files_by_class[c].append(filename)
            
        print(f"  • Generated {count:3d} scans for Class: {c}")
        
    total_raw = sum(RAW_CLASS_COUNTS.values())
    print(f"\nTotal Raw Dataset Size: {total_raw} scans across 4 classes.")
    
    # 2. Compute 70 / 20 / 10 Stratified Split
    print(f"\n==================================================")
    print(f"2. Performing Stratified 70 / 20 / 10 Split...")
    print(f"==================================================")
    
    split_stats = {"train": {}, "val": {}, "test": {}, "total": {}}
    
    for c in CLASSES:
        files = all_files_by_class[c]
        random.seed(42)
        random.shuffle(files)
        
        n_total = len(files)
        n_train = int(0.70 * n_total)
        n_val = int(0.20 * n_total)
        n_test = n_total - n_train - n_val
        
        train_files = files[:n_train]
        val_files = files[n_train:n_train + n_val]
        test_files = files[n_train + n_val:]
        
        # Copy to train/val/test
        for f in train_files:
            shutil.copy(os.path.join(raw_dir, c, f), os.path.join(train_dir, c, f))
        for f in val_files:
            shutil.copy(os.path.join(raw_dir, c, f), os.path.join(val_dir, c, f))
        for f in test_files:
            shutil.copy(os.path.join(raw_dir, c, f), os.path.join(test_dir, c, f))
            
        split_stats["train"][c] = len(train_files)
        split_stats["val"][c] = len(val_files)
        split_stats["test"][c] = len(test_files)
        split_stats["total"][c] = n_total
        
        print(f"  [{c:12}] Total: {n_total:3d} | Train (70%): {n_train:3d} | Val (20%): {n_val:2d} | Test (10%): {n_test:2d}")
        
    # 3. Compute Class Imbalance Metrics & Weights for Cross-Entropy
    print(f"\n==================================================")
    print(f"3. Class Imbalance Analysis & Loss Weight Formulation...")
    print(f"==================================================")
    
    train_counts = [split_stats["train"][c] for c in CLASSES]
    total_train = sum(train_counts)
    num_classes = len(CLASSES)
    
    # Standard Inverse Class Frequency Weight: w_c = N_total / (num_classes * N_c)
    class_weights = {}
    for c in CLASSES:
        nc = split_stats["train"][c]
        wc = total_train / (num_classes * nc)
        class_weights[c] = round(wc, 4)
        pct = (nc / total_train) * 100
        print(f"  • {c:12}: {nc:3d} scans ({pct:4.1f}%) -> CrossEntropy Loss Weight: {wc:.4f}")
        
    max_count = max(train_counts)
    min_count = min(train_counts)
    imbalance_ratio = max_count / min_count
    print(f"\nImbalance Ratio (Max/Min Class): {imbalance_ratio:.2f}:1")
    print(f"Minority Class ('pituitary') is penalized {class_weights['pituitary']/class_weights['glioma']:.2f}x more heavily during training.")
    
    # Save metadata JSON for the training and evaluation scripts
    metadata = {
        "classes": CLASSES,
        "raw_counts": RAW_CLASS_COUNTS,
        "split_ratios": {"train": 0.70, "val": 0.20, "test": 0.10},
        "split_counts": split_stats,
        "class_weights": class_weights,
        "imbalance_ratio": round(imbalance_ratio, 2)
    }
    
    meta_path = os.path.join(base_dir, "dataset_stats.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
        
    print(f"\n✓ Dataset Manifest & Weights saved to: {meta_path}")
    print(f"==================================================")

if __name__ == "__main__":
    build_dataset_and_split()
