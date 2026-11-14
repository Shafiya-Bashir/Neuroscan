"""
NEUROSCAN-AI: From-Scratch Deep CNN Training Pipeline with Class Imbalance Resolution
Author: Dar Rehan Rasool
Features:
  - 100% From-Scratch Model (No Pre-trained Weights)
  - 70/20/10 Train/Val/Test Split Enforcement
  - Two-Tier Class Imbalance Resolution:
      1. WeightedRandomSampler for Mini-Batch Equilibrium
      2. Inverse Class-Frequency Weighted CrossEntropyLoss
  - Device Agnostic: Apple Silicon MPS / CUDA / CPU
  - Comprehensive Epoch-by-Epoch Metric Tracking
"""

import os
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from model_scratch import NeuroScanCustomCNN

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def main():
    device = get_device()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("=" * 60)
    print("NEUROSCAN-AI: FROM-SCRATCH TRAINING PIPELINE")
    print(f"Target Hardware Acceleration: {device.type.upper()}")
    print("=" * 60)

    # 1. Load Dataset Statistics & Class Weights
    stats_path = os.path.join(base_dir, "dataset", "dataset_stats.json")
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Missing {stats_path}. Please execute prepare_data.py first.")

    with open(stats_path, "r") as f:
        stats = json.load(f)

    classes = stats["classes"]
    weights_dict = stats["class_weights"]
    class_weights = torch.tensor([weights_dict[c] for c in classes], dtype=torch.float32).to(device)

    print("\n[Stage 1] Class Distribution & Inverse Frequency Weights:")
    for c in classes:
        print(f"  • {c:<12}: {stats['split_counts']['train'][c]:>3} train scans | Loss Weight = {weights_dict[c]:.4f}")
    print(f"  -> Imbalance Ratio (Max/Min): {stats['imbalance_ratio']}")

    # 2. Data Transforms (Anatomical MRI Data Augmentation)
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=12),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 3. Load Datasets
    train_dir = os.path.join(base_dir, "dataset", "train")
    val_dir = os.path.join(base_dir, "dataset", "val")

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    # 4. Resolve Imbalance with WeightedRandomSampler
    # Calculate sample weights for every individual item in the training set
    train_targets = [sample[1] for sample in train_dataset.samples]
    class_counts = [stats['split_counts']['train'][classes[i]] for i in range(len(classes))]
    sample_weights = [1.0 / class_counts[target] for target in train_targets]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"\n[Stage 2] Data Loaders Initialized:")
    print(f"  • Train Set: {len(train_dataset)} scans ({len(train_loader)} batches/epoch with Balanced Sampler)")
    print(f"  • Val Set  : {len(val_dataset)} scans ({len(val_loader)} batches/epoch)")

    # 5. Initialize Model From Scratch
    model = NeuroScanCustomCNN(num_classes=len(classes)).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[Stage 3] Model Architecture Initialized (Kaiming Normal):")
    print(f"  • Model Type: NeuroScanCustomCNN (4-Stage ConvNet)")
    print(f"  • Trainable Parameters: {total_params:,} (Zero Pre-trained Weights)")

    # 6. Criterion & Optimization
    # Combined with class weights to penalize minority misclassification
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    epochs = 15
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    history = {
        "epochs": [],
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "learning_rate": []
    }

    best_val_acc = 0.0
    weights_dir = os.path.join(base_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)
    best_weights_path = os.path.join(weights_dir, "neuroscan_scratch_model.pth")

    print("\n" + "=" * 60)
    print("BEGINNING 15-EPOCH FROM-SCRATCH TRAINING")
    print("=" * 60)

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        running_train_loss = 0.0
        correct_train = 0
        total_train = 0

        current_lr = optimizer.param_groups[0]["lr"]

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += torch.sum(preds == labels).item()
            total_train += labels.size(0)

        epoch_train_loss = running_train_loss / total_train
        epoch_train_acc = (correct_train / total_train) * 100.0

        # Validation Phase
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                running_val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val += torch.sum(preds == labels).item()
                total_val += labels.size(0)

        epoch_val_loss = running_val_loss / total_val
        epoch_val_acc = (correct_val / total_val) * 100.0

        scheduler.step()

        # Record History
        history["epochs"].append(epoch)
        history["train_loss"].append(round(epoch_train_loss, 4))
        history["val_loss"].append(round(epoch_val_loss, 4))
        history["train_acc"].append(round(epoch_train_acc, 2))
        history["val_acc"].append(round(epoch_val_acc, 2))
        history["learning_rate"].append(current_lr)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] "
              f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:6.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:6.2f}% | "
              f"LR: {current_lr:.6f}")

        # Checkpoint Best Model
        if epoch_val_acc >= best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": best_val_acc,
                "val_loss": epoch_val_loss,
                "classes": classes,
                "class_weights": weights_dict
            }, best_weights_path)

    elapsed = time.time() - start_time
    print("=" * 60)
    print(f"✓ Training Completed in {elapsed:.1f}s")
    print(f"✓ Optimal Model Checkpoint Saved: {best_weights_path} (Val Acc: {best_val_acc:.2f}%)")

    # Save History JSON for Graph Generation
    reports_dir = os.path.join(base_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    history_path = os.path.join(reports_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"✓ Epoch-by-Epoch Metric History Saved: {history_path}")

if __name__ == "__main__":
    main()
