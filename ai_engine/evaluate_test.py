"""
NEUROSCAN-AI: Final Evaluation Pipeline on Untouched 10% Test Cohort
Author: Dar Rehan Rasool
Computes:
  - Confusion Matrix (4x4)
  - Per-Class Precision, Recall, F1-Score, Specificity
  - One-vs-Rest ROC-AUC & PR-AUC
  - Saves metrics to ai_engine/reports/evaluation_metrics.json
"""

import os
import json
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from model_scratch import NeuroScanCustomCNN

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    print("=" * 60)
    print("NEUROSCAN-AI: TEST SET EVALUATION (UNTOUCHED 10% COHORT)")
    print(f"Device: {device}")
    print("=" * 60)

    # 1. Load Dataset Stats
    stats_path = os.path.join(base_dir, "dataset", "dataset_stats.json")
    with open(stats_path, "r") as f:
        stats = json.load(f)
    classes = stats["classes"]

    # 2. Test Transform (Deterministic, No Augmentation)
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dir = os.path.join(base_dir, "dataset", "test")
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

    print(f"Loaded {len(test_dataset)} untouched test scans across {len(classes)} classes.")

    # 3. Load Model Checkpoint
    weights_path = os.path.join(base_dir, "weights", "neuroscan_scratch_model.pth")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Missing trained weights at {weights_path}.")

    checkpoint = torch.load(weights_path, map_location=device)
    model = NeuroScanCustomCNN(num_classes=len(classes)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"✓ Successfully loaded trained model from Epoch {checkpoint['epoch']} (Val Acc: {checkpoint['val_acc']:.2f}%)")

    # 4. Run Inference on Test Cohort
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_targets.extend(labels.numpy())

    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # 5. Compute Metrics
    acc = np.mean(all_preds == all_targets) * 100.0
    cm = confusion_matrix(all_targets, all_preds)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    report = classification_report(all_targets, all_preds, target_names=classes, output_dict=True)

    # Compute ROC and PR curves per class (One-vs-Rest)
    roc_data = {}
    pr_data = {}

    for i, c in enumerate(classes):
        binary_targets = (all_targets == i).astype(int)
        c_probs = all_probs[:, i]

        fpr, tpr, _ = roc_curve(binary_targets, c_probs)
        roc_auc = auc(fpr, tpr)
        roc_data[c] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "auc": round(float(roc_auc), 4)
        }

        precision, recall, _ = precision_recall_curve(binary_targets, c_probs)
        ap = average_precision_score(binary_targets, c_probs)
        pr_data[c] = {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "average_precision": round(float(ap), 4)
        }

    print("\n" + "=" * 60)
    print(f"TEST ACCURACY: {acc:.2f}%")
    print("=" * 60)
    print("\nPer-Class Breakdown:")
    for c in classes:
        print(f"  • {c:<12}: Precision: {report[c]['precision']*100:5.1f}% | "
              f"Recall: {report[c]['recall']*100:5.1f}% | "
              f"F1-Score: {report[c]['f1-score']*100:5.1f}% | "
              f"ROC-AUC: {roc_data[c]['auc']:.4f}")

    # 6. Save Evaluation Metrics JSON
    results = {
        "test_accuracy": round(float(acc), 2),
        "total_test_samples": int(len(test_dataset)),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalized": cm_norm.round(4).tolist(),
        "classification_report": report,
        "roc_curves": roc_data,
        "pr_curves": pr_data,
        "classes": classes
    }

    eval_path = os.path.join(base_dir, "reports", "evaluation_metrics.json")
    with open(eval_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Saved Comprehensive Test Evaluation to: {eval_path}")

if __name__ == "__main__":
    main()
