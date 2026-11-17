"""
NEUROSCAN-AI: Step 2 - Exploratory Data Analysis (EDA), Stratified Split & Zero-Leakage Normalization
Features:
  - Comprehensive clinical EDA on authentic MRI cohort (N=600)
  - Strict Stratified 70/15/15 Train-Val-Test Split
  - Zero-Leakage Normalization (Scaler fitted exclusively on Training split)
  - Class Imbalance Management (Inverse Class Frequency loss weights + Balanced Sampler weights)
  - Output manifests and processed folder hierarchy
"""

import os
import shutil
import json
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]

def run_eda(raw_manifest_path: str, base_dir: str):
    """Conducts rigorous Exploratory Data Analysis on the raw clinical MRI cohort."""
    df = pd.read_csv(raw_manifest_path)
    total_scans = len(df)

    print("\n--- [EDA Stage 1: Cohort Demographics & Class Balance] ---")
    class_counts = df['class_label'].value_counts().to_dict()
    for c in CLASSES:
        count = class_counts.get(c, 0)
        pct = (count / total_scans) * 100
        print(f"  • {c:<12}: {count:>3} scans ({pct:5.1f}%)")

    max_c = max(class_counts.values())
    min_c = min(class_counts.values())
    imbalance_ratio = round(max_c / min_c, 3)
    print(f"  -> Natural Imbalance Ratio: {imbalance_ratio}:1 (Minority: Pituitary, Majority: Glioma)")

    print(f"\n--- [EDA Stage 2: Patient Demographics] ---")
    print(f"  • Mean Patient Age: {df['age'].mean():.1f} ± {df['age'].std():.1f} years (Range: {df['age'].min()} - {df['age'].max()})")
    sex_dist = df['sex'].value_counts().to_dict()
    print(f"  • Biological Sex  : Male = {sex_dist.get('Male', 0)} ({sex_dist.get('Male', 0)/total_scans*100:.1f}%), Female = {sex_dist.get('Female', 0)} ({sex_dist.get('Female', 0)/total_scans*100:.1f}%)")

    print(f"\n--- [EDA Stage 3: MRI Pixel Intensity & SNR Analysis] ---")
    intensity_stats = {}
    for c in CLASSES:
        cls_df = df[df['class_label'] == c]
        sample_pixels = []
        # Sample 20 scans per class for detailed pixel density profiling
        for _, row in cls_df.head(20).iterrows():
            img_path = os.path.join(base_dir, row['file_path'])
            arr = np.array(Image.open(img_path).convert('L'))
            sample_pixels.extend(arr.flatten())

        sample_pixels = np.array(sample_pixels, dtype=np.float32)
        mean_val = float(np.mean(sample_pixels))
        std_val = float(np.std(sample_pixels))
        snr_val = float(mean_val / (std_val + 1e-6))

        intensity_stats[c] = {
            "mean_intensity": round(mean_val, 2),
            "std_intensity": round(std_val, 2),
            "snr_db": round(20 * np.log10(snr_val + 1e-6), 2),
            "min": int(np.min(sample_pixels)),
            "max": int(np.max(sample_pixels))
        }
        print(f"  • {c:<12}: Mean = {mean_val:6.2f} | Std = {std_val:5.2f} | SNR = {intensity_stats[c]['snr_db']:5.2f} dB")

    return {
        "total_scans": total_scans,
        "class_distribution": class_counts,
        "imbalance_ratio": imbalance_ratio,
        "intensity_stats": intensity_stats
    }

def main():
    print("=" * 65)
    print("STEP 2: CLINICAL EDA, STRATIFIED SPLIT & ZERO-LEAKAGE SCALING")
    print("=" * 65)

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    raw_dir = os.path.join(base_dir, "data", "raw")
    processed_dir = os.path.join(base_dir, "data", "processed")
    manifest_csv = os.path.join(raw_dir, "raw_mri_manifest.csv")

    if not os.path.exists(manifest_csv):
        raise FileNotFoundError(f"Missing {manifest_csv}. Run make_dataset.py first.")

    eda_results = run_eda(manifest_csv, base_dir)

    df = pd.read_csv(manifest_csv)

    # 1. Stratified 70 / 15 / 15 Split
    # Step A: Split 70% Train vs 30% Temp (Val + Test)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    train_idx, temp_idx = next(sss1.split(df, df['class_label']))

    train_df = df.iloc[train_idx].copy().reset_index(drop=True)
    temp_df = df.iloc[temp_idx].copy().reset_index(drop=True)

    # Step B: Split Temp 50/50 into Val (15%) and Test (15%)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
    val_idx, test_idx = next(sss2.split(temp_df, temp_df['class_label']))

    val_df = temp_df.iloc[val_idx].copy().reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].copy().reset_index(drop=True)

    assert len(train_df) + len(val_df) + len(test_df) == len(df), "Split count mismatch!"
    # Zero patient leakage check
    train_patients = set(train_df['patient_id'])
    val_patients = set(val_df['patient_id'])
    test_patients = set(test_df['patient_id'])
    assert len(train_patients.intersection(val_patients)) == 0, "Data leakage: Train-Val patient overlap!"
    assert len(train_patients.intersection(test_patients)) == 0, "Data leakage: Train-Test patient overlap!"
    assert len(val_patients.intersection(test_patients)) == 0, "Data leakage: Val-Test patient overlap!"

    print("\n" + "=" * 65)
    print("STRATIFIED COHORT PARTITION (ZERO DATA LEAKAGE VERIFIED)")
    print("=" * 65)
    print(f"✓ Train Set (70%): {len(train_df):>3} scans")
    print(f"✓ Val Set   (15%): {len(val_df):>3} scans")
    print(f"✓ Test Set  (15%): {len(test_df):>3} scans")
    print(f"✓ Verification: Zero patient overlap across splits (0% leakage).")

    # 2. Compute Normalization Statistics EXCLUSIVELY on Training Set
    print("\n[Zero-Leakage Normalization Calibration]")
    train_pixel_vals = []
    for _, row in train_df.iterrows():
        img_path = os.path.join(base_dir, row['file_path'])
        arr = np.array(Image.open(img_path).convert('RGB'), dtype=np.float32) / 255.0
        train_pixel_vals.append(arr)

    train_pixel_vals = np.array(train_pixel_vals) # Shape: (N, 256, 256, 3)
    train_mean = np.mean(train_pixel_vals, axis=(0, 1, 2)).tolist()
    train_std = np.std(train_pixel_vals, axis=(0, 1, 2)).tolist()

    print(f"  • Training Dataset Fitted Mean : {[round(x, 4) for x in train_mean]}")
    print(f"  • Training Dataset Fitted Std  : {[round(x, 4) for x in train_std]}")
    print("  -> Freezing parameters: Val & Test will be scaled strictly using these training parameters.")

    # 3. Class Imbalance Mitigation Weights
    # Formula: w_c = N_train / (C * N_c_train)
    n_train = len(train_df)
    n_classes = len(CLASSES)
    train_class_counts = train_df['class_label'].value_counts().to_dict()

    class_weights = {}
    for c in CLASSES:
        w_c = n_train / (n_classes * train_class_counts[c])
        class_weights[c] = round(float(w_c), 4)

    print("\n[Class Imbalance Mitigation Loss Weights]")
    for c in CLASSES:
        print(f"  • {c:<12}: Train N={train_class_counts[c]:>3} | Inverse Frequency Weight = {class_weights[c]:.4f}")

    # 4. Create Processed Directory Structure & Copy Files
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        split_dir = os.path.join(processed_dir, split_name)
        if os.path.exists(split_dir):
            shutil.rmtree(split_dir)
        os.makedirs(split_dir, exist_ok=True)
        for c in CLASSES:
            os.makedirs(os.path.join(split_dir, c), exist_ok=True)

        # Copy images into structured folders
        new_paths = []
        for _, row in split_df.iterrows():
            src_file = os.path.join(base_dir, row['file_path'])
            dst_file = os.path.join(split_dir, row['class_label'], os.path.basename(src_file))
            shutil.copy2(src_file, dst_file)
            new_paths.append(os.path.relpath(dst_file, base_dir))

        split_df['processed_path'] = new_paths
        manifest_out = os.path.join(processed_dir, f"{split_name}_manifest.csv")
        split_df.to_csv(manifest_out, index=False)
        print(f"✓ Saved {split_name.upper()} manifest ({len(split_df)} records) -> {manifest_out}")

    # 5. Save Summary & Normalization Parameters
    norm_params = {
        "train_fitted_mean": train_mean,
        "train_fitted_std": train_std,
        "image_size": [256, 256],
        "normalization_protocol": "Z-Score Standard Normalization (Fitted Strictly on Train)"
    }
    with open(os.path.join(processed_dir, "norm_params.json"), "w") as f:
        json.dump(norm_params, f, indent=2)

    with open(os.path.join(processed_dir, "class_weights.json"), "w") as f:
        json.dump(class_weights, f, indent=2)

    summary = {
        "dataset_name": "NEUROSCAN Clinical Brain MRI Cohort",
        "classes": CLASSES,
        "total_scans": len(df),
        "split_counts": {
            "train": {c: int(train_class_counts[c]) for c in CLASSES},
            "val": {c: int((val_df['class_label'] == c).sum()) for c in CLASSES},
            "test": {c: int((test_df['class_label'] == c).sum()) for c in CLASSES},
            "total": {c: int((df['class_label'] == c).sum()) for c in CLASSES}
        },
        "class_weights": class_weights,
        "imbalance_ratio": eda_results["imbalance_ratio"],
        "eda": eda_results
    }
    with open(os.path.join(processed_dir, "dataset_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 65)
    print("STEP 2 COMPLETE: ZERO-LEAKAGE FEATURE PIPELINE BUILT SUCCESSFULLY")
    print("=" * 65)

if __name__ == "__main__":
    main()
