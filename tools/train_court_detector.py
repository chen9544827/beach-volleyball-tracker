# tools/train_court_detector.py
# -*- coding: utf-8 -*-
"""
Train YOLO Keypoint Model for Court Detection

Trains a YOLOv8-pose model to detect 6 court keypoints:
    far_left, far_right, near_left, near_right, net_left, net_right

Uses transfer learning from yolov8m-pose.pt (already in project).
Key training settings tuned for court detection:
    - mosaic=0.0 (court must be fully visible)
    - Large image size (1280) for keypoint precision
    - High pose loss weight for keypoint accuracy

Usage:
    # Default training
    python tools/train_court_detector.py

    # Custom parameters
    python tools/train_court_detector.py \
        --data dataset/court_keypoints/court_keypoints.yaml \
        --base-model models/yolov8m-pose.pt \
        --epochs 200 \
        --imgsz 1280 \
        --batch 4

    # Resume training
    python tools/train_court_detector.py --resume runs/pose/court_detector/weights/last.pt
"""

import os
import sys
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def train(data_yaml, base_model, epochs, imgsz, batch, patience,
          pose_weight, project, name, resume_path=None):
    """
    Train YOLO keypoint model for court detection.

    Args:
        data_yaml: Path to dataset YAML
        base_model: Path to pretrained YOLO pose model
        epochs: Maximum training epochs
        imgsz: Training image size
        batch: Batch size
        patience: Early stopping patience
        pose_weight: Keypoint loss weight
        project: Output project directory
        name: Experiment name
        resume_path: Path to checkpoint for resuming training
    """
    from ultralytics import YOLO

    if resume_path:
        print(f"Resuming training from: {resume_path}")
        model = YOLO(resume_path)
        model.train(resume=True)
        return

    print("=" * 70)
    print("Court Keypoint Detector Training")
    print("=" * 70)
    print(f"Base model:   {base_model}")
    print(f"Dataset:      {data_yaml}")
    print(f"Image size:   {imgsz}")
    print(f"Batch size:   {batch}")
    print(f"Epochs:       {epochs}")
    print(f"Patience:     {patience}")
    print(f"Pose weight:  {pose_weight}")
    print()

    # Validate paths
    if not os.path.exists(data_yaml):
        print(f"[ERROR] Dataset YAML not found: {data_yaml}")
        print("  Run tools/prepare_court_dataset.py first to generate the dataset.")
        sys.exit(1)

    if not os.path.exists(base_model):
        print(f"[ERROR] Base model not found: {base_model}")
        sys.exit(1)

    # Load pretrained model
    model = YOLO(base_model)

    # Train with court-detection-optimized parameters
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=patience,

        # Loss weights
        pose=pose_weight,    # High weight for keypoint precision
        box=7.5,             # Default bbox weight

        # Augmentation - conservative for court detection
        mosaic=0.0,          # Disabled: court must be fully visible
        mixup=0.0,           # Disabled: court structure must be intact
        fliplr=0.5,          # Horizontal flip (with correct flip_idx)
        flipud=0.0,          # No vertical flip (camera is always upright)
        degrees=3.0,         # Slight rotation
        translate=0.1,       # Slight translation
        scale=0.3,           # Moderate scale variation
        perspective=0.0005,  # Very slight perspective (already in data)
        hsv_h=0.015,         # Color jitter
        hsv_s=0.5,
        hsv_v=0.3,

        # Training config
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=5,
        cos_lr=True,
        amp=True,

        # Output
        project=project,
        name=name,
        exist_ok=True,
        save=True,
        save_period=50,
        plots=True,
        verbose=True,
    )

    # Copy best model to models/
    best_path = os.path.join(project, name, 'weights', 'best.pt')
    target_path = os.path.join(PROJECT_ROOT, 'models', 'court_best.pt')
    if os.path.exists(best_path):
        import shutil
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        shutil.copy2(best_path, target_path)
        print(f"\n[OK] Best model copied to: {target_path}")

    print("\nTraining complete!")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLO keypoint model for court detection"
    )
    parser.add_argument("--data", type=str,
                        default="dataset/court_keypoints/court_keypoints.yaml",
                        help="Path to dataset YAML")
    parser.add_argument("--base-model", type=str,
                        default="models/yolov8m-pose.pt",
                        help="Path to pretrained YOLO pose model")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Maximum training epochs (default: 200)")
    parser.add_argument("--imgsz", type=int, default=1280,
                        help="Training image size (default: 1280)")
    parser.add_argument("--batch", type=int, default=4,
                        help="Batch size (default: 4, adjust for GPU memory)")
    parser.add_argument("--patience", type=int, default=50,
                        help="Early stopping patience (default: 50)")
    parser.add_argument("--pose-weight", type=float, default=15.0,
                        help="Keypoint loss weight (default: 15.0)")
    parser.add_argument("--project", type=str, default="runs/pose",
                        help="Output project directory")
    parser.add_argument("--name", type=str, default="court_detector",
                        help="Experiment name")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume training from checkpoint")

    args = parser.parse_args()

    train(
        data_yaml=args.data,
        base_model=args.base_model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        pose_weight=args.pose_weight,
        project=args.project,
        name=args.name,
        resume_path=args.resume,
    )


if __name__ == "__main__":
    main()
