# tools/train_pose_finetune.py
# -*- coding: utf-8 -*-
"""
Fine-tune YOLOv8m-pose on volleyball player dataset

將 Roboflow COCO JSON 資料集轉換為 YOLO pose 格式，
並從 yolov8m-pose.pt 進行 fine-tune。

用法：
    python tools/train_pose_finetune.py

    # 自訂參數
    python tools/train_pose_finetune.py \
        --data-dir "dataset/My First Project.v3i.coco" \
        --base-model models/yolov8m-pose.pt \
        --epochs 100 \
        --batch 8
"""

import os
import json
import shutil
import argparse
import random
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# ─── 預設設定 ─────────────────────────────────────────────────────────────────
DEFAULT_DATA_DIR   = "dataset/My First Project.v3i.coco"
DEFAULT_BASE_MODEL = "models/yolov8m-pose.pt"
DEFAULT_OUTPUT_DIR = "runs/pose/pose_finetune"
DEFAULT_EPOCHS     = 100
DEFAULT_BATCH      = 8
DEFAULT_IMGSZ      = 640
DEFAULT_VAL_SPLIT  = 0.15   # 15% 作為 validation
# ─────────────────────────────────────────────────────────────────────────────

# COCO 17 關鍵點水平翻轉對稱索引（left ↔ right）
COCO17_FLIP_IDX = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


def convert_coco_to_yolo_pose(
    coco_json_path: Path,
    images_dir: Path,
    output_labels_dir: Path,
    image_ids: list = None
) -> int:
    """
    將 COCO JSON 轉換為 YOLO pose 格式 (.txt)

    YOLO pose 格式（每行一個物件）：
        class cx cy w h  kp1x kp1y kp1v  kp2x kp2y kp2v  ...  kp17x kp17y kp17v
        （所有座標 normalized 到 0~1，kpv = visibility：0/1/2）

    Args:
        coco_json_path: COCO 標籤 JSON 路徑
        images_dir: 圖片資料夾
        output_labels_dir: 輸出 .txt 資料夾
        image_ids: 若指定，只處理這些 image_id（用於 train/val 分割）

    Returns:
        成功轉換的圖片數
    """
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    with open(coco_json_path, encoding="utf-8") as f:
        coco = json.load(f)

    id_to_image = {img["id"]: img for img in coco["images"]}

    # image_id → annotations
    id_to_anns: dict = {}
    for ann in coco["annotations"]:
        # 只保留有 bbox 且 category_id == 1 (person) 的標記
        if ann.get("category_id") != 1:
            continue
        if not ann.get("bbox"):
            continue
        id_to_anns.setdefault(ann["image_id"], []).append(ann)

    target_ids = set(image_ids) if image_ids else set(id_to_image.keys())
    converted = 0

    for img_id in target_ids:
        img_info = id_to_image.get(img_id)
        if img_info is None:
            continue

        anns = id_to_anns.get(img_id, [])
        filename = img_info["file_name"]
        iw = img_info["width"]
        ih = img_info["height"]

        lines = []
        for ann in anns:
            bx, by, bw, bh = ann["bbox"]

            # 跳過無效 bbox
            if bw <= 0 or bh <= 0:
                continue

            # Normalize bbox → cx cy w h
            cx = (bx + bw / 2) / iw
            cy = (by + bh / 2) / ih
            nw = bw / iw
            nh = bh / ih

            # class = 0（person）
            parts = [f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"]

            # 關鍵點：[x, y, v] × 17
            kps = ann.get("keypoints", [])
            if len(kps) == 51:
                for i in range(17):
                    kx = kps[i * 3] / iw
                    ky = kps[i * 3 + 1] / ih
                    kv = int(kps[i * 3 + 2])
                    parts.append(f"{kx:.6f} {ky:.6f} {kv}")
            else:
                # 沒有關鍵點，補零
                for _ in range(17):
                    parts.append("0.0 0.0 0")

            lines.append(" ".join(parts))

        # 儲存 label txt（即使空的也要儲存，YOLO 需要）
        stem = Path(filename).stem
        label_path = output_labels_dir / f"{stem}.txt"
        with open(label_path, "w") as f:
            f.write("\n".join(lines))

        converted += 1

    return converted


def prepare_dataset(data_dir: Path, val_split: float, seed: int = 42) -> Path:
    """
    準備 YOLO pose 訓練資料夾，並建立 dataset.yaml

    輸出結構：
        data_dir/yolo_pose/
            train/
                images/  (symlink 或複製)
                labels/
            val/
                images/
                labels/
            dataset.yaml
    """
    coco_json = data_dir / "train" / "_annotations.coco.json"
    images_src = data_dir / "train"
    out_dir = data_dir / "yolo_pose"

    print(f"\n[準備資料集]")
    print(f"  來源 COCO JSON : {coco_json}")
    print(f"  輸出目錄       : {out_dir}")

    with open(coco_json, encoding="utf-8") as f:
        coco = json.load(f)

    # 只取有 person 標記的 image_id
    valid_ids = {
        ann["image_id"]
        for ann in coco["annotations"]
        if ann.get("category_id") == 1 and ann.get("bbox")
    }
    all_ids = sorted(valid_ids)
    print(f"  有效圖片數     : {len(all_ids)}")

    # train / val 分割
    random.seed(seed)
    random.shuffle(all_ids)
    val_n = max(1, int(len(all_ids) * val_split))
    val_ids = all_ids[:val_n]
    train_ids = all_ids[val_n:]
    print(f"  train: {len(train_ids)} | val: {len(val_ids)}")

    id_to_filename = {img["id"]: img["file_name"] for img in coco["images"]}

    for split, ids in [("train", train_ids), ("val", val_ids)]:
        split_dir = out_dir / split
        images_dst = split_dir / "images"
        labels_dst = split_dir / "labels"
        images_dst.mkdir(parents=True, exist_ok=True)

        # 複製圖片
        print(f"  複製 {split} 圖片...")
        for img_id in ids:
            fname = id_to_filename.get(img_id)
            if fname is None:
                continue
            src = images_src / fname
            if not src.exists():
                src = images_src / Path(fname).name
            if src.exists():
                shutil.copy2(src, images_dst / Path(fname).name)

        # 轉換標籤
        print(f"  轉換 {split} 標籤...")
        n = convert_coco_to_yolo_pose(coco_json, images_src, labels_dst, image_ids=ids)
        print(f"    → {n} 張標籤完成")

    # 建立 dataset.yaml
    yaml_path = out_dir / "dataset.yaml"
    abs_out = out_dir.resolve()
    yaml_content = f"""# Auto-generated by train_pose_finetune.py
path: {abs_out.as_posix()}
train: train/images
val: val/images

# 關鍵點設定（COCO 17 keypoints）
kpt_shape: [17, 3]
flip_idx: {COCO17_FLIP_IDX}

# 類別
nc: 1
names: ['person']
"""
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"  [OK] dataset.yaml 已建立: {yaml_path}")
    return yaml_path


def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8m-pose on volleyball player dataset")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                        help=f"COCO 資料集目錄（預設: {DEFAULT_DATA_DIR}）")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL,
                        help=f"起始模型（預設: {DEFAULT_BASE_MODEL}）")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                        help=f"訓練 epoch 數（預設: {DEFAULT_EPOCHS}）")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH,
                        help=f"Batch size（預設: {DEFAULT_BATCH}）")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ,
                        help=f"訓練圖片大小（預設: {DEFAULT_IMGSZ}）")
    parser.add_argument("--val-split", type=float, default=DEFAULT_VAL_SPLIT,
                        help=f"Validation 比例（預設: {DEFAULT_VAL_SPLIT}）")
    parser.add_argument("--name", default="pose_finetune",
                        help="實驗名稱（儲存在 runs/pose/<name>/）")
    parser.add_argument("--device", default="0",
                        help="GPU 裝置（預設: 0，多 GPU: 0,1）")
    parser.add_argument("--resume", metavar="WEIGHTS",
                        help="從指定 checkpoint 繼續訓練")
    parser.add_argument("--prepare-only", action="store_true",
                        help="只準備資料集，不執行訓練")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"[ERROR] 找不到資料集目錄: {data_dir}")
        return

    # ── 步驟 1：準備資料集 ──────────────────────────────────────────────────
    yaml_path = prepare_dataset(data_dir, args.val_split, seed=args.seed)

    if args.prepare_only:
        print("\n[prepare-only] 資料集準備完成，未執行訓練。")
        return

    # ── 步驟 2：Fine-tune ───────────────────────────────────────────────────
    from ultralytics import YOLO

    base_model = args.base_model
    if args.resume:
        print(f"\n[Fine-tune] 從 checkpoint 繼續: {args.resume}")
        model = YOLO(args.resume)
    else:
        if not Path(base_model).exists():
            print(f"[ERROR] 找不到基礎模型: {base_model}")
            return
        print(f"\n[Fine-tune] 從 {base_model} 開始 fine-tune")
        model = YOLO(base_model)

    print(f"  資料集 YAML : {yaml_path}")
    print(f"  Epochs      : {args.epochs}")
    print(f"  Batch       : {args.batch}")
    print(f"  Image size  : {args.imgsz}")
    print(f"  Device      : {args.device}")

    results = model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        name=args.name,
        project="runs/pose",
        workers=0,             # Windows 必須設 0
        seed=args.seed,
        # ── Augmentation（球網遮擋已在圖片中，關閉部分內建增強避免衝突）──
        mosaic=0.5,            # 少量 mosaic 增強多尺度
        flipud=0.0,            # 沙灘排球場景不上下翻轉
        fliplr=0.5,            # 左右翻轉OK
        degrees=5.0,           # 小幅旋轉
        translate=0.1,
        scale=0.3,
        # ── 關鍵點損失權重 ──────────────────────────────────────────────────
        pose=12.0,             # 提高 pose 損失權重（預設 12）
        # ── Early stopping ─────────────────────────────────────────────────
        patience=30,
        # ── 其他 ────────────────────────────────────────────────────────────
        verbose=True,
        exist_ok=True,
    )

    print(f"\n[完成] 模型儲存於: runs/pose/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()
