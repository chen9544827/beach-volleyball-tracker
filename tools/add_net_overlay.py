# tools/add_net_overlay.py
# -*- coding: utf-8 -*-
"""
為 Roboflow COCO JSON 資料集加上模擬球網遮擋效果

對每個球員的 bounding box 疊加半透明網狀紋路，
讓模型學習透過球網辨識球員與關節。

用法：
    python tools/add_net_overlay.py --dataset-dir path/to/roboflow_export

資料夾結構預期：
    dataset_dir/
        train/
            images/
            _annotations.coco.json
        valid/
            images/
            _annotations.coco.json
        test/   (可選)
            images/
            _annotations.coco.json
"""

import os
import json
import argparse
import random
from pathlib import Path

import numpy as np
import cv2


# ─── 預設網紋參數 ─────────────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    "net_color_bgr": (220, 220, 220),  # 網線顏色（淺灰白）
    "net_alpha": 0.55,                  # 網線不透明度（0=透明，1=不透明）
    "cell_size_ratio": 0.03,            # 格子大小（相對於 bbox 高度）
    "line_thickness": 1,                # 網線粗細（像素）
    "top_band_ratio": 0.06,             # 頂部白帶高度（相對於 bbox 高度，0=不加）
    "top_band_alpha": 0.7,              # 頂部白帶不透明度
    "cover_ratio_min": 0.5,             # bbox 被覆蓋的最小比例（從下往上）
    "cover_ratio_max": 1.0,             # bbox 被覆蓋的最大比例
    "apply_prob": 0.8,                  # 每個 bbox 被覆蓋的機率
}
# ─────────────────────────────────────────────────────────────────────────────


def draw_net_on_region(
    image: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    cfg: dict
) -> np.ndarray:
    """
    在指定矩形區域內繪製半透明排球網紋路

    Args:
        image: 原始 BGR 圖片（in-place 修改）
        x1, y1: 區域左上角
        x2, y2: 區域右下角
        cfg: 網紋參數字典

    Returns:
        繪製後的圖片
    """
    h = y2 - y1
    w = x2 - x1
    if h <= 0 or w <= 0:
        return image

    # 決定覆蓋範圍（從 bbox 下方往上覆蓋 cover_ratio 的高度）
    cover = random.uniform(cfg["cover_ratio_min"], cfg["cover_ratio_max"])
    net_y1 = y1 + int(h * (1.0 - cover))
    net_y2 = y2
    net_h = net_y2 - net_y1
    if net_h <= 0:
        return image

    cell = max(6, int(h * cfg["cell_size_ratio"]))
    color = cfg["net_color_bgr"]
    alpha = cfg["net_alpha"]
    thickness = cfg["line_thickness"]

    overlay = image.copy()

    # 水平線
    y = net_y1
    while y <= net_y2:
        cv2.line(overlay, (x1, y), (x2, y), color, thickness)
        y += cell

    # 垂直線
    x = x1
    while x <= x2:
        cv2.line(overlay, (x, net_y1), (x, net_y2), color, thickness)
        x += cell

    # 混合網格
    roi = (slice(net_y1, net_y2), slice(x1, x2))
    image[roi] = cv2.addWeighted(
        overlay[roi], alpha,
        image[roi], 1.0 - alpha,
        0
    )

    # 頂部白色橫帶（模擬網帶）
    if cfg["top_band_ratio"] > 0:
        band_h = max(3, int(h * cfg["top_band_ratio"]))
        band_y2 = min(net_y1 + band_h, net_y2)
        band_roi = (slice(net_y1, band_y2), slice(x1, x2))
        band_overlay = image.copy()
        band_overlay[band_roi] = (255, 255, 255)
        image[band_roi] = cv2.addWeighted(
            band_overlay[band_roi], cfg["top_band_alpha"],
            image[band_roi], 1.0 - cfg["top_band_alpha"],
            0
        )

    return image


def find_annotation_file(split_dir: Path) -> Path | None:
    """找出 split 資料夾內的 COCO JSON 標籤檔"""
    candidates = [
        split_dir / "_annotations.coco.json",
        split_dir / "annotations.json",
        split_dir / f"{split_dir.name}.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    # 搜尋任何 .json 檔
    jsons = list(split_dir.glob("*.json"))
    if jsons:
        return jsons[0]
    return None


def find_image(split_dir: Path, filename: str) -> Path | None:
    """找出圖片路徑，相容多種資料夾結構"""
    for base in [split_dir / "images", split_dir]:
        p = base / filename
        if p.exists():
            return p
        p = base / Path(filename).name
        if p.exists():
            return p
    return None


def process_split(split_dir: Path, cfg: dict, dry_run: bool = False) -> dict:
    """處理一個 split 資料夾（train / valid / test）"""
    anno_path = find_annotation_file(split_dir)
    if anno_path is None:
        print(f"  [SKIP] 找不到標籤檔: {split_dir}")
        return {}

    print(f"  標籤檔: {anno_path.name}")
    with open(anno_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    id_to_filename = {img["id"]: img["file_name"] for img in coco["images"]}

    # image_id → list of COCO bbox [x, y, w, h]
    id_to_bboxes: dict = {}
    for ann in coco.get("annotations", []):
        bbox = ann.get("bbox")
        if bbox:
            id_to_bboxes.setdefault(ann["image_id"], []).append(bbox)

    stats = {"processed": 0, "skipped": 0, "bboxes_drawn": 0}

    for img_id, filename in id_to_filename.items():
        bboxes = id_to_bboxes.get(img_id, [])
        if not bboxes:
            stats["skipped"] += 1
            continue

        img_path = find_image(split_dir, filename)
        if img_path is None:
            print(f"  [WARN] 找不到圖片: {filename}")
            stats["skipped"] += 1
            continue

        if dry_run:
            print(f"  [DRY-RUN] {img_path.name}  ({len(bboxes)} bbox)")
            stats["processed"] += 1
            stats["bboxes_drawn"] += len(bboxes)
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  [WARN] 無法讀取: {img_path.name}")
            stats["skipped"] += 1
            continue

        ih, iw = image.shape[:2]
        drawn = 0

        for bbox in bboxes:
            if random.random() > cfg["apply_prob"]:
                continue
            bx, by, bw, bh = bbox
            x1 = max(0, int(bx))
            y1 = max(0, int(by))
            x2 = min(iw, int(bx + bw))
            y2 = min(ih, int(by + bh))
            image = draw_net_on_region(image, x1, y1, x2, y2, cfg)
            drawn += 1

        cv2.imwrite(str(img_path), image)
        stats["processed"] += 1
        stats["bboxes_drawn"] += drawn

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="為 COCO 資料集的球員 bounding box 加上模擬球網遮擋"
    )
    parser.add_argument("--dataset-dir", default=None,
                        help="Roboflow 匯出的根目錄（含 train/valid/test 子資料夾）")
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"],
                        help="要處理的 split（預設: train valid test）")
    parser.add_argument("--seed", type=int, default=42,
                        help="隨機種子")
    parser.add_argument("--dry-run", action="store_true",
                        help="只顯示會做什麼，不實際修改圖片")
    parser.add_argument("--alpha", type=float, default=DEFAULT_CONFIG["net_alpha"],
                        help=f"網線不透明度 0~1（預設 {DEFAULT_CONFIG['net_alpha']}）")
    parser.add_argument("--cell-ratio", type=float, default=DEFAULT_CONFIG["cell_size_ratio"],
                        help=f"格子大小比例（預設 {DEFAULT_CONFIG['cell_size_ratio']}）")
    parser.add_argument("--prob", type=float, default=DEFAULT_CONFIG["apply_prob"],
                        help=f"每個 bbox 被覆蓋的機率 0~1（預設 {DEFAULT_CONFIG['apply_prob']}）")
    parser.add_argument("--preview", metavar="IMG",
                        help="只對單張圖片預覽效果（不修改原檔，另存為 preview_net.jpg）")
    args = parser.parse_args()

    random.seed(args.seed)

    # 建立 config
    cfg = DEFAULT_CONFIG.copy()
    cfg["net_alpha"] = args.alpha
    cfg["cell_size_ratio"] = args.cell_ratio
    cfg["apply_prob"] = args.prob

    # ── 單張預覽模式 ────────────────────────────────────────────────────────
    if args.preview:
        img_path = Path(args.preview)
        if not img_path.exists():
            print(f"[ERROR] 找不到圖片: {img_path}")
            return
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"[ERROR] 無法讀取: {img_path}")
            return
        ih, iw = image.shape[:2]
        # 預覽：在整張圖中間畫一個全幅 bbox
        x1, y1 = iw // 4, ih // 4
        x2, y2 = iw * 3 // 4, ih * 3 // 4
        cfg["apply_prob"] = 1.0
        image = draw_net_on_region(image, x1, y1, x2, y2, cfg)
        out = img_path.parent / "preview_net.jpg"
        cv2.imwrite(str(out), image)
        print(f"[OK] 預覽圖片已儲存: {out}")
        return

    # ── 批次處理模式 ─────────────────────────────────────────────────────────
    if args.dataset_dir is None:
        print("[ERROR] 批次模式需要指定 --dataset-dir")
        return
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        print(f"[ERROR] 找不到資料夾: {dataset_dir}")
        return

    print("=" * 60)
    print("Volleyball Net Overlay Augmentation")
    print("=" * 60)
    print(f"資料集路徑  : {dataset_dir}")
    print(f"網線不透明度: {cfg['net_alpha']}")
    print(f"格子大小比例: {cfg['cell_size_ratio']}")
    print(f"覆蓋機率    : {cfg['apply_prob']}")
    print(f"覆蓋高度範圍: {cfg['cover_ratio_min']*100:.0f}% ~ {cfg['cover_ratio_max']*100:.0f}%")
    if args.dry_run:
        print("[DRY-RUN 模式] 不實際修改檔案")

    total_processed = 0
    total_bboxes = 0

    for split in args.splits:
        split_dir = dataset_dir / split
        if not split_dir.exists():
            print(f"\n[SKIP] 找不到 split: {split_dir}")
            continue

        print(f"\n處理 [{split}] split ...")
        stats = process_split(split_dir, cfg, dry_run=args.dry_run)
        if stats:
            p = stats["processed"]
            s = stats["skipped"]
            b = stats["bboxes_drawn"]
            print(f"  [OK] 處理 {p} 張 | 跳過 {s} 張 | 繪製 {b} 個 bbox")
            total_processed += p
            total_bboxes += b

    print("\n" + "=" * 60)
    print(f"完成！共處理 {total_processed} 張圖片，繪製 {total_bboxes} 個網格遮罩")
    if args.dry_run:
        print("（DRY-RUN 模式，未修改任何圖片）")
    print("=" * 60)


if __name__ == "__main__":
    main()
