#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
準備 VballNet fine-tune 訓練資料集

整合兩種來源：
  1. 作者原始資料集（data_20250711_2330）：影片 + CSV，座標在原始解析度
  2. 我們的 CVAT 標注（dataset/vball_finetune/train）：已完成轉換，直接複製

輸出格式（vball-net train_v1.py 所需）：
  <output_dir>/train/
    <track_id>/       ← 512x288 PNG 幀
    <track_id>_ball.csv ← 座標在 512x288 空間

用法：
    python tools/prepare_vballnet_dataset.py \
        --author-dataset dataset/data_20250711_2330/data \
        --our-dataset    dataset/vball_finetune/train \
        --output         vball-net/data/frames \
        --mode           train

    # 只處理作者資料集的 test split
    python tools/prepare_vballnet_dataset.py \
        --author-dataset dataset/data_20250711_2330/data \
        --output         vball-net/data/frames \
        --mode           test

    # 只更新我們的 CVAT 資料（不重新處理作者資料）
    python tools/prepare_vballnet_dataset.py \
        --our-dataset    dataset/vball_finetune/train \
        --output         vball-net/data/frames \
        --mode           train --skip-author
"""

import os
import sys
import csv
import shutil
import argparse
from pathlib import Path

import cv2

TRAIN_W = 512
TRAIN_H = 288


def scale_csv(src_csv: str, dst_csv: str, orig_w: int, orig_h: int):
    """
    讀取原始座標 CSV（Frame,Visibility,X,Y），縮放到 512x288，寫入 dst_csv。
    若 orig_w==512 且 orig_h==288 則直接複製（不縮放）。
    """
    sx = TRAIN_W / orig_w
    sy = TRAIN_H / orig_h
    need_scale = not (orig_w == TRAIN_W and orig_h == TRAIN_H)

    rows = list(csv.DictReader(open(src_csv, encoding='utf-8')))
    with open(dst_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Frame', 'Visibility', 'X', 'Y'])
        for r in rows:
            vis = int(r['Visibility'])
            if vis and need_scale:
                x = int(round(float(r['X']) * sx))
                y = int(round(float(r['Y']) * sy))
                x = max(0, min(x, TRAIN_W - 1))
                y = max(0, min(y, TRAIN_H - 1))
            else:
                x = int(float(r['X']))
                y = int(float(r['Y']))
            writer.writerow([r['Frame'], vis, x, y])


def extract_frames(video_path: str, out_dir: str, total_frames: int) -> int:
    """影片 → 512x288 PNG 幀，輸出到 out_dir/0.png, 1.png, ..."""
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret or count >= total_frames:
            break
        resized = cv2.resize(frame, (TRAIN_W, TRAIN_H))
        cv2.imwrite(os.path.join(out_dir, f"{count}.png"), resized)
        count += 1
    cap.release()
    return count


def process_author_clip(video_path: str, csv_path: str, track_id: str,
                         out_dir: str, force: bool = False) -> bool:
    """
    處理作者資料集的單一片段。
    frame_dir = out_dir/<track_id>/
    ball_csv  = out_dir/<track_id>_ball.csv
    """
    frame_dir = os.path.join(out_dir, track_id)
    ball_csv  = os.path.join(out_dir, f"{track_id}_ball.csv")

    # 讀取 CSV 取得 total_frames
    rows = list(csv.DictReader(open(csv_path, encoding='utf-8')))
    total_frames = len(rows)
    if total_frames == 0:
        print(f"  [SKIP] {track_id}: 空 CSV")
        return False

    # 取得影片解析度
    cap = cv2.VideoCapture(video_path)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # 已存在且不強制重做
    if not force and os.path.exists(ball_csv):
        existing_pngs = len([f for f in os.listdir(frame_dir) if f.endswith('.png')]) if os.path.isdir(frame_dir) else 0
        if existing_pngs >= total_frames:
            print(f"  [SKIP] {track_id}: 已存在 ({existing_pngs} 幀)")
            return True

    # 縮放 CSV
    scale_csv(csv_path, ball_csv, orig_w, orig_h)

    # 提取幀
    n = extract_frames(video_path, frame_dir, total_frames)

    vis_count = sum(1 for r in rows if r['Visibility'] == '1')
    print(f"  [OK] {track_id}: {n} 幀, {vis_count}/{total_frames} visible ({100*vis_count/total_frames:.0f}%), {orig_w}x{orig_h}→512x288")
    return True


def process_author_dataset(author_data_dir: str, out_dir: str, mode: str, force: bool):
    """處理作者資料集的指定 split（train 或 test）。"""
    split_dir = os.path.join(author_data_dir, mode)
    if not os.path.isdir(split_dir):
        print(f"[WARNING] 找不到作者 {mode} 目錄: {split_dir}")
        return

    folders = sorted(os.listdir(split_dir))
    total_ok = 0
    for folder in folders:
        folder_path = os.path.join(split_dir, folder)
        csv_dir   = os.path.join(folder_path, 'csv')
        video_dir = os.path.join(folder_path, 'video')
        if not os.path.isdir(csv_dir) or not os.path.isdir(video_dir):
            continue

        csv_files = sorted(f for f in os.listdir(csv_dir) if f.endswith('_ball.csv'))
        print(f"\n[{folder}] {len(csv_files)} 片段")

        for csv_file in csv_files:
            # track_id = csv 檔名去掉 _ball.csv
            track_id = csv_file[:-len('_ball.csv')]
            csv_path   = os.path.join(csv_dir, csv_file)
            video_file = track_id + '.mp4'
            video_path = os.path.join(video_dir, video_file)

            if not os.path.exists(video_path):
                # 嘗試其他副檔名
                for ext in ['.avi', '.mov', '.mkv']:
                    alt = os.path.join(video_dir, track_id + ext)
                    if os.path.exists(alt):
                        video_path = alt
                        break
                else:
                    print(f"  [SKIP] {track_id}: 找不到影片")
                    continue

            ok = process_author_clip(video_path, csv_path, track_id, out_dir, force)
            if ok:
                total_ok += 1

    print(f"\n作者資料集 {mode}: 處理完成 {total_ok} 片段")


def copy_our_cvat_data(our_dataset_dir: str, out_dir: str, force: bool):
    """複製我們的 CVAT 轉換資料（已是 512x288 格式）到輸出目錄。"""
    if not os.path.isdir(our_dataset_dir):
        print(f"[WARNING] 找不到 CVAT 資料目錄: {our_dataset_dir}")
        return

    copied = 0
    for item in sorted(os.listdir(our_dataset_dir)):
        src = os.path.join(our_dataset_dir, item)

        if item.endswith('_ball.csv'):
            dst = os.path.join(out_dir, item)
            if force or not os.path.exists(dst):
                shutil.copy2(src, dst)
                track_id = item[:-len('_ball.csv')]
                rows = list(csv.DictReader(open(dst)))
                vis = sum(1 for r in rows if r['Visibility'] == '1')
                print(f"  [CSV] {item}: {len(rows)} 幀, {vis} visible ({100*vis/len(rows):.0f}%)")
                copied += 1
            else:
                print(f"  [SKIP] {item}: 已存在")

        elif os.path.isdir(src):
            # PNG 幀目錄
            dst_dir = os.path.join(out_dir, item)
            if force or not os.path.isdir(dst_dir):
                if os.path.isdir(dst_dir):
                    shutil.rmtree(dst_dir)
                shutil.copytree(src, dst_dir)
                n_png = len([f for f in os.listdir(dst_dir) if f.endswith('.png')])
                print(f"  [DIR] {item}/: {n_png} PNG 幀")
                copied += 1
            else:
                n_png = len([f for f in os.listdir(dst_dir) if f.endswith('.png')])
                print(f"  [SKIP] {item}/: 已存在 ({n_png} 幀)")

    print(f"\nCVAT 資料: 複製完成 {copied} 項目")


def verify_dataset(out_dir: str):
    """驗證輸出目錄的配對完整性。"""
    track_dirs = sorted(d for d in os.listdir(out_dir) if os.path.isdir(os.path.join(out_dir, d)))
    ok = 0
    missing_csv = []
    missing_frames = []

    for track_id in track_dirs:
        csv_path  = os.path.join(out_dir, f"{track_id}_ball.csv")
        frame_dir = os.path.join(out_dir, track_id)

        if not os.path.exists(csv_path):
            missing_csv.append(track_id)
            continue

        rows = list(csv.DictReader(open(csv_path)))
        n_png = len([f for f in os.listdir(frame_dir) if f.endswith('.png')])

        if n_png < len(rows):
            missing_frames.append(f"{track_id}: CSV={len(rows)}, PNG={n_png}")
            continue
        ok += 1

    print(f"\n驗證結果: {ok} OK, {len(missing_csv)} 缺 CSV, {len(missing_frames)} 幀不足")
    if missing_csv:
        print(f"  缺 CSV: {missing_csv[:5]}")
    if missing_frames:
        print(f"  幀不足: {missing_frames[:5]}")

    # 統計
    all_csvs = [f for f in os.listdir(out_dir) if f.endswith('_ball.csv')]
    total_frames = 0
    total_vis = 0
    for csv_file in all_csvs:
        rows = list(csv.DictReader(open(os.path.join(out_dir, csv_file))))
        total_frames += len(rows)
        total_vis += sum(1 for r in rows if r['Visibility'] == '1')

    print(f"\n資料集統計:")
    print(f"  片段數: {len(all_csvs)}")
    print(f"  總幀數: {total_frames:,}")
    print(f"  有球幀: {total_vis:,} ({100*total_vis/max(total_frames,1):.1f}%)")
    print(f"  無球幀: {total_frames-total_vis:,} ({100*(total_frames-total_vis)/max(total_frames,1):.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="準備 VballNet fine-tune 訓練資料集")
    parser.add_argument("--author-dataset", type=str, default=None,
                        help="作者資料集根目錄（含 train/ test/ 子目錄）")
    parser.add_argument("--our-dataset", type=str, default=None,
                        help="我們的 CVAT 轉換資料目錄（已是 512x288 格式）")
    parser.add_argument("--output", type=str, default="vball-net/data/frames",
                        help="輸出根目錄（預設: vball-net/data/frames）")
    parser.add_argument("--mode", type=str, choices=["train", "test", "both"],
                        default="train", help="處理 split（預設: train）")
    parser.add_argument("--skip-author", action="store_true",
                        help="跳過作者資料集，只複製 CVAT 資料")
    parser.add_argument("--force", action="store_true",
                        help="強制重新處理（覆蓋已存在的輸出）")
    parser.add_argument("--verify-only", action="store_true",
                        help="只驗證資料集完整性，不做轉換")
    args = parser.parse_args()

    modes = ["train", "test"] if args.mode == "both" else [args.mode]

    for mode in modes:
        out_dir = os.path.join(args.output, mode)
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n{'='*60}")
        print(f"處理 {mode} split → {out_dir}")
        print(f"{'='*60}")

        if args.verify_only:
            verify_dataset(out_dir)
            continue

        # 處理作者資料集
        if not args.skip_author and args.author_dataset:
            print(f"\n--- 作者資料集 ---")
            process_author_dataset(args.author_dataset, out_dir, mode, args.force)

        # 複製 CVAT 資料（只在 train mode）
        if args.our_dataset and mode == "train":
            print(f"\n--- 我們的 CVAT 標注 ---")
            copy_our_cvat_data(args.our_dataset, out_dir, args.force)

        # 驗證
        verify_dataset(out_dir)

    print("\n完成!")


if __name__ == "__main__":
    main()
