#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VballNet PyTorch 訓練腳本
讀取 vball-net/data/frames/{train,test}/ 中的 PNG+CSV 格式資料
支援多種架構：VballNetV1a (243K), VballNetV1b (1M, ASPP+DeepSup), VballNetV1c (490K, GRU)
支援 GPU (CUDA)，保留原始 ONNX 模型不動

用法:
    conda activate tff_env
    python train_vballnet_pt.py                                  # 預設 V1a
    python train_vballnet_pt.py --model_name VballNetV1b         # 1M 參數版本
    python train_vballnet_pt.py --model_name VballNetV1c         # GRU 版本
    python train_vballnet_pt.py --epochs 50 --batch 4
    python train_vballnet_pt.py --resume vball-net-pytorch/outputs/.../checkpoints/best.pth
"""

import os
import sys
import argparse
import time
import json
import math
from pathlib import Path
from datetime import datetime

import numpy as np
import cv2
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ── GPU: 確保 CUDA DLL 可被找到（Windows 需要）────────────────────────────
_torch_lib = Path(sys.executable).parent.parent / 'Lib/site-packages/torch/lib'
if _torch_lib.exists():
    os.add_dll_directory(str(_torch_lib))

# ── 路徑設定 ─────────────────────────────────────────────────────────────
SCRIPT_DIR    = Path(__file__).parent
DATA_DIR      = SCRIPT_DIR / 'vball-net' / 'data' / 'frames'
PT_SRC_DIR    = SCRIPT_DIR / 'vball-net-pytorch' / 'src'
OUTPUT_DIR    = SCRIPT_DIR / 'vball-net-pytorch' / 'outputs'
sys.path.insert(0, str(PT_SRC_DIR))

# ── 超參數 ────────────────────────────────────────────────────────────────
IMG_H, IMG_W = 288, 512
SIGMA        = 3       # Gaussian heatmap sigma (px)
SEQ          = 9       # 序列長度（in_dim = out_dim = 9）


# ════════════════════════════════════════════════════════════════════════════
# Dataset：讀取 PNG + CSV，on-the-fly 生成高斯 heatmap
# ════════════════════════════════════════════════════════════════════════════
def make_heatmap(x: float, y: float, h: int, w: int, sigma: float = SIGMA) -> np.ndarray:
    """建立以 (x,y) 為中心的 Gaussian heatmap（numpy 向量化，不可見時回傳全零）。"""
    hm = np.zeros((h, w), dtype=np.float32)
    if x < 0 or y < 0:
        return hm
    cx, cy = int(round(x)), int(round(y))
    r = int(3 * sigma)
    x1, x2 = max(cx - r, 0), min(cx + r + 1, w)
    y1, y2 = max(cy - r, 0), min(cy + r + 1, h)
    if x1 >= x2 or y1 >= y2:
        return hm
    xs = np.arange(x1, x2, dtype=np.float32) - cx
    ys = np.arange(y1, y2, dtype=np.float32) - cy
    xx, yy = np.meshgrid(xs, ys)
    hm[y1:y2, x1:x2] = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return hm


class VballCsvPngDataset(Dataset):
    """
    讀取 `root_dir/<track_id>/0.png ...` + `root_dir/<track_id>_ball.csv`
    CSV 格式: Frame, Visibility, X, Y
    preload=True: 一次性將所有 PNG 載入 RAM（~3-4GB），加快訓練速度
    回傳:
        frames  : (SEQ, H, W)  float32 [0,1] grayscale
        heatmaps: (SEQ, H, W)  float32 [0,1] Gaussian
    """

    def __init__(self, root_dir: Path, seq: int = SEQ, augment: bool = True,
                 preload: bool = True):
        self.root_dir = Path(root_dir)
        self.seq      = seq
        self.augment  = augment
        self.items    = self._build_index()
        # 預載入所有圖片到 RAM
        self.img_cache: dict = {}
        if preload:
            self._preload_images()
        print(f"[Dataset] {self.root_dir.name}: {len(self.items)} sequences")

    def _build_index(self):
        """
        回傳 (track_id, frames_np, coords_np) 預載入到記憶體，避免每次讀 CSV。
        frames_np: (N,) int array of frame indices
        coords_np: (N, 3) float array of [visibility, x, y]
        """
        items = []  # list of (track_id, frames_np, coords_np, start)
        for csv_path in sorted(self.root_dir.glob('*_ball.csv')):
            track_id   = csv_path.stem[:-5]
            frames_dir = self.root_dir / track_id
            if not frames_dir.is_dir():
                continue
            df = pd.read_csv(csv_path)
            n  = len(df)
            if n < self.seq:
                continue
            frames_arr = df['Frame'].to_numpy(dtype=np.int32)
            vis_arr    = df['Visibility'].to_numpy(dtype=np.float32)
            x_arr      = df['X'].to_numpy(dtype=np.float32)
            y_arr      = df['Y'].to_numpy(dtype=np.float32)
            coords     = np.stack([vis_arr, x_arr, y_arr], axis=1)  # (N, 3)
            for start in range(0, n - self.seq + 1):
                items.append((track_id, frames_arr, coords, start))
        return items

    def __len__(self):
        return len(self.items)

    def _preload_images(self):
        """多執行緒平行讀取所有 PNG 到 RAM，key=(track_id, frame_id)。"""
        from concurrent.futures import ThreadPoolExecutor
        import threading

        print(f"[Dataset] 預載入圖片到 RAM（多執行緒）...")
        lock = threading.Lock()

        # 收集所有 PNG 路徑
        all_pngs: list[tuple[str, int, Path]] = []
        track_ids = {item[0] for item in self.items}
        for tid in sorted(track_ids):
            fd = self.root_dir / tid
            for png in fd.glob('*.png'):
                all_pngs.append((tid, int(png.stem), png))

        def load_one(args):
            tid, fid, png = args
            img = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
            if img is None:
                img = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
            if img.shape != (IMG_H, IMG_W):
                img = cv2.resize(img, (IMG_W, IMG_H))
            with lock:
                self.img_cache[(tid, fid)] = img

        with ThreadPoolExecutor(max_workers=8) as ex:
            list(ex.map(load_one, all_pngs))

        total_mb = sum(v.nbytes for v in self.img_cache.values()) / 1024**2
        print(f"[Dataset] 預載完成：{len(self.img_cache)} 張，RAM {total_mb:.0f} MB")

    def _read_img(self, track_id: str, frame_id: int, fallback_idx: int) -> np.ndarray:
        key = (track_id, frame_id)
        if key in self.img_cache:
            return self.img_cache[key]
        # 未預載：直接讀檔
        img_path = self.root_dir / track_id / f"{frame_id}.png"
        if not img_path.exists():
            img_path = self.root_dir / track_id / f"{fallback_idx}.png"
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
        if img.shape != (IMG_H, IMG_W):
            img = cv2.resize(img, (IMG_W, IMG_H))
        return img

    def __getitem__(self, idx):
        track_id, frames_arr, coords, start = self.items[idx]

        imgs, hms = [], []
        flip = self.augment and (np.random.rand() < 0.5)

        for i in range(self.seq):
            frame_id = int(frames_arr[start + i])
            vis, x, y = coords[start + i]

            # ── 讀取灰階幀（優先從 cache）──────────────────────────────
            img = self._read_img(track_id, frame_id, start + i)

            # ── 生成 heatmap ────────────────────────────────────────────
            if vis > 0 and not np.isnan(x):
                hm = make_heatmap(float(x), float(y), IMG_H, IMG_W)
            else:
                hm = np.zeros((IMG_H, IMG_W), dtype=np.float32)

            # ── 左右翻轉 augmentation ───────────────────────────────────
            if flip:
                img = np.fliplr(img).copy()
                hm  = np.fliplr(hm).copy()

            imgs.append(img.astype(np.float32) / 255.0)
            hms.append(hm)

        frames   = np.stack(imgs, axis=0)   # (SEQ, H, W)
        heatmaps = np.stack(hms,  axis=0)   # (SEQ, H, W)
        return torch.from_numpy(frames), torch.from_numpy(heatmaps)


# ════════════════════════════════════════════════════════════════════════════
# 損失函數：Weighted Binary Cross-Entropy（與原論文一致）
# ════════════════════════════════════════════════════════════════════════════
class WeightedBCE(nn.Module):
    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        pred = torch.clamp(pred, self.eps, 1 - self.eps)
        w    = pred
        loss = -((1 - w)**2 * target * torch.log(pred) +
                  w**2 * (1 - target) * torch.log(1 - pred))
        return loss.mean()


# ════════════════════════════════════════════════════════════════════════════
# 主程式
# ════════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description='VballNet PyTorch 訓練（PNG+CSV 格式）')
    p.add_argument('--data',       type=str,   default=str(DATA_DIR / 'train'))
    p.add_argument('--val_data',   type=str,   default=str(DATA_DIR / 'test'))
    p.add_argument('--epochs',     type=int,   default=50)
    p.add_argument('--batch',      type=int,   default=4)
    p.add_argument('--lr',         type=float, default=1e-3)
    p.add_argument('--workers',    type=int,   default=0,
                   help='DataLoader workers (Windows 建議 0)')
    p.add_argument('--resume',     type=str,   default=None,
                   help='從 checkpoint (.pth) 繼續訓練')
    p.add_argument('--out',        type=str,   default=str(OUTPUT_DIR))
    p.add_argument('--name',       type=str,   default='finetune')
    p.add_argument('--patience',   type=int,   default=5,
                   help='ReduceLROnPlateau patience')
    p.add_argument('--min_lr',     type=float, default=1e-6)
    p.add_argument('--model_name', type=str,   default='VballNetV1a',
                   choices=['VballNetV1a', 'VballNetV1b', 'VballNetV1c'],
                   help='模型架構 (V1a=243K, V1b=1M ASPP+DeepSup, V1c=490K GRU)')
    return p.parse_args()


def get_device():
    if torch.cuda.is_available():
        dev = torch.device('cuda')
        print(f"[GPU] {torch.cuda.get_device_name(0)}")
    else:
        dev = torch.device('cpu')
        print('[GPU] GPU 不可用，使用 CPU')
    return dev


def save_checkpoint(state: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def run_epoch(model, loader, criterion, optimizer, device, train: bool,
              log_interval: int = 200, model_name: str = 'VballNetV1a'):
    model.train(train)
    total_loss, n_batches = 0.0, 0
    phase = 'train' if train else 'val'
    with torch.set_grad_enabled(train):
        for frames, heatmaps in loader:
            frames   = frames.to(device)
            heatmaps = heatmaps.to(device)
            out = model(frames)
            # V1c 回傳 (output, hn) tuple；其他回傳單一 tensor
            if isinstance(out, tuple):
                preds = out[0]
            else:
                preds = out                   # (B, 9, H, W)
            loss     = criterion(preds, heatmaps)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            n_batches  += 1
            if train and log_interval > 0 and n_batches % log_interval == 0:
                print(f"  [{phase}] batch {n_batches}/{len(loader)}"
                      f"  loss={total_loss/n_batches:.6f}", flush=True)
    return total_loss / max(n_batches, 1)


def build_model(model_name: str, img_h: int, img_w: int, seq: int):
    """根據 model_name 建立並回傳模型。"""
    if model_name == 'VballNetV1a':
        from model.vballnet_v1a import VballNetV1a
        return VballNetV1a(height=img_h, width=img_w, in_dim=seq, out_dim=seq)
    elif model_name == 'VballNetV1b':
        from model.vballnet_v1b import VballNetV1b
        return VballNetV1b(height=img_h, width=img_w, in_dim=seq, out_dim=seq)
    elif model_name == 'VballNetV1c':
        from model.vballnet_v1c import VballNetV1c
        return VballNetV1c(height=img_h, width=img_w, in_dim=seq, out_dim=seq)
    else:
        raise ValueError(f"未知模型: {model_name}")


def main():
    args   = parse_args()
    device = get_device()

    # ── 建立輸出目錄 ─────────────────────────────────────────────────────
    ts      = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = Path(args.out) / f"{args.name}_{args.model_name}_seq9_grayscale_{ts}"
    (out_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    print(f"[OUT] {out_dir}")

    # ── 資料集 ───────────────────────────────────────────────────────────
    train_ds = VballCsvPngDataset(Path(args.data),     seq=SEQ, augment=True)
    val_ds   = VballCsvPngDataset(Path(args.val_data), seq=SEQ, augment=False)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch,
                              shuffle=True, num_workers=args.workers,
                              pin_memory=(device.type == 'cuda'))
    val_loader   = DataLoader(val_ds,   batch_size=args.batch,
                              shuffle=False, num_workers=args.workers,
                              pin_memory=(device.type == 'cuda'))

    # ── 模型 ─────────────────────────────────────────────────────────────
    model = build_model(args.model_name, IMG_H, IMG_W, SEQ)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] {args.model_name}  params={total_params:,}  device={device}")

    # ── 損失 + Optimizer + Scheduler ─────────────────────────────────────
    criterion = WeightedBCE()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                  patience=args.patience, min_lr=args.min_lr)

    # ── 從 checkpoint 繼續 ───────────────────────────────────────────────
    start_epoch = 0
    best_val    = float('inf')
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_val    = ckpt.get('best_val', float('inf'))
        print(f"[Resume] epoch={start_epoch}  best_val={best_val:.6f}")

    # ── 訓練迴圈 ─────────────────────────────────────────────────────────
    history = []
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device,
                               train=True,  model_name=args.model_name)
        val_loss   = run_epoch(model, val_loader,   criterion, optimizer, device,
                               train=False, model_name=args.model_name)
        scheduler.step(val_loss)
        lr_now = optimizer.param_groups[0]['lr']
        elapsed = time.time() - t0

        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss

        print(f"Epoch {epoch+1:3d}/{args.epochs}  "
              f"train={train_loss:.6f}  val={val_loss:.6f}  "
              f"lr={lr_now:.2e}  {'[BEST]' if is_best else ''}  {elapsed:.0f}s")

        # 儲存 checkpoint
        state = {
            'epoch':                epoch,
            'model_state_dict':     model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss':           train_loss,
            'val_loss':             val_loss,
            'best_val':             best_val,
        }
        save_checkpoint(state, out_dir / 'checkpoints' / 'last.pth')
        if is_best:
            save_checkpoint(state, out_dir / 'checkpoints' / 'best.pth')

        history.append({'epoch': epoch+1, 'train': train_loss,
                        'val': val_loss, 'lr': lr_now})
        with open(out_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)

    print(f"\n[Done] best_val={best_val:.6f}  model saved to {out_dir / 'checkpoints'}")
    print(f"\n下一步: 轉換為 ONNX")
    print(f"  cd {SCRIPT_DIR}")
    print(f"  python export_vballnet_pt.py --weights {out_dir/'checkpoints'/'best.pth'}")


if __name__ == '__main__':
    main()
