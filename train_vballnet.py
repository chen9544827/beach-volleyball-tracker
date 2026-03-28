#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VballNet 訓練啟動腳本
從專案根目錄執行，自動設定工作目錄到 vball-net/

用法（在 tff_env 環境下）：
    conda activate tff_env
    python train_vballnet.py [train_v1.py 的所有參數]

範例：
    # 從頭訓練 seq9 grayscale（推薦）
    python train_vballnet.py --model_name VballNetV1 --seq 9 --grayscale --epochs 50

    # 繼續訓練
    python train_vballnet.py --model_name VballNetV1 --seq 9 --grayscale --epochs 50 --resume

    # 限制 GPU 記憶體（如 8192MB）
    python train_vballnet.py --model_name VballNetV1 --seq 9 --grayscale --epochs 50 --gpu_memory_limit 8192
"""

import os
import sys

# 切換工作目錄到 vball-net/（使 ./data/frames 路徑正確）
script_dir = os.path.dirname(os.path.abspath(__file__))
vball_net_dir = os.path.join(script_dir, 'vball-net')
vball_net_src  = os.path.join(vball_net_dir, 'src')

os.chdir(vball_net_dir)
sys.path.insert(0, vball_net_src)

# 驗證資料目錄
train_dir = os.path.join('data', 'frames', 'train')
test_dir  = os.path.join('data', 'frames', 'test')
if not os.path.isdir(train_dir):
    print(f"[ERROR] 找不到訓練資料目錄: {os.path.abspath(train_dir)}")
    sys.exit(1)

train_clips = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
test_clips  = [d for d in os.listdir(test_dir)  if os.path.isdir(os.path.join(test_dir,  d))] if os.path.isdir(test_dir) else []

print(f"訓練資料: {len(train_clips)} 片段 @ {os.path.abspath(train_dir)}")
print(f"測試資料: {len(test_clips)}  片段 @ {os.path.abspath(test_dir)}")
print(f"工作目錄: {os.getcwd()}")
print()

# 轉移到 train_v1.py
import importlib.util
spec = importlib.util.spec_from_file_location("train_v1", os.path.join(vball_net_src, "train_v1.py"))
mod  = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
mod.main()
