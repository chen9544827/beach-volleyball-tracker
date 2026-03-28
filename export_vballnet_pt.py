#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
將 PyTorch checkpoint (.pth) 匯出為 ONNX 格式
支援 VballNetV1a / VballNetV1b / VballNetV1c

用法:
    conda activate tff_env
    python export_vballnet_pt.py --weights vball-net-pytorch/outputs/.../checkpoints/best.pth
    python export_vballnet_pt.py --weights ... --model_name VballNetV1b
"""

import os
import sys
import argparse
from pathlib import Path

# Windows CUDA DLL
_torch_lib = Path(sys.executable).parent.parent / 'Lib/site-packages/torch/lib'
if _torch_lib.exists():
    os.add_dll_directory(str(_torch_lib))

import torch

PT_SRC = Path(__file__).parent / 'vball-net-pytorch' / 'src'
sys.path.insert(0, str(PT_SRC))

IMG_H, IMG_W, SEQ = 288, 512, 9


def build_model(model_name: str):
    if model_name == 'VballNetV1a':
        from model.vballnet_v1a import VballNetV1a
        return VballNetV1a(height=IMG_H, width=IMG_W, in_dim=SEQ, out_dim=SEQ)
    elif model_name == 'VballNetV1b':
        from model.vballnet_v1b import VballNetV1b
        return VballNetV1b(height=IMG_H, width=IMG_W, in_dim=SEQ, out_dim=SEQ)
    elif model_name == 'VballNetV1c':
        from model.vballnet_v1c import VballNetV1c
        return VballNetV1c(height=IMG_H, width=IMG_W, in_dim=SEQ, out_dim=SEQ)
    else:
        raise ValueError(f"未知模型: {model_name}")


def export(weights_path: Path, output_path: Path, model_name: str = 'VballNetV1a'):
    print(f"載入 checkpoint: {weights_path}")
    ckpt = torch.load(weights_path, map_location='cpu')
    epoch = ckpt.get('epoch', '?')
    val_loss = ckpt.get('val_loss', '?')
    print(f"  epoch={epoch}  val_loss={val_loss}")

    model = build_model(model_name)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"[Model] {model_name}  params={sum(p.numel() for p in model.parameters()):,}")

    dummy = torch.zeros(1, SEQ, IMG_H, IMG_W)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # V1c forward 回傳 (output, hn)；包裝成只輸出 heatmap 的 wrapper
    if model_name == 'VballNetV1c':
        class V1cWrapper(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
            def forward(self, x):
                out, _ = self.m(x)
                return out
        export_model = V1cWrapper(model)
    else:
        export_model = model

    print(f"匯出 ONNX: {output_path}")
    torch.onnx.export(
        export_model, dummy, str(output_path),
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
        opset_version=17,
        dynamo=False,
    )
    print("完成！")

    # 驗證
    import onnx
    m = onnx.load(str(output_path))
    onnx.checker.check_model(m)
    inp = m.graph.input[0].type.tensor_type.shape
    out = m.graph.output[0].type.tensor_type.shape
    print(f"  input shape:  {[d.dim_value for d in inp.dim]}")
    print(f"  output shape: {[d.dim_value for d in out.dim]}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--weights',    type=str, required=True)
    p.add_argument('--output',     type=str, default=None,
                   help='輸出 ONNX 路徑（預設與 .pth 同目錄）')
    p.add_argument('--model_name', type=str, default=None,
                   choices=['VballNetV1a', 'VballNetV1b', 'VballNetV1c'],
                   help='模型架構（預設從 checkpoint 目錄名稱自動推斷）')
    return p.parse_args()


def _infer_model_name(weights_path: Path) -> str:
    """從 checkpoint 路徑名稱自動推斷模型。"""
    name = str(weights_path)
    for m in ['VballNetV1b', 'VballNetV1c', 'VballNetV1a']:
        if m in name:
            return m
    return 'VballNetV1a'


if __name__ == '__main__':
    args = parse_args()
    wp = Path(args.weights)
    model_name = args.model_name or _infer_model_name(wp)
    print(f"[模型] {model_name}")
    suffix = f'_{model_name}_seq9.onnx'
    op = Path(args.output) if args.output else wp.parent / (wp.stem + suffix)
    export(wp, op, model_name=model_name)
