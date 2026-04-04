# tools/export_vballnet_hires.py
# -*- coding: utf-8 -*-
"""
將 VballNetV1.keras 以更高解析度重新 export 成 ONNX

因模型內含 Reshape((num_frames, channels, H, W)) 層硬編碼原始解析度，
無法直接改 input_signature。改用「重建模型 + 轉移權重」方式：
1. 載入原始 .keras 取得訓練權重
2. 用新解析度重建相同架構的模型
3. 逐層轉移 ConvNet 權重（與解析度無關）
4. 用 tf2onnx 匯出新解析度的 ONNX

用法:
    conda activate tff_env
    cd vball-net/src
    python ../../tools/export_vballnet_hires.py \
        --model_path ../../models/vb-models/VballNetV1_150.keras \
        --height 576 --width 1024 \
        --output_dir ../../models/vb-models
"""

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VBALL_SRC = os.path.join(PROJECT_ROOT, 'vball-net', 'src')
sys.path.insert(0, VBALL_SRC)


def parse_args():
    parser = argparse.ArgumentParser(
        description='VballNetV1.keras -> 高解析度 ONNX（重建 + 轉移權重）'
    )
    parser.add_argument('--model_path', type=str, required=True,
                        help='原始 .keras 路徑')
    parser.add_argument('--height', type=int, default=576,
                        help='新輸出高度（預設 576）')
    parser.add_argument('--width', type=int, default=1024,
                        help='新輸出寬度（預設 1024）')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='ONNX 輸出目錄（預設與 model_path 同層）')
    return parser.parse_args()


def main():
    args = parse_args()

    import tensorflow as tf
    import tf2onnx
    import onnx
    from model.VballNetV1 import VballNetV1, MotionPromptLayer, FusionLayerTypeA, FusionLayerTypeB
    from utils import custom_loss

    custom_objects = {
        'MotionPromptLayer': MotionPromptLayer,
        'FusionLayerTypeA': FusionLayerTypeA,
        'FusionLayerTypeB': FusionLayerTypeB,
        'custom_loss': custom_loss,
    }

    print(f"[1/4] 載入原始模型: {args.model_path}")
    model_old = tf.keras.models.load_model(
        args.model_path, custom_objects=custom_objects
    )

    # 從原始模型推斷設定
    old_input_shape = model_old.input_shape   # (None, in_dim, H_old, W_old)
    in_dim = old_input_shape[1]               # e.g. 9
    old_h = old_input_shape[2]               # e.g. 288
    old_w = old_input_shape[3]               # e.g. 512
    out_dim = model_old.output_shape[1]       # e.g. 3

    # 推斷 fusion_layer_type
    fusion_type = "TypeB"
    for layer in model_old.layers:
        if 'fusion_layer_type_a' in layer.name:
            fusion_type = "TypeA"
            break
        elif 'fusion_layer_type_b' in layer.name:
            fusion_type = "TypeB"
            break

    print(f"    原始設定: in_dim={in_dim}, out_dim={out_dim}, "
          f"解析度={old_h}x{old_w}, fusion={fusion_type}")
    print(f"    目標解析度: {args.height}x{args.width}")

    print(f"[2/4] 重建新解析度模型")
    model_new = VballNetV1(
        height=args.height,
        width=args.width,
        in_dim=in_dim,
        out_dim=out_dim,
        fusion_layer_type=fusion_type
    )

    print(f"[3/4] 轉移權重")
    # 取得兩個模型的可訓練權重
    old_weights = model_old.get_weights()
    new_weights = model_new.get_weights()

    if len(old_weights) != len(new_weights):
        print(f"[WARNING] 權重數量不一致: 舊={len(old_weights)}, 新={len(new_weights)}")
        print("嘗試逐層名稱匹配...")
        # 建立名稱對應
        old_layer_weights = {l.name: l.get_weights()
                              for l in model_old.layers if l.get_weights()}
        transferred = 0
        for layer in model_new.layers:
            if layer.name in old_layer_weights:
                try:
                    layer.set_weights(old_layer_weights[layer.name])
                    transferred += 1
                except Exception as e:
                    print(f"  [SKIP] {layer.name}: {e}")
        print(f"  成功轉移 {transferred} 層")
    else:
        # 直接轉移（形狀相同，因為都是 conv 層）
        model_new.set_weights(old_weights)
        print(f"  成功轉移全部 {len(old_weights)} 個權重張量")

    print(f"[4/4] 匯出 ONNX at {args.height}x{args.width}")
    input_shape = (None, in_dim, args.height, args.width)
    input_signature = [tf.TensorSpec(shape=input_shape, dtype=tf.float32, name='input')]

    onnx_model, _ = tf2onnx.convert.from_keras(
        model_new,
        input_signature=input_signature,
        opset=13
    )

    # 輸出路徑
    output_dir = args.output_dir or os.path.dirname(args.model_path)
    base_name = os.path.splitext(os.path.basename(args.model_path))[0]
    out_path = os.path.join(output_dir, f"{base_name}_h{args.height}_w{args.width}.onnx")

    onnx.checker.check_model(onnx_model)
    onnx.save_model(onnx_model, out_path)
    print(f"[OK] 匯出完成: {out_path}")
    print(f"     輸入: (batch, {in_dim}, {args.height}, {args.width})")
    print(f"     輸出: (batch, {out_dim}, {args.height}, {args.width})")


if __name__ == '__main__':
    main()
