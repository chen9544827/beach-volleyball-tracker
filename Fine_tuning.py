#!/usr/bin/env python3
# finetune_schemeB.py
# 在已训练的 ball-only 模型基础上，结合少量 ball 样本和全量 player 样本进行增量微调

from ultralytics import YOLO

# ====== 在这里直接设置参数 ======
model_path       = "runs/detect/ball_only/weights/best.pt"    # 预训练的 ball-only 模型路径
data_yaml        = "combined_dataset/data.yaml"               # 合并后双类数据的 data.yaml
epochs           = 30                                          # 训练世代
imgsz            = 960                                         # 输入图片尺寸
batch_size       = 16                                          # 批量大小
device           = "0"                                        # 'cpu' 或 GPU id
# ================================

def main():
    # 加载预训练模型（包含 ball-only 权重）
    model = YOLO(model_path)

    # 增量微调：只冻结 backbone（block0），保持 Detect 层包括分类头可训练
    # 使用 train 中的 freeze 参数更可靠
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        freeze=[0],             # 冻结 backbone；解除 Detect 层所有通道更新
        name="ball_player_schemeB",
        exist_ok=True
    )

if __name__ == "__main__":
    main()
