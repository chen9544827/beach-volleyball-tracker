#!/usr/bin/env python3
"""
finetune_combined.py

在已准备好的双类数据集（ball + player）上微调 YOLO11 模型。
请确保：
  1. 已安装 ultralytics 包
  2. data.yaml 在 data_dir 下，内容示例：
     train: train/images
     val:   val/images
     nc: 2
     names: ['ball','player']
  3. 你已放置合适的 COCO 预训练权重 yolo11s.pt

用法：
  python finetune_combined.py

脚本会从 COCO 权重加载 yolo11s.pt，并在 combined_dataset 上进行微调。
"""
from ultralytics import YOLO

# ====== 配置部分 ======
# 预训练权重文件
base_weights    = 'yolo11s.pt'                # 或者 'yolov8s.pt'
# 数据集配置文件
data_yaml       = 'combined_dataset/data.yaml'
# 训练参数
epochs          = 50                          # 训练总轮数
imgsz           = 960                         # 输入尺寸
batch_size      = 16                          # 批量大小
device_id       = '0'                         # GPU id，如 '0' 或 'cpu'
# =====================

def main():
    device = f"cuda:{device_id}" if device_id.isdigit() else device_id

    # 1. 加载 COCO 预训练模型
    model = YOLO(base_weights)

    # 2. 开始微调
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        name='finetune_ball_player',
        exist_ok=True  # 如果同名 runs 目录已存在则覆盖
    )

    print('微调完成，权重保存在 runs/train/finetune_ball_player/weights/')

if __name__ == '__main__':
    main()
