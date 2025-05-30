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
freeze_backbone  = True                                        # 是否冻结 backbone，仅微调 Detect 层
# ================================

def main():
    # 加载预训练模型（只检测 ball）
    model = YOLO(model_path)

    # 可选冻结 backbone
    if freeze_backbone:
        # 冻结除 Detect 层外的所有参数
        for name, param in model.model.named_parameters():
            param.requires_grad = False
        # 解冻 Detect 层
        for param in model.model.model[-1].parameters():
            param.requires_grad = True
        print("Backbone frozen, only Detect layer will be trained.")

    # 开始训练
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        name="ball_player_schemeB",
        exist_ok=True
    )

if __name__ == "__main__":
    main()
