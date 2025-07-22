#!/usr/bin/env python3
"""
finetune_combined.py
...
"""
from ultralytics import YOLO

# ====== 配置部分 ======
# ✨ ---【核心修改】--- ✨
# 預訓練權重文件，請務必使用您現有的模型，而不是從頭開始！
base_weights    = 'models/ball_best.pt' # <--- 修改這裡，指向您現有的球模型

# 資料集配置文件，指向您剛剛準備好的微調資料集
data_yaml       = 'D:/Github/beach-volleyball-tracker/dataset/finntunning_vollyball_123.v2i.yolov8/data.yaml' # <--- 修改這裡

# 訓練參數
epochs          = 30                          # ✨ 微調通常不需要太多輪，25-50輪即可
imgsz           = 640                         # 建議使用與您主模型相同的尺寸
batch_size      = 8                           # ✨ 圖片不多，batch size可以設小一點
device_id       = '0'                         # GPU id，如 '0' 或 'cpu'
# =====================

def main():
    device = f"cuda:{device_id}" if device_id.isdigit() else device_id

    # 1. 載入您現有的 ball_best.pt 模型
    model = YOLO(base_weights)

    # 2. 開始微調
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        name='finetune_ball_on_background_issue', # 給這次微調一個描述性的名字
        exist_ok=True
    )

    print('微調完成，新的、更強大的模型保存在 runs/train/finetune_ball_on_background_issue/weights/best.pt')
    print('請將這個 best.pt 複製出來，並重新命名為 ball_best.pt 來取代舊模型。')

if __name__ == '__main__':
    main()