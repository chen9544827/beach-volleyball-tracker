from ultralytics import YOLO

# 1. 加载预训练 YOLO11 small
model = YOLO("yolo11s.pt")  

# 2. 微调训练
model.train(
    data="dataset_combined/data.yaml",  # 刚才生成的配置文件
    epochs=50,                          # 训练世代
    imgsz=960,                          # 输入尺寸
    batch=16,                           # 批量大小
    device=0,                           # GPU 0
    augment=False                       # 已在 Roboflow 做过增强，可关
)
