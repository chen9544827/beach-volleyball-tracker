import os
import random
import shutil
from glob import glob

# ---------- 配置区：请根据实际下载路径修改 ----------
indoor_root = "path/to/indoor_dataset"   # 室内数据集根目录，含 train/valid/test 子目录
beach_root  = "path/to/beach_dataset"    # 沙滩数据集根目录，含 train/valid/test 子目录
output_root = "dataset_combined"         # 合并后输出根目录
sampling_ratio = 1  # 室内采样数量 = 沙滩数量 × sampling_ratio

# 支持的图片扩展名
IMG_EXTS = ['.jpg', '.jpeg', '.png']

# 数据集拆分
splits = ["train", "valid", "test"]

for split in splits:
    # 定义各自的 images/labels 目录
    indoor_img_dir = os.path.join(indoor_root, split, "images")
    indoor_lbl_dir = os.path.join(indoor_root, split, "labels")
    beach_img_dir  = os.path.join(beach_root,  split, "images")
    beach_lbl_dir  = os.path.join(beach_root,  split, "labels")

    # 列出沙滩图片并统计
    beach_images = [f for f in glob(os.path.join(beach_img_dir, "*")) \
                    if os.path.splitext(f)[1].lower() in IMG_EXTS]
    beach_count = len(beach_images)

    # 计算室内采样数量
    indoor_images_all = [f for f in glob(os.path.join(indoor_img_dir, "*")) \
                         if os.path.splitext(f)[1].lower() in IMG_EXTS]
    sample_count = min(beach_count * sampling_ratio, len(indoor_images_all))

    # 随机采样室内图片
    sampled_indoor = random.sample(indoor_images_all, sample_count)

    # 创建输出目录结构
    out_img_dir = os.path.join(output_root, "images", split)
    out_lbl_dir = os.path.join(output_root, "labels", split)
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    # 拷贝采样后的室内图片及对应标签
    for img_path in sampled_indoor:
        fname = os.path.basename(img_path)
        shutil.copy(img_path, os.path.join(out_img_dir, fname))
        lbl_name = os.path.splitext(fname)[0] + ".txt"
        src_lbl = os.path.join(indoor_lbl_dir, lbl_name)
        if os.path.isfile(src_lbl):
            shutil.copy(src_lbl, os.path.join(out_lbl_dir, lbl_name))

    # 拷贝所有沙滩图片及对应标签
    for img_path in beach_images:
        fname = os.path.basename(img_path)
        shutil.copy(img_path, os.path.join(out_img_dir, fname))
        lbl_name = os.path.splitext(fname)[0] + ".txt"
        src_lbl = os.path.join(beach_lbl_dir, lbl_name)
        if os.path.isfile(src_lbl):
            shutil.copy(src_lbl, os.path.join(out_lbl_dir, lbl_name))

    print(f"[{split}] 采样室内: {sample_count} 张, 合并沙滩: {beach_count} 张, 总计: {sample_count + beach_count} 张")

print("数据合并完成，路径：", output_root)
