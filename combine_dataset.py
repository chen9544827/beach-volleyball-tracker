import os, random, shutil
from glob import glob

# ---------- 配置区，请替换为你的本地路径 ----------
indoor_root = "path/to/indoor_dataset"   # 室内数据集根目录
beach_root  = "path/to/beach_dataset"    # 沙滩数据集根目录
output_root = "dataset_combined"         # 下采样后输出根目录
sampling_ratio = 1  # 采样比例：1 表示室内数量=沙滩数量，2 表示室内数量=沙滩数量*2
# -------------------------------------------

splits = ["train", "valid", "test"]

for split in splits:
    indoor_img_dir = os.path.join(indoor_root, "images", split)
    indoor_lbl_dir = os.path.join(indoor_root, "labels", split)
    beach_img_dir  = os.path.join(beach_root, "images", split)

    # 计算沙滩图数量并确定采样数量
    beach_count = len(glob(os.path.join(beach_img_dir, "*.jpg")))
    sample_count = min(beach_count * sampling_ratio,
                       len(glob(os.path.join(indoor_img_dir, "*.jpg"))))

    # 随机采样室内图
    indoor_images = glob(os.path.join(indoor_img_dir, "*.jpg"))
    sampled_imgs = random.sample(indoor_images, sample_count)

    # 创建输出目录
    out_img_dir = os.path.join(output_root, "images", split)
    out_lbl_dir = os.path.join(output_root, "labels", split)
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    # 拷贝采样后的图片与对应标签
    for img_path in sampled_imgs:
        fname = os.path.basename(img_path)
        lbl_src = os.path.join(indoor_lbl_dir, fname.replace('.jpg', '.txt'))
        shutil.copy(img_path, os.path.join(out_img_dir, fname))
        if os.path.exists(lbl_src):
            shutil.copy(lbl_src, os.path.join(out_lbl_dir, fname.replace('.jpg', '.txt')))

    print(f"{split}: sampled {sample_count} images from indoor data")

print("下采样完成，合并目录：", output_root)
