#!/usr/bin/env python3
# merge_datasets.py
# 结合两个单类YOLO数据集（ball & player），支持train/valid/test三分法

import os
import shutil
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge two single-class YOLO datasets into one multi-class dataset with train/valid/test splits."
    )
    parser.add_argument(
        "--ball_dir", type=str, required=True,
        help="只含 ball 标签的数据集根目录（包含 train/valid/test 子目录）"
    )
    parser.add_argument(
        "--player_dir", type=str, required=True,
        help="只含 player 标签的数据集根目录（包含 train/valid/test 子目录）"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="输出合并后数据集的根目录"
    )
    return parser.parse_args()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def copy_and_remap_labels(src_label, dst_label, remap=False):
    with open(src_label, 'r') as f_src, open(dst_label, 'w') as f_dst:
        for line in f_src:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, *coords = parts
            if remap:
                cls = '1'  # player 的 class id 从 0 重映射为 1
            f_dst.write(' '.join([cls] + coords) + '\n')


def merge_subset(subset, ball_dir, player_dir, out_dir):
    # 输入子集目录
    ball_img_dir = os.path.join(ball_dir, subset, 'images')
    ball_lbl_dir = os.path.join(ball_dir, subset, 'labels')
    plr_img_dir  = os.path.join(player_dir, subset, 'images')
    plr_lbl_dir  = os.path.join(player_dir, subset, 'labels')

    # 输出子集目录
    out_img_dir = os.path.join(out_dir, subset, 'images')
    out_lbl_dir = os.path.join(out_dir, subset, 'labels')
    ensure_dir(out_img_dir)
    ensure_dir(out_lbl_dir)

    idx = 0
    # 合并 ball 数据
    for fname in sorted(os.listdir(ball_img_dir)):
        if not fname.lower().endswith(('.jpg', '.png')):
            continue
        base = f"{idx:06d}"
        # 复制图片
        src_img = os.path.join(ball_img_dir, fname)
        dst_img = os.path.join(out_img_dir, base + os.path.splitext(fname)[1])
        shutil.copy(src_img, dst_img)
        # 复制标签
        src_lbl = os.path.join(ball_lbl_dir, os.path.splitext(fname)[0] + '.txt')
        dst_lbl = os.path.join(out_lbl_dir, base + '.txt')
        copy_and_remap_labels(src_lbl, dst_lbl, remap=False)
        idx += 1

    # 合并 player 数据
    for fname in sorted(os.listdir(plr_img_dir)):
        if not fname.lower().endswith(('.jpg', '.png')):
            continue
        base = f"{idx:06d}"
        # 复制图片
        src_img = os.path.join(plr_img_dir, fname)
        dst_img = os.path.join(out_img_dir, base + os.path.splitext(fname)[1])
        shutil.copy(src_img, dst_img)
        # 复制并重映射标签
        src_lbl = os.path.join(plr_lbl_dir, os.path.splitext(fname)[0] + '.txt')
        dst_lbl = os.path.join(out_lbl_dir, base + '.txt')
        copy_and_remap_labels(src_lbl, dst_lbl, remap=True)
        idx += 1

    print(f"Subset '{subset}' merged {idx} images.")


def main():
    args = parse_args()
    subsets = ['train', 'valid', 'test']
    for subset in subsets:
        merge_subset(subset, args.ball_dir, args.player_dir, args.output_dir)
    print(f"All subsets merged into {args.output_dir}")

if __name__ == '__main__':
    main()
