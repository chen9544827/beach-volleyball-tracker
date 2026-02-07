# batch_segment_pipeline.py
# -*- coding: utf-8 -*-
"""
批次影片分段管線

自動流程：
1. 掃描 original_video 資料夾，用 filename_parser 解析並分組
2. 對每個分組，檢查是否有對應的 ROI 模板
   - 有 ROI -> 自動執行影片分段
   - 沒有 ROI -> 啟動 roi_config_generator_v2.py 讓使用者框選
3. 分段結果按分組存放

用法:
  python batch_segment_pipeline.py --video-dir input_video/original_video
  python batch_segment_pipeline.py --video-dir input_video/original_video --roi-only
  python batch_segment_pipeline.py --video-dir input_video/original_video --slice-only
"""

import os
import sys
import subprocess
import argparse

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from core.filename_parser import parse_filename, group_videos, scan_video_directory, print_group_summary


def find_roi_config(venue_name, venues_dir):
    """
    找場地的 ROI 模板

    嘗試多種名稱匹配（大小寫不敏感）

    Args:
        venue_name: 場地名 (e.g., 'Edmonton')
        venues_dir: 場地模板目錄

    Returns:
        ROI config 路徑，找不到返回 None
    """
    if not os.path.isdir(venues_dir):
        return None

    # 直接匹配
    direct = os.path.join(venues_dir, f"{venue_name}.json")
    if os.path.exists(direct):
        return direct

    # 大小寫不敏感搜尋
    venue_lower = venue_name.lower()
    for fname in os.listdir(venues_dir):
        if fname.lower() == f"{venue_lower}.json":
            return os.path.join(venues_dir, fname)

    return None


def find_output_segments(group_key, output_base_dir):
    """
    檢查某分組是否已有分段輸出

    Args:
        group_key: 分組鍵 (e.g., 'Edmonton_WT19_C4')
        output_base_dir: 輸出根目錄

    Returns:
        已存在的片段數量
    """
    group_dir = os.path.join(output_base_dir, group_key)
    if not os.path.isdir(group_dir):
        return 0

    count = 0
    for fname in os.listdir(group_dir):
        if fname.endswith('.mp4'):
            count += 1
    return count


def run_roi_setup(video_path, venue_name, venues_dir):
    """
    啟動 ROI 設定工具

    Args:
        video_path: 影片路徑（用作參考）
        venue_name: 場地名
        venues_dir: 場地模板目錄

    Returns:
        是否成功設定
    """
    output_path = os.path.join(venues_dir, f"{venue_name}.json")
    # output for video-level config (required by roi_config_generator_v2)
    video_config_path = os.path.join("roi_configs", "videos",
                                      f"{os.path.splitext(os.path.basename(video_path))[0]}_roi_config.json")

    print(f"\n{'='*60}")
    print(f"  ROI Setup: {venue_name}")
    print(f"  Video: {os.path.basename(video_path)}")
    print(f"  Venue template: {output_path}")
    print(f"{'='*60}")
    print(f"  GUI will open. Drag to select score ROIs.")
    print(f"  Step 1: Drag Team1 score area (green)")
    print(f"  Step 2: Drag Team2 score area (red)")
    print(f"  Press 's' to save, 'r' to reset, 'q' to quit.\n")

    os.makedirs(os.path.dirname(video_config_path), exist_ok=True)

    cmd = [
        sys.executable,
        os.path.join(PROJECT_ROOT, "video_processing", "roi_config_generator_v2.py"),
        "--video", video_path,
        "--output", video_config_path,
        "--venues-dir", venues_dir,
    ]

    # roi_config_generator_v2.py uses input() to ask for venue selection.
    # We auto-feed: "n" (create new) + venue_name
    # If venues already exist, it shows a numbered list + [n]/[q],
    # so "n" then venue_name works in both cases.
    stdin_input = f"n\n{venue_name}\n"

    try:
        result = subprocess.run(cmd, cwd=PROJECT_ROOT,
                                input=stdin_input, text=True)
        if result.returncode == 0 and os.path.exists(output_path):
            print(f"  [OK] ROI saved: {output_path}")
            return True
        else:
            print(f"  [WARNING] ROI setup may not have completed successfully.")
            return os.path.exists(output_path)
    except Exception as e:
        print(f"  [ERROR] Failed to run ROI setup: {e}")
        return False


def run_video_slicing(video_path, roi_config_path, output_dir):
    """
    執行單一影片的分段

    Args:
        video_path: 影片路徑
        roi_config_path: ROI 設定路徑
        output_dir: 輸出目錄

    Returns:
        是否成功
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    segment_output_dir = os.path.join(output_dir, video_name)

    # 檢查是否已分段
    if os.path.isdir(segment_output_dir):
        existing = [f for f in os.listdir(segment_output_dir) if f.endswith('.mp4')]
        if existing:
            print(f"    [SKIP] Already segmented ({len(existing)} segments): {video_name}")
            return True

    print(f"    [SLICE] Segmenting: {video_name}")

    cmd = [
        sys.executable,
        os.path.join(PROJECT_ROOT, "video_processing", "video_slicer_by_score.py"),
        "--input", video_path,
        "--output_dir", segment_output_dir,
        "--roi_config", roi_config_path,
    ]

    try:
        result = subprocess.run(cmd, cwd=PROJECT_ROOT)
        return result.returncode == 0
    except Exception as e:
        print(f"    [ERROR] Slicing failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="批次影片分段管線 - 自動解析檔名、匹配 ROI、分段影片",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  # 完整流程（缺 ROI 的會開 GUI 讓你框選）
  python batch_segment_pipeline.py --video-dir input_video/original_video

  # 只設定 ROI（不做分段）
  python batch_segment_pipeline.py --video-dir input_video/original_video --roi-only

  # 只做分段（跳過沒有 ROI 的）
  python batch_segment_pipeline.py --video-dir input_video/original_video --slice-only

  # 查看分組狀態（不執行任何操作）
  python batch_segment_pipeline.py --video-dir input_video/original_video --dry-run
        """
    )
    parser.add_argument("--video-dir", type=str, required=True,
                        help="原始影片目錄")
    parser.add_argument("--venues-dir", type=str, default="roi_configs/venues",
                        help="ROI 場地模板目錄 (default: roi_configs/venues)")
    parser.add_argument("--output-dir", type=str, default="output_data/video_segments",
                        help="分段輸出目錄 (default: output_data/video_segments)")
    parser.add_argument("--roi-only", action="store_true",
                        help="只設定缺少的 ROI，不做分段")
    parser.add_argument("--slice-only", action="store_true",
                        help="只做分段，跳過沒有 ROI 的分組")
    parser.add_argument("--dry-run", action="store_true",
                        help="只顯示分組狀態，不執行任何操作")

    args = parser.parse_args()

    # Step 1: 掃描並分組
    print("=" * 60)
    print("  Batch Segment Pipeline")
    print("=" * 60)

    videos = scan_video_directory(args.video_dir)
    if not videos:
        print(f"[ERROR] No videos found in: {args.video_dir}")
        return

    groups = group_videos(videos)
    print_group_summary(groups)

    # 確保目錄存在
    os.makedirs(args.venues_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # Step 2: 檢查每個分組的 ROI 狀態
    print(f"\n{'='*60}")
    print("  ROI Status Check")
    print(f"{'='*60}")

    groups_with_roi = {}
    groups_without_roi = {}

    for group_key in sorted(groups.keys()):
        if group_key == '_unparsed':
            continue

        entries = groups[group_key]
        venue = entries[0]['parsed']['venue']
        roi_path = find_roi_config(venue, args.venues_dir)

        if roi_path:
            groups_with_roi[group_key] = {
                'entries': entries,
                'venue': venue,
                'roi_path': roi_path,
            }
            print(f"  [OK]   {group_key}: ROI found ({os.path.basename(roi_path)})")
        else:
            groups_without_roi[group_key] = {
                'entries': entries,
                'venue': venue,
            }
            print(f"  [MISS] {group_key}: No ROI config for venue '{venue}'")

    unparsed = groups.get('_unparsed', [])
    if unparsed:
        print(f"  [SKIP] {len(unparsed)} unparsed videos")

    print(f"\n  Summary: {len(groups_with_roi)} ready, {len(groups_without_roi)} need ROI setup")

    if args.dry_run:
        print("\n[DRY RUN] No actions taken.")
        return

    # Step 3: 設定缺少的 ROI
    if groups_without_roi and not args.slice_only:
        print(f"\n{'='*60}")
        print(f"  ROI Setup ({len(groups_without_roi)} venues)")
        print(f"{'='*60}")

        for group_key, info in groups_without_roi.items():
            venue = info['venue']
            # 取該分組的第一部影片作為參考
            ref_video = info['entries'][0]['path']

            success = run_roi_setup(ref_video, venue, args.venues_dir)
            if success:
                roi_path = find_roi_config(venue, args.venues_dir)
                if roi_path:
                    groups_with_roi[group_key] = {
                        'entries': info['entries'],
                        'venue': venue,
                        'roi_path': roi_path,
                    }
                    print(f"  [OK] {venue} ROI setup complete!")

    if args.roi_only:
        print("\n[ROI-ONLY] ROI setup complete. Skipping video slicing.")
        return

    # Step 4: 執行分段
    if groups_with_roi:
        print(f"\n{'='*60}")
        print(f"  Video Slicing ({len(groups_with_roi)} groups)")
        print(f"{'='*60}")

        total_videos = 0
        total_sliced = 0
        total_skipped = 0

        for group_key in sorted(groups_with_roi.keys()):
            info = groups_with_roi[group_key]
            print(f"\n  [{group_key}] ({len(info['entries'])} videos, ROI: {os.path.basename(info['roi_path'])})")

            group_output = os.path.join(args.output_dir, group_key)
            os.makedirs(group_output, exist_ok=True)

            for entry in info['entries']:
                total_videos += 1
                success = run_video_slicing(
                    entry['path'],
                    info['roi_path'],
                    group_output
                )
                if success:
                    total_sliced += 1

        print(f"\n{'='*60}")
        print(f"  Slicing Complete")
        print(f"  Total: {total_videos}, Sliced: {total_sliced}")
        print(f"{'='*60}")

    # Summary
    still_missing = [g for g in groups_without_roi if g not in groups_with_roi]
    if still_missing:
        print(f"\n[WARNING] {len(still_missing)} groups still missing ROI:")
        for g in still_missing:
            print(f"  - {g} (venue: {groups_without_roi[g]['venue']})")
        print(f"\nRun again or use --roi-only to set them up.")


if __name__ == "__main__":
    main()
