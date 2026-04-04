"""
批次影片分割腳本
使用 ROI 配置批次處理多個影片
"""
import os
import subprocess
import argparse
import glob


def main():
    parser = argparse.ArgumentParser(
        description="批次分割影片",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  python batch_video_slicing.py \\
      --video-dir input_video/original_video \\
      --roi-config-dir roi_configs \\
      --output-dir output_data/video_segments_batch

說明:
  此工具會自動為每個影片尋找對應的 ROI 配置檔案，並執行影片分割。
  ROI 配置檔名格式：{影片名稱}_roi_config.json

注意:
  此腳本需要在 Anaconda base 環境中執行。
        """
    )
    parser.add_argument("--video-dir", type=str, required=True,
                        help="影片目錄")
    parser.add_argument("--roi-config-dir", type=str, required=True,
                        help="ROI 配置目錄")
    parser.add_argument("--output-dir", type=str, default="output_data/video_segments_batch",
                        help="輸出根目錄（預設: output_data/video_segments_batch）")
    parser.add_argument("--min-segment-duration", type=int, default=10,
                        help="最小片段時長（秒，預設: 10）")
    parser.add_argument("--long-segment-threshold", type=int, default=90,
                        help="長片段閾值（秒，預設: 90）")
    parser.add_argument("--diff-threshold", type=int, default=9000,
                        help="SAD 差異閾值（預設: 9000）")
    parser.add_argument("--video-extension", type=str, default=".mp4",
                        help="影片檔案副檔名（預設: .mp4）")
    args = parser.parse_args()

    # 驗證目錄存在
    if not os.path.exists(args.video_dir):
        print(f"[ERROR] 影片目錄不存在: {args.video_dir}")
        return

    if not os.path.exists(args.roi_config_dir):
        print(f"[ERROR] ROI 配置目錄不存在: {args.roi_config_dir}")
        return

    # 找出所有影片
    videos = sorted(glob.glob(os.path.join(args.video_dir, f"*{args.video_extension}")))

    if not videos:
        print(f"[ERROR] 在 {args.video_dir} 中找不到 {args.video_extension} 影片檔案")
        return

    print(f"\n{'='*60}")
    print(f"批次影片分割腳本")
    print(f"{'='*60}")
    print(f"影片目錄: {args.video_dir}")
    print(f"ROI 配置目錄: {args.roi_config_dir}")
    print(f"輸出目錄: {args.output_dir}")
    print(f"找到 {len(videos)} 個影片檔案")
    print(f"{'='*60}\n")

    success_count = 0
    failed_videos = []
    skipped_videos = []

    for i, video_path in enumerate(videos, 1):
        video_name = os.path.basename(video_path)
        video_basename = video_name.replace(args.video_extension, '')

        print(f"\n{'='*60}")
        print(f"[{i}/{len(videos)}] 處理中: {video_name}")
        print(f"{'='*60}")

        # 尋找對應的 ROI 配置
        roi_config_path = os.path.join(args.roi_config_dir,
                                       f"{video_basename}_roi_config.json")

        if not os.path.exists(roi_config_path):
            print(f"[ERROR] 找不到 ROI 配置: {roi_config_path}")
            print(f"        請先使用 roi_config_generator.py 生成配置")
            failed_videos.append((video_name, "No ROI config"))
            continue

        # 設定輸出目錄
        output_subdir = os.path.join(args.output_dir, video_basename)

        # 檢查是否已處理過
        if os.path.exists(output_subdir):
            summary_path = os.path.join(output_subdir, "slicing_summary.csv")
            if os.path.exists(summary_path):
                print(f"[SKIP] 輸出目錄已存在: {output_subdir}")
                print(f"       如需重新處理，請先刪除該目錄")
                skipped_videos.append(video_name)
                continue

        print(f"[INFO] 使用 ROI 配置: {roi_config_path}")
        print(f"[INFO] 輸出至: {output_subdir}")

        # 執行影片分割
        cmd = [
            "python", "video_processing/video_slicer_by_score.py",
            "--input", video_path,
            "--output_dir", output_subdir,
            "--roi_config", roi_config_path,
            "--min_segment_duration", str(args.min_segment_duration),
            "--long_segment_threshold", str(args.long_segment_threshold),
            "--diff_threshold", str(args.diff_threshold)
        ]

        # 需要在 Anaconda 環境中執行
        cmd_str = " && ".join([
            'source "C:/Users/Aa954/anaconda3/etc/profile.d/conda.sh"',
            "conda activate base",
            " ".join(cmd)
        ])

        try:
            print(f"\n[INFO] 執行命令...")
            # 使用 errors='replace' 處理編碼問題（Windows 輸出可能是 cp950）
            result = subprocess.run(cmd_str, shell=True, check=True,
                                   capture_output=True, text=True,
                                   encoding='utf-8', errors='replace')

            print(result.stdout)
            print(f"[OK] 分割完成: {video_basename}")
            success_count += 1

        except subprocess.CalledProcessError as e:
            print(f"[ERROR] 分割失敗: {video_basename}")
            # 改善錯誤訊息顯示
            if e.stderr:
                print(f"        錯誤訊息: {e.stderr}")
            if e.stdout:
                print(f"        輸出: {e.stdout}")
            if not e.stderr and not e.stdout:
                print(f"        返回碼: {e.returncode}")
            failed_videos.append((video_name, f"Exit code {e.returncode}"))

        except Exception as e:
            print(f"[ERROR] 執行失敗: {video_basename}")
            print(f"        錯誤: {str(e)}")
            failed_videos.append((video_name, str(type(e).__name__)))

    # 輸出摘要
    print(f"\n{'='*60}")
    print(f"批次處理完成！")
    print(f"{'='*60}")
    print(f"  成功: {success_count}/{len(videos)}")
    print(f"  跳過: {len(skipped_videos)}/{len(videos)} (已處理過)")
    print(f"  失敗: {len(failed_videos)}/{len(videos)}")

    if skipped_videos:
        print(f"\n跳過清單:")
        for video in skipped_videos:
            print(f"    - {video}")

    if failed_videos:
        print(f"\n失敗清單:")
        for video, reason in failed_videos:
            print(f"    - {video} ({reason})")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
