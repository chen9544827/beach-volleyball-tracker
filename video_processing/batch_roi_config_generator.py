"""
批次 ROI 配置生成器
自動為多個影片依序生成 ROI 配置檔案
"""
import os
import argparse
import sys

# 添加父目錄到路徑以導入 roi_config_generator
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from roi_config_generator import ROIConfigGenerator


def main():
    parser = argparse.ArgumentParser(
        description="批次生成 ROI 配置",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  python batch_roi_config_generator.py \\
      --video-dir input_video/original_video \\
      --output-dir roi_configs

說明:
  此工具會依序為每個影片開啟互動視窗，讓用戶設定 ROI。
  如果某個影片的配置檔案已存在，則會自動跳過。
        """
    )
    parser.add_argument("--video-dir", type=str, required=True,
                        help="影片目錄")
    parser.add_argument("--output-dir", type=str, default="roi_configs",
                        help="ROI 配置輸出目錄（預設: roi_configs）")
    parser.add_argument("--video-extension", type=str, default=".mp4",
                        help="影片檔案副檔名（預設: .mp4）")
    args = parser.parse_args()

    # 驗證影片目錄存在
    if not os.path.exists(args.video_dir):
        print(f"[ERROR] 影片目錄不存在: {args.video_dir}")
        return

    # 列出所有影片
    videos = sorted([
        f for f in os.listdir(args.video_dir)
        if f.lower().endswith(args.video_extension.lower())
    ])

    if not videos:
        print(f"[ERROR] 在 {args.video_dir} 中找不到 {args.video_extension} 影片檔案")
        return

    print(f"\n{'='*60}")
    print(f"批次 ROI 配置生成器")
    print(f"{'='*60}")
    print(f"影片目錄: {args.video_dir}")
    print(f"輸出目錄: {args.output_dir}")
    print(f"找到 {len(videos)} 個影片檔案")
    print(f"{'='*60}\n")

    # 確保輸出目錄存在
    os.makedirs(args.output_dir, exist_ok=True)

    success_count = 0
    skipped_count = 0
    failed_count = 0

    for i, video_name in enumerate(videos, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(videos)}] {video_name}")
        print(f"{'='*60}")

        video_path = os.path.join(args.video_dir, video_name)
        config_name = video_name.replace(args.video_extension, '_roi_config.json')
        output_path = os.path.join(args.output_dir, config_name)

        # 檢查是否已存在配置
        if os.path.exists(output_path):
            print(f"[SKIP] 配置已存在: {output_path}")
            print(f"       如需重新設定，請先刪除該檔案")
            skipped_count += 1
            continue

        # 執行互動式配置
        generator = ROIConfigGenerator(video_path, output_path)
        success = generator.run()

        if success:
            success_count += 1
            print(f"[OK] 配置完成: {config_name}")

            # 詢問是否繼續
            if i < len(videos):
                print(f"\n還有 {len(videos) - i} 個影片待處理")
                user_input = input("按 Enter 繼續下一個影片，或輸入 'q' 退出: ")
                if user_input.lower() == 'q':
                    print("\n[INFO] 用戶中止批次處理")
                    break
        else:
            failed_count += 1
            print(f"[ERROR] 配置失敗: {video_name}")

            # 詢問是否繼續
            if i < len(videos):
                user_input = input("按 Enter 繼續下一個影片，或輸入 'q' 退出: ")
                if user_input.lower() == 'q':
                    print("\n[INFO] 用戶中止批次處理")
                    break

    # 輸出摘要
    print(f"\n{'='*60}")
    print(f"批次處理完成！")
    print(f"{'='*60}")
    print(f"  成功: {success_count}/{len(videos)}")
    print(f"  跳過: {skipped_count}/{len(videos)} (配置已存在)")
    print(f"  失敗: {failed_count}/{len(videos)}")
    print(f"  配置檔案已儲存至: {args.output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
