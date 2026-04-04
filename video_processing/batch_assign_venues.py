"""
批次指定場地模板工具
快速為多個影片指定場地模板，無需重複設定 ROI
"""
import os
import json
import argparse
import glob
import cv2


def load_venue_templates(venues_dir):
    """載入所有場地模板"""
    venues = {}
    venue_files = glob.glob(os.path.join(venues_dir, "*.json"))

    for venue_file in venue_files:
        try:
            with open(venue_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
                venue_name = config.get('venue_name', os.path.basename(venue_file).replace('.json', ''))
                venues[venue_name] = config
        except Exception as e:
            print(f"[WARNING] 無法載入場地模板 {venue_file}: {e}")

    return venues


def display_venue_selection(venues):
    """顯示場地選擇介面"""
    venue_list = list(venues.keys())

    print("\n可用的場地模板：")
    for i, venue in enumerate(venue_list, 1):
        print(f"  [{i}] {venue}")
    print(f"  [s] 跳過此影片")
    print(f"  [q] 退出程式")

    return venue_list


def preview_video_with_roi(video_path, venue_config):
    """預覽影片第一幀與 ROI"""
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("[ERROR] 無法讀取影片")
        return None

    # 繪製 ROI
    roi1 = venue_config['score_roi_team1']
    roi2 = venue_config['score_roi_team2']

    x1, y1, w1, h1 = roi1['x'], roi1['y'], roi1['width'], roi1['height']
    cv2.rectangle(frame, (x1, y1), (x1+w1, y1+h1), (0, 255, 0), 2)
    cv2.putText(frame, 'Team1', (x1, y1-10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    x2, y2, w2, h2 = roi2['x'], roi2['y'], roi2['width'], roi2['height']
    cv2.rectangle(frame, (x2, y2), (x2+w2, y2+h2), (0, 0, 255), 2)
    cv2.putText(frame, 'Team2', (x2, y2-10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # 顯示場地名稱（右上角）
    venue_name = venue_config.get('venue_name', 'Unknown')
    venue_text = f"Venue: {venue_name}"

    frame_height, frame_width = frame.shape[:2]
    text_size = cv2.getTextSize(venue_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    text_x = frame_width - text_size[0] - 20
    text_y = 35

    # 繪製背景（黑色）
    cv2.rectangle(frame, (text_x - 10, 10), (frame_width - 10, 50), (0, 0, 0), -1)
    # 繪製文字
    cv2.putText(frame, venue_text, (text_x, text_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return frame


def assign_venue_to_video(video_path, venue_name, venue_config, output_path):
    """為影片指定場地模板"""
    # 讀取影片資訊
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    video_name = os.path.basename(video_path)
    resolution = {"width": frame_width, "height": frame_height}

    # 創建影片配置
    config = {
        "video_name": video_name,
        "venue": venue_name,
        "video_resolution": resolution,
        "score_roi_team1": venue_config['score_roi_team1'],
        "score_roi_team2": venue_config['score_roi_team2'],
        "notes": f"使用場地模板: {venue_name}"
    }

    # 儲存配置
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"[OK] 已指定場地 '{venue_name}': {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="批次指定場地模板工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  python batch_assign_venues.py \\
      --video-dir input_video/original_video \\
      --venues-dir roi_configs/venues \\
      --output-dir roi_configs/videos

說明:
  此工具會依序為每個影片顯示預覽，並讓你選擇對應的場地模板。
  適合快速為大量影片指定已存在的場地配置。
        """
    )
    parser.add_argument("--video-dir", type=str, required=True,
                        help="影片目錄")
    parser.add_argument("--venues-dir", type=str, default="roi_configs/venues",
                        help="場地模板目錄（預設: roi_configs/venues）")
    parser.add_argument("--output-dir", type=str, default="roi_configs/videos",
                        help="輸出目錄（預設: roi_configs/videos）")
    parser.add_argument("--video-extension", type=str, default=".mp4",
                        help="影片檔案副檔名（預設: .mp4）")
    parser.add_argument("--preview", action="store_true",
                        help="顯示預覽視窗（需要 GUI）")
    args = parser.parse_args()

    # 驗證目錄
    if not os.path.exists(args.video_dir):
        print(f"[ERROR] 影片目錄不存在: {args.video_dir}")
        return

    if not os.path.exists(args.venues_dir):
        print(f"[ERROR] 場地模板目錄不存在: {args.venues_dir}")
        print(f"[INFO] 請先使用 roi_config_generator_v2.py 創建場地模板")
        return

    # 載入場地模板
    venues = load_venue_templates(args.venues_dir)

    if not venues:
        print(f"[ERROR] 場地模板目錄中沒有模板")
        print(f"[INFO] 請先使用 roi_config_generator_v2.py 創建場地模板")
        return

    # 列出所有影片
    videos = sorted(glob.glob(os.path.join(args.video_dir, f"*{args.video_extension}")))

    if not videos:
        print(f"[ERROR] 在 {args.video_dir} 中找不到 {args.video_extension} 影片檔案")
        return

    print(f"\n{'='*60}")
    print(f"批次指定場地模板工具")
    print(f"{'='*60}")
    print(f"影片目錄: {args.video_dir}")
    print(f"場地模板目錄: {args.venues_dir}")
    print(f"輸出目錄: {args.output_dir}")
    print(f"找到 {len(videos)} 個影片檔案")
    print(f"可用場地: {', '.join(venues.keys())}")
    print(f"{'='*60}\n")

    success_count = 0
    skipped_count = 0
    venue_list = list(venues.keys())

    for i, video_path in enumerate(videos, 1):
        video_name = os.path.basename(video_path)
        video_basename = video_name.replace(args.video_extension, '')

        print(f"\n{'='*60}")
        print(f"[{i}/{len(videos)}] {video_name}")
        print(f"{'='*60}")

        # 檢查是否已存在配置
        output_path = os.path.join(args.output_dir, f"{video_basename}_roi_config.json")
        if os.path.exists(output_path):
            # 讀取現有配置
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    existing_config = json.load(f)
                    existing_venue = existing_config.get('venue', 'Unknown')
                    print(f"[INFO] 已存在配置: 場地 = {existing_venue}")

                overwrite = input("是否覆蓋？(y/n/s=跳過): ")
                if overwrite.lower() == 's' or overwrite.lower() == 'n':
                    skipped_count += 1
                    continue
            except:
                pass

        # 顯示場地選擇
        venue_list_display = display_venue_selection(venues)

        # 預覽（如果啟用）
        preview_window = None
        if args.preview:
            print("\n[INFO] 載入預覽...")

        while True:
            choice = input(f"\n請選擇場地 (1-{len(venue_list_display)}/s/q): ")

            if choice.lower() == 'q':
                print("\n[INFO] 用戶中止批次處理")
                return

            if choice.lower() == 's':
                print(f"[SKIP] 跳過 {video_name}")
                skipped_count += 1
                break

            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(venue_list_display):
                    selected_venue = venue_list_display[idx]
                    venue_config = venues[selected_venue]

                    # 顯示預覽
                    if args.preview:
                        frame = preview_video_with_roi(video_path, venue_config)
                        if frame is not None:
                            window_name = f"Preview - {video_name}"
                            cv2.imshow(window_name, frame)
                            print(f"\n[INFO] 預覽場地: {selected_venue}")
                            print("      按任意鍵確認，或重新選擇場地")

                            key = cv2.waitKey(0) & 0xFF
                            cv2.destroyAllWindows()

                            if key == ord('q'):
                                print("[INFO] 取消此影片")
                                continue

                    # 確認
                    confirm = input(f"確認使用場地 '{selected_venue}'？(y/n): ")
                    if confirm.lower() == 'y':
                        assign_venue_to_video(video_path, selected_venue, venue_config, output_path)
                        success_count += 1
                        break
                else:
                    print("[ERROR] 無效的選擇")
            else:
                print("[ERROR] 無效的輸入")

    # 輸出摘要
    print(f"\n{'='*60}")
    print(f"批次處理完成！")
    print(f"{'='*60}")
    print(f"  成功: {success_count}/{len(videos)}")
    print(f"  跳過: {skipped_count}/{len(videos)}")
    print(f"  配置檔案已儲存至: {args.output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
