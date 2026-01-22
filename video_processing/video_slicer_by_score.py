# video_processing/video_slicer_by_score.py (v5 - 支援外部配置檔)
import cv2
import os
import argparse
import numpy as np
import csv
import json

# --- 預設 ROI 設定 (Fallback) ---
# 當未提供配置檔時使用這些預設值
DEFAULT_SCORE_ROI_TEAM1 = (280, 29, 59, 51)  # 隊伍1 (例如:上方/左方) 的分數區域 (x, y, w, h)
DEFAULT_SCORE_ROI_TEAM2 = (287, 92, 59, 50)  # 隊伍2 (例如:下方/右方) 的分數區域 (x, y, w, h)

def load_scoreboard_config(config_path):
    """從 JSON 檔案載入記分板配置"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        team1 = config.get('score_roi_team1', {})
        team2 = config.get('score_roi_team2', {})
        
        roi_team1 = (team1['x'], team1['y'], team1['w'], team1['h'])
        roi_team2 = (team2['x'], team2['y'], team2['w'], team2['h'])
        
        print(f"✓ 已從配置檔載入 ROI 座標: {config_path}")
        return roi_team1, roi_team2
    except Exception as e:
        print(f"⚠️  載入配置檔失敗: {e}")
        print(f"   使用預設 ROI 座標")
        return DEFAULT_SCORE_ROI_TEAM1, DEFAULT_SCORE_ROI_TEAM2

def parse_arguments():
    parser = argparse.ArgumentParser(description="根據兩個獨立分數ROI的影像變化來分割影片，並判斷得分方。")
    parser.add_argument("--input", type=str, required=True, help="輸入的長時間影片檔案路徑")
    parser.add_argument("--output_dir", type=str, default="output_data/video_segments_with_score", help="儲存分割後影片片段與報告的根目錄")
    parser.add_argument("--scoreboard_config", type=str, help="記分板配置檔路徑 (scoreboard_config.json)")
    parser.add_argument("--min_segment_duration", type=int, default=10, help="有效比賽片段的最小持續時間 (秒)")
    parser.add_argument("--long_segment_threshold", type=int, default=90, help="長片段的閾值 (秒)")
    parser.add_argument("--roi_check_interval", type=float, default=0.5, help="每隔多少秒檢查一次ROI變化 (秒)")
    parser.add_argument("--diff_threshold", type=int, default=9000, 
                        help="單個ROI影像差異閾值 (SAD)。這是觸發分數變化的最低門檻，建議使用測試工具來決定此數值。")
    return parser.parse_args()

def get_roi_image(frame, roi_coords, frame_width, frame_height):
    x, y, w, h = roi_coords
    if not (0 <= x < frame_width and 0 <= y < frame_height and x + w <= frame_width and y + h <= frame_height and w > 0 and h > 0):
        return None
    roi_img = frame[y:y+h, x:x+w]
    return cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)

def finalize_segment_and_log_score(temp_filename, segment_frames_written, fps, min_duration_sec, long_threshold_sec,
                                     normal_dir, long_dir, segment_id_counter, scoring_team):
    if not os.path.exists(temp_filename) or segment_frames_written == 0:
        if os.path.exists(temp_filename):
            try: os.remove(temp_filename)
            except OSError as e: print(f"刪除空臨時檔 {temp_filename} 時出錯: {e}")
        return None

    duration_sec = segment_frames_written / fps
    final_base_name = f"segment_{segment_id_counter:03d}_{scoring_team}.mp4"

    segment_data = {
        "Segment_Name": final_base_name,
        "Duration_Seconds": round(duration_sec, 2),
        "Scoring_Team": scoring_team,
        "Status": "",
        "Final_Path": ""
    }

    if duration_sec < min_duration_sec:
        print(f"片段 {final_base_name} ({duration_sec:.1f}s) 過短 (<{min_duration_sec}s)，已刪除。")
        try: os.remove(temp_filename)
        except OSError as e: print(f"刪除過短片段時出錯 {temp_filename}: {e}")
        segment_data["Status"] = "Deleted (Too Short)"
        return segment_data
    
    elif duration_sec > long_threshold_sec:
        target_filename = os.path.join(long_dir, final_base_name)
        segment_data["Status"] = "Long Segment"
        segment_data["Final_Path"] = target_filename
        print(f"片段 {final_base_name} ({duration_sec:.1f}s) 為長片段 (>{long_threshold_sec}s)，移動到: {target_filename}")
        try: os.rename(temp_filename, target_filename)
        except OSError as e: print(f"移動長片段時出錯 {temp_filename} -> {target_filename}: {e}")
    
    else:
        target_filename = os.path.join(normal_dir, final_base_name)
        segment_data["Status"] = "Normal"
        segment_data["Final_Path"] = target_filename
        print(f"片段 {final_base_name} ({duration_sec:.1f}s) 為普通片段，移動到: {target_filename}")
        try: os.rename(temp_filename, target_filename)
        except OSError as e: print(f"移動普通片段時出錯 {temp_filename} -> {target_filename}: {e}")

    return segment_data

def write_summary_csv(summary_data, output_dir):
    if not summary_data:
        print("沒有可供寫入 CSV 的分析數據。")
        return
        
    csv_path = os.path.join(output_dir, "slicing_summary.csv")
    headers = ["Segment_Name", "Duration_Seconds", "Scoring_Team", "Status", "Final_Path"]
    
    try:
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(summary_data)
        print(f"\n✅ 分割摘要報告已成功儲存至: {os.path.abspath(csv_path)}")
    except IOError as e:
        print(f"\n❌ 寫入 CSV 報告失敗: {e}")

def main():
    args = parse_arguments()
    
    # 載入 ROI 配置
    if args.scoreboard_config and os.path.exists(args.scoreboard_config):
        SCORE_ROI_TEAM1, SCORE_ROI_TEAM2 = load_scoreboard_config(args.scoreboard_config)
        roi_source = f"配置檔: {args.scoreboard_config}"
    else:
        SCORE_ROI_TEAM1 = DEFAULT_SCORE_ROI_TEAM1
        SCORE_ROI_TEAM2 = DEFAULT_SCORE_ROI_TEAM2
        roi_source = "預設值 (硬編碼)"
        if args.scoreboard_config:
            print(f"⚠️  找不到配置檔: {args.scoreboard_config}")
    
    output_root_abs = os.path.abspath(args.output_dir)
    os.makedirs(output_root_abs, exist_ok=True)
    normal_segments_dir = os.path.join(output_root_abs, "normal_segments")
    long_segments_dir = os.path.join(output_root_abs, "long_segments")
    os.makedirs(normal_segments_dir, exist_ok=True)
    os.makedirs(long_segments_dir, exist_ok=True)
    temp_dir = os.path.join(output_root_abs, "temp")
    os.makedirs(temp_dir, exist_ok=True)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened(): print(f"錯誤: 無法打開影片 {args.input}"); return
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: print("錯誤: 無法獲取影片的FPS。"); cap.release(); return
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    ret, first_frame = cap.read()
    if not ret: print("錯誤：無法讀取影片的第一幀。"); cap.release(); return

    x1, y1, w1, h1 = SCORE_ROI_TEAM1
    cv2.rectangle(first_frame, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2)
    cv2.putText(first_frame, 'Team1 ROI', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    x2, y2, w2, h2 = SCORE_ROI_TEAM2
    cv2.rectangle(first_frame, (x2, y2), (x2 + w2, y2 + h2), (0, 0, 255), 2)
    cv2.putText(first_frame, 'Team2 ROI', (x2, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # 顯示 ROI 來源
    cv2.putText(first_frame, f'ROI Source: {roi_source}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    print("\n--- ROI 預覽 ---")
    print(f"ROI 來源: {roi_source}")
    print(f"Team1: (x={x1}, y={y1}, w={w1}, h={h1})")
    print(f"Team2: (x={x2}, y={y2}, w={w2}, h={h2})")
    print("請檢查 Team1 (綠色) 與 Team2 (紅色) 的框是否正確。")
    print("確認後，關閉圖片視窗即可繼續執行...")
    cv2.imshow('ROI Preview - Press any key to continue', first_frame); cv2.waitKey(0); cv2.destroyAllWindows()
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    print("\n--- 開始處理影片 ---")
    print(f"輸入影片: {args.input}, FPS: {fps:.2f}"); print(f"輸出到: {output_root_abs}")
    print(f"ROI檢查間隔: {args.roi_check_interval}s, 差異閾值 (SAD): {args.diff_threshold}")
    print(f"片段最小時長: {args.min_segment_duration}s, 長片段閾值: {args.long_segment_threshold}s\n")
    
    previous_roi1_gray, previous_roi2_gray = None, None
    is_game_active, video_writer, current_temp_video_path = False, None, None
    segment_id_counter, frames_written_this_segment = 0, 0
    slicing_summary = []
    frame_idx, roi_check_interval_frames = 0, max(1, int(fps * args.roi_check_interval))
    first_valid_rois_captured = False

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1

        if is_game_active and video_writer is not None:
            video_writer.write(frame)
            frames_written_this_segment += 1

        if frame_idx % roi_check_interval_frames == 0:
            current_roi1_gray = get_roi_image(frame, SCORE_ROI_TEAM1, frame_width, frame_height)
            current_roi2_gray = get_roi_image(frame, SCORE_ROI_TEAM2, frame_width, frame_height)

            if current_roi1_gray is None or current_roi2_gray is None: continue

            if not first_valid_rois_captured:
                previous_roi1_gray, previous_roi2_gray = current_roi1_gray, current_roi2_gray
                first_valid_rois_captured = True
                print(f"影格 {frame_idx}: 已捕獲初始ROI狀態，開始錄製第一個片段。")
                is_game_active = True
                segment_id_counter = 1
                current_temp_video_path = os.path.join(temp_dir, f"segment_{segment_id_counter:03d}_temp.mp4")
                video_writer = cv2.VideoWriter(current_temp_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
                frames_written_this_segment = 0
                continue

            sad1 = np.sum(cv2.absdiff(current_roi1_gray, previous_roi1_gray))
            sad2 = np.sum(cv2.absdiff(current_roi2_gray, previous_roi2_gray))
            
            # ✨ --- 核心修改點 --- ✨
            # 只有當任一邊的 SAD 超過閾值時，才進行後續判斷與輸出
            if sad1 > args.diff_threshold or sad2 > args.diff_threshold:
                scoring_team = "Team1" if sad1 > sad2 else "Team2"
                
                # 將SAD值與判斷結果合併成一行輸出
                print(f"Frame {frame_idx}: 偵測到變化！SAD1={sad1}, SAD2={sad2} -> 得分方: {scoring_team}")

                if video_writer is not None:
                    video_writer.release()
                    segment_log = finalize_segment_and_log_score(current_temp_video_path, frames_written_this_segment, fps,
                                                                 args.min_segment_duration, args.long_segment_threshold,
                                                                 normal_segments_dir, long_segments_dir, segment_id_counter, scoring_team)
                    if segment_log: slicing_summary.append(segment_log)
                
                segment_id_counter += 1
                current_temp_video_path = os.path.join(temp_dir, f"segment_{segment_id_counter:03d}_temp.mp4")
                video_writer = cv2.VideoWriter(current_temp_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
                frames_written_this_segment = 0
                print(f"--- 開始錄製新片段 (ID: {segment_id_counter:03d}) ---")

            previous_roi1_gray, previous_roi2_gray = current_roi1_gray, current_roi2_gray
            
    if video_writer is not None:
        print(f"--- 影片結束，結束最後一個片段 (ID: {segment_id_counter:03d}) ---")
        video_writer.release()
        segment_log = finalize_segment_and_log_score(current_temp_video_path, frames_written_this_segment, fps,
                                                     args.min_segment_duration, args.long_segment_threshold,
                                                     normal_segments_dir, long_segments_dir, segment_id_counter, "End_Of_Video")
        if segment_log: slicing_summary.append(segment_log)

    cap.release()
    cv2.destroyAllWindows()
    
    write_summary_csv(slicing_summary, output_root_abs)
    print("\n影片分割與得分分析處理完成!")

if __name__ == "__main__":
    main()