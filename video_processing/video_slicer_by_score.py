# video_slicer_by_score.py (v4 - 移除雙方同時得分邏輯，增加優先序)
import cv2
import os
import argparse
import numpy as np
import csv

# --- 設定 ---
# 這些 ROI 座標需要您根據您的影片進行精確調整
SCORE_ROI_TEAM1 = (280, 29, 59, 51)  # 範例值：(x, y, 寬, 高)
SCORE_ROI_TEAM2 = (287, 92, 59, 50)  # 範例值

def parse_arguments():
    parser = argparse.ArgumentParser(description="根據指定ROI的分數變化來分割影片，並記錄得分方。")
    parser.add_argument("--input", type=str, required=True, help="輸入的長時間影片檔案路徑")
    parser.add_argument("--output_dir", type=str, default="../output_data/video_segments_output_scored", help="儲存分割後影片片段的根目錄")
    parser.add_argument("--min_segment_duration", type=int, default=10, help="有效比賽片段的最小持續時間 (秒)")
    parser.add_argument("--long_segment_threshold", type=int, default=90, help="長片段的閾值 (秒)")
    parser.add_argument("--roi_check_interval", type=float, default=0.5, help="每隔多少秒檢查一次ROI變化 (秒)")
    parser.add_argument("--diff_threshold", type=int, default=600, help="單個ROI影像差異閾值 (SAD)，需要根據實際情況調校。")
    return parser.parse_args()

def finalize_segment_processing(temp_filename, segment_frames_written, fps, min_duration_sec, long_threshold_sec,
                                normal_dir, long_dir, segment_id_counter, trigger_event):
    if not os.path.exists(temp_filename) or segment_frames_written == 0:
        if os.path.exists(temp_filename):
            try: os.remove(temp_filename)
            except OSError: pass
        return None

    duration_sec = segment_frames_written / fps
    final_base_name = f"segment_{segment_id_counter:03d}_{trigger_event}.mp4"

    if duration_sec < min_duration_sec:
        print(f"片段 {final_base_name} ({duration_sec:.1f}s) 過短 (<{min_duration_sec}s)，已刪除。")
        try: os.remove(temp_filename)
        except OSError: pass
        return None

    target_dir = long_dir if duration_sec > long_threshold_sec else normal_dir
    target_filename = os.path.join(target_dir, final_base_name)
    
    print(f"片段 {final_base_name} ({duration_sec:.1f}s) 移動到: {target_filename}")
    try:
        os.rename(temp_filename, target_filename)
    except OSError as e:
        print(f"移動片段時出錯 {temp_filename} -> {target_filename}: {e}")
        return None
        
    return {"filename": final_base_name, "duration": round(duration_sec, 2), "trigger_event": trigger_event}

def get_roi_image(frame, roi_coords, frame_width, frame_height):
    x, y, w, h = roi_coords
    if not (0 <= x < frame_width and 0 <= y < frame_height and x + w <= frame_width and y + h <= frame_height and w > 0 and h > 0):
        return None
    return cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)

def main():
    args = parse_arguments()
    
    output_root_abs = os.path.abspath(args.output_dir)
    normal_segments_dir = os.path.join(output_root_abs, "normal_segments")
    long_segments_dir = os.path.join(output_root_abs, "long_segments")
    temp_dir = os.path.join(output_root_abs, "temp_segments")
    for d in [output_root_abs, normal_segments_dir, long_segments_dir, temp_dir]:
        os.makedirs(d, exist_ok=True)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened(): print(f"錯誤: 無法打開影片 {args.input}"); return
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: print("錯誤: 無法獲取影片的FPS。"); cap.release(); return
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"輸入影片: {args.input}, FPS: {fps:.2f}")
    print(f"輸出到: {output_root_abs}")

    previous_roi1_gray, previous_roi2_gray = None, None
    is_game_active, video_writer, current_temp_video_path = False, None, None
    segment_id_counter, frames_written_this_segment, frame_idx = 0, 0, 0
    roi_check_interval_frames = int(fps * args.roi_check_interval) or 1
    
    first_valid_rois_captured = False
    slicing_log_data = []
    last_trigger_event = "Initial_Start" 

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1

        if frame_idx % roi_check_interval_frames == 0:
            current_roi1_gray = get_roi_image(frame, SCORE_ROI_TEAM1, frame_width, frame_height)
            current_roi2_gray = get_roi_image(frame, SCORE_ROI_TEAM2, frame_width, frame_height)

            if current_roi1_gray is None or current_roi2_gray is None:
                if is_game_active and video_writer:
                    video_writer.write(frame); frames_written_this_segment += 1
                continue

            if not first_valid_rois_captured:
                previous_roi1_gray = current_roi1_gray.copy()
                previous_roi2_gray = current_roi2_gray.copy()
                first_valid_rois_captured = True
                print(f"影格 {frame_idx}: 已捕獲初始ROI狀態，開始錄製第一個片段。")
                
                is_game_active = True
                segment_id_counter += 1
                current_temp_video_path = os.path.join(temp_dir, f"temp_segment_{segment_id_counter:03d}.mp4")
                video_writer = cv2.VideoWriter(current_temp_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
                frames_written_this_segment = 0
                
                if is_game_active and video_writer:
                    video_writer.write(frame); frames_written_this_segment += 1
                continue
            
            sad1 = np.sum(cv2.absdiff(current_roi1_gray, previous_roi1_gray))
            sad2 = np.sum(cv2.absdiff(current_roi2_gray, previous_roi2_gray))
            
            roi1_has_changed = sad1 > args.diff_threshold
            roi2_has_changed = sad2 > args.diff_threshold

            # --- ✨ 核心修改: 移除 "Both_Teams_Score"，建立優先序 ---
            current_trigger_event = None
            if roi1_has_changed:
                current_trigger_event = "Team1_Scores"
            elif roi2_has_changed:
                current_trigger_event = "Team2_Scores"

            if current_trigger_event:
                print(f"影格 {frame_idx}: ROI 偵測到變化 -> {current_trigger_event} (SAD1={int(sad1)}, SAD2={int(sad2)})")
                
                if is_game_active and video_writer is not None:
                    video_writer.release()
                    finalized_info = finalize_segment_processing(
                        current_temp_video_path, frames_written_this_segment, fps,
                        args.min_segment_duration, args.long_segment_threshold,
                        normal_segments_dir, long_segments_dir, segment_id_counter,
                        last_trigger_event
                    )
                    if finalized_info:
                        slicing_log_data.append(finalized_info)
                
                segment_id_counter += 1
                current_temp_video_path = os.path.join(temp_dir, f"temp_segment_{segment_id_counter:03d}.mp4")
                video_writer = cv2.VideoWriter(current_temp_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
                frames_written_this_segment = 0
                last_trigger_event = current_trigger_event
            
            previous_roi1_gray = current_roi1_gray.copy()
            previous_roi2_gray = current_roi2_gray.copy()
        
        if is_game_active and video_writer is not None:
            video_writer.write(frame)
            frames_written_this_segment += 1
            
    if video_writer is not None:
        video_writer.release()
        finalized_info = finalize_segment_processing(
            current_temp_video_path, frames_written_this_segment, fps,
            args.min_segment_duration, args.long_segment_threshold,
            normal_segments_dir, long_segments_dir, segment_id_counter,
            last_trigger_event
        )
        if finalized_info:
            slicing_log_data.append(finalized_info)

    cap.release()
    
    if slicing_log_data:
        csv_output_path = os.path.join(output_root_abs, "slicing_summary.csv")
        print(f"\n--- 正在將 {len(slicing_log_data)} 筆記錄寫入 CSV 日誌 ---")
        try:
            with open(csv_output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=["filename", "duration", "trigger_event"])
                writer.writeheader()
                writer.writerows(slicing_log_data)
            print(f"✅ CSV 日誌已成功儲存至: {csv_output_path}")
        except Exception as e:
            print(f"❌ 寫入 CSV 日誌時發生錯誤: {e}")

    print("\n影片分割處理完成!")

if __name__ == "__main__":
    main()