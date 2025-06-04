# video_slicer_by_score_comparison.py (或您的新檔名)
import cv2
import os
import argparse
import numpy as np

# --- 設定 ---
SCORE_ROI_TEAM1 = (280, 29, 59, 51)  # 範例值
SCORE_ROI_TEAM2 = (287, 92, 59, 50)  # 範例值

def parse_arguments():
    parser = argparse.ArgumentParser(description="根據兩個獨立分數ROI的影像變化來分割影片。")
    parser.add_argument("--input", type=str, required=True, help="輸入的長時間影片檔案路徑")
    parser.add_argument("--output_dir", type=str, default="../output_data/video_segments_output_comp_split_v2", help="儲存分割後影片片段的根目錄")
    parser.add_argument("--min_segment_duration", type=int, default=10, help="有效比賽片段的最小持續時間 (秒)")
    parser.add_argument("--long_segment_threshold", type=int, default=90, help="長片段的閾值 (秒)")
    parser.add_argument("--roi_check_interval", type=float, default=0.5, help="每隔多少秒檢查一次ROI變化 (秒)")
    # 移除了 --no_change_timeout 參數，因為現在的邏輯是基於變化來分割
    parser.add_argument("--diff_threshold", type=int, default=600, 
                        help="單個ROI影像差異閾值 (SAD)。需要調校！")
    return parser.parse_args()

def finalize_segment_processing(temp_filename, segment_frames_written, fps, min_duration_sec, long_threshold_sec,
                                normal_dir, long_dir, segment_id_counter):
    # (此函數與之前版本完全相同)
    if not os.path.exists(temp_filename) or segment_frames_written == 0:
        if os.path.exists(temp_filename):
            try: os.remove(temp_filename)
            except OSError as e: print(f"刪除空臨時檔 {temp_filename} 時出錯: {e}")
        return

    duration_sec = segment_frames_written / fps
    final_base_name = f"segment_{segment_id_counter:03d}.mp4"

    if duration_sec < min_duration_sec:
        print(f"片段 {final_base_name} ({duration_sec:.1f}s) 過短 (<{min_duration_sec}s)，已刪除。")
        try: os.remove(temp_filename)
        except OSError as e: print(f"刪除過短片段時出錯 {temp_filename}: {e}")
    elif duration_sec > long_threshold_sec:
        target_filename = os.path.join(long_dir, final_base_name)
        print(f"片段 {final_base_name} ({duration_sec:.1f}s) 為長片段 (>{long_threshold_sec}s)，移動到: {target_filename}")
        try: os.rename(temp_filename, target_filename)
        except OSError as e: print(f"移動長片段時出錯 {temp_filename} -> {target_filename}: {e}")
    else: # 普通片段
        target_filename = os.path.join(normal_dir, final_base_name)
        print(f"片段 {final_base_name} ({duration_sec:.1f}s) 為普通片段，移動到: {target_filename}")
        try: os.rename(temp_filename, target_filename)
        except OSError as e: print(f"移動普通片段時出錯 {temp_filename} -> {target_filename}: {e}")

def get_roi_image(frame, roi_coords, frame_width, frame_height):
    # (此函數與之前版本相同)
    x, y, w, h = roi_coords
    if not (0 <= x < frame_width and 0 <= y < frame_height and x + w <= frame_width and y + h <= frame_height and w > 0 and h > 0):
        return None
    roi_img = frame[y:y+h, x:x+w]
    return cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)

def main():
    args = parse_arguments()
    # ... (輸出目錄創建邏輯與之前相同) ...
    output_root_abs = os.path.abspath(args.output_dir)
    os.makedirs(output_root_abs, exist_ok=True)
    normal_segments_dir = os.path.join(output_root_abs, "normal_segments")
    long_segments_dir = os.path.join(output_root_abs, "long_segments")
    os.makedirs(normal_segments_dir, exist_ok=True)
    os.makedirs(long_segments_dir, exist_ok=True)
    temp_dir = os.path.join(output_root_abs, "temp_segments")
    os.makedirs(temp_dir, exist_ok=True)

    cap = cv2.VideoCapture(args.input)
    # ... (影片打開和FPS獲取邏輯與之前相同) ...
    if not cap.isOpened(): print(f"錯誤: 無法打開影片 {args.input}"); return
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: print("錯誤: 無法獲取影片的FPS。"); cap.release(); return
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"輸入影片: {args.input}, FPS: {fps:.2f}")
    print(f"輸出到: {output_root_abs}")
    print(f"ROI檢查間隔: {args.roi_check_interval}s, 差異閾值 (SAD): {args.diff_threshold}")
    print(f"片段最小時長: {args.min_segment_duration}s, 長片段閾值: {args.long_segment_threshold}s")


    previous_roi1_gray = None
    previous_roi2_gray = None
    
    # is_game_active 仍然用來標記是否正在錄製一個片段
    is_game_active = False 
    
    video_writer = None
    current_temp_video_path = None
    segment_id_counter = 0
    frames_written_this_segment = 0

    frame_idx = 0
    roi_check_interval_frames = int(fps * args.roi_check_interval)
    if roi_check_interval_frames == 0: roi_check_interval_frames = 1
    
    # 移除了 no_change_timeout_checks 和 frames_since_last_change_or_valid_roi
    # 因為現在的邏輯是：只要有變化就切，沒有「超時結束片段」的概念了

    # ROI 座標 (確保它們在影片範圍內)
    rois = {"team1": SCORE_ROI_TEAM1, "team2": SCORE_ROI_TEAM2}
    for team_name, (rx, ry, rw, rh) in rois.items():
        if not (0 <= rx < frame_width and 0 <= ry < frame_height and \
                rx + rw <= frame_width and ry + rh <= frame_height and rw > 0 and rh > 0):
            print(f"錯誤：{team_name} 的 ROI {rois[team_name]} 超出影片邊界或尺寸無效。")
            cap.release()
            return

    first_valid_rois_captured = False # 標記是否已捕獲到第一組有效的ROI作為基線

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # 如果正在錄製一個片段，則寫入影格
        # 這個寫入判斷移到ROI檢查之後，確保在開始新片段時能包含觸發變化的那一組影格
        # if is_game_active and video_writer is not None:
        #     video_writer.write(frame)
        #     frames_written_this_segment += 1

        if frame_idx % roi_check_interval_frames == 0:
            current_roi1_gray = get_roi_image(frame, SCORE_ROI_TEAM1, frame_width, frame_height)
            current_roi2_gray = get_roi_image(frame, SCORE_ROI_TEAM2, frame_width, frame_height)

            if current_roi1_gray is None or current_roi2_gray is None:
                print(f"影格 {frame_idx}: 一個或多個ROI擷取失敗，跳過此檢查點。")
                # 如果正在錄製，可以考慮是否因為ROI擷取失敗而結束片段，或暫時忽略
                # 目前邏輯：如果正在錄製，會繼續錄製，直到下一次成功的ROI比對
                if is_game_active and video_writer is not None: # 即使ROI失敗也寫入，確保連續性
                    video_writer.write(frame)
                    frames_written_this_segment += 1
                continue # 跳過本次變化的判斷

            if not first_valid_rois_captured: # 捕獲第一組有效的ROI作為比較的基線
                previous_roi1_gray = current_roi1_gray.copy()
                previous_roi2_gray = current_roi2_gray.copy()
                first_valid_rois_captured = True
                print(f"影格 {frame_idx}: 已捕獲初始ROI狀態。")
                if is_game_active and video_writer is not None: # 即使ROI失敗也寫入
                    video_writer.write(frame)
                    frames_written_this_segment += 1
                continue


            roi1_has_changed = False
            diff1 = cv2.absdiff(current_roi1_gray, previous_roi1_gray)
            sad1 = np.sum(diff1)
            if sad1 > args.diff_threshold:
                roi1_has_changed = True

            roi2_has_changed = False
            diff2 = cv2.absdiff(current_roi2_gray, previous_roi2_gray)
            sad2 = np.sum(diff2)
            if sad2 > args.diff_threshold:
                roi2_has_changed = True
            
            # print(f"Frame {frame_idx}: SAD1={sad1}, SAD2={sad2}")

            overall_roi_has_changed = roi1_has_changed or roi2_has_changed
            
            if overall_roi_has_changed:
                print(f"影格 {frame_idx}: ROI 偵測到變化 (ROI1 changed: {roi1_has_changed}, ROI2 changed: {roi2_has_changed})")
                
                # 如果之前正在錄製一個片段，先結束並儲存它
                if is_game_active and video_writer is not None:
                    print(f"--- 因ROI變化，結束片段 (ID: {segment_id_counter:03d}) ---")
                    video_writer.release()
                    finalize_segment_processing(current_temp_video_path, frames_written_this_segment, fps,
                                                args.min_segment_duration, args.long_segment_threshold,
                                                normal_segments_dir, long_segments_dir, segment_id_counter)
                    # is_game_active = False # 馬上要開始新的，所以不用設為False
                
                # 開始一個新的片段
                is_game_active = True # 確保是 active
                segment_id_counter += 1
                current_temp_video_path = os.path.join(temp_dir, f"segment_{segment_id_counter:03d}_temp.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(current_temp_video_path, fourcc, fps, (frame_width, frame_height))
                frames_written_this_segment = 0 # 為新片段重置計數
                print(f"=== 因ROI變化，開始新片段 (ID: {segment_id_counter:03d}) 寫入到: {current_temp_video_path} ===")
            
            # 更新 previous_roi_gray 以便下次比較
            previous_roi1_gray = current_roi1_gray.copy()
            previous_roi2_gray = current_roi2_gray.copy()
        
        # 將影格寫入邏輯移到這裡，確保在ROI檢查和片段開始/結束邏輯之後執行
        # 這樣可以包含觸發變化的那一組影格（從ROI檢查點開始的整個interval的影格）
        if is_game_active and video_writer is not None:
            video_writer.write(frame)
            frames_written_this_segment += 1
            
    # --- 迴圈結束後 ---
    if video_writer is not None: # 處理影片末尾未結束的片段
        print(f"--- 影片結束，結束最後片段 (ID: {segment_id_counter:03d}) ---")
        video_writer.release()
        finalize_segment_processing(current_temp_video_path, frames_written_this_segment, fps,
                                    args.min_segment_duration, args.long_segment_threshold,
                                    normal_segments_dir, long_segments_dir, segment_id_counter)

    cap.release()
    cv2.destroyAllWindows()
    print("\n影片分割處理完成!") # ... (其他打印)

if __name__ == "__main__":
    main()