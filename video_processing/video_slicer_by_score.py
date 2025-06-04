# video_slicer_by_score_comparison.py
import cv2
import os
import argparse
import numpy as np

# --- 設定 ---
# !! 請根據你的影像，精確定義這兩個只包含單個隊伍分數數字的ROI !!
SCORE_ROI_TEAM1 = (280, 29, 59, 51)  # 假設這是隊伍1 (例如 GER, 上方) 分數的ROI (x,y,w,h) - 你需要精確調整！
SCORE_ROI_TEAM2 = (287, 92, 59, 50)  # 假設這是隊伍2 (例如 CAN, 下方) 分數的ROI (x,y,w,h) - 你需要精確調整！
                                    # 注意：Y座標和高度需要仔細量測以確保不重疊且準確

def parse_arguments():
    parser = argparse.ArgumentParser(description="根據兩個獨立分數ROI的影像變化來分割影片。")
    # ... (其他參數與之前版本相同: --input, --output_dir, --min_segment_duration, --long_segment_threshold, --roi_check_interval, --no_change_timeout) ...
    parser.add_argument("--input", type=str, required=True, help="輸入的長時間影片檔案路徑")
    parser.add_argument("--output_dir", type=str, default="video_segments_output_comp_split", help="儲存分割後影片片段的根目錄")
    parser.add_argument("--min_segment_duration", type=int, default=5, help="有效比賽片段的最小持續時間 (秒)")
    parser.add_argument("--long_segment_threshold", type=int, default=45, help="長片段的閾值 (秒)")
    parser.add_argument("--roi_check_interval", type=float, default=2, help="每隔多少秒檢查一次ROI變化 (秒)")
    parser.add_argument("--no_change_timeout", type=int, default=120, help="ROI區域多久無顯著變化則認為非比賽時段 (秒)")
    parser.add_argument("--diff_threshold", type=int, default=20000, help="單個ROI影像差異閾值 (SAD)，超過此值認為有變化。需要針對更小的ROI重新調校！")
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
        print(f"片段 {final_base_name} ({duration_sec:.1f}s) 過短，已刪除。")
        try: os.remove(temp_filename)
        except OSError as e: print(f"刪除過短片段時出錯 {temp_filename}: {e}")
    elif duration_sec > long_threshold_sec:
        target_filename = os.path.join(long_dir, final_base_name)
        print(f"片段 {final_base_name} ({duration_sec:.1f}s) 為長片段，移動到: {target_filename}")
        try: os.rename(temp_filename, target_filename)
        except OSError as e: print(f"移動長片段時出錯 {temp_filename} -> {target_filename}: {e}")
    else:
        target_filename = os.path.join(normal_dir, final_base_name)
        print(f"片段 {final_base_name} ({duration_sec:.1f}s) 為普通片段，移動到: {target_filename}")
        try: os.rename(temp_filename, target_filename)
        except OSError as e: print(f"移動普通片段時出錯 {temp_filename} -> {target_filename}: {e}")


def get_roi_image(frame, roi_coords, frame_width, frame_height):
    """安全地擷取ROI影像，並轉換為灰階"""
    x, y, w, h = roi_coords
    if not (0 <= x < frame_width and 0 <= y < frame_height and x + w <= frame_width and y + h <= frame_height and w > 0 and h > 0):
        # print(f"警告：ROI {roi_coords} 超出影片邊界或尺寸無效。")
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
    print(f"輸入影片: {args.input}, FPS: {fps:.2f}") # ... (其他打印)

    previous_roi1_gray = None
    previous_roi2_gray = None
    
    is_game_active = False
    frames_since_last_change_or_valid_roi = 0 # 修改變數名以反映邏輯
    
    video_writer = None
    current_temp_video_path = None
    segment_id_counter = 0
    frames_written_this_segment = 0

    frame_idx = 0
    roi_check_interval_frames = int(fps * args.roi_check_interval)
    if roi_check_interval_frames == 0: roi_check_interval_frames = 1
    
    no_change_timeout_checks = int(args.no_change_timeout / args.roi_check_interval)
    if no_change_timeout_checks == 0: no_change_timeout_checks = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        if frame_idx % roi_check_interval_frames == 0:
            current_roi1_gray = get_roi_image(frame, SCORE_ROI_TEAM1, frame_width, frame_height)
            current_roi2_gray = get_roi_image(frame, SCORE_ROI_TEAM2, frame_width, frame_height)

            # 確保兩個ROI都成功擷取才能進行比較
            if current_roi1_gray is None or current_roi2_gray is None:
                print(f"影格 {frame_idx}: 一個或多個ROI擷取失敗，跳過此檢查點。")
                # 這種情況下，可以選擇增加 frames_since_last_change_or_valid_roi，或保持不變
                frames_since_last_change_or_valid_roi += 1 
            else:
                roi1_has_changed = False
                if previous_roi1_gray is not None:
                    diff1 = cv2.absdiff(current_roi1_gray, previous_roi1_gray)
                    sad1 = np.sum(diff1)
                    if sad1 > args.diff_threshold:
                        roi1_has_changed = True
                else: # 第一次檢查ROI1
                    roi1_has_changed = True 

                roi2_has_changed = False
                if previous_roi2_gray is not None:
                    diff2 = cv2.absdiff(current_roi2_gray, previous_roi2_gray)
                    sad2 = np.sum(diff2)
                    if sad2 > args.diff_threshold:
                        roi2_has_changed = True
                else: # 第一次檢查ROI2
                    roi2_has_changed = True
                
                # print(f"Frame {frame_idx}: SAD1={sad1 if previous_roi1_gray is not None else 'N/A'}, SAD2={sad2 if previous_roi2_gray is not None else 'N/A'}")


                # 只要任何一個ROI發生變化，就認為整體分數板有變化
                overall_roi_has_changed = roi1_has_changed or roi2_has_changed
                
                # 如果是第一次成功獲取ROI，也視為"變化"以啟動第一個片段
                if previous_roi1_gray is None and previous_roi2_gray is None and \
                   current_roi1_gray is not None and current_roi2_gray is not None:
                   overall_roi_has_changed = True


                previous_roi1_gray = current_roi1_gray.copy() if current_roi1_gray is not None else None
                previous_roi2_gray = current_roi2_gray.copy() if current_roi2_gray is not None else None

                if overall_roi_has_changed:
                    # print(f"Frame {frame_idx}: ROI 偵測到變化 (ROI1 changed: {roi1_has_changed}, ROI2 changed: {roi2_has_changed})")
                    frames_since_last_change_or_valid_roi = 0
                    if not is_game_active: # 從非活躍變為活躍
                        is_game_active = True
                        segment_id_counter += 1
                        current_temp_video_path = os.path.join(temp_dir, f"segment_{segment_id_counter:03d}_temp.mp4")
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        video_writer = cv2.VideoWriter(current_temp_video_path, fourcc, fps, (frame_width, frame_height))
                        frames_written_this_segment = 0
                        print(f"=== 開始新片段 (ID: {segment_id_counter:03d}) 因ROI變化 ===")
                else: # ROI 無顯著變化
                    frames_since_last_change_or_valid_roi += 1
                    # print(f"Frame {frame_idx}: ROI 無顯著變化。無變化檢查次數: {frames_since_last_change_or_valid_roi}")
                    if frames_since_last_change_or_valid_roi >= no_change_timeout_checks:
                        if is_game_active: 
                            print(f"--- 結束片段 (ID: {segment_id_counter:03d}) 因ROI長時間無變化 ({frames_since_last_change_or_valid_roi * args.roi_check_interval:.1f}s) ---")
                            is_game_active = False
                            if video_writer is not None:
                                video_writer.release()
                                finalize_segment_processing(current_temp_video_path, frames_written_this_segment, fps,
                                                            args.min_segment_duration, args.long_segment_threshold,
                                                            normal_segments_dir, long_segments_dir, segment_id_counter)
                                video_writer = None
                                current_temp_video_path = None
                                frames_written_this_segment = 0
        
        if is_game_active and video_writer is not None:
            video_writer.write(frame)
            frames_written_this_segment += 1
        
        # (調試時可取消註解以顯示畫面和ROI框)
        # display_frame_debug = frame.copy()
        # x1,y1,w1,h1 = SCORE_ROI_TEAM1; cv2.rectangle(display_frame_debug, (x1,y1), (x1+w1, y1+h1), (0,255,0), 1)
        # x2,y2,w2,h2 = SCORE_ROI_TEAM2; cv2.rectangle(display_frame_debug, (x2,y2), (x2+w2, y2+h2), (0,0,255), 1)
        # cv2.imshow("Video Slicer by Split ROI Comparison", display_frame_debug)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
            
    if video_writer is not None: # 處理影片末尾的片段
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