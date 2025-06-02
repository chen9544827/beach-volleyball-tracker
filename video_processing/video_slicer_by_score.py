import cv2
import os
import argparse
import pytesseract
import re
import time # 用於調試或增加延遲（如果需要）

# --- OCR 相關設定與函數 ---
# 如果 Tesseract OCR 未加入到系統 PATH，請取消下面這行的註解並修改為你的 Tesseract 安裝路徑
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe' # Windows範例

# !! 請根據你的影像，精確定義這兩個只包含單個分數數字的ROI !!
# 這些可以作為腳本的參數，或者從一個小設定檔讀取，或者像這樣硬編碼（但不推薦用於多個影片）
SCORE_ROI_TEAM1 = (280, 29, 59, 55)  # 範例值: x, y, width, height (隊伍1/上方分數)
SCORE_ROI_TEAM2 = (287, 92, 59, 55) # 範例值: (隊伍2/下方分數)

# 通用的OCR設定，適用於個位數或多位數的單行數字
OCR_CONFIG = r'--psm 7 -c tessedit_char_whitelist=0123456789'

def preprocess_image_for_ocr_single_score(image_roi):
    # (從之前的討論複製過來，確保它能良好運作)
    if image_roi is None or image_roi.size == 0:
        return None
    gray = cv2.cvtColor(image_roi, cv2.COLOR_BGR2GRAY)
    scale_factor = 3.0
    width = int(gray.shape[1] * scale_factor)
    height = int(gray.shape[0] * scale_factor)
    if width == 0 or height == 0: # 避免resize到0尺寸
        return None
    resized = cv2.resize(gray, (width, height), interpolation=cv2.INTER_CUBIC)
    _, thresh_img = cv2.threshold(resized, 190, 255, cv2.THRESH_BINARY_INV) # 根據你的測試調整閾值
    return thresh_img

def extract_single_score_from_ocr(ocr_text):
    # (從之前的討論複製過來)
    if ocr_text is None: return None
    numbers = re.findall(r'\d+', ocr_text)
    if numbers:
        try:
            return int(numbers[0])
        except (ValueError, IndexError):
            return None
    return None
# --- OCR 相關設定與函數結束 ---

def parse_arguments():
    parser = argparse.ArgumentParser(description="根據OCR分數變化分割影片，並將長片段放入特定資料夾。")
    parser.add_argument("--input", type=str, required=True, help="輸入的長時間影片檔案路徑")
    parser.add_argument("--output_dir", type=str, default="video_segments_output", help="儲存分割後影片片段的根目錄")
    parser.add_argument("--min_segment_duration", type=int, default=10, help="有效比賽片段的最小持續時間 (秒)")
    parser.add_argument("--long_segment_threshold", type=int, default=90, help="長片段的閾值 (秒)，超過此時間的片段會放入 'long_segments' 資料夾")
    parser.add_argument("--score_check_interval", type=float, default=1.0, help="每隔多少秒檢查一次分數 (可以是小數)")
    parser.add_argument("--no_score_change_timeout", type=int, default=120, help="分數多久保持不變（或OCR失敗）則認為非比賽時段 (秒)")
    # 可以考慮增加ROI的參數
    # parser.add_argument("--roi1", type=str, help="隊伍1分數ROI 'x,y,w,h'")
    # parser.add_argument("--roi2", type=str, help="隊伍2分數ROI 'x,y,w,h'")
    return parser.parse_args()

def finalize_segment_processing(temp_filename, segment_frames_written, fps, min_duration_sec, long_threshold_sec,
                                normal_dir, long_dir, segment_id_counter):
    if not os.path.exists(temp_filename) or segment_frames_written == 0:
        if os.path.exists(temp_filename): # 如果檔案存在但沒有幀，也刪除
            try:
                os.remove(temp_filename)
            except OSError as e:
                print(f"刪除空臨時檔 {temp_filename} 時出錯: {e}")
        print(f"片段 {segment_id_counter:03d} 的臨時檔無效或無幀，已跳過/刪除。")
        return

    duration_sec = segment_frames_written / fps
    final_base_name = f"segment_{segment_id_counter:03d}.mp4" # 或者使用其他你喜歡的影片格式和編碼

    if duration_sec < min_duration_sec:
        print(f"片段 {final_base_name} ({duration_sec:.1f}s) 過短 (少於 {min_duration_sec}s)，將被刪除。")
        try:
            os.remove(temp_filename)
        except OSError as e:
            print(f"刪除過短片段時出錯 {temp_filename}: {e}")
    elif duration_sec > long_threshold_sec:
        target_filename = os.path.join(long_dir, final_base_name)
        print(f"片段 {final_base_name} ({duration_sec:.1f}s) 為長片段 (超過 {long_threshold_sec}s)，移動到: {target_filename}")
        try:
            os.rename(temp_filename, target_filename)
        except OSError as e:
            print(f"移動長片段時出錯 {temp_filename} -> {target_filename}: {e}")
    else:
        target_filename = os.path.join(normal_dir, final_base_name)
        print(f"片段 {final_base_name} ({duration_sec:.1f}s) 為普通片段，移動到: {target_filename}")
        try:
            os.rename(temp_filename, target_filename)
        except OSError as e:
            print(f"移動普通片段時出錯 {temp_filename} -> {target_filename}: {e}")

def main():
    args = parse_arguments()

    # 創建輸出目錄結構
    os.makedirs(args.output_dir, exist_ok=True)
    normal_segments_dir = os.path.join(args.output_dir, "normal_segments")
    long_segments_dir = os.path.join(args.output_dir, "long_segments")
    os.makedirs(normal_segments_dir, exist_ok=True)
    os.makedirs(long_segments_dir, exist_ok=True)
    
    # 暫存切割片段的臨時目錄 (可選，或者直接在 output_dir 下加 _temp 後綴)
    temp_dir = os.path.join(args.output_dir, "temp_segments")
    os.makedirs(temp_dir, exist_ok=True)


    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"錯誤: 無法打開影片 {args.input}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print("錯誤: 無法獲取影片的FPS，請確保影片檔案有效。")
        cap.release()
        return
        
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"輸入影片: {args.input}")
    print(f"FPS: {fps}, 尺寸: {frame_width}x{frame_height}")
    print(f"輸出到: {args.output_dir}")
    print(f"  - 普通片段: {normal_segments_dir}")
    print(f"  - 長片段 (> {args.long_segment_threshold}s): {long_segments_dir}")
    print(f"片段最小時長: {args.min_segment_duration}s")
    print(f"分數檢查間隔: {args.score_check_interval}s")
    print(f"無分數變化/OCR失敗超時: {args.no_score_change_timeout}s")

    last_scores = {"team1": None, "team2": None}
    
    is_game_active = False
    frames_since_last_change_or_valid_ocr = 0 # 用於追蹤分數不變或OCR失敗的時間
    
    video_writer = None
    current_temp_video_path = None
    segment_id_counter = 0
    frames_written_this_segment = 0

    frame_idx = 0
    score_check_interval_frames = int(fps * args.score_check_interval)
    if score_check_interval_frames == 0: score_check_interval_frames = 1 # 至少每幀都檢查（如果interval設太小）
    
    no_change_timeout_frames = int(fps * args.no_score_change_timeout)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_idx += 1

        # 定期檢查分數
        if frame_idx % score_check_interval_frames == 0:
            current_scores_this_check = {"team1": None, "team2": None}
            
            # Team 1 Score OCR
            x1, y1, w1, h1 = SCORE_ROI_TEAM1
            roi1 = frame[y1:y1+h1, x1:x1+w1] if y1+h1 <= frame_height and x1+w1 <= frame_width else None
            if roi1 is not None and roi1.size > 0:
                preprocessed_roi1 = preprocess_image_for_ocr_single_score(roi1)
                if preprocessed_roi1 is not None:
                    ocr_text1 = pytesseract.image_to_string(preprocessed_roi1, config=OCR_CONFIG, lang='eng')
                    current_scores_this_check["team1"] = extract_single_score_from_ocr(ocr_text1)

            # Team 2 Score OCR
            x2, y2, w2, h2 = SCORE_ROI_TEAM2
            roi2 = frame[y2:y2+h2, x2:x2+w2] if y2+h2 <= frame_height and x2+w2 <= frame_width else None
            if roi2 is not None and roi2.size > 0:
                preprocessed_roi2 = preprocess_image_for_ocr_single_score(roi2)
                if preprocessed_roi2 is not None:
                    ocr_text2 = pytesseract.image_to_string(preprocessed_roi2, config=OCR_CONFIG, lang='eng')
                    current_scores_this_check["team2"] = extract_single_score_from_ocr(ocr_text2)
            
            print(f"影格 {frame_idx}: OCR -> T1: {current_scores_this_check['team1']}, T2: {current_scores_this_check['team2']} "
                  f"(上次有效: T1:{last_scores['team1']}, T2:{last_scores['team2']})")

            scores_are_valid_this_check = current_scores_this_check["team1"] is not None and \
                                          current_scores_this_check["team2"] is not None
            
            scores_have_changed = (scores_are_valid_this_check and 
                                   (current_scores_this_check["team1"] != last_scores["team1"] or \
                                    current_scores_this_check["team2"] != last_scores["team2"]))

            if scores_have_changed:
                print(f"  -> 分數發生變化。")
                frames_since_last_change_or_valid_ocr = 0
                last_scores = current_scores_this_check.copy()
                if not is_game_active: # 從非活躍變為活躍
                    is_game_active = True
                    segment_id_counter += 1
                    current_temp_video_path = os.path.join(temp_dir, f"segment_{segment_id_counter:03d}_temp.mp4")
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # 或 'XVID' for .avi
                    video_writer = cv2.VideoWriter(current_temp_video_path, fourcc, fps, (frame_width, frame_height))
                    frames_written_this_segment = 0
                    print(f"=== 開始新片段 (ID: {segment_id_counter:03d}) 寫入到: {current_temp_video_path} ===")
            
            elif scores_are_valid_this_check: # 分數有效，但未變化
                frames_since_last_change_or_valid_ocr += score_check_interval_frames
                print(f"  -> 分數有效但未變。無變化時長: {frames_since_last_change_or_valid_ocr/fps:.1f}s")
            
            else: # OCR 失敗或分數無效
                frames_since_last_change_or_valid_ocr += score_check_interval_frames
                print(f"  -> OCR失敗或分數無效。無變化/失敗時長: {frames_since_last_change_or_valid_ocr/fps:.1f}s")

            # 檢查是否超時 (分數長時間不變或OCR持續失敗)
            if frames_since_last_change_or_valid_ocr >= no_change_timeout_frames:
                if is_game_active: # 如果之前是活躍的，現在結束它
                    print(f"--- 結束片段 (ID: {segment_id_counter:03d}) 因超時 ---")
                    is_game_active = False
                    if video_writer is not None:
                        video_writer.release()
                        finalize_segment_processing(current_temp_video_path, frames_written_this_segment, fps,
                                                    args.min_segment_duration, args.long_segment_threshold,
                                                    normal_segments_dir, long_segments_dir, segment_id_counter)
                        video_writer = None
                        current_temp_video_path = None
                        frames_written_this_segment = 0
        
        # 如果當前是活躍的比賽片段，則寫入影格
        if is_game_active and video_writer is not None:
            video_writer.write(frame)
            frames_written_this_segment += 1

        # (調試時可取消註解以顯示畫面)
        # cv2.imshow("Video Slicer", frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    # --- 迴圈結束後 ---
    if video_writer is not None: # 處理影片末尾未結束的片段
        print(f"--- 影片結束，結束最後片段 (ID: {segment_id_counter:03d}) ---")
        video_writer.release()
        finalize_segment_processing(current_temp_video_path, frames_written_this_segment, fps,
                                    args.min_segment_duration, args.long_segment_threshold,
                                    normal_segments_dir, long_segments_dir, segment_id_counter)

    cap.release()
    cv2.destroyAllWindows()
    print("\n影片分割處理完成!")
    print(f"普通片段保存在: {normal_segments_dir}")
    print(f"長片段 (> {args.long_segment_threshold}s) 保存在: {long_segments_dir}")
    
    # (可選) 清理空的temp目錄
    try:
        if not os.listdir(temp_dir) : # 如果temp目錄是空的
            os.rmdir(temp_dir)
    except OSError:
        pass


if __name__ == "__main__":
    main()