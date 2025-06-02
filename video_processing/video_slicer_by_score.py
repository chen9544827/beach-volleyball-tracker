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
    parser.add_argument("--output_dir", type=str, default="output_data/video_segments_output", help="儲存分割後影片片段的根目錄")
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
    final_base_name = f"segment_{segment_id_counter:03d}.avi" #x 或者使用其他你喜歡的影片格式和編碼

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
    temp_dir = os.path.join(args.output_dir, "temp_segments")
    os.makedirs(temp_dir, exist_ok=True)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"錯誤: 無法打開影片 {args.input}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print("錯誤: 無法獲取影片的FPS。")
        cap.release()
        return
        
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"輸入影片: {args.input}") # ... (其他打印訊息) ...

    # --- 狀態變數初始化 ---
    last_valid_scores = {"team1": None, "team2": None} # 上一次成功辨識的有效分數
    
    is_game_active = False # 目前是否處於活躍的比賽片段（即正在錄製）
    frames_ocr_failed_or_static = 0 # OCR失敗或分數未變的連續檢查次數的計時器（以檢查間隔為單位）
    
    video_writer = None
    current_temp_video_path = None
    segment_id_counter = 0
    frames_written_this_segment = 0

    frame_idx = 0 # 總影格計數器
    score_check_interval_frames = int(fps * args.score_check_interval)
    if score_check_interval_frames == 0: score_check_interval_frames = 1
    
    # 將超時時間轉換為 "檢查次數" 的超時，而不是影格數超時
    no_change_timeout_checks = int(args.no_score_change_timeout / args.score_check_interval)
    if no_change_timeout_checks == 0: no_change_timeout_checks = 1

    print(f"FPS: {fps}, 分數檢查間隔(影格): {score_check_interval_frames}, 超時(檢查次數): {no_change_timeout_checks}")


    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # 定期檢查分數
        if frame_idx % score_check_interval_frames == 0:
            current_ocr_scores = {"team1": None, "team2": None} # 本次OCR的直接結果
            
            # --- OCR辨識 (同前) ---
            x1, y1, w1, h1 = SCORE_ROI_TEAM1
            roi1 = frame[y1:y1+h1, x1:x1+w1] if y1+h1 <= frame_height and x1+w1 <= frame_width else None
            if roi1 is not None and roi1.size > 0:
                preprocessed_roi1 = preprocess_image_for_ocr_single_score(roi1)
                if preprocessed_roi1 is not None:
                    ocr_text1 = pytesseract.image_to_string(preprocessed_roi1, config=OCR_CONFIG, lang='eng')
                    current_ocr_scores["team1"] = extract_single_score_from_ocr(ocr_text1)

            x2, y2, w2, h2 = SCORE_ROI_TEAM2
            roi2 = frame[y2:y2+h2, x2:x2+w2] if y2+h2 <= frame_height and x2+w2 <= frame_width else None
            if roi2 is not None and roi2.size > 0:
                preprocessed_roi2 = preprocess_image_for_ocr_single_score(roi2)
                if preprocessed_roi2 is not None:
                    ocr_text2 = pytesseract.image_to_string(preprocessed_roi2, config=OCR_CONFIG, lang='eng')
                    current_ocr_scores["team2"] = extract_single_score_from_ocr(ocr_text2)
            
            print(f"影格 {frame_idx}: OCR -> T1: {current_ocr_scores['team1']}, T2: {current_ocr_scores['team2']} "
                  f"(上次有效: T1:{last_valid_scores['team1']}, T2:{last_valid_scores['team2']})")

            # --- 判斷邏輯 ---
            ocr_successful_this_check = current_ocr_scores["team1"] is not None and \
                                        current_ocr_scores["team2"] is not None
            
            # 決定用於比較的當前分數 (如果OCR失敗，則使用上次的有效分數)
            scores_to_compare = {}
            if ocr_successful_this_check:
                scores_to_compare = current_ocr_scores.copy()
                frames_ocr_failed_or_static = 0 # OCR成功，重置計時器
                # 如果 last_valid_scores 之前是 None (例如影片剛開始OCR就成功)，則初始化它
                if last_valid_scores["team1"] is None and last_valid_scores["team2"] is None:
                    last_valid_scores = current_ocr_scores.copy()

            elif last_valid_scores["team1"] is not None and last_valid_scores["team2"] is not None:
                # OCR失敗，但之前有有效分數，則沿用上次有效分數進行比較，並增加計時器
                scores_to_compare = last_valid_scores.copy() 
                frames_ocr_failed_or_static += 1
                print(f"  -> OCR 本次失敗或不完整，沿用上次有效分數。失敗/靜態次數: {frames_ocr_failed_or_static}")
            else:
                # OCR失敗，且之前也沒有有效分數 (例如影片一開始就連續OCR失敗)
                frames_ocr_failed_or_static += 1
                print(f"  -> OCR 本次失敗且無先前有效分數。失敗/靜態次數: {frames_ocr_failed_or_static}")
                # 在這種情況下，scores_to_compare 可能是空的或不完整的，後續的 scores_have_changed 會是 False

            # 檢查分數是否變化 (只有在 scores_to_compare 有效時才有意義)
            scores_have_changed = False
            if (last_valid_scores["team1"] is not None and last_valid_scores["team2"] is not None and
                scores_to_compare.get("team1") is not None and scores_to_compare.get("team2") is not None):
                if (scores_to_compare["team1"] != last_valid_scores["team1"] or \
                    scores_to_compare["team2"] != last_valid_scores["team2"]):
                    scores_have_changed = True
            
            # 如果分數發生變化
            if scores_have_changed:
                print(f"  -> 分數發生實質變化。")
                frames_ocr_failed_or_static = 0 # 重置計時器
                
                if is_game_active and video_writer is not None: # 如果之前正在錄製，結束上一個片段
                    print(f"--- 因分數變化，結束片段 (ID: {segment_id_counter:03d}) ---")
                    video_writer.release()
                    finalize_segment_processing(current_temp_video_path, frames_written_this_segment, fps,
                                                args.min_segment_duration, args.long_segment_threshold,
                                                normal_segments_dir, long_segments_dir, segment_id_counter)
                    video_writer = None # 準備開始新片段

                # 開始一個新片段
                is_game_active = True
                segment_id_counter += 1
                current_temp_video_path = os.path.join(temp_dir, f"segment_{segment_id_counter:03d}_temp.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(current_temp_video_path, fourcc, fps, (frame_width, frame_height))
                frames_written_this_segment = 0 # 為新片段重置計數
                print(f"=== 因分數變化，開始新片段 (ID: {segment_id_counter:03d}) 寫入到: {current_temp_video_path} ===")
                
                last_valid_scores = scores_to_compare.copy() # 更新上次有效分數

            # 如果分數未變，但OCR是成功的 (說明是有效的比賽進行中但未得分)
            elif ocr_successful_this_check and not scores_have_changed:
                 # 如果之前不是活躍的 (例如影片開始後第一次成功讀到分數但與初始None不同)，也應該開始片段
                if not is_game_active:
                    is_game_active = True
                    segment_id_counter += 1
                    current_temp_video_path = os.path.join(temp_dir, f"segment_{segment_id_counter:03d}_temp.mp4")
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(current_temp_video_path, fourcc, fps, (frame_width, frame_height))
                    frames_written_this_segment = 0
                    print(f"=== 首次有效分數/從非活躍恢復，開始新片段 (ID: {segment_id_counter:03d}) ===")
                
                # 雖然分數未變，但OCR成功了，所以重置 "失敗/靜態" 計時器
                frames_ocr_failed_or_static = 0 
                last_valid_scores = scores_to_compare.copy() # 更新為當前成功讀取的分數
                print(f"  -> 分數有效且未變，比賽進行中。")

            # 如果OCR失敗或分數未變，增加計時器
            # (這部分邏輯已在前面 scores_to_compare 的賦值部分處理了 frames_ocr_failed_or_static 的增加)

            # 檢查是否因長時間無變化/OCR失敗而超時
            if frames_ocr_failed_or_static >= no_change_timeout_checks:
                if is_game_active: # 如果之前是活躍的，現在結束它
                    print(f"--- 結束片段 (ID: {segment_id_counter:03d}) 因OCR持續失敗或分數長時間無變化 ({frames_ocr_failed_or_static * args.score_check_interval:.1f}s) ---")
                    is_game_active = False # 標記為非活躍
                    if video_writer is not None:
                        video_writer.release()
                        finalize_segment_processing(current_temp_video_path, frames_written_this_segment, fps,
                                                    args.min_segment_duration, args.long_segment_threshold,
                                                    normal_segments_dir, long_segments_dir, segment_id_counter)
                        video_writer = None
                        current_temp_video_path = None
                        frames_written_this_segment = 0
                # 即使之前不是 is_game_active，也重置計時器，避免其無限增長
                # frames_ocr_failed_or_static = 0 # 或者讓它保持，直到下一次成功OCR或分數變化

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
    print("\n影片分割處理完成!") # ... (其他打印訊息) ...

if __name__ == "__main__":
    main()
