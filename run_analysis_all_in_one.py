# -*- coding: utf-8 -*-
# run_analysis_all_in_one.py (最終通用版 v4 - 增加發球區域判斷)
# 核心偵測邏輯採用您的版本，並結合回溯尋人法 + 精確距離計算 + 發球區域判斷。

import os
import subprocess
import argparse
import sys
import json
import numpy as np
import cv2
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import io

# --- (此處省略了與上一版相同的 tqdm 程式碼) ---
try:
    from tqdm import tqdm
except ImportError:
    print("模組 'tqdm' 未找到，正在嘗試自動安裝...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"])
        from tqdm import tqdm; print("tqdm 安裝成功！")
    except Exception as e:
        print(f"自動安裝 tqdm 失敗，請手動執行 'pip install tqdm'。錯誤: {e}"); sys.exit(1)

# ==============================================================================
#  輔助函數 (Helper Functions)
# ==============================================================================
def get_ball_center_from_data(frame_data):
    if frame_data and frame_data.get('ball_detections'):
        valid_balls = [b for b in frame_data['ball_detections'] if not b.get('is_in_background_zone', False)]
        if not valid_balls: return None
        best_ball = max(valid_balls, key=lambda b: b.get('confidence', 0))
        box = best_ball['box_coords']
        return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])

def get_distance_to_player_box(point, player_box):
    px, py = point
    x1, y1, x2, y2 = player_box
    closest_x = max(x1, min(px, x2))
    closest_y = max(y1, min(py, y2))
    distance = np.sqrt((px - closest_x)**2 + (py - closest_y)**2)
    return distance

def get_player_center(player_data):
    if not player_data or 'center_point' not in player_data: return None
    return np.array(player_data['center_point'])

# ✨ NEW HELPER FUNCTION: Determines the serving zone (A, B, C)
def get_serving_zone(server_center, court_polygon):
    """
    根據發球員的中心點和球場的四個角落，判斷發球區域 (A, B, C)。
    - 假設:
        - court_polygon 的點順序為 [左上, 左下, 右下, 右上]。
        - 發球員總是在Y座標較大的「下半場」發球。
    - 傳回: 'A', 'B', 'C', 或 'Unknown'
    """
    if server_center is None or len(court_polygon) != 4:
        return 'Unknown'

    # 我們關心的是下半場的底線，也就是 P1 和 P2
    p1 = np.array(court_polygon[1])  # 左下角
    p2 = np.array(court_polygon[2])  # 右下角

    # 計算底線上 A/B 和 B/C 區的分割點
    zone_ab_divider = p1 + (p2 - p1) / 3
    zone_bc_divider = p1 + 2 * (p2 - p1) / 3

    server_x = server_center[0]

    # 為了應對非水平的底線，我們使用分割點的X座標作為判斷依據
    if server_x < zone_ab_divider[0]:
        return 'A'
    elif server_x < zone_bc_divider[0]:
        return 'B'
    else:
        return 'C'

# ==============================================================================
#  (analyze_serve_events 和 process_single_video 函數與上一版相同，此處省略)
# ==============================================================================
def analyze_serve_events(all_frames_data, config, log_prefix=""):
    # This function is identical to the previous version.
    hit_v_thresh = config.get("hit_v", 40.0)
    max_plausible_speed = config.get("max_speed", 200.0)
    toss_initial_vy_thresh = config.get("toss_vy", 8.0)
    frames_to_validate_toss = config.get("frames_to_validate", 8)
    min_upward_confirms = config.get("min_upward_confirms", 3)
    vertical_ratio_thresh = config.get("vertical_ratio", 1.5)
    hit_horizontal_ratio_thresh = config.get("hit_h_ratio", 2.5)
    max_frames_to_apex = config.get("max_frames_to_apex", 75)
    max_frames_to_hit = config.get("max_frames_to_hit", 40)
    max_lost_frames_apex = config.get("max_lost_frames_apex", 50)
    serve_events, state, event_candidate = [], "SEARCHING_TOSS", {}
    print(f"\n{log_prefix}[分析階段] 正在使用您提供的邏輯偵測擊球事件...")
    for i in range(1, len(all_frames_data)):
        prev_ball_pos = get_ball_center_from_data(all_frames_data[i-1])
        curr_ball_pos = get_ball_center_from_data(all_frames_data[i])
        if prev_ball_pos is None or curr_ball_pos is None:
            if state in ["AWAITING_APEX", "AWAITING_HIT", "CONFIRMING_TOSS"]: event_candidate['lost_frames_count'] = event_candidate.get('lost_frames_count', 0) + 1
            if state == "AWAITING_APEX" and event_candidate.get('lost_frames_count', 0) > max_lost_frames_apex: state = "SEARCHING_TOSS"
            elif state == "AWAITING_HIT" and event_candidate.get('lost_frames_count', 0) > 15: state = "SEARCHING_TOSS"
            elif state == "CONFIRMING_TOSS" and event_candidate.get('lost_frames_count', 0) > 5: state = "SEARCHING_TOSS"
            continue
        speed = np.linalg.norm(curr_ball_pos - prev_ball_pos)
        if speed > max_plausible_speed: continue
        if 'lost_frames_count' in event_candidate: event_candidate['lost_frames_count'] = 0
        if state == "SEARCHING_TOSS":
            vy = prev_ball_pos[1] - curr_ball_pos[1]; vx = curr_ball_pos[0] - prev_ball_pos[0]
            if vy > toss_initial_vy_thresh and (abs(vy) / (abs(vx) + 1e-6)) > vertical_ratio_thresh:
                state = "CONFIRMING_TOSS"; event_candidate = {'confirm_start_frame': i, 'upward_frames_count': 1, 'lost_frames_count': 0, 'last_pos': curr_ball_pos}
        elif state == "CONFIRMING_TOSS":
            if (i - event_candidate['confirm_start_frame']) > frames_to_validate_toss: state = "SEARCHING_TOSS"; continue
            vy = event_candidate['last_pos'][1] - curr_ball_pos[1]
            if vy > 0: event_candidate['upward_frames_count'] += 1
            event_candidate['last_pos'] = curr_ball_pos
            if event_candidate['upward_frames_count'] >= min_upward_confirms:
                state = "AWAITING_APEX"; event_candidate['toss_start_frame'] = event_candidate['confirm_start_frame']; event_candidate['lost_frames_count'] = 0
        elif state == "AWAITING_APEX":
            if (i - event_candidate['toss_start_frame']) > max_frames_to_apex: state = "SEARCHING_TOSS"; continue
            vy = prev_ball_pos[1] - curr_ball_pos[1]
            if vy < -1: state = "AWAITING_HIT"; event_candidate['apex_frame'] = i; event_candidate['lost_frames_count'] = 0
        elif state == "AWAITING_HIT":
            if (i - event_candidate['apex_frame']) > max_frames_to_hit: state = "SEARCHING_TOSS"; continue
            if speed > hit_v_thresh:
                hit_vx = curr_ball_pos[0] - prev_ball_pos[0]; hit_vy = curr_ball_pos[1] - prev_ball_pos[1]; horizontal_ratio = abs(hit_vx) / (abs(hit_vy) + 1e-6)
                if horizontal_ratio < hit_horizontal_ratio_thresh:
                    print(f"{log_prefix}  > [第 {i} 幀] 偵測到有效擊球事件 (速度: {speed:.1f})")
                    serve_events.append({"hit_frame_id": i, "hit_position": curr_ball_pos.tolist(), "hit_speed": speed})
                    state = "SEARCHING_TOSS"
    return serve_events

def process_single_video(video_path, base_output_dir, overwrite, frame_args, analysis_args):
    # This function is identical to the previous version.
    video_base_name = os.path.splitext(os.path.basename(video_path))[0]
    log_prefix = f"[{video_base_name}] "
    try:
        video_specific_output_dir = os.path.join(base_output_dir, video_base_name)
        tracking_output_path = os.path.join(video_specific_output_dir, 'tracking_output')
        analysis_output_path = os.path.join(video_specific_output_dir, 'analysis_output')
        final_video_output = os.path.join(analysis_output_path, f"{video_base_name}_analysis.mp4")
        json_input_path = os.path.join(tracking_output_path, f"{video_base_name}_all_frames_data_with_pose.json")
        if not overwrite and os.path.exists(final_video_output):
            return {"video": video_base_name, "status": "skipped", "log": "最終輸出檔案已存在。", "events": [], "tracking_data_path": None, "original_video_path": video_path}
        os.makedirs(analysis_output_path, exist_ok=True)
        print(f"{log_prefix}開始執行 Stage 1: 物件追蹤...")
        track_command = [sys.executable, os.path.join("video_processing", "track_ball_and_player.py"), "--input", video_path, "--output_dir", tracking_output_path]
        if frame_args.get('save_all_frames'): track_command.append("--save_all_frames")
        result_track = subprocess.run(track_command, capture_output=True, text=True, encoding='utf-8')
        if result_track.returncode != 0: raise subprocess.CalledProcessError(result_track.returncode, track_command, output=result_track.stdout, stderr=result_track.stderr)
        print(f"{log_prefix}物件追蹤完成。")
        if not os.path.exists(json_input_path): raise FileNotFoundError(f"追蹤後所需的JSON檔案不存在: {json_input_path}")
        with open(json_input_path, 'r', encoding='utf-8') as f: all_frames_data = json.load(f)
        serve_events = analyze_serve_events(all_frames_data, config=analysis_args, log_prefix=log_prefix)
        if serve_events: print(f"{log_prefix}偵測到 {len(serve_events)} 個擊球事件。")
        else: print(f"{log_prefix}未偵測到符合條件的擊球事件。")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): raise IOError(f"無法開啟影片檔案: {video_path}")
        fps, w, h = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(final_video_output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        while cap.isOpened():
            ret, frame = cap.read();
            if not ret: break
            writer.write(frame)
        cap.release(); writer.release()
        with open(os.path.join(video_specific_output_dir, "processing.log"), 'w', encoding='utf-8') as f:
            f.write(f"--- STAGE 1: TRACKING LOG ---\n{result_track.stdout}\n{result_track.stderr}\n\n--- STAGE 2: ANALYSIS RESULTS ---\n")
            f.write(json.dumps(serve_events, indent=2))
        return {"video": video_base_name, "status": "success", "log": final_video_output, "events": serve_events, "tracking_data_path": json_input_path, "original_video_path": video_path}
    except Exception as e:
        error_message = f"處理影片 {video_base_name} 時發生嚴重錯誤: {e}"
        if isinstance(e, subprocess.CalledProcessError): error_message += f"\n--- 子程序標準輸出 ---\n{e.output}\n\n--- 子程序標準錯誤（Traceback） ---\n{e.stderr}"
        return {"video": video_base_name, "status": "failed", "log": error_message, "events": [], "original_video_path": video_path}

# ==============================================================================
# ✨ 核心修改：Summary 產生器，增加發球區域判斷 ✨
# ==============================================================================
def create_first_hit_summary(results, base_output_dir, search_offset, court_polygon):
    print("\n" + "="*80); print(f"--- 正在產生首次擊球總結報告 (增加發球區域判斷)... ---")
    summary_dir = os.path.join(base_output_dir, "first_hit_summary_with_zone")
    os.makedirs(summary_dir, exist_ok=True)
    summary_txt_path = os.path.join(summary_dir, f"first_hit_frames_offset_{search_offset}.txt")
    
    with open(summary_txt_path, 'w', encoding='utf-8') as f:
        f.write(f"--- 每個影片片段首次偵測到的擊球資訊 (回溯 {search_offset} 幀) ---\n\n")
        
        for result in sorted(results, key=lambda r: r['video']):
            if result['status'] != 'success' or not result.get('events'): continue
            
            video_base_name = result['video']
            first_event = result['events'][0]
            hit_frame_id = first_event['hit_frame_id']
            tracking_path = result.get('tracking_data_path')

            if not tracking_path or not os.path.exists(tracking_path): continue
            with open(tracking_path, 'r', encoding='utf-8') as json_f: all_frames_data = json.load(json_f)

            player_search_frame_id = max(0, hit_frame_id - search_offset)
            server_data = None
            
            if player_search_frame_id < len(all_frames_data):
                search_frame_data = all_frames_data[player_search_frame_id]
                ball_pos_at_search_frame = get_ball_center_from_data(search_frame_data)
                all_players_at_search_frame = search_frame_data.get('player_detections', [])
                
                if ball_pos_at_search_frame is not None and all_players_at_search_frame:
                    server_data = min(all_players_at_search_frame, key=lambda p: get_distance_to_player_box(ball_pos_at_search_frame, p['box_coords']))
            
            # ✨ NEW LOGIC: Determine serving zone
            serving_zone = "Unknown"
            if server_data:
                server_center = get_player_center(server_data)
                serving_zone = get_serving_zone(server_center, court_polygon)
                print(f"[{video_base_name}] 成功鎖定發球員，判斷其位於區域: {serving_zone}")
            else:
                 print(f"[{video_base_name}] 未能鎖定發球員，無法判斷區域。")

            f.write(f"{video_base_name}: 擊球偵測幀={hit_frame_id}, 發球區域={serving_zone}\n")
            
            original_video_path = result.get('original_video_path')
            if not original_video_path: continue

            try:
                cap = cv2.VideoCapture(original_video_path)
                if not cap.isOpened(): continue
                cap.set(cv2.CAP_PROP_POS_FRAMES, hit_frame_id)
                ret, hit_frame_image = cap.read()
                if ret and server_data:
                    # 在擊球幀上繪製標記
                    p1 = (int(server_data['box_coords'][0]), int(server_data['box_coords'][1]))
                    p2 = (int(server_data['box_coords'][2]), int(server_data['box_coords'][3]))
                    cv2.rectangle(hit_frame_image, p1, p2, (255, 0, 255), 3)
                    
                    # ✨ NEW VISUALIZATION: Display the zone on the image
                    text = f"SERVER (Zone: {serving_zone})"
                    cv2.putText(hit_frame_image, text, (p1[0], p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 255), 3)
                    
                    # 畫出球場和分區線用於偵錯
                    if court_polygon:
                        pts = np.array(court_polygon, np.int32).reshape((-1, 1, 2))
                        cv2.polylines(hit_frame_image, [pts], isClosed=True, color=(255, 255, 0), thickness=2)
                        p1_base = np.array(court_polygon[1])
                        p2_base = np.array(court_polygon[2])
                        div1 = tuple(map(int, p1_base + (p2_base - p1_base) / 3))
                        div2 = tuple(map(int, p1_base + 2 * (p2_base - p1_base) / 3))
                        cv2.line(hit_frame_image, (div1[0], div1[1] - 20), (div1[0], div1[1] + 20), (0, 255, 255), 3)
                        cv2.line(hit_frame_image, (div2[0], div2[1] - 20), (div2[0], div2[1] + 20), (0, 255, 255), 3)
                    
                    img_name = f"{video_base_name}_frame_{hit_frame_id:06d}_zoned.jpg"
                    cv2.imwrite(os.path.join(summary_dir, img_name), hit_frame_image)
                cap.release()
            except Exception as e:
                print(f"[錯誤] 繪製 {video_base_name} 的區域圖片時失敗: {e}")
                
    print(f"\n總結報告產生完畢！"); print(f"詳細資訊請見: {os.path.abspath(summary_dir)}")

def main():
    parser = argparse.ArgumentParser(description="[最終通用版 v4] 自動化分析排球影片，增加發球區域判斷功能。")
    # ✨ NEW ARGUMENT: --court_config is now required.
    parser.add_argument("--court_config", type=str, required=True, help="定義了球場邊界的 court_config.json 檔案路徑。")
    
    # (此處省略了與上一版相同的其他參數定義)
    parser.add_argument("--input_folder", type=str, required=True, help="包含影片檔案的輸入資料夾。")
    parser.add_argument("--output_folder", type=str, default="volleyball_analysis_results", help="儲存所有分析結果的根目錄。")
    parser.add_argument("--workers", type=int, default=None, help="指定平行處理的核心數，預設為電腦所有核心。")
    parser.add_argument("--overwrite", action="store_true", help="強制重新執行所有影片的分析，覆蓋現有結果。")
    analysis_group = parser.add_argument_group('Analysis Parameters (您的偵測邏輯參數)')
    analysis_group.add_argument("--search_offset", type=int, default=3, help="從偵測到擊球的幀往前『回溯』幾幀來尋找發球員。")
    analysis_group.add_argument("--hit_v", type=float, default=40.0, help="偵測擊球的最小瞬時速度。")
    analysis_group.add_argument("--max_speed", type=float, default=200.0, help="過濾不可能的高速移動。")
    analysis_group.add_argument("--toss_vy", type=float, default=8.0, help="觸發『疑似拋球』的最小初始垂直向上速度。")
    analysis_group.add_argument("--frames_to_validate", type=int, default=8, help="確認拋球動作的幀數窗口。")
    analysis_group.add_argument("--min_upward_confirms", type=int, default=3, help="在確認窗口中，球至少要持續向上幾幀。")
    analysis_group.add_argument("--vertical_ratio", type=float, default=1.5, help="拋球時，垂直速度必須是水平速度的最小倍數。")
    analysis_group.add_argument("--hit_h_ratio", type=float, default=2.5, help="擊球時，水平分量與垂直分量的最大比例(避免將向上的高球誤判為擊球)。")
    analysis_group.add_argument("--max_frames_to_apex", type=int, default=75, help="從拋球到球達頂點的最長幀數。")
    analysis_group.add_argument("--max_frames_to_hit", type=int, default=40, help="從球達頂點到擊球的最長幀數。")
    analysis_group.add_argument("--max_lost_frames_apex", type=int, default=50, help="在等待頂點時，允許球消失的最大幀數。")
    frame_group = parser.add_argument_group('Frame Saving Options')
    frame_group.add_argument("--save_all_frames", action="store_true", help="(可選) 讓追蹤腳本儲存所有原始畫格。")
    
    args = parser.parse_args()
    
    # ✨ NEW LOGIC: Load court configuration once at the beginning.
    try:
        with open(args.court_config, 'r') as f:
            court_config = json.load(f)
        court_polygon = court_config.get("court_boundary_polygon")
        if not court_polygon or len(court_polygon) != 4:
            raise ValueError("court_boundary_polygon not found or is not a 4-point list in the config file.")
    except Exception as e:
        print(f"讀取或解析 --court_config 檔案 '{args.court_config}' 時發生錯誤: {e}")
        sys.exit(1)

    frame_args = {'save_all_frames': args.save_all_frames}
    analysis_args = {k: v for k, v in vars(args).items() if k not in ['input_folder', 'output_folder', 'workers', 'overwrite', 'save_all_frames', 'court_config']}
    
    # (此處省略了與上一版相同的 ProcessPoolExecutor 程式碼)
    script_start_time = datetime.now()
    video_files = find_video_files(args.input_folder)
    if not video_files: print(f"[錯誤] 在 '{args.input_folder}' 中找不到任何支援的影片檔案。"); return
    base_output_dir = os.path.abspath(args.output_folder)
    os.makedirs(base_output_dir, exist_ok=True)
    max_workers = args.workers if args.workers and args.workers > 0 else (os.cpu_count() or 1)
    print(f"[資訊] 找到 {len(video_files)} 個影片。將使用最多 {max_workers} 個核心進行平行處理。")
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_video, vp, base_output_dir, args.overwrite, frame_args, analysis_args): vp for vp in video_files}
        with tqdm(total=len(futures), desc="整體進度", unit="video") as pbar:
            for future in as_completed(futures):
                pbar.update(1)
                try: 
                    results.append(future.result())
                except Exception as exc: 
                    video_path = futures[future]
                    results.append({"video": os.path.basename(video_path), "status": "failed", "log": f"執行時發生嚴重錯誤: {exc}", "events": [], "original_video_path": video_path})

    # ✨ NEW LOGIC: Pass the loaded court_polygon to the summary function.
    create_first_hit_summary(results, base_output_dir, args.search_offset, court_polygon)
    
    # (此處省略了與上一版相同的最終報告生成程式碼)
    print("\n" + "="*80); print("--- 所有任務已完成，正在生成最終摘要報告... ---")
    summary_path = os.path.join(base_output_dir, "summary_report.txt")
    success_videos = [r for r in results if r['status'] == 'success']
    failed_videos = [r for r in results if r['status'] == 'failed']
    skipped_videos = [r for r in results if r['status'] == 'skipped']
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"批次處理執行摘要\n執行時間: {script_start_time.strftime('%Y-%m-%d %H:%M:%S')}\n總耗時: {datetime.now() - script_start_time}\n" + "="*60 + "\n\n")
        f.write(f"--- ✅ 處理成功 ({len(success_videos)}) ---\n"); [f.write(f"- {res['video']}\n") for res in success_videos]
        f.write(f"\n--- ⏭️ 自動跳過 ({len(skipped_videos)}) ---\n"); [f.write(f"- {res['video']}\n") for res in skipped_videos]
        f.write(f"\n--- ❌ 處理失敗 ({len(failed_videos)}) ---\n"); [f.write(f"--- Video: {res['video']} ---\n{res['log']}\n\n") for res in failed_videos]
    print(f"摘要報告已生成於: {summary_path}"); print(f"執行結果: {len(success_videos)} 成功, {len(failed_videos)} 失敗, {len(skipped_videos)} 跳過。"); print("="*80)

def find_video_files(directory):
    supported_formats = ('.mp4', '.avi', 'mov', '.mkv')
    video_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(supported_formats):
                video_files.append(os.path.join(root, file))
    return video_files

if __name__ == '__main__':
    main()