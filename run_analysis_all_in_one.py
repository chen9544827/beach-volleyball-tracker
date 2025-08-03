# run_analysis_all_in_one.py (最終穩定版 v11 - 增加關鍵幀儲存)

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
import locale

# --- (此處省略了與上一版相同的套件匯入和輔助函數) ---
try:
    from tqdm import tqdm
except ImportError:
    print("模組 'tqdm' 未找到，正在嘗試自動安裝...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"])
        from tqdm import tqdm; print("tqdm 安裝成功！")
    except Exception as e:
        print(f"自動安裝 tqdm 失敗，請手動執行 'pip install tqdm'。錯誤: {e}"); sys.exit(1)
try:
    import matplotlib.pyplot as plt
except ImportError:
    print("偵錯模式需要 'matplotlib' 模組，正在嘗試自動安裝...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
        import matplotlib.pyplot as plt
        print("matplotlib 安裝成功！")
    except Exception as e:
        print(f"自動安裝 matplotlib 失敗，請手動執行 'pip install matplotlib'。錯誤: {e}")
        plt = None
from analysis.jump_serve_analyzer import analyze_jump_serve_by_pose, get_player_center
def get_ball_center_from_data(frame_data):
    if frame_data and frame_data.get('ball_detections'):
        valid_balls = [b for b in frame_data['ball_detections'] if not b.get('is_in_background_zone', False)]
        if not valid_balls: return None
        best_ball = max(valid_balls, key=lambda b: b.get('confidence', 0))
        box = best_ball['box_coords']; return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
def get_distance_to_player_box(point, player_box):
    px, py = point; x1, y1, x2, y2 = player_box
    closest_x = max(x1, min(px, x2)); closest_y = max(y1, min(py, y2))
    return np.sqrt((px - closest_x)**2 + (py - closest_y)**2)
def get_serving_zone(server_center, court_polygon):
    if server_center is None or court_polygon is None or len(court_polygon) != 4: return 'Unknown'
    p1 = np.array(court_polygon[1]); p2 = np.array(court_polygon[2])
    zone_ab_divider = p1 + (p2 - p1) / 3; zone_bc_divider = p1 + 2 * (p2 - p1) / 3
    if server_center[0] < zone_ab_divider[0]: return 'A'
    elif server_center[0] < zone_bc_divider[0]: return 'B'
    else: return 'C'
def analyze_serve_events(all_frames_data, config, log_prefix=""):
    # (此處為您提供的發球偵測邏輯，保持不變)
    hit_v_thresh = config.get("hit_v", 40.0); max_plausible_speed = config.get("max_speed", 200.0); toss_initial_vy_thresh = config.get("toss_vy", 8.0); frames_to_validate_toss = config.get("frames_to_validate", 8); min_upward_confirms = config.get("min_upward_confirms", 3); vertical_ratio_thresh = config.get("vertical_ratio", 1.5); hit_horizontal_ratio_thresh = config.get("hit_h_ratio", 2.5); max_frames_to_apex = config.get("max_frames_to_apex", 75); max_frames_to_hit = config.get("max_frames_to_hit", 40); max_lost_frames_apex = config.get("max_lost_frames_apex", 50)
    serve_events, state, event_candidate = [], "SEARCHING_TOSS", {}
    print(f"\n{log_prefix}[分析階段] 正在使用您提供的邏輯偵測擊球事件...")
    for i in range(1, len(all_frames_data)):
        prev_ball_pos = get_ball_center_from_data(all_frames_data[i-1]); curr_ball_pos = get_ball_center_from_data(all_frames_data[i])
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
def process_single_video(video_path, base_output_dir, overwrite, config):
    video_base_name = os.path.splitext(os.path.basename(video_path))[0]
    log_prefix = f"[{video_base_name}] "
    try:
        video_specific_output_dir = os.path.join(base_output_dir, video_base_name)
        tracking_output_path = os.path.join(video_specific_output_dir, 'tracking_output')
        json_input_path = os.path.join(tracking_output_path, f"{video_base_name}_all_frames_data_with_pose.json")
        if not overwrite and os.path.exists(os.path.join(video_specific_output_dir, 'final_analysis')):
            return {"video": video_base_name, "status": "skipped", "log": "最終分析資料夾已存在。", "tracking_data_path": json_input_path, "original_video_path": video_path}
        
        if overwrite or not os.path.exists(json_input_path):
            print(f"{log_prefix}開始執行 Stage 1: 物件追蹤...")
            track_command = [sys.executable, os.path.join("video_processing", "track_ball_and_player.py"), "--input", video_path, "--output_dir", tracking_output_path]
            system_encoding = locale.getpreferredencoding()
            result_track = subprocess.run(track_command, capture_output=True, text=True, encoding=system_encoding, errors='ignore')
            if result_track.returncode != 0:
                raise subprocess.CalledProcessError(result_track.returncode, track_command, output=result_track.stdout, stderr=result_track.stderr)
            print(f"{log_prefix}物件追蹤完成。")
        else:
            print(f"{log_prefix}物件追蹤 JSON 已存在，跳過 Stage 1。")
        
        print(f"{log_prefix}開始執行 Stage 2: 事件分析...")
        if not os.path.exists(json_input_path):
            raise FileNotFoundError(f"追蹤後所需的JSON檔案不存在: {json_input_path}")
        with open(json_input_path, 'r', encoding='utf-8') as f:
            all_frames_data = json.load(f)
        serve_events = analyze_serve_events(all_frames_data, config=config, log_prefix=log_prefix)
        
        return {"video": video_base_name, "status": "success", "log": "追蹤和初步分析完成", "events": serve_events, "tracking_data_path": json_input_path, "original_video_path": video_path}
    except Exception as e:
        error_message = f"處理影片 {video_base_name} 時發生嚴重錯誤: {e}"
        if isinstance(e, subprocess.CalledProcessError):
            error_message += f"\n--- 子程序標準輸出 ---\n{e.output}\n\n--- 子程序標準錯誤（Traceback） ---\n{e.stderr}"
        return {"video": video_base_name, "status": "failed", "log": error_message, "events": [], "original_video_path": video_path}

# ==============================================================================
# ✨ 核心修改：Summary 產生器，增加關鍵幀儲存功能 ✨
# ==============================================================================
def create_final_summary(results, base_output_dir, config):
    print("\n" + "="*80); print(f"--- 正在產生最終分析報告與影像... ---")
    
    # --- 新增的資料夾路徑 ---
    # 總報告資料夾
    summary_dir = os.path.join(base_output_dir, "final_summary")
    os.makedirs(summary_dir, exist_ok=True)
    # 集中審核關鍵幀的資料夾
    review_frames_dir = os.path.join(base_output_dir, "key_frames_for_review")
    os.makedirs(review_frames_dir, exist_ok=True)
    
    summary_txt_path = os.path.join(summary_dir, f"summary_report.txt")
    
    court_polygon = config.get("court_polygon")
    DEBUG_MODE = config.get("debug_jump_serve", False)
    if DEBUG_MODE and plt is None:
        print("\n錯誤：偵錯模式需要 matplotlib，但安裝失敗。") ; sys.exit(1)

    with open(summary_txt_path, 'w', encoding='utf-8') as f:
        f.write(f"--- 發球分析總結報告 ---\n\n")
        
        for result in sorted(results, key=lambda r: r['video']):
            if result['status'] != 'success' or not result.get('events'): continue
            
            video_base_name = result['video']
            first_event = result['events'][0]
            hit_frame_id = first_event['hit_frame_id']
            tracking_path = result.get('tracking_data_path')
            
            if not tracking_path or not os.path.exists(tracking_path): continue
            with open(tracking_path, 'r', encoding='utf-8') as json_f: all_frames_data = json.load(json_f)
            
            player_search_frame_id = max(0, hit_frame_id - config.get('search_offset'))
            server_data = None
            if player_search_frame_id < len(all_frames_data):
                search_frame_data = all_frames_data[player_search_frame_id]
                ball_pos = get_ball_center_from_data(search_frame_data)
                players = search_frame_data.get('player_detections', [])
                if ball_pos is not None and players:
                    server_data = min(players, key=lambda p: get_distance_to_player_box(ball_pos, p['box_coords']))
            
            serving_zone, serve_type, debug_info = "Unknown", "Unknown", {}
            if server_data:
                server_center = get_player_center(server_data)
                serving_zone = get_serving_zone(server_center, court_polygon)
                serve_type, debug_info = analyze_jump_serve_by_pose(all_frames_data, server_data, hit_frame_id, config)
                print(f"[{video_base_name}] 分析完成。區域: {serving_zone}, 類型: {serve_type}")
            else:
                print(f"[{video_base_name}] 未能鎖定發球員。")

            f.write(f"{video_base_name}: 擊球幀={hit_frame_id}, 發球區域={serving_zone}, 發球類型={serve_type}, 偵錯狀態='{debug_info.get('status', 'N/A')}'\n")
            
            # --- ✨ 全新關鍵幀儲存邏輯 ✨ ---
            original_video_path = result.get('original_video_path')
            if not original_video_path: continue
            
            # 為每個片段建立自己的關鍵幀資料夾
            segment_key_frames_dir = os.path.join(base_output_dir, video_base_name, "key_frames")
            os.makedirs(segment_key_frames_dir, exist_ok=True)
            
            try:
                cap = cv2.VideoCapture(original_video_path)
                if not cap.isOpened(): continue
                
                # 迴圈儲存 hit_frame 前後各3幀
                for frame_offset in range(-3, 4):
                    frame_to_capture = hit_frame_id + frame_offset
                    if frame_to_capture < 0: continue
                    
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_to_capture)
                    ret, frame = cap.read()
                    if ret:
                        # 命名規則
                        tag = ""
                        if frame_offset == 0: tag = "_HIT"
                        elif frame_offset < 0: tag = f"_PRE_{abs(frame_offset)}"
                        else: tag = f"_POST_{frame_offset}"
                        
                        img_name = f"{video_base_name}_frame_{frame_to_capture:06d}{tag}.jpg"
                        
                        # 儲存到兩個位置
                        cv2.imwrite(os.path.join(segment_key_frames_dir, img_name), frame)
                        cv2.imwrite(os.path.join(review_frames_dir, img_name), frame)
                
                cap.release()
            except Exception as e:
                print(f"[錯誤] 儲存 {video_base_name} 的關鍵幀時失敗: {e}")
            
            # --- 偵錯繪圖邏輯 (保持不變) ---
            if DEBUG_MODE and debug_info:
                # ... (此處的 matplotlib 繪圖邏輯與上一版完全相同)
                plt.figure(figsize=(10, 6)); trajectory = debug_info.get("hip_y_trajectory", [])
                plot_title = f'Jump Serve Debug Plot for: {video_base_name}\n'
                if "Success" in debug_info.get("status", ""):
                    displacement = debug_info.get("displacement"); threshold = debug_info.get("threshold")
                    plot_title += (f'Result: {serve_type} (Displacement: {displacement:.1f}, Threshold: {threshold})')
                else: plot_title += f'Result: {serve_type} (Status: {debug_info.get("status", "Unknown Error")})'
                plt.title(plot_title)
                if trajectory:
                    action_window = debug_info.get("action_window", range(len(trajectory)))
                    plt.plot(action_window[:len(trajectory)], trajectory, marker='o', linestyle='-', label='Hip Y-Trajectory')
                    peak_y, crouch_y = debug_info.get("peak_y"), debug_info.get("crouch_y")
                    if peak_y is not None: plt.axhline(y=peak_y, color='g', linestyle='--', label=f'Peak Height (Y={peak_y:.1f})')
                    if crouch_y is not None: plt.axhline(y=crouch_y, color='r', linestyle='--', label=f'Crouch Height (Y={crouch_y:.1f})')
                    plt.xlabel('Frame Number'); plt.ylabel('Hip Y-Coordinate (pixels)'); plt.legend(); plt.grid(True); plt.gca().invert_yaxis()
                    plot_filename = f"{video_base_name}_jump_debug_plot.png"; plt.savefig(os.path.join(summary_dir, plot_filename)); plt.close()
                else: print(f"[{video_base_name}] 偵錯警告：無法繪製軌跡圖，因為 trajectory 為空。狀態: {debug_info.get('status')}")

    print(f"\n總結報告與影像產生完畢！"); print(f"詳細資訊請見: {os.path.abspath(summary_dir)} 和 {os.path.abspath(review_frames_dir)}")

def main():
    # (main 函數與上一版完全相同，不需修改)
    parser = argparse.ArgumentParser(description="[最終穩定版 v11] 自動化分析排球影片，增加關鍵幀儲存功能。")
    parser.add_argument("--workers", type=int, default=2, help="同時執行的最大程序數量。建議從 1 或 2 開始。")
    parser.add_argument("--debug_jump_serve", action="store_true", help="啟用跳發判斷的視覺化偵錯模式，會產生臀部軌跡圖。")
    parser.add_argument("--input_folder", type=str, required=True, help="包含待分析影片的資料夾路徑。")
    parser.add_argument("--court_config", type=str, help="定義了球場邊界的 court_config.json 檔案路徑 (可選)。")
    parser.add_argument("--output_folder", type=str, default="volleyball_analysis_results", help="儲存所有分析結果的根目錄。")
    parser.add_argument("--overwrite", action="store_true", help="強制重新執行所有影片的分析，覆蓋現有結果。")
    analysis_group = parser.add_argument_group('Analysis Parameters')
    analysis_group.add_argument("--jump_height_threshold", type=int, default=20, help="判定為跳躍的最小臀部垂直位移像素值。")
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
    args = parser.parse_args()
    config = {k: v for k, v in vars(args).items() if k not in ['input_folder', 'output_folder', 'overwrite', 'court_config', 'workers']}
    court_polygon = None
    if args.court_config:
        try:
            with open(args.court_config, 'r') as f: court_config_data = json.load(f)
            court_polygon = court_config_data.get("court_boundary_polygon")
            if not court_polygon or len(court_polygon) != 4: raise ValueError("court_boundary_polygon not found or is not a 4-point list.")
            config['court_polygon'] = court_polygon
        except Exception as e:
            print(f"警告：讀取或解析 --court_config 檔案 '{args.court_config}' 時發生錯誤: {e}。將無法進行區域分析。")
            config['court_polygon'] = None
    script_start_time = datetime.now()
    video_files = find_video_files(args.input_folder)
    if not video_files: print(f"[錯誤] 在 '{args.input_folder}' 中找不到任何支援的影片檔案。"); return
    base_output_dir = os.path.abspath(args.output_folder)
    os.makedirs(base_output_dir, exist_ok=True)
    max_workers = args.workers
    print(f"[資訊] 找到 {len(video_files)} 個影片。將使用最多 {max_workers} 個程序同時處理。")
    all_results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_video, vp, base_output_dir, args.overwrite, config): vp for vp in video_files}
        with tqdm(total=len(futures), desc="整體進度", unit="video") as pbar:
            for future in as_completed(futures):
                pbar.update(1)
                try:
                    result = future.result()
                    all_results.append(result)
                except Exception as exc:
                    video_path = futures[future]
                    all_results.append({"video": os.path.basename(video_path), "status": "failed", "log": f"執行時發生嚴重錯誤: {exc}", "events": [], "original_video_path": video_path})
    create_final_summary(all_results, base_output_dir, config)
    print("\n" + "="*80); print("--- 所有任務已完成，正在生成最終摘要報告... ---")
    final_summary_path = os.path.join(base_output_dir, "master_summary_report.txt")
    success_videos = [r for r in all_results if r['status'] == 'success']
    failed_videos = [r for r in all_results if r['status'] == 'failed']
    skipped_videos = [r for r in all_results if r['status'] == 'skipped']
    with open(final_summary_path, 'w', encoding='utf-8') as f:
        f.write(f"批次處理執行摘要\n執行時間: {script_start_time.strftime('%Y-%m-%d %H:%M:%S')}\n總耗時: {datetime.now() - script_start_time}\n" + "="*60 + "\n\n")
        f.write(f"--- ✅ 處理成功 ({len(success_videos)}) ---\n"); [f.write(f"- {res['video']}\n") for res in success_videos]
        f.write(f"\n--- ⏭️ 自動跳過 ({len(skipped_videos)}) ---\n"); [f.write(f"- {res['video']}\n") for res in skipped_videos]
        f.write(f"\n--- ❌ 處理失敗 ({len(failed_videos)}) ---\n"); [f.write(f"--- Video: {res['video']} ---\n{res['log']}\n\n") for res in failed_videos]
    print(f"總摘要報告已生成於: {final_summary_path}"); print("="*80)

def find_video_files(directory):
    supported_formats = ('.mp4', '.avi', '.mov', '.mkv')
    video_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(supported_formats):
                video_files.append(os.path.join(root, file))
    return video_files

if __name__ == '__main__':
    main()