# batch_test_serve.py
# -*- coding: utf-8 -*-
"""
批次測試發球偵測和發球員識別

測試整個資料夾的影片，輸出統計結果
"""

import os
import sys
import json
import argparse
import cv2
import glob
import numpy as np
from datetime import datetime

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from core.serve_detector import analyze_serve_events_v2
from core.server_identifier import analyze_serve_player, get_keypoint


def load_tracking_data(json_path: str) -> dict:
    """載入追蹤數據"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def load_court_config(config_path: str) -> dict:
    """載入場地設定（包含排除區域）"""
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return None


def get_frame_by_id(frames_data: list, frame_id: int) -> dict:
    """根據 frame_id 取得幀資料"""
    for frame in frames_data:
        if frame.get('frame_id') == frame_id:
            return frame
    return None


def draw_serve_analysis(frame, frame_data, ball_position, server_result, frame_label="", is_reference=False, exclusion_zones=None):
    """在影格上繪製發球分析結果
    
    注意：使用發球員的座標範圍來匹配，而非 index（因為每幀的球員順序可能不同）
    """
    height, width = frame.shape[:2]
    
    # 繪製排除區域（紫色半透明）
    if exclusion_zones:
        for zone in exclusion_zones:
            if isinstance(zone, dict) and 'polygon' in zone:
                polygon = zone['polygon']
                zone_name = zone.get('name', 'Exclusion Zone')
            elif isinstance(zone, list):
                polygon = zone
                zone_name = 'Exclusion Zone'
            else:
                continue
            
            pts = np.array(polygon, dtype=np.int32)
            # 畫紫色邊框
            cv2.polylines(frame, [pts], True, (255, 0, 255), 3)
            # 標示名稱
            if len(polygon) > 0:
                cv2.putText(frame, zone_name, (int(polygon[0][0]), int(polygon[0][1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
    
    if ball_position:
        x, y = int(ball_position[0]), int(ball_position[1])
        cv2.circle(frame, (x, y), 20, (0, 255, 255), 3)
        cv2.putText(frame, "BALL", (x - 20, y - 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    players = frame_data.get('player_detections', [])
    
    # 取得 FOUND 幀中發球員的座標（用於跨幀匹配）
    server_info = server_result.get('server')
    server_center = None
    if server_info:
        server_center = server_info.get('center_point')
    
    # 找出當前幀中最接近發球員位置的球員
    matched_server_idx = None
    if server_center:
        min_dist = float('inf')
        for i, player in enumerate(players):
            center = player.get('center_point', [0, 0])
            dist = ((center[0] - server_center[0])**2 + (center[1] - server_center[1])**2)**0.5
            if dist < min_dist and dist < 150:  # 150 像素內才算匹配
                min_dist = dist
                matched_server_idx = i
    
    for i, player in enumerate(players):
        box = player.get('box_coords')
        if not box:
            continue
        
        if i == matched_server_idx:
            color = (0, 255, 0)
            label = f"SERVER (conf: {server_result.get('final_confidence', 0):.2f})"
        else:
            color = (255, 0, 0)
            label = f"Player {i}"
        
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
        cv2.putText(frame, label, (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        left_wrist = get_keypoint(player, 9)
        right_wrist = get_keypoint(player, 10)
        
        if left_wrist:
            cv2.circle(frame, (int(left_wrist[0]), int(left_wrist[1])), 8, (0, 0, 255), -1)
        if right_wrist:
            cv2.circle(frame, (int(right_wrist[0]), int(right_wrist[1])), 8, (0, 0, 255), -1)
    
    # 如果發球員在當前幀沒有被偵測到，顯示其 FOUND 幀的位置
    if matched_server_idx is None and server_center:
        cv2.circle(frame, (int(server_center[0]), int(server_center[1])), 30, (0, 255, 0), 3)
        cv2.putText(frame, "SERVER (not detected)", (int(server_center[0]) - 80, int(server_center[1]) - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    info_lines = [
        f"{frame_label}",
        f"Server: Player at ({int(server_center[0])}, {int(server_center[1])})" if server_center else "Server: Unknown",
        f"Confidence: {server_result.get('final_confidence', 0):.2f}",
    ]
    
    if is_reference:
        info_lines.append(f"(Server determined from FOUND frame)")
    
    for i, line in enumerate(info_lines):
        cv2.putText(frame, line, (10, 30 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return frame


def process_single_video(video_path: str, json_path: str, output_dir: str, 
                         save_images: bool = True, verbose: bool = False,
                         court_config: dict = None) -> dict:
    """
    處理單個影片
    
    Args:
        video_path: 影片路徑
        json_path: JSON 追蹤資料路徑
        output_dir: 輸出目錄
        save_images: 是否儲存圖片
        verbose: 是否顯示詳細訊息
        court_config: 場地設定（包含 exclusion_zones）
    
    Returns:
        處理結果字典
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    result = {
        'video_name': video_name,
        'status': 'unknown',
        'serve_detected': False,
        'toss_frame': None,
        'hit_frame': None,
        'server_index': None,
        'confidence': 0,
        'error': None
    }
    
    try:
        # 載入追蹤數據
        data = load_tracking_data(json_path)
        frames_data = data.get('frames', [])
        
        if not frames_data:
            result['status'] = 'no_frames'
            result['error'] = 'JSON 中沒有幀資料'
            return result
        
        # 發球偵測
        serve_events = analyze_serve_events_v2(
            frames_data,
            config={'hit_v': 40.0, 'toss_vy': 8.0},
            use_dynamic_threshold=True,
            first_only=True,
            log_prefix="" if verbose else None
        )
        
        if not serve_events:
            result['status'] = 'no_serve'
            return result
        
        serve_event = serve_events[0]
        result['serve_detected'] = True
        result['toss_frame'] = serve_event.get('toss_start_frame')
        result['hit_frame'] = serve_event.get('hit_frame_id')
        result['hit_speed'] = serve_event.get('hit_speed', 0)
        
        # 取得影片高度
        cap = cv2.VideoCapture(video_path)
        image_height = 720  # 預設值
        if cap.isOpened():
            image_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 發球員識別（使用 lookback 方法）
        server_result = analyze_serve_player(
            frames_data=frames_data,
            serve_event=serve_event,
            method='lookback',
            image_height=image_height,
            exclusion_zones=court_config.get('exclusion_zones') if court_config else None
        )
        
        result['server_index'] = server_result.get('final_server_index')
        result['confidence'] = server_result.get('final_confidence', 0)
        result['found_frame'] = server_result.get('found_frame_id')
        result['frames_searched'] = server_result.get('frames_searched', 0)
        result['status'] = 'success'
        
        # 儲存圖片
        if save_images:
            if cap.isOpened():
                # 找到的幀圖片（球員與球重疊的幀）
                found_frame_id = server_result.get('found_frame_id')
                ball_position = server_result.get('ball_position')
                
                if found_frame_id:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, found_frame_id)
                    ret, frame = cap.read()
                    if ret:
                        found_frame_data = get_frame_by_id(frames_data, found_frame_id)
                        if found_frame_data:
                            frame = draw_serve_analysis(
                                frame, found_frame_data,
                                ball_position,
                                server_result,
                                f"FOUND Frame {found_frame_id} (searched back {server_result.get('frames_searched', 0)} frames)",
                                exclusion_zones=court_config.get('exclusion_zones') if court_config else None
                            )
                            output_path = os.path.join(output_dir, f"{video_name}_server_FOUND.jpg")
                            cv2.imwrite(output_path, frame)
                
                toss_frame_id = serve_event.get('toss_start_frame')
                hit_frame_id = serve_event.get('hit_frame_id')
                
                # 拋球幀圖片（參考）
                if toss_frame_id:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, toss_frame_id)
                    ret, frame = cap.read()
                    if ret:
                        toss_frame_data = get_frame_by_id(frames_data, toss_frame_id)
                        if toss_frame_data:
                            frame = draw_serve_analysis(
                                frame, toss_frame_data,
                                serve_event.get('toss_position'),
                                server_result,
                                f"TOSS Frame {toss_frame_id}",
                                is_reference=True,
                                exclusion_zones=court_config.get('exclusion_zones') if court_config else None
                            )
                            output_path = os.path.join(output_dir, f"{video_name}_server_TOSS.jpg")
                            cv2.imwrite(output_path, frame)
                
                # 擊球幀圖片
                if hit_frame_id:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, hit_frame_id)
                    ret, frame = cap.read()
                    if ret:
                        hit_frame_data = get_frame_by_id(frames_data, hit_frame_id)
                        if hit_frame_data:
                            frame = draw_serve_analysis(
                                frame, hit_frame_data,
                                serve_event.get('hit_position'),
                                server_result,
                                f"HIT Frame {hit_frame_id}",
                                is_reference=True,
                                exclusion_zones=court_config.get('exclusion_zones') if court_config else None
                            )
                            output_path = os.path.join(output_dir, f"{video_name}_server_HIT.jpg")
                            cv2.imwrite(output_path, frame)
                
                cap.release()
        
    except Exception as e:
        result['status'] = 'error'
        result['error'] = str(e)
    
    return result


def find_matching_files(video_dir: str, json_dir: str) -> list:
    """
    找出匹配的影片和 JSON 檔案
    
    Returns:
        [(video_path, json_path, video_name), ...]
    """
    matches = []
    
    # 找出所有影片
    video_patterns = ['*.mp4', '*.avi', '*.mov', '*.mkv']
    video_files = []
    for pattern in video_patterns:
        video_files.extend(glob.glob(os.path.join(video_dir, pattern)))
    
    for video_path in sorted(video_files):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # 找對應的 JSON
        json_patterns = [
            f"{video_name}_all_frames_data_with_pose.json",
            f"{video_name}_all_frames_data.json",
            f"{video_name}.json"
        ]
        
        json_path = None
        for pattern in json_patterns:
            potential_path = os.path.join(json_dir, pattern)
            if os.path.exists(potential_path):
                json_path = potential_path
                break
        
        if json_path:
            matches.append((video_path, json_path, video_name))
    
    return matches


def batch_test(video_dir: str, json_dir: str, output_dir: str, 
               save_images: bool = True, verbose: bool = False,
               court_config_path: str = None):
    """
    批次測試整個資料夾
    
    Args:
        video_dir: 影片目錄
        json_dir: JSON 追蹤資料目錄
        output_dir: 輸出目錄
        save_images: 是否儲存圖片
        verbose: 是否顯示詳細訊息
        court_config_path: 場地設定 JSON 路徑
    """
    print("="*70)
    print("批次測試發球偵測和發球員識別")
    print("="*70)
    print(f"影片目錄: {video_dir}")
    print(f"JSON 目錄: {json_dir}")
    print(f"輸出目錄: {output_dir}")
    
    # 載入場地設定
    court_config = load_court_config(court_config_path)
    if court_config:
        exclusion_count = len(court_config.get('exclusion_zones', []))
        print(f"場地設定: {court_config_path} (排除區域: {exclusion_count} 個)")
    else:
        print(f"場地設定: 未指定（不排除任何區域）")
    print()
    
    # 建立輸出目錄
    os.makedirs(output_dir, exist_ok=True)
    
    # 找出匹配的檔案
    matches = find_matching_files(video_dir, json_dir)
    
    if not matches:
        print("❌ 找不到匹配的影片和 JSON 檔案！")
        print(f"   請確認：")
        print(f"   - 影片目錄中有 .mp4 檔案")
        print(f"   - JSON 目錄中有對應的 *_all_frames_data_with_pose.json 檔案")
        return
    
    print(f"找到 {len(matches)} 個匹配的檔案")
    print("-"*70)
    
    # 處理每個影片
    results = []
    for i, (video_path, json_path, video_name) in enumerate(matches, 1):
        print(f"[{i}/{len(matches)}] 處理: {video_name}")
        
        result = process_single_video(
            video_path, json_path, output_dir,
            save_images=save_images,
            verbose=verbose,
            court_config=court_config
        )
        results.append(result)
        
        # 簡短輸出
        if result['status'] == 'success':
            found = result.get('found_frame', 'N/A')
            searched = result.get('frames_searched', 0)
            print(f"    ✅ 找到幀: {found} (往回 {searched} 幀), "
                  f"發球員: Player {result['server_index']}, "
                  f"信心度: {result['confidence']:.2f}")
        elif result['status'] == 'no_serve':
            print(f"    ⚠️ 未偵測到發球")
        else:
            print(f"    ❌ 錯誤: {result.get('error', result['status'])}")
    
    # 統計結果
    print()
    print("="*70)
    print("統計結果")
    print("="*70)
    
    total = len(results)
    success = sum(1 for r in results if r['status'] == 'success')
    no_serve = sum(1 for r in results if r['status'] == 'no_serve')
    errors = sum(1 for r in results if r['status'] == 'error')
    
    print(f"總計: {total} 個影片")
    print(f"  ✅ 成功偵測: {success} ({success/total*100:.1f}%)")
    print(f"  ⚠️ 未偵測到發球: {no_serve} ({no_serve/total*100:.1f}%)")
    print(f"  ❌ 錯誤: {errors} ({errors/total*100:.1f}%)")
    
    # 信心度統計
    confidences = [r['confidence'] for r in results if r['status'] == 'success']
    if confidences:
        print()
        print(f"發球員識別信心度:")
        print(f"  平均: {sum(confidences)/len(confidences):.2f}")
        print(f"  最高: {max(confidences):.2f}")
        print(f"  最低: {min(confidences):.2f}")
    
    # 儲存結果到 JSON
    summary = {
        'test_time': datetime.now().isoformat(),
        'video_dir': video_dir,
        'json_dir': json_dir,
        'total': total,
        'success': success,
        'no_serve': no_serve,
        'errors': errors,
        'results': results
    }
    
    summary_path = os.path.join(output_dir, 'batch_test_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print()
    print(f"詳細結果已儲存: {summary_path}")
    
    # 列出需要檢查的影片
    need_check = [r for r in results if r['status'] != 'success' or r['confidence'] < 0.5]
    if need_check:
        print()
        print("-"*70)
        print("需要人工檢查的影片:")
        for r in need_check:
            if r['status'] == 'success':
                print(f"  ⚠️ {r['video_name']}: 信心度低 ({r['confidence']:.2f})")
            else:
                print(f"  ❌ {r['video_name']}: {r['status']}")
    
    print()
    print("="*70)
    print("批次測試完成！")
    print("="*70)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="批次測試發球偵測和發球員識別")
    parser.add_argument("--video-dir", type=str, required=True, 
                        help="影片目錄路徑")
    parser.add_argument("--json-dir", type=str, required=True,
                        help="JSON 追蹤數據目錄路徑")
    parser.add_argument("--output", type=str, default="batch_test_output",
                        help="輸出目錄 (預設: batch_test_output)")
    parser.add_argument("--court-config", type=str, default=None,
                        help="場地設定 JSON 路徑 (包含排除區域)")
    parser.add_argument("--no-images", action="store_true",
                        help="不儲存圖片（只輸出統計）")
    parser.add_argument("--verbose", action="store_true",
                        help="顯示詳細處理過程")
    
    args = parser.parse_args()
    
    batch_test(
        video_dir=args.video_dir,
        json_dir=args.json_dir,
        output_dir=args.output,
        save_images=not args.no_images,
        verbose=args.verbose,
        court_config_path=args.court_config
    )


if __name__ == "__main__":
    main()