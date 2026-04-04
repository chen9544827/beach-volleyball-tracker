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
from core.server_identifier import analyze_serve_player, get_keypoint, determine_serving_side
from core.jump_serve_detector import classify_serve_type
from core.data_validator import DataValidator, safe_load_json, validate_center_point
from core.error_messages import ValidationError, format_error
from core.filename_parser import parse_filename, extract_group_key_from_path
from core.court_zones import CourtZones
from core.reception_detector import ReceptionDetector
from core.result_exporter import build_result_row, export_to_csv, export_to_excel, export_summary_json
import logging

# 設定日誌
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')

# 跳過條件：整個片段中最高連續有球幀數低於此值則跳過
MIN_CONSECUTIVE_BALL_FRAMES = 10


def load_tracking_data(json_path: str) -> dict:
    """
    載入追蹤數據並驗證

    Args:
        json_path: JSON 檔案路徑

    Returns:
        追蹤資料字典

    Raises:
        ValidationError: 當檔案載入失敗或資料驗證失敗時
    """
    # 安全載入 JSON
    data, error = safe_load_json(json_path)
    if error:
        raise ValidationError(f"載入 JSON 失敗: {error}")

    # 驗證資料結構
    validator = DataValidator(verbose=False)
    is_valid, errors = validator.validate_tracking_json(data)

    if not is_valid:
        error_msg = f"資料驗證失敗:\n  - " + "\n  - ".join(errors)
        raise ValidationError(error_msg)

    return data


def load_court_config(config_path: str) -> dict:
    """
    載入場地設定（包含排除區域）

    Args:
        config_path: 場地設定檔路徑

    Returns:
        場地設定字典，失敗時返回 None
    """
    if not config_path or not os.path.exists(config_path):
        logging.warning(f"找不到場地設定檔: {config_path}")
        return None

    data, error = safe_load_json(config_path)
    if error:
        logging.error(f"載入場地設定失敗: {error}")
        return None

    # 驗證場地設定結構
    validator = DataValidator(verbose=False)
    is_valid, errors = validator.validate_court_config(data)
    if not is_valid:
        logging.error(f"場地設定驗證失敗: {', '.join(errors)}")
        return None

    return data


def get_frame_by_id(frames_data: list, frame_id: int) -> dict:
    """根據 frame_id 取得幀資料"""
    for frame in frames_data:
        if frame.get('frame_id') == frame_id:
            return frame
    return None


def draw_serve_analysis(frame, frame_data, ball_position, server_result, frame_label="", is_reference=False, exclusion_zones=None, jump_result=None):
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
    
    # 驗證 frame_data 並取得球員資料
    players = frame_data.get('player_detections', []) if frame_data else []

    # 取得 FOUND 幀中發球員的座標（用於跨幀匹配）
    server_info = server_result.get('server')
    server_center = None
    if server_info:
        server_center = validate_center_point(server_info.get('center_point'))

    # 找出當前幀中最接近發球員位置的球員
    matched_server_idx = None
    matched_server_player = None
    if server_center:
        min_dist = float('inf')
        for i, player in enumerate(players):
            center = validate_center_point(player.get('center_point'))
            if not center:
                continue
            dist = ((center[0] - server_center[0])**2 + (center[1] - server_center[1])**2)**0.5
            if dist < min_dist and dist < 150:  # 150 像素內才算匹配
                min_dist = dist
                matched_server_idx = i
                matched_server_player = player
    
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
        
        # 繪製腳踝位置（如果是發球員）
        if i == matched_server_idx:
            left_ankle = get_keypoint(player, 15)
            right_ankle = get_keypoint(player, 16)
            if left_ankle:
                cv2.circle(frame, (int(left_ankle[0]), int(left_ankle[1])), 8, (0, 165, 255), -1)  # 橘色
            if right_ankle:
                cv2.circle(frame, (int(right_ankle[0]), int(right_ankle[1])), 8, (0, 165, 255), -1)
    
    # 如果發球員在當前幀沒有被偵測到，顯示其 FOUND 幀的位置
    if matched_server_idx is None and server_center:
        cv2.circle(frame, (int(server_center[0]), int(server_center[1])), 30, (0, 255, 0), 3)
        cv2.putText(frame, "SERVER (not detected)", (int(server_center[0]) - 80, int(server_center[1]) - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # 在右上角顯示大型跳發/站發標籤
    if jump_result:
        serve_type = jump_result.get('serve_type', 'unknown')
        jump_height = jump_result.get('jump_height', 0)
        
        if serve_type == 'jump':
            # 跳發 - 黃底黑字
            label = "JUMP SERVE"
            bg_color = (0, 255, 255)  # 黃色
            text_color = (0, 0, 0)    # 黑色
        else:
            # 站發 - 灰底白字
            label = "STANDING"
            bg_color = (128, 128, 128)  # 灰色
            text_color = (255, 255, 255)  # 白色
        
        # 計算標籤位置（右上角）
        label_x = width - 250
        label_y = 30
        
        # 繪製背景矩形
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
        cv2.rectangle(frame, 
                      (label_x - 10, label_y - text_h - 10),
                      (label_x + text_w + 10, label_y + baseline + 10),
                      bg_color, -1)
        
        # 繪製文字
        cv2.putText(frame, label, (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, text_color, 3)
        
        # 如果是跳發，顯示跳躍高度
        if serve_type == 'jump':
            height_text = f"Height: {jump_height:.0f}px"
            cv2.putText(frame, height_text, (label_x, label_y + 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
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
                         court_config: dict = None,
                         court_zones: CourtZones = None) -> dict:
    """
    處理單個影片

    Args:
        video_path: 影片路徑
        json_path: JSON 追蹤資料路徑
        output_dir: 輸出目錄
        save_images: 是否儲存圖片
        verbose: 是否顯示詳細訊息
        court_config: 場地設定（包含 exclusion_zones）
        court_zones: CourtZones 實例（用於區域判定）

    Returns:
        處理結果字典
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # 提前取得影片高度（供 analyze_serve_events_v2 解析度縮放使用）
    image_height = 720  # 預設值
    _cap = cv2.VideoCapture(video_path)
    if _cap.isOpened():
        image_height = int(_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        _cap.release()

    result = {
        'video_name': video_name,
        'status': 'unknown',
        'serve_detected': False,
        'toss_frame': None,
        'hit_frame': None,
        'server_index': None,
        'confidence': 0,
        'error': None,
        # New fields
        'serve_zone': None,
        'serving_side': None,
        'reception_detected': False,
        'reception_frame': None,
        'reception_zone': None,
        'receiver_index': None,
        'time_to_reception': None,
        'reception_confidence': 0,
        'ball_crossed_net': False,
        'is_ace': False,
        'landing_zone': None,
        'landing_position': None,
        'max_consecutive_ball_frames': 0,
        'court_detection_quality': None,
    }

    # 記錄場地偵測品質（來自 auto_court_detector 的 court_config 元資料）
    if court_config and court_config.get('court_detection_quality'):
        result['court_detection_quality'] = court_config['court_detection_quality']

    try:
        # 載入追蹤數據（包含驗證）
        data = load_tracking_data(json_path)
        frames_data = data.get('frames', [])

        if not frames_data:
            result['status'] = 'no_frames'
            result['error'] = 'JSON 中沒有幀資料'
            return result

        # 計算球偵測率
        frames_with_ball = sum(1 for fr in frames_data if fr.get('ball_detections'))
        result['ball_detection_rate'] = frames_with_ball / len(frames_data) if frames_data else 0

        # 新版跳過條件：最高連續有球幀數 < 10 才跳過
        # （取代舊版「追蹤中斷 > 50% 則跳過」規則）
        max_consec = DataValidator.max_consecutive_ball_frames(frames_data)
        result['max_consecutive_ball_frames'] = max_consec
        if max_consec < MIN_CONSECUTIVE_BALL_FRAMES:
            result['status'] = 'insufficient_ball_data'
            result['error'] = (
                f'最高連續有球幀數僅 {max_consec} 幀，'
                f'低於門檻 {MIN_CONSECUTIVE_BALL_FRAMES}，跳過此片段'
            )
            return result

    except ValidationError as e:
        result['status'] = 'error'
        result['error'] = str(e)
        return result

    except Exception as e:
        result['status'] = 'error'
        result['error'] = f"載入資料時發生未預期的錯誤: {str(e)}"
        return result

    try:
        
        # 發球偵測
        # net_y 傳入 serve_detector 以啟用 [Method C] net_y 約束：
        # 拋球起點需在 net_y + margin 以上，過濾接球反彈的假陽性
        serve_config = {'hit_v': 40.0, 'toss_vy': 8.0}
        if court_config and 'net_y' in court_config:
            serve_config['net_y'] = court_config['net_y']
        serve_events = analyze_serve_events_v2(
            frames_data,
            config=serve_config,
            use_dynamic_threshold=True,
            first_only=True,
            log_prefix="" if verbose else None,
            image_height=image_height,
            ball_detection_rate=result.get('ball_detection_rate', 1.0)
        )
        
        if not serve_events:
            result['status'] = 'no_serve'
            return result
        
        serve_event = serve_events[0]
        result['serve_detected'] = True
        result['toss_frame'] = serve_event.get('toss_start_frame')
        result['hit_frame'] = serve_event.get('hit_frame_id')
        result['hit_speed'] = serve_event.get('hit_speed', 0)

        # 發球員識別（使用 lookback 方法）
        det_rate = result.get('ball_detection_rate', 1.0)
        server_min_conf = 0.25 if det_rate < 0.5 else 0.4
        server_result = analyze_serve_player(
            frames_data=frames_data,
            serve_event=serve_event,
            method='lookback',
            image_height=image_height,
            exclusion_zones=court_config.get('exclusion_zones') if court_config else None,
            min_confidence=server_min_conf
        )
        
        result['server_index'] = server_result.get('final_server_index')
        result['confidence'] = server_result.get('final_confidence', 0)
        result['found_frame'] = server_result.get('found_frame_id')
        result['frames_searched'] = server_result.get('frames_searched', 0)
        result['status'] = 'success'
        result['toss_height_px'] = serve_event.get('toss_height_px')
        result['min_toss_height_px'] = serve_event.get('min_toss_height_px')
        
        # 跳發偵測
        jump_result = classify_serve_type(
            frames_data=frames_data,
            serve_event=serve_event,
            server_result=server_result,
            court_config=court_config,
            verbose=verbose,
            image_height=image_height
        )
        
        result['serve_type'] = jump_result.get('serve_type', 'unknown')
        result['is_jump_serve'] = jump_result.get('is_jump_serve', False)
        result['jump_confidence'] = jump_result.get('confidence', 0)
        result['jump_height'] = jump_result.get('jump_height', 0)
        result['ground_y'] = jump_result.get('ground_y')

        # Serving side determination
        net_y = court_config.get('net_y', image_height * 0.35) if court_config else image_height * 0.35
        hit_position = serve_event.get('hit_position')
        if hit_position:
            serving_side = determine_serving_side(
                hit_position, image_height=image_height,
                net_position_ratio=net_y / image_height if court_config else 0.35
            )
            result['serving_side'] = serving_side

            # Serve zone detection
            if court_zones and server_result.get('server'):
                server_pos = validate_center_point(
                    server_result['server'].get('center_point')
                )
                if server_pos:
                    result['serve_zone'] = court_zones.get_serve_zone(server_pos, serving_side)

            # Reception detection
            try:
                reception_detector = ReceptionDetector()
                reception_result = reception_detector.analyze_reception(
                    frames_data=frames_data,
                    serve_event=serve_event,
                    net_y=net_y,
                    serving_side=serving_side,
                    court_zones=court_zones,
                )
                result['reception_detected'] = reception_result.get('reception_detected', False)
                result['reception_frame'] = reception_result.get('reception_frame')
                result['reception_zone'] = reception_result.get('reception_zone')
                result['receiver_index'] = reception_result.get('receiver_index')
                result['time_to_reception'] = reception_result.get('time_to_reception')
                result['reception_confidence'] = reception_result.get('confidence', 0)
                result['ball_crossed_net'] = reception_result.get('ball_crossed_net', False)
                result['is_ace'] = reception_result.get('is_ace', False)
                result['landing_zone'] = reception_result.get('landing_zone')
                result['landing_position'] = reception_result.get('landing_position')
            except Exception as e:
                if verbose:
                    print(f"    [WARNING] Reception detection failed: {e}")

        # 儲存圖片
        if save_images:
            cap = cv2.VideoCapture(video_path)
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
                                exclusion_zones=court_config.get('exclusion_zones') if court_config else None,
                                jump_result=jump_result
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
                                exclusion_zones=court_config.get('exclusion_zones') if court_config else None,
                                jump_result=jump_result
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
                            annotated = draw_serve_analysis(
                                frame.copy(), hit_frame_data,
                                serve_event.get('hit_position'),
                                server_result,
                                f"HIT Frame {hit_frame_id}",
                                is_reference=True,
                                exclusion_zones=court_config.get('exclusion_zones') if court_config else None,
                                jump_result=jump_result
                            )
                            output_path = os.path.join(output_dir, f"{video_name}_server_HIT.jpg")
                            cv2.imwrite(output_path, annotated)

                            # 發球瞬間截圖：存至 serve_images/ 子目錄，同時保留乾淨原圖
                            serve_images_dir = os.path.join(output_dir, "serve_images")
                            os.makedirs(serve_images_dir, exist_ok=True)
                            serve_type_tag = jump_result.get('serve_type', 'unknown') if jump_result else 'unknown'
                            moment_filename = f"{video_name}_serve_moment_f{hit_frame_id}_{serve_type_tag}.jpg"
                            moment_path = os.path.join(serve_images_dir, moment_filename)
                            cv2.imwrite(moment_path, annotated)
                            result['serve_moment_image'] = moment_path

                cap.release()
        
    except Exception as e:
        result['status'] = 'error'
        # 清理錯誤訊息中的無法編碼字符
        error_str = str(e).encode('cp950', errors='replace').decode('cp950')
        result['error'] = error_str
    
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
               court_config_path: str = None, court_config_dir: str = None,
               export_excel: bool = False):
    """
    批次測試整個資料夾

    Args:
        video_dir: 影片目錄
        json_dir: JSON 追蹤資料目錄
        output_dir: 輸出目錄
        save_images: 是否儲存圖片
        verbose: 是否顯示詳細訊息
        court_config_path: 場地設定 JSON 路徑（全域，所有影片共用）
        court_config_dir: court_config 目錄（按 group_key 自動匹配）
    """
    print("="*70)
    print("批次測試發球偵測和發球員識別")
    print("="*70)
    print(f"影片目錄: {video_dir}")
    print(f"JSON 目錄: {json_dir}")
    print(f"輸出目錄: {output_dir}")
    
    # 載入場地設定
    # 全域 court_config（所有影片共用）
    global_court_config = load_court_config(court_config_path)
    global_court_zones = None
    use_per_video_config = bool(court_config_dir and os.path.isdir(court_config_dir if court_config_dir else ''))

    if global_court_config:
        exclusion_count = len(global_court_config.get('exclusion_zones', []))
        print(f"場地設定: {court_config_path} (排除區域: {exclusion_count} 個)")
        try:
            global_court_zones = CourtZones(global_court_config)
            print(f"場地分區: 已建立 (net_y={global_court_zones.net_y})")
        except Exception as e:
            print(f"場地分區: 建立失敗 ({e})")
    elif use_per_video_config:
        print(f"場地設定: 按 group_key 自動匹配 ({court_config_dir})")
    else:
        print(f"場地設定: 未指定（不排除任何區域）")
    print()
    
    # 建立輸出目錄
    os.makedirs(output_dir, exist_ok=True)
    
    # 找出匹配的檔案
    matches = find_matching_files(video_dir, json_dir)
    
    if not matches:
        print("[ERROR] 找不到匹配的影片和 JSON 檔案！")
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

        # 決定本影片使用的 court_config
        court_config = global_court_config
        court_zones = global_court_zones

        if not court_config and use_per_video_config:
            parsed = parse_filename(video_name)
            group_key = None
            if parsed and parsed.get('group_key'):
                group_key = parsed['group_key']
            else:
                # fallback: 從路徑中提取 group_key
                group_key = extract_group_key_from_path(video_path)
            if group_key:
                auto_path = os.path.join(court_config_dir, f"{group_key}.json")
                court_config = load_court_config(auto_path)
                if court_config:
                    # 若自動場地偵測品質不可靠，回退至全域 court_config
                    cq = court_config.get('court_detection_quality')
                    if cq == 'unreliable':
                        flags = court_config.get('court_detection_flags', [])
                        flag_str = ', '.join(flags) if flags else 'unknown'
                        print(f"    [WARNING] 自動場地偵測不可靠 ({flag_str})，"
                              f"回退至預設 court_config")
                        court_config = global_court_config
                        court_zones = global_court_zones
                    else:
                        try:
                            court_zones = CourtZones(court_config)
                        except Exception:
                            court_zones = None
                        if verbose:
                            quality_tag = f' [{cq}]' if cq else ''
                            print(f"    [AutoMatch] {group_key}{quality_tag}")

        result = process_single_video(
            video_path, json_path, output_dir,
            save_images=save_images,
            verbose=verbose,
            court_config=court_config,
            court_zones=court_zones
        )
        results.append(result)
        
        # 簡短輸出
        if result['status'] == 'success':
            found = result.get('found_frame', 'N/A')
            searched = result.get('frames_searched', 0)
            serve_type = result.get('serve_type', 'unknown')
            serve_emoji = '[JUMP]' if serve_type == 'jump' else '[STAND]'
            reception_info = ""
            if result.get('reception_detected'):
                rz = result.get('reception_zone', '?')
                reception_info = f", Reception: zone {rz}"
            sz_info = ""
            if result.get('serve_zone'):
                sz_info = f", ServeZone: {result['serve_zone']}"
            print(f"    [SUCCESS] 找到幀: {found} (往回 {searched} 幀), "
                  f"發球員: Player {result['server_index']}, "
                  f"信心度: {result['confidence']:.2f}, "
                  f"{serve_emoji} {serve_type}{sz_info}{reception_info}")
            moment_img = result.get('serve_moment_image')
            if moment_img:
                print(f"    [IMAGE]   發球瞬間圖片: {moment_img}")
        elif result['status'] == 'no_serve':
            print(f"    [WARNING] 未偵測到發球")
        else:
            error_msg = str(result.get('error', result['status']))
            # 移除無法編碼的字符
            error_msg = error_msg.encode('cp950', errors='replace').decode('cp950')
            print(f"    [ERROR] 錯誤: {error_msg}")
    
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
    print(f"  [SUCCESS] 成功偵測: {success} ({success/total*100:.1f}%)")
    print(f"  [WARNING] 未偵測到發球: {no_serve} ({no_serve/total*100:.1f}%)")
    print(f"  [ERROR] 錯誤: {errors} ({errors/total*100:.1f}%)")
    
    # 信心度統計
    confidences = [r['confidence'] for r in results if r['status'] == 'success']
    if confidences:
        print()
        print(f"發球員識別信心度:")
        print(f"  平均: {sum(confidences)/len(confidences):.2f}")
        print(f"  最高: {max(confidences):.2f}")
        print(f"  最低: {min(confidences):.2f}")
    
    # 跳發統計
    jump_serves = [r for r in results if r.get('is_jump_serve', False)]
    standing_serves = [r for r in results if r['status'] == 'success' and not r.get('is_jump_serve', False)]
    
    print()
    print(f"發球類型統計:")
    print(f"  [JUMP] 跳發 (Jump Serve): {len(jump_serves)} ({len(jump_serves)/max(1,success)*100:.1f}%)")
    print(f"  [STAND] 站發 (Standing Serve): {len(standing_serves)} ({len(standing_serves)/max(1,success)*100:.1f}%)")
    
    if jump_serves:
        jump_heights = [r.get('jump_height', 0) for r in jump_serves]
        print(f"  跳發高度:")
        print(f"    平均: {sum(jump_heights)/len(jump_heights):.1f} 像素")
        print(f"    最高: {max(jump_heights):.1f} 像素")

    # 發球區統計
    serve_zone_counts = {}
    for r in results:
        sz = r.get('serve_zone')
        if sz is not None:
            serve_zone_counts[sz] = serve_zone_counts.get(sz, 0) + 1
    if serve_zone_counts:
        print()
        print(f"發球區統計:")
        for z in sorted(serve_zone_counts.keys()):
            zone_names = {1: 'Left', 2: 'Center', 3: 'Right'}
            print(f"  Zone {z} ({zone_names.get(z, '?')}): {serve_zone_counts[z]}")

    # 接球統計
    receptions = [r for r in results if r.get('reception_detected')]
    print()
    print(f"接球偵測統計:")
    print(f"  偵測到接球: {len(receptions)} / {success} ({len(receptions)/max(1,success)*100:.1f}%)")
    if receptions:
        reception_zone_counts = {}
        for r in receptions:
            rz = r.get('reception_zone')
            if rz is not None:
                reception_zone_counts[rz] = reception_zone_counts.get(rz, 0) + 1
        if reception_zone_counts:
            print(f"  接球區分布:")
            zone_names_6 = {1: 'Front-L', 2: 'Front-C', 3: 'Front-R',
                            4: 'Back-L', 5: 'Back-C', 6: 'Back-R'}
            for z in sorted(reception_zone_counts.keys()):
                print(f"    Zone {z} ({zone_names_6.get(z, '?')}): {reception_zone_counts[z]}")

    # 儲存結果到 JSON
    summary = {
        'test_time': datetime.now().isoformat(),
        'video_dir': video_dir,
        'json_dir': json_dir,
        'total': total,
        'success': success,
        'no_serve': no_serve,
        'errors': errors,
        'jump_serves': len(jump_serves),
        'standing_serves': len(standing_serves),
        'results': results
    }
    
    summary_path = os.path.join(output_dir, 'batch_test_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print()
    print(f"詳細結果已儲存: {summary_path}")

    # Export CSV with structured results
    try:
        export_rows = []
        for r in results:
            parsed = parse_filename(r.get('video_name', ''))
            row = build_result_row(
                video_name=r.get('video_name', ''),
                parsed_filename=parsed,
                serve_result=r,
                reception_result=r,
                ball_detection_rate=r.get('ball_detection_rate', 0),
                status=r.get('status', 'unknown'),
            )
            # 強制降級：自動場地偵測品質不佳時覆蓋 quality_grade
            court_quality = r.get('court_detection_quality')
            if court_quality == 'unreliable':
                row['quality_grade'] = 'F'
            elif court_quality == 'degraded' and row.get('quality_grade') in ('A', 'B'):
                row['quality_grade'] = 'C'
            export_rows.append(row)

        csv_path = os.path.join(output_dir, 'batch_results.csv')
        export_to_csv(export_rows, csv_path)
        print(f"CSV 結果已儲存: {csv_path}")

        if export_excel:
            excel_path = os.path.join(output_dir, 'batch_results.xlsx')
            export_to_excel(export_rows, excel_path)
            print(f"Excel 結果已儲存: {excel_path}")

        # Also export summary JSON
        summary_export_path = os.path.join(output_dir, 'batch_results_summary.json')
        export_summary_json(export_rows, summary_export_path)
        print(f"摘要已儲存: {summary_export_path}")
    except Exception as e:
        print(f"[WARNING] Export failed: {e}")
    
    # 列出需要檢查的影片
    need_check = [r for r in results if r['status'] != 'success' or r['confidence'] < 0.5]
    if need_check:
        print()
        print("-"*70)
        print("需要人工檢查的影片:")
        for r in need_check:
            if r['status'] == 'success':
                print(f"  [WARNING] {r['video_name']}: 信心度低 ({r['confidence']:.2f})")
            else:
                print(f"  [ERROR] {r['video_name']}: {r['status']}")
    
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
    parser.add_argument("--court-config-dir", type=str, default=None,
                        help="court_config 目錄，自動按檔名 group_key 匹配")
    parser.add_argument("--no-images", action="store_true",
                        help="不儲存圖片（只輸出統計）")
    parser.add_argument("--verbose", action="store_true",
                        help="顯示詳細處理過程")
    parser.add_argument("--export-excel", action="store_true",
                        help="額外匯出 Excel 格式（需要 openpyxl）")
    
    args = parser.parse_args()
    
    batch_test(
        video_dir=args.video_dir,
        json_dir=args.json_dir,
        output_dir=args.output,
        save_images=not args.no_images,
        verbose=args.verbose,
        court_config_path=args.court_config,
        court_config_dir=args.court_config_dir,
        export_excel=args.export_excel,
    )


if __name__ == "__main__":
    main()