# video_processing/test_serve_detection.py
import json
import os
import sys
import cv2
import numpy as np

# --- 設定Python導入路徑 ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- 從模組導入函數 ---
try:
    from court_definition.court_config_generator import load_court_geometry
    from video_processing.event_analyzer import (
        get_player_center,
        get_baselines_from_ordered_corners,
        is_point_outside_line_segment_extended,
        is_player_behind_baseline,
        find_serve_events
    )
except ImportError as e:
    print(f"導入函數時發生錯誤: {e}")
    sys.exit(1)

# --- (可選) 視覺化輔助函數 ---
def visualize_court_and_points(image_to_draw_on, court_polygon, baselines, points_to_test_dict, title="Court Visualization"):
    """在影像上繪製場地、底線和測試點"""
    vis_img = image_to_draw_on.copy()
    # 繪製場地邊界 (藍色)
    if court_polygon and len(court_polygon) == 4:
        cv2.polylines(vis_img, [np.array(court_polygon, dtype=np.int32)], True, (255, 0, 0), 2)

    # 繪製底線 (紅色)
    if baselines and baselines[0] and baselines[1]: # far_baseline, near_baseline
        cv2.line(vis_img, baselines[0][0], baselines[0][1], (0, 0, 255), 2) # Far
        cv2.line(vis_img, baselines[1][0], baselines[1][1], (0, 0, 255), 2) # Near

    # 繪製測試點
    for point_name, data in points_to_test_dict.items():
        coords = data["coords"]
        is_behind = data.get("is_behind_baseline_result", None) # 獲取測試結果
        color = (0, 255, 0) if is_behind is True else ((0, 0, 255) if is_behind is False else (255,255,255)) # 綠:True, 紅:False, 白:未測
        cv2.circle(vis_img, coords, 7, color, -1)
        cv2.putText(vis_img, point_name, (coords[0] + 10, coords[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    cv2.imshow(title, vis_img)
    print(f"顯示 '{title}'。按任意鍵繼續下一個測試或結束...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# --- 主要測試執行函數 ---
def run_all_tests():
    print("--- 開始 event_analyzer.py 函數測試 ---")

    # --- 測試參數設定 ---
    video_name_for_data = "segment_170"  # <<--- ★★★ 修改為你生成 _all_frames_data.json 的影片基本名 ★★★
    config_file_name = "court_config.json"
    
    # 影片基本資訊 (必須與生成 _all_frames_data.json 的影片一致)
    test_video_fps = 25      # <<--- ★★★ 修改為你的影片 FPS ★★★
    test_video_width = 640    # <<--- ★★★ 修改為你的影片寬度 ★★★
    test_video_height = 640    # <<--- ★★★ 修改為你的影片高度 ★★★

    # 檔案路徑
    all_frames_data_file_path = os.path.join(project_root, "output_data", "tracking_output",
                                           f"{video_name_for_data}",
                                           f"{video_name_for_data}_all_frames_data.json")
    court_config_file_path = os.path.join(project_root, config_file_name)
    sample_video_for_vis_path = os.path.join(project_root, "input_video", f"{video_name_for_data}.avi") # 假設與json同名

    # 載入場地設定
    court_geometry = load_court_geometry(config_load_path=court_config_file_path)
    if court_geometry is None or "court_boundary_polygon" not in court_geometry:
        print(f"錯誤：無法從 '{court_config_file_path}' 載入有效的場地邊界資訊。測試終止。")
        return
    court_polygon = court_geometry["court_boundary_polygon"]
    if not isinstance(court_polygon, list) or len(court_polygon) != 4:
        print(f"錯誤：court_boundary_polygon 格式不正確或不是4個點。測試終止。")
        return
        
    # 計算場地近似中心 (用於 is_player_behind_baseline)
    court_center_approx = None
    try:
        poly_np = np.array(court_polygon, dtype=np.int32)
        M = cv2.moments(poly_np)
        if M["m00"] != 0:
            court_center_approx = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        elif court_polygon:
            court_center_approx = (
                int(sum(p[0] for p in court_polygon) / len(court_polygon)),
                int(sum(p[1] for p in court_polygon) / len(court_polygon))
            )
    except Exception: pass # 忽略計算錯誤，後面函數會處理 None


    # --- 1. 測試 get_player_center ---
    print("\n--- 1. 測試 get_player_center ---")
    box1_pc = (100, 100, 200, 200, 0.9)
    center1_x_pc, center1_y_pc = get_player_center(box1_pc)
    print(f"  Box1: {box1_pc} -> Center: ({center1_x_pc}, {center1_y_pc}) (預期: 150.0, 150.0)")
    assert center1_x_pc == 150.0 and center1_y_pc == 150.0, "get_player_center 測試失敗 for box1"
    
    box2_pc = [50, 50, 150, 100] # 測試沒有信心度的情況
    center2_x_pc, center2_y_pc = get_player_center(box2_pc)
    print(f"  Box2: {box2_pc} -> Center: ({center2_x_pc}, {center2_y_pc}) (預期: 100.0, 75.0)")
    assert center2_x_pc == 100.0 and center2_y_pc == 75.0, "get_player_center 測試失敗 for box2"
    print("get_player_center 測試通過 (基本情況)。")


    # --- 2. 測試 get_baselines_from_ordered_corners ---
    print("\n--- 2. 測試 get_baselines_from_ordered_corners ---")
    # 使用從 court_config.json 載入的 court_polygon
    far_baseline, near_baseline = get_baselines_from_ordered_corners(court_polygon)
    if far_baseline and near_baseline:
        print(f"  從場地邊界 {court_polygon} 提取的：")
        print(f"    遠端底線 (P0-P3): {far_baseline}")
        print(f"    近端底線 (P1-P2): {near_baseline}")
        # 這裡你可以根據你的 court_polygon 手動驗證 P0,P1,P2,P3 是否正確對應
    else:
        print(f"  錯誤：未能從 {court_polygon} 提取底線。")
    # 可以在視覺化時畫出這兩條線來確認


    # --- 3. & 4. 測試 is_point_outside_line_segment_extended 和 is_player_behind_baseline ---
    print("\n--- 3. & 4. 測試 is_player_behind_baseline (及其輔助函數) ---")
    # 載入第一幀用於視覺化
    cap_vis = cv2.VideoCapture(sample_video_for_vis_path)
    vis_frame = None
    if cap_vis.isOpened():
        ret_vis, vis_frame = cap_vis.read()
        if not ret_vis: vis_frame = None
    cap_vis.release()

    if vis_frame is None:
        print(f"警告：無法載入影片 '{sample_video_for_vis_path}' 的第一幀進行視覺化測試。")
        # 即使沒有視覺化，也可以進行邏輯測試
        vis_frame_for_drawing = np.zeros((test_video_height, test_video_width, 3), dtype=np.uint8) # 創建黑色背景
    else:
        vis_frame_for_drawing = vis_frame

    # ★★★★★ 根據你的 court_polygon 和視角，精心設計以下測試點 ★★★★★
    points_to_test_ipbb = {
        "InsideCenter": {"coords": court_center_approx if court_center_approx else (test_video_width//2, test_video_height//2)},
        "BehindNearBaseline": {"coords": ((near_baseline[0][0] + near_baseline[1][0]) // 2, 
                                          (near_baseline[0][1] + near_baseline[1][1]) // 2 + 30) if near_baseline else (0,0)},
        "BehindFarBaseline": {"coords": ((far_baseline[0][0] + far_baseline[1][0]) // 2, 
                                         (far_baseline[0][1] + far_baseline[1][1]) // 2 - 30) if far_baseline else (0,0)},
        "OutsideSidelineNear": {"coords": (near_baseline[0][0] - 50, (near_baseline[0][1] + near_baseline[1][1]) // 2) if near_baseline else (0,0)},
        # 添加更多你認為重要的邊界測試點...
    }
    
    results_ipbb = {}
    for name, data in points_to_test_ipbb.items():
        if data["coords"] == (0,0) and (not far_baseline or not near_baseline):
            print(f"  跳過測試點 '{name}' 因底線未定義。")
            results_ipbb[name] = None # 標記為未執行
            continue
        
        # is_player_behind_baseline 需要 frame_h, frame_w, court_center_approx
        result = is_player_behind_baseline(data["coords"], court_polygon, test_video_height, test_video_width, court_center_approx)
        print(f"  測試點 '{name}' at {data['coords']} -> is_behind_baseline: {result}")
        results_ipbb[name] = result
        points_to_test_ipbb[name]["is_behind_baseline_result"] = result # 更新字典以便視覺化

    if vis_frame is not None or court_polygon: # 只要有參考物就顯示
        visualize_court_and_points(vis_frame_for_drawing, court_polygon, (far_baseline, near_baseline), points_to_test_ipbb, "is_player_behind_baseline Test")


    # --- 5. 測試 find_serve_events ---
    print("\n--- 5. 測試 find_serve_events ---")
    if not os.path.exists(all_frames_data_file_path):
        print(f"錯誤: 發球事件測試所需的數據檔案 '{all_frames_data_file_path}' 不存在。跳過此測試。")
    else:
        try:
            with open(all_frames_data_file_path, 'r') as f:
                all_frames_data_loaded = json.load(f)
            print(f"  已從 '{all_frames_data_file_path}' 載入 {len(all_frames_data_loaded)} 幀數據。")

            serve_events = find_serve_events(all_frames_data_loaded, court_geometry, 
                                             test_video_fps, test_video_width, test_video_height)
            
            if serve_events:
                print(f"  --- 偵測到 {len(serve_events)} 個發球事件 ---")
                for idx, event in enumerate(serve_events):
                    p_info = event.get('serving_player_info', {})
                    p_box = p_info.get('box_coords', [0,0,0,0])
                    b_start = event.get('ball_start_position', (0,0))
                    print(f"    事件 {idx+1}: Frame {event.get('serve_frame_idx')}, PlayerBox {p_box[:2]}, BallStart {b_start}")
            else:
                print("  在此數據中未偵測到發球事件。")

        except Exception as e:
            print(f"  執行 find_serve_events 時發生錯誤: {e}")

    print("\n--- 所有測試結束 ---")

if __name__ == "__main__":
    run_all_tests()