#!/usr/bin/env python3
import os
import cv2
import argparse
import numpy as np
from ultralytics import YOLO
import sys
import json

# --- 1. 設定Python導入路徑，以便找到 court_definition 模組 ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_dir) # video_processing/ 的上一層是 project_root
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- 2. 從 court_definition 模組導入必要的函數 ---
try:
    # define_court_boundaries_manually 主要由 generator 腳本自身使用
    # track_ball_and_player 主要需要 load_court_geometry
    from court_definition.court_config_generator import load_court_geometry, define_court_boundaries_manually # 保留 define 以便首次設定
    from video_processing.event_analyzer import find_serve_events 
    # 如果 test_serve_detection.py 需要其他輔助函數，它們仍在 event_analyzer.py 中可以被導入
except ImportError as e:
    print(f"錯誤: 無法導入 court_definition 模組中的函數: {e}")
    print("請確保 court_definition 資料夾和其中的 court_config_generator.py 檔案存在，並且路徑設定正確。")
    print("同時，建議在 court_definition 資料夾中加入一個空的 __init__.py 檔案。")
    sys.exit(1)

# --- 3. 定義命令行參數 ---
def parse_args():
    parser = argparse.ArgumentParser(description="對沙灘排球影片進行球和球員追蹤分析。")
    parser.add_argument("--input", type=str, required=True, help="輸入的短影片片段路徑")
    parser.add_argument("--output_dir", type=str, default="output_data/tracking_output", 
                        help="追蹤結果的輸出根目錄 (相對於專案根目錄)")
    # 模型路徑預設相對於專案根目錄下的 models/ 資料夾
    parser.add_argument("--ball_model", type=str, default="model/ball_best.pt", 
                        help="排球偵測模型路徑 (相對於專案根目錄)")
    parser.add_argument("--player_model", type=str, default="model/yolov8s-pose.pt", 
                        help="球員偵測模型路徑 (相對於專案根目錄)")
    parser.add_argument("--conf", type=float, default=0.3, help="物件偵測的置信度閾值")
    parser.add_argument("--device", type=str, default="0", help="推理設備: 'cpu' 或 GPU id")
    parser.add_argument("--config_file_name", type=str, default="court_config.json",
                        help="場地設定檔名稱 (位於專案根目錄)")
    return parser.parse_args()

# --- 背景分割器初始化 (與你原有的相同) ---
def init_bg_subtractor():
    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    return fgbg, kernel

# --- 球體偵測函數 (與你原有的相同，假設仍使用 fgmask) ---
def detect_ball(frame, ball_model, conf_thresh, background_ball_zones=None):
    res_ball = ball_model(frame, conf=conf_thresh, classes=[0])[0]
    ball_boxes = []
    for box in res_ball.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        # 檢查是否在背景球過濾區域內
        if background_ball_zones:
            in_background_zone = False
            for zone in background_ball_zones:
                if (zone["x1"] <= center_x <= zone["x2"] and 
                    zone["y1"] <= center_y <= zone["y2"]):
                    in_background_zone = True
                    break
            if in_background_zone:
                continue
        
        conf_score = float(box.conf[0].cpu().numpy())
        ball_boxes.append((x1, y1, x2, y2, conf_score))
    return ball_boxes

# --- 4. 修改後的球員偵測與姿態估計函數 ---
def detect_players_with_pose(frame, player_pose_model, conf_thresh,
                               court_center_xy,
                               court_boundary_polygon_np,
                               exclusion_zones_np_list):
    """
    使用YOLOv8-Pose偵測球員並提取姿態關鍵點。
    然後結合場地幾何資訊進行篩選。
    返回:
        - simple_player_boxes: list of (x1,y1,x2,y2,conf) 用於快速繪圖/舊標籤格式
        - detailed_player_info: list of dictionaries, 每個字典包含box, conf, pose, is_inside, dist_center等
    """
    results = player_pose_model(frame, conf=conf_thresh, classes=[0])[0] # 假設 'person' class is 0

    candidate_persons_detailed = []
    if results.boxes and results.keypoints: # 確保同時有邊界框和關鍵點結果
        for i in range(len(results.boxes)):
            box_data = results.boxes[i]
            keypoints_data = results.keypoints[i] # 對應的關鍵點數據

            x1, y1, x2, y2 = map(int, box_data.xyxy[0].cpu().numpy())
            confidence = float(box_data.conf[0].cpu().numpy()) # 人物偵測的信心度

            # 提取姿態關鍵點 (x, y, confidence_kpt)
            # keypoints_data.data 是一個 tensor，需要轉換
            # keypoints_data.xy 返回一個 (N, K, 2) 的 tensor, N是人數(通常是1因為我們在循環內), K是關鍵點數, 2是xy
            # keypoints_data.conf 返回一個 (N, K) 的 tensor
            person_keypoints_list = []
            if keypoints_data is not None and keypoints_data.xy is not None and keypoints_data.conf is not None:
                kpts_xy = keypoints_data.xy[0].cpu().numpy() # (K, 2)
                kpts_conf = keypoints_data.conf[0].cpu().numpy() # (K,)
                for kp_idx in range(kpts_xy.shape[0]):
                    person_keypoints_list.append([
                        float(kpts_xy[kp_idx, 0]),
                        float(kpts_xy[kp_idx, 1]),
                        float(kpts_conf[kp_idx])
                    ])

            person_cx = (x1 + x2) / 2
            person_cy = (y1 + y2) / 2
            person_center_for_test = (float(person_cx), float(person_cy))

            in_exclusion_zone = False
            if exclusion_zones_np_list:
                for zone_poly in exclusion_zones_np_list:
                    if cv2.pointPolygonTest(zone_poly, person_center_for_test, False) >= 0:
                        in_exclusion_zone = True
                        break
            if in_exclusion_zone:
                continue

            is_inside_court = False
            if isinstance(court_boundary_polygon_np, np.ndarray) and court_boundary_polygon_np.ndim == 2 and court_boundary_polygon_np.shape[0] >= 3:
                is_inside_court = cv2.pointPolygonTest(court_boundary_polygon_np, person_center_for_test, False) >= 0
            
            distance_to_court_center = float('inf')
            if court_center_xy:
                distance_to_court_center = np.sqrt((person_cx - court_center_xy[0])**2 + \
                                                   (person_cy - court_center_xy[1])**2)
            
            candidate_persons_detailed.append({
                "box_coords": (x1, y1, x2, y2),
                "confidence_person": confidence, # 這是人物偵測的信心度
                "is_inside_court": is_inside_court,
                "distance_to_center": distance_to_court_center,
                "center_point": (person_cx, person_cy),
                "pose_keypoints_data": { # 儲存原始的關鍵點數據
                    "keypoints_xyc_list": person_keypoints_list, # [[x,y,conf], ...]
                    # "normalized_keypoints_xyn_list": keypoints_data.xyn[0].cpu().numpy().tolist() if keypoints_data.xyn is not None else [],
                    # "original_keypoints_object": keypoints_data # 或者儲存整個 Ultralytics Keypoints 物件 (如果方便後續處理)
                }
            })
        
    def sort_key_for_player_selection(person):
        priority_inside = 0 if person["is_inside_court"] else 1
        return (priority_inside, person["distance_to_center"])

    sorted_persons = sorted(candidate_persons_detailed, key=sort_key_for_player_selection)
    selected_players_data = sorted_persons[:4] # 選取最多4位
    
    output_player_boxes = []
    for player_data in selected_players_data:
        # 返回的格式與原來的 detect_player 保持一致，方便 draw_and_save
        # 但如果 all_frames_data 需要更多資訊，這裡可以返回 player_data (字典)
        output_player_boxes.append((*player_data["box_coords"], player_data["confidence_person"]))
        
    return output_player_boxes, selected_players_data # 同時返回詳細數據用於 all_frames_data

# --- 繪製函數 (可以增加姿態繪製) ---
def draw_all_detections(frame_to_draw, ball_boxes, player_boxes_simple, detailed_player_info_list, court_poly_np, exclusion_zones_np_list):
    # 繪製場地邊界 (可選)
    if court_poly_np is not None:
        cv2.polylines(frame_to_draw, [court_poly_np], True, (0,255,0), 1)
    # 繪製排除區域 (可選)
    for ex_zone in exclusion_zones_np_list:
        cv2.polylines(frame_to_draw, [ex_zone], True, (255,0,255), 1)

    # 繪製球 (藍色)
    for (x1,y1,x2,y2,conf) in ball_boxes:
        cv2.rectangle(frame_to_draw, (x1,y1), (x2,y2), (255,0,0), 2)
        cv2.putText(frame_to_draw, f"B {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

    # 繪製球員邊界框 (黃色) 和姿態 
    # Ultralytics YOLOv8 的 Results 物件有 plot() 方法可以直接繪製 bbox 和 pose
    # 但既然我們自己篩選了球員，就需要手動繪製
    if detailed_player_info_list: # 使用詳細資訊來繪製姿態
        for p_data in detailed_player_info_list: # 這裡 p_data 是 selected_players_detailed_info 中的元素
            x1, y1, x2, y2 = p_data["box_coords"]
            conf = p_data["confidence_person"]
            cv2.rectangle(frame_to_draw, (x1,y1), (x2,y2), (0,255,255), 2) # 球員框用黃色
            cv2.putText(frame_to_draw, f"P {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

            # 繪製姿態關鍵點和骨骼 (COCO 17 個關鍵點的連接方式)
            kpts_list = p_data["pose_keypoints_data"]["keypoints_xyc_list"]
            if kpts_list and len(kpts_list) == 17: # 確保是17個COCO關鍵點
                # COCO的骨骼連接對
                skeleton = [
                    [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7],
                    [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]
                ]
                # 顏色定義
                limb_color = (0, 255, 0) # 綠色骨骼
                kpt_color = (0, 0, 255)  # 紅色關鍵點

                # 畫關鍵點
                for kpt_x, kpt_y, kpt_c in kpts_list:
                    if kpt_c > 0.5: # 只畫信心度高的關鍵點
                        cv2.circle(frame_to_draw, (int(kpt_x), int(kpt_y)), 3, kpt_color, -1)
                
                # 畫骨骼
                for bone in skeleton:
                    idx1, idx2 = bone[0]-1, bone[1]-1 # 轉為0-based index
                    if idx1 < len(kpts_list) and idx2 < len(kpts_list):
                        pt1_x, pt1_y, pt1_c = kpts_list[idx1]
                        pt2_x, pt2_y, pt2_c = kpts_list[idx2]
                        if pt1_c > 0.5 and pt2_c > 0.5: # 只有當兩個端點都可信時才畫線
                            cv2.line(frame_to_draw, (int(pt1_x), int(pt1_y)), (int(pt2_x), int(pt2_y)), limb_color, 2)
    # 返回YOLO標籤 (如果需要)
    yolo_labels = []
    frame_w, frame_h = frame_to_draw.shape[1], frame_to_draw.shape[0]
    for (x1,y1,x2,y2,conf) in ball_boxes:
        cx = ((x1 + x2) / 2) / frame_w; cy = ((y1 + y2) / 2) / frame_h
        bw = (x2 - x1) / frame_w; bh = (y2 - y1) / frame_h
        yolo_labels.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}") # 球標籤0
    for (x1,y1,x2,y2,conf) in player_boxes_simple: # 使用 simple_player_boxes 生成標籤
        cx = ((x1 + x2) / 2) / frame_w; cy = ((y1 + y2) / 2) / frame_h
        bw = (x2 - x1) / frame_w; bh = (y2 - y1) / frame_h
        yolo_labels.append(f"1 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}") # 球員標籤1
    return yolo_labels

# --- 5. 主函數 ---
def main():
    args = parse_args()

    # --- 載入和處理場地幾何設定 ---
    config_file_path = os.path.join(project_root, args.config_file_name) 
    background_ball_zones = []
    if os.path.exists(config_file_path):
        with open(config_file_path, 'r') as f:
            court_config = json.load(f)
            background_ball_zones = court_config.get("background_ball_zones", [])
    
    if not os.path.exists(config_file_path):
        print(f"警告：場地設定檔 '{os.path.abspath(config_file_path)}' 不存在。")
        # 為了讓 track_ball_and_player 更專注於分析，建議在此不調用 define_court_boundaries_manually
        # 而是提示用戶先運行 court_config_generator.py
        print(f"請先從專案根目錄執行: python court_definition/court_config_generator.py <用於定義的範例影片路徑>")
        print("來生成 court_config.json 檔案。")
        print("如果希望在沒有場地設定的情況下繼續 (僅進行基礎偵測)，請手動創建一個空的 court_config.json 或修改程式碼。")
        # return # 可以選擇終止，或者允許在沒有場地資訊的情況下運行（但功能會受限）
        court_geometry = {} # 允許在沒有設定檔的情況下繼續，但功能會受限
    else:
        court_geometry = load_court_geometry(config_load_path=config_file_path)
        if court_geometry is None: # load_court_geometry 可能因格式錯誤返回 None
            print(f"錯誤：無法從 '{os.path.abspath(config_file_path)}' 正確載入場地幾何資訊。")
            court_geometry = {} # 設為空字典以避免後續 NoneType 錯誤，但功能受限
        else:
            print(f"成功載入場地資訊。")
    
    # --- 準備幾何資訊給偵測函數 ---
    court_center_xy_tuple = None
    court_boundary_poly_np = None
    exclusion_zones_np_list = []

    if "court_boundary_polygon" in court_geometry and court_geometry["court_boundary_polygon"]:
        points = court_geometry["court_boundary_polygon"]
        if isinstance(points, list) and len(points) >= 3:
            court_boundary_poly_np = np.array(points, dtype=np.int32)
            try:
                M = cv2.moments(court_boundary_poly_np)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    court_center_xy_tuple = (cx, cy)
                elif points: 
                    cx = sum(p[0] for p in points) / len(points)
                    cy = sum(p[1] for p in points) / len(points)
                    court_center_xy_tuple = (cx, cy)
                if court_center_xy_tuple: print(f"計算得到的場地中心: ({court_center_xy_tuple[0]:.0f}, {court_center_xy_tuple[1]:.0f})")
            except Exception as e:
                print(f"計算場地中心時出錯: {e}.")
                if points and len(points) > 0:
                    cx = sum(p[0] for p in points) / len(points)
                    cy = sum(p[1] for p in points) / len(points)
                    court_center_xy_tuple = (cx, cy)
                    if court_center_xy_tuple: print(f"備用場地中心 (平均值): ({court_center_xy_tuple[0]:.0f}, {court_center_xy_tuple[1]:.0f})")
        else: print("警告: 'court_boundary_polygon' 無效。")
    else: print("警告: 未定義 'court_boundary_polygon'。")

    loaded_exclusion_zones = court_geometry.get("exclusion_zones", [])
    if isinstance(loaded_exclusion_zones, list):
        for zone_points in loaded_exclusion_zones:
            if isinstance(zone_points, list) and len(zone_points) >= 3:
                exclusion_zones_np_list.append(np.array(zone_points, dtype=np.int32))
        if exclusion_zones_np_list: print(f"載入了 {len(exclusion_zones_np_list)} 個排除區域。")
    elif loaded_exclusion_zones: print("警告: 'exclusion_zones' 格式不正確，應為列表的列表。")
    
    # --- 初始化模型 ---
    device_to_use = f"cuda:{args.device}" if args.device.isdigit() else args.device
    # 將模型路徑與專案根目錄結合，確保能正確找到
    abs_ball_model_path = os.path.join(project_root, args.ball_model)
    abs_player_model_path = os.path.join(project_root, args.player_model)
    try:
        print(f"正在加載球體模型: {abs_ball_model_path} 到設備 {device_to_use}")
        ball_model = YOLO(abs_ball_model_path).to(device_to_use)
        print(f"正在加載球員模型: {abs_player_model_path} 到設備 {device_to_use}")
        player_model = YOLO(abs_player_model_path).to(device_to_use)
        print("模型加載成功。")
    except Exception as e:
        print(f"錯誤: 加載YOLO模型失敗: {e}")
        return

    # --- 修改：創建輸出目錄，使其直接使用影片名稱 ---
    video_name_base = os.path.splitext(os.path.basename(args.input))[0]

    # 確保 args.output_dir (例如 "output_data/tracking_output") 是相對於專案根目錄的
    # specific_video_output_dir 將直接是 args.output_dir
    base_output_folder_for_tracking = os.path.join(project_root, args.output_dir)
    specific_video_output_dir = os.path.join(base_output_folder_for_tracking, video_name_base)

    os.makedirs(specific_video_output_dir, exist_ok=True)  # 創建輸出資料夾
    print(f"所有輸出將儲存到: {specific_video_output_dir}")

    # (可選) 如果你仍然需要 frames 和 labels 子目錄，在這裡創建它們
    frames_output_sub_dir = os.path.join(specific_video_output_dir, "frames")
    labels_output_sub_dir = os.path.join(specific_video_output_dir, "labels")
    os.makedirs(frames_output_sub_dir, exist_ok=True)
    os.makedirs(labels_output_sub_dir, exist_ok=True)

    annotated_video_path = None
    annotated_video_path = os.path.join(specific_video_output_dir, f"{video_name_base}_annotated.avi")
    # --- 輸出目錄修改結束 ---


    # --- 處理影片 ---
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"錯誤: 無法打開輸入影片 {args.input}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(annotated_video_path, fourcc, fps, (frame_w, frame_h))
    print(f"將儲存標註影片到: {annotated_video_path}")

    fgbg, kernel = init_bg_subtractor() # 初始化背景分割器 (如果 detect_ball 需要)
    all_frames_data = []
    frame_count = 0
    print(f"\n開始處理影片: {args.input}")

    while True:
        ret, frame_orig = cap.read()
        if not ret:
            break
        
        frame_count += 1
        frame_to_draw_on = frame_orig.copy() # 用於繪製的副本

        # 1) 運動前景分割 (如果 detect_ball 需要)
        fgmask = None # 初始化
        if fgbg is not None and kernel is not None : # 確保已初始化
             fgmask = fgbg.apply(frame_orig)
             fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

        # 2) 排球检测 (假設它仍使用 fgmask)
        ball_boxes = detect_ball(frame_orig, ball_model, args.conf, background_ball_zones)
        
        # 3) 選手检测 (使用增強版函數)-
        player_boxes_for_drawing, detailed_player_infos = detect_players_with_pose(
            frame_orig, player_model, args.conf, # 使用 player_model
            court_center_xy_tuple,
            court_boundary_poly_np,
            exclusion_zones_np_list
        )
        
        # 4) 儲存數據到 all_frames_data
        current_frame_info = {
            "frame_id": frame_count,
            "ball_detections": [], 
            "player_detections": [] # 現在會包含姿態資訊
        }
        for (x1,y1,x2,y2,conf) in ball_boxes:
            current_frame_info["ball_detections"].append({
                "center_x": float((x1+x2)/2), # 確保是 float
                "center_y": float((y1+y2)/2), # 確保是 float
                "confidence": round(float(conf), 4), # 確保是 float
                "box": [int(x1), int(y1), int(x2), int(y2)] # <--- 明確轉換為 Python int
            })
        
        for p_detailed_info in detailed_player_infos: # 使用詳細資訊儲存
            current_frame_info["player_detections"].append({
                "box_coords": p_detailed_info["box_coords"], 
                "confidence_person": round(p_detailed_info["confidence_person"], 4),
                "is_inside_court": p_detailed_info["is_inside_court"],
                "distance_to_center": round(p_detailed_info["distance_to_center"], 2) if p_detailed_info["distance_to_center"] != float('inf') else None,
                "center_point": (round(p_detailed_info["center_point"][0], 2), round(p_detailed_info["center_point"][1], 2) ),
                "pose_keypoints": p_detailed_info["pose_keypoints_data"] # 儲存姿態數據
            })
        all_frames_data.append(current_frame_info)

        # --- 繪製與儲存標籤和幀 (如果需要) ---
        # 如果你仍然需要單獨的 frames/ 和 labels/ 子目錄用於YOLO格式數據集：
        raw_frame_path = os.path.join(frames_output_sub_dir, f"frame_{frame_count:06d}.jpg")
        cv2.imwrite(raw_frame_path, frame_orig)

        # --- 修改：總是進行繪製，並由 out_writer 是否存在決定是否寫入 ---
        yolo_labels_for_frame = draw_all_detections(
            frame_to_draw_on,
            ball_boxes,
            player_boxes_for_drawing,
            detailed_player_infos,
            court_boundary_poly_np,
            exclusion_zones_np_list
        )
        if out_writer: # 只有 out_writer 成功初始化才寫入
            out_writer.write(frame_to_draw_on)
        # --- 繪製邏輯修改結束 ---
        
        if frame_count % int(fps if fps > 0 else 30) == 0:
            print(f"  已處理 {frame_count} 幀...")
    
    # --- 迴圈結束後 ---
    cap.release()
    if out_writer: 
        out_writer.release()
        print(f"已儲存帶標註（含姿態）的影片到: {annotated_video_path}") # annotated_video_path 在前面定義
    cv2.destroyAllWindows()
    
    # --- 儲存 all_frames_data (檔名和邏輯與上一回應相同) ---
    if all_frames_data:
        all_frames_data_filename = os.path.join(specific_video_output_dir, f"{video_name_base}_all_frames_data_with_pose.json")
        try:
            with open(all_frames_data_filename, 'w') as f:
                json.dump(all_frames_data, f, indent=2)
            print(f"已將 all_frames_data (含姿態) 儲存到: {all_frames_data_filename}")
        except Exception as e:
            print(f"儲存 all_frames_data 時發生錯誤: {e}")
    
    print(f"影片 '{args.input}' 處理完成。共處理 {frame_count} 幀。")
    print(f"所有幀的偵測數據已收集 (共 {len(all_frames_data)} 條記錄)。")

    # --- 後續分析 ---
    if all_frames_data and court_geometry:
        print(f"\n開始進行事件分析...")
        # serve_events = find_serve_events(all_frames_data, court_geometry, fps, frame_w, frame_h)
        # ...
    # ...

    # --- 後續分析 ---
    if all_frames_data and court_geometry:
        print(f"\n開始進行事件分析...")
        serve_events = find_serve_events(all_frames_data, court_geometry)
        print(f"分析完成，共找到 {len(serve_events)} 個發球事件")

if __name__ == "__main__":
    main()
