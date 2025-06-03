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
    parser.add_argument("--player_model", type=str, default="model/player_yolo.pt", 
                        help="球員偵測模型路徑 (相對於專案根目錄)")
    parser.add_argument("--conf", type=float, default=0.3, help="物件偵測的置信度閾值")
    parser.add_argument("--device", type=str, default="0", help="推理設備: 'cpu' 或 GPU id")
    parser.add_argument("--save_annotated_video", action='store_true',default=True, help="是否儲存帶有標註的影片")
    parser.add_argument("--config_file_name", type=str, default="court_config.json", 
                        help="場地設定檔名稱 (位於專案根目錄)")
    return parser.parse_args()

# --- 背景分割器初始化 (與你原有的相同) ---
def init_bg_subtractor():
    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    return fgbg, kernel

# --- 球體偵測函數 (與你原有的相同，假設仍使用 fgmask) ---
def detect_ball(frame, ball_model, fgmask, kernel, conf_thresh):
    res_ball = ball_model(frame, conf=conf_thresh, classes=[0])[0] # 假設球是 class 0
    ball_boxes = []
    for box in res_ball.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        roi = fgmask[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        motion_ratio = np.count_nonzero(roi) / roi.size
        if motion_ratio < 0.02: # 運動比例閾值，過濾靜止球
            continue
        conf_score = float(box.conf[0].cpu().numpy())
        ball_boxes.append((x1, y1, x2, y2, conf_score))
    return ball_boxes

# --- 4. 增強版的球員偵測函數 ---
def detect_players_enhanced(frame, player_model, conf_thresh, 
                            court_center_xy, 
                            court_boundary_polygon_np, 
                            exclusion_zones_np_list):
    results = player_model(frame, conf=conf_thresh, classes=[0])[0] # 假設 person class is 0
    
    candidate_persons = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        confidence = float(box.conf[0].cpu().numpy())
        
        person_cx = (x1 + x2) / 2
        person_cy = (y1 + y2) / 2
        person_center_for_test = (float(person_cx), float(person_cy))

        in_exclusion_zone = False
        if exclusion_zones_np_list: # 檢查是否為 None 或空
            for zone_poly in exclusion_zones_np_list:
                if cv2.pointPolygonTest(zone_poly, person_center_for_test, False) >= 0:
                    in_exclusion_zone = True
                    break
        if in_exclusion_zone:
            continue

        is_inside_court = False
        # 確保 court_boundary_polygon_np 是有效的 NumPy 陣列且至少有3個點
        if isinstance(court_boundary_polygon_np, np.ndarray) and court_boundary_polygon_np.ndim == 2 and court_boundary_polygon_np.shape[0] >= 3:
            is_inside_court = cv2.pointPolygonTest(court_boundary_polygon_np, person_center_for_test, False) >= 0
        
        distance_to_court_center = float('inf') 
        if court_center_xy: # 確保 court_center_xy 不是 None
            distance_to_court_center = np.sqrt((person_cx - court_center_xy[0])**2 + \
                                               (person_cy - court_center_xy[1])**2)
        
        candidate_persons.append({
            "box_coords": (x1, y1, x2, y2),
            "confidence": confidence,
            "is_inside_court": is_inside_court,
            "distance_to_center": distance_to_court_center,
            "center_point": (person_cx, person_cy) # 保留以備調試或後續使用
        })
        
    def sort_key_for_player_selection(person):
        priority_inside = 0 if person["is_inside_court"] else 1
        return (priority_inside, person["distance_to_center"])

    sorted_persons = sorted(candidate_persons, key=sort_key_for_player_selection)
    selected_players_data = sorted_persons[:4] # 選取最多4位
    
    output_player_boxes = []
    for player_data in selected_players_data:
        # 返回的格式與原來的 detect_player 保持一致，方便 draw_and_save
        # 但如果 all_frames_data 需要更多資訊，這裡可以返回 player_data (字典)
        output_player_boxes.append((*player_data["box_coords"], player_data["confidence"]))
        
    return output_player_boxes, selected_players_data # 同時返回詳細數據用於 all_frames_data

# --- 繪製與儲存標籤 (與你原有的 draw_and_save 類似，但幀的保存移到主迴圈) ---
def draw_boxes_and_get_labels(frame_to_draw, ball_boxes, player_boxes, w, h):
    txt_lines = []
    # 繪製排球 (class 0, 藍色)
    for (x1, y1, x2, y2, conf) in ball_boxes:
        cx = ((x1 + x2) / 2) / w
        cy = ((y1 + y2) / 2) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        txt_lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}") # 球標籤為0
        cv2.rectangle(frame_to_draw, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame_to_draw, f"Ball {conf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # 繪製選手 (class 1, 黃色)
    for (x1, y1, x2, y2, conf) in player_boxes: # player_boxes 現在是 (x1,y1,x2,y2,conf) 格式
        cx = ((x1 + x2) / 2) / w
        cy = ((y1 + y2) / 2) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        txt_lines.append(f"1 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}") # 球員標籤為1
        cv2.rectangle(frame_to_draw, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(frame_to_draw, f"Player {conf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return txt_lines

# --- 5. 主函數 ---
def main():
    args = parse_args()

    # --- 載入和處理場地幾何設定 ---
    config_file_path = os.path.join(project_root, args.config_file_name) 

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
    # 而 specific_output_dir 將是 args.output_dir 下以影片名命名的子資料夾
    base_output_folder_for_tracking = os.path.join(project_root, args.output_dir)
    specific_video_output_dir = os.path.join(base_output_folder_for_tracking, video_name_base)
    
    os.makedirs(specific_video_output_dir, exist_ok=True) # 創建以影片名為基礎的資料夾
    print(f"所有輸出將儲存到: {specific_video_output_dir}")

    # (可選) 如果你仍然需要 frames 和 labels 子目錄，在這裡創建它們
    # frames_output_sub_dir = os.path.join(specific_video_output_dir, "frames")
    # labels_output_sub_dir = os.path.join(specific_video_output_dir, "labels")
    # os.makedirs(frames_output_sub_dir, exist_ok=True)
    # os.makedirs(labels_output_sub_dir, exist_ok=True)

    annotated_video_path = None
    if args.save_annotated_video:
        # 標註影片直接存在 specific_video_output_dir 下
        annotated_video_path = os.path.join(specific_video_output_dir, f"{video_name_base}_annotated.mp4")
    # --- 輸出目錄修改結束 ---


    # --- 處理影片 ---
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"錯誤: 無法打開輸入影片 {args.input}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_writer = None
    if args.save_annotated_video and annotated_video_path:
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
        fgmask = fgbg.apply(frame_orig)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

        # 2) 排球检测 (假設它仍使用 fgmask)
        ball_boxes = detect_ball(frame_orig, ball_model, fgmask, kernel, args.conf)
        
        # 3) 選手检测 (使用增強版函數)
        player_boxes_simple, player_boxes_detailed = detect_players_enhanced( # 修改為接收兩個返回值
            frame_orig, player_model, args.conf,
            court_center_xy_tuple,
            court_boundary_poly_np,
            exclusion_zones_np_list
        )
        
        # 4) 儲存數據到 all_frames_data
        current_frame_info = {
            "frame_id": frame_count,
            "ball_detections": [], 
            "player_detections": [] 
        }
        for (x1,y1,x2,y2,conf) in ball_boxes:
            current_frame_info["ball_detections"].append({
                "center_x": (x1+x2)/2, "center_y": (y1+y2)/2, 
                "confidence": round(conf, 4), "box": [x1,y1,x2,y2]
            })
        # 使用 player_boxes_detailed 來儲存更豐富的球員資訊
        for p_data in player_boxes_detailed:
            current_frame_info["player_detections"].append({
                "box_coords": p_data["box_coords"], 
                "confidence": round(p_data["confidence"], 4),
                "is_inside_court": p_data["is_inside_court"],
                "distance_to_center": round(p_data["distance_to_center"], 2) if p_data["distance_to_center"] != float('inf') else None,
                "center_point": (round(p_data["center_point"][0], 2), round(p_data["center_point"][1], 2) )
            })
        all_frames_data.append(current_frame_info)

        # --- 繪製與儲存標籤和幀 (如果需要) ---
        # 如果你仍然需要單獨的 frames/ 和 labels/ 子目錄用於YOLO格式數據集：
        # raw_frame_path = os.path.join(frames_output_sub_dir, f"frame_{frame_count:06d}.jpg")
        # cv2.imwrite(raw_frame_path, frame_orig)
        # yolo_labels = draw_boxes_and_get_labels(frame_to_draw_on, ball_boxes, player_boxes_simple, frame_w, frame_h)
        # label_path = os.path.join(labels_output_sub_dir, f"frame_{frame_count:06d}.txt")
        # with open(label_path, 'w') as f:
        #     f.write("\n".join(yolo_labels))
        
        # 否則，如果只需要標註影片，上面的繪製邏輯可以簡化或放在 if args.save_annotated_video 內部
        if args.save_annotated_video:
            # 繪製球和球員 (這個繪製邏輯可以來自 draw_boxes_and_get_labels，或者直接在這裡寫)
            # 為了簡潔，我們假設 draw_boxes_and_get_labels 只返回標籤，繪製在這裡完成
            if court_boundary_poly_np is not None:
                cv2.polylines(frame_to_draw_on, [court_boundary_poly_np], True, (0,255,0), 1)
            for ex_zone in exclusion_zones_np_list:
                cv2.polylines(frame_to_draw_on, [ex_zone], True, (255,0,255), 1)

            for (x1,y1,x2,y2,conf) in ball_boxes:
                cv2.rectangle(frame_to_draw_on, (x1,y1), (x2,y2), (255,0,0), 2)
                cv2.putText(frame_to_draw_on, f"B {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
            for (x1,y1,x2,y2,conf) in player_boxes_simple: # 使用 player_boxes_simple 進行繪製
                cv2.rectangle(frame_to_draw_on, (x1,y1), (x2,y2), (0,255,255), 2)
                cv2.putText(frame_to_draw_on, f"P {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
            
            if out_writer:
                out_writer.write(frame_to_draw_on)
        
        if frame_count % int(fps if fps > 0 else 30) == 0:
            print(f"  已處理 {frame_count} 幀...")

    # --- 迴圈結束後 ---

    cap.release()
    if out_writer: 
        out_writer.release()
        print(f"已儲存標註影片到: {annotated_video_path}")
    cv2.destroyAllWindows()
    print(f"影片 '{args.input}' 處理完成。共處理 {frame_count} 幀。")
    
    # --- 新增：儲存 all_frames_data 到 specific_video_output_dir ---
    if all_frames_data:
        all_frames_data_filename = os.path.join(specific_video_output_dir, f"{video_name_base}_all_frames_data.json")
        try:
            with open(all_frames_data_filename, 'w') as f:
                json.dump(all_frames_data, f, indent=2)
            print(f"已將 all_frames_data 儲存到: {all_frames_data_filename}")
        except Exception as e:
            print(f"儲存 all_frames_data 時發生錯誤: {e}")
    # -------------------------------------------------------------


# --- 新增：儲存 all_frames_data 到 specific_video_output_dir ---
    if all_frames_data:
        all_frames_data_filename = os.path.join(specific_video_output_dir, f"{video_name_base}_all_frames_data.json")
        try:
            with open(all_frames_data_filename, 'w') as f:
                json.dump(all_frames_data, f, indent=2) # indent=2 讓JSON檔案更易讀
            print(f"已將 all_frames_data 儲存到: {all_frames_data_filename}")
        except Exception as e:
            print(f"儲存 all_frames_data 時發生錯誤: {e}")
    # -------------------------------------------------------------
    
    print(f"所有幀的偵測數據已收集 (共 {len(all_frames_data)} 條記錄)。")


    # --- 後續分析 ---
    if all_frames_data and court_geometry:
        print(f"\n開始進行事件分析...")
        # serve_events = find_serve_events(all_frames_data, court_geometry, fps, frame_w, frame_h)
        # ...
    # ...

if __name__ == "__main__":
    main()