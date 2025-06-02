#!/usr/bin/env python3
import os
import cv2
import argparse
import numpy as np
from ultralytics import YOLO
import sys
# --- 設定Python導入路徑，以便找到 court_definition 模組 ---
# 獲取目前腳本 (track_ball_and_player.py) 所在的目錄 (video_processing/)
current_script_dir = os.path.dirname(os.path.abspath(__file__))
# 獲取專案根目錄 (beach-volleyball-tracker/)
project_root = os.path.dirname(current_script_dir)
# 將專案根目錄添加到 sys.path，這樣Python就能找到其他子目錄中的模組
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from court_definition.court_config_generator import define_court_boundaries_manually, load_court_geometry

def parse_args():
    p = argparse.ArgumentParser(description="同时使用两个模型：排球检测（带运动过滤）和选手检测（仅标出场上4位选手）")
    p.add_argument("--input",        type=str, required=True,  help="输入视频文件路径")
    p.add_argument("--output_dir",   type=str, default="output_video", help="输出根目录")
    p.add_argument("--ball_model",   type=str, default="model/ball_best.pt", help="排球检测模型路径")
    p.add_argument("--player_model", type=str, default="model/player_yolo.pt", help="选手检测模型路径")
    p.add_argument("--conf",         type=float, default=0.3,   help="置信度阈值")
    p.add_argument("--device",       type=str, default="0",     help="推理设备: 'cpu' 或 GPU id")
    return p.parse_args()


def init_models(ball_model_path, player_model_path, device):
    """
    加载两个模型：ball_model 用于排球检测，player_model 用于选手检测
    """
    ball_model   = YOLO(ball_model_path).to(device)
    player_model = YOLO(player_model_path).to(device)
    return ball_model, player_model


def init_bg_subtractor():
    """
    初始化 MOG2 背景分割器及形态学内核，用于运动前景提取
    """
    fgbg   = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    return fgbg, kernel


def detect_ball(frame, ball_model, fgmask, kernel, conf_thresh):
    """
    在单帧图像上检测排球，并做运动过滤。
    返回 list of tuples: (x1, y1, x2, y2, confidence)
    """
    # 推理只保留 class 0 (排球)
    res_ball = ball_model(frame, conf=conf_thresh, classes=[0])[0]
    ball_boxes = []
    for box in res_ball.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        # 对应 fgmask 区域
        roi = fgmask[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        motion_ratio = np.count_nonzero(roi) / roi.size
        if motion_ratio < 0.02:
            continue  # 过滤静止球
        conf_score = float(box.conf[0].cpu().numpy())
        ball_boxes.append((x1, y1, x2, y2, conf_score))
    return ball_boxes


def detect_player_by_proximity(frame, player_model, conf_thresh, court_center_x, court_center_y):
    """
    偵測畫面中的所有 'person'，並選出離場地中心最近的4位作為球員。
    不再使用運動過濾。
    返回 list of tuples: (x1, y1, x2, y2, confidence)
    """
    # 推理，假設 'person' 類別的索引是 0 (COCO 模型中的典型值)
    results = player_model(frame, conf=conf_thresh, classes=[0])[0] 
    
    detected_persons_with_distance = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy()) # 將座標轉為整數
        confidence = float(box.conf[0].cpu().numpy())
        
        # 計算偵測到的 person 的中心點
        person_cx = (x1 + x2) / 2
        person_cy = (y1 + y2) / 2
        
        # 計算該 person 中心到場地中心的距離
        distance = np.sqrt((person_cx - court_center_x)**2 + (person_cy - court_center_y)**2)
        
        detected_persons_with_distance.append({
            "box_coords": (x1, y1, x2, y2),
            "confidence": confidence,
            "distance_to_center": distance
        })
        
    # 根據到場地中心的距離進行排序 (由近到遠)
    sorted_persons = sorted(detected_persons_with_distance, key=lambda p: p["distance_to_center"])
    
    # 選取距離最近的4位
    top_4_players_data = sorted_persons[:4]
    
    # 準備回傳結果，格式與原來的 detect_player 保持一致
    output_player_boxes = []
    for player_data in top_4_players_data:
        output_player_boxes.append((*player_data["box_coords"], player_data["confidence"]))
        
    return output_player_boxes

# 你可以選擇保留舊的 detect_player 函數並將新的命名為 detect_player_by_proximity，
# 或者直接用新邏輯覆蓋舊的 detect_player 函數。
# 如果覆蓋，記得更新 main 函數中的調用。為了清楚，這裡我們用新名字。


def draw_and_save(frame, ball_boxes, player_boxes, w, h, frame_idx, frame_dir, label_dir):
    """
    在图像上绘制两个通道的框，并保存帧图与对应的 YOLO 格式标签文件
    """
    # 保存原始帧
    frame_path = os.path.join(frame_dir, f"frame_{frame_idx:06d}.jpg")
    cv2.imwrite(frame_path, frame)

    txt_lines = []
    # 绘制排球 (class 0)
    for (x1, y1, x2, y2, conf) in ball_boxes:
        cx = ((x1 + x2) / 2) / w
        cy = ((y1 + y2) / 2) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        txt_lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, f"ball {conf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # 绘制选手 (class 1)
    for (x1, y1, x2, y2, conf) in player_boxes:
        cx = ((x1 + x2) / 2) / w
        cy = ((y1 + y2) / 2) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        txt_lines.append(f"1 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"player {conf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # 保存标签
    label_path = os.path.join(label_dir, f"frame_{frame_idx:06d}.txt")
    with open(label_path, 'w') as f:
        f.write("\n".join(txt_lines))


def main():
    args = parse_args()


    config_file_path = os.path.join(project_root, "court_config.json") 


     # 檢查設定檔是否存在，如果不存在，或使用者選擇重新定義，則運行定義程序
    if not os.path.exists(config_file_path):
        print(f"場地設定檔 '{config_file_path}' 不存在。")
        print("現在將啟動首次場地邊界定義程序...")
        if not define_court_boundaries_manually(args.input, config_file_path):
            print("場地邊界定義未完成或已取消。程式無法繼續。")
            return # 結束程式
    else:
        # 如果設定檔已存在，可以詢問使用者是否要重新定義
        user_choice = input(f"場地設定檔 '{config_file_path}' 已存在。是否要重新定義場地邊界? (y/N): ").strip().lower()
        if user_choice == 'y':
            print("重新定義場地邊界...")
            if not define_court_boundaries_manually(args.input, config_file_path):
                print("場地邊界定義未完成或已取消。將嘗試使用現有設定檔。")
        else:
            print(f"將使用現有的場地設定檔: {config_file_path}")


       # 載入場地幾何資訊
    court_geometry = load_court_geometry(config_file_path)
    if court_geometry is None:
        print("無法載入場地幾何資訊。請確保已正確定義或設定檔無誤。程式終止。")
        return
    
      # --- 新增：計算場地中心點 ---
    court_center_x, court_center_y = None, None
    if "court_boundary_polygon" in court_geometry and \
       isinstance(court_geometry["court_boundary_polygon"], list) and \
       len(court_geometry["court_boundary_polygon"]) == 4: # 假設是4個點定義的邊界
        
        points = court_geometry["court_boundary_polygon"]
        # 計算多邊形 (這裡假設為四邊形) 的幾何中心 (質心)
        # 對於凸四邊形，頂點平均值是一個合理的近似
        court_center_x = sum(p[0] for p in points) / len(points)
        court_center_y = sum(p[1] for p in points) / len(points)
        print(f"計算得到的場地中心點: ({court_center_x:.2f}, {court_center_y:.2f})")
    else:
        print("錯誤：'court_boundary_polygon' 未在設定檔中正確定義或不是4個點。")
        # 可以選擇終止程式，或使用影像中心作為備用方案（但不推薦用於精確判斷）
        # 獲取影像寬高 (w, h) 後，可用 w/2, h/2
        # 此處選擇終止，因為場地中心對此功能很重要
        print("程式因無法確定場地中心而終止。")
        return
    # --- 場地中心點計算結束 ---

    print(f"成功載入場地資訊: {court_geometry}")
    # --- 場地定義與載入結束 ---


    # 创建输出目录结构
    base     = os.path.splitext(os.path.basename(args.input))[0]
    out_root = os.path.join(args.output_dir, base)
    os.makedirs(out_root, exist_ok=True)
    frame_dir = os.path.join(out_root, "frames")
    label_dir = os.path.join(out_root, "labels")
    os.makedirs(frame_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    # 设备选择
    device = args.device
    if device.isdigit():
        device = f"cuda:{device}"

    # 加载两个模型
    ball_model, player_model = init_models(args.ball_model, args.player_model, device)

    # 打开视频并获取参数
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {args.input}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 输出视频 writer
    out_vid = os.path.join(out_root, f"{base}_output.avi")
    fourcc  = cv2.VideoWriter_fourcc(*'mp4v')
    writer  = cv2.VideoWriter(out_vid, fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"无法打开输出视频: {out_vid}")

    # 初始化背景分割器（用于排球检测的运动过滤）
    fgbg, kernel = init_bg_subtractor()

    frame_idx = 0
    all_frames_data =[]# <--- 新增：用於儲存所有幀的資訊

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1) 运动前景分割
        fgmask = fgbg.apply(frame)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

        # 2) 排球检测并做运动过滤
        ball_boxes = detect_ball(frame, ball_model, fgmask, kernel, args.conf)
 
        # 3) 选手检测，仅保留置信度最高的4个
        player_boxes = detect_player_by_proximity(frame, player_model, args.conf, court_center_x, court_center_y)


                # <--- 新增：儲存當前幀的球和球員資訊 --->
        current_frame_info = {
            "frame_id": frame_idx,
            "ball_detections": [], # 儲存球的中心點或邊界框
            "player_detections": [] # 儲存球員的中心點或邊界框
        }

        # 處理球的資訊 (範例：儲存球的中心點)
        if ball_boxes:
            for (x1, y1, x2, y2, conf) in ball_boxes:
                ball_center_x = (x1 + x2) / 2
                ball_center_y = (y1 + y2) / 2
                current_frame_info["ball_detections"].append({
                    "center_x": ball_center_x,
                    "center_y": ball_center_y,
                    "confidence": conf
                    # 你也可以儲存原始的 x1, y1, x2, y2
                })
        
        # 處理球員的資訊 (範例：儲存球員的邊界框)
        if player_boxes:
            for (x1, y1, x2, y2, conf) in player_boxes:
                current_frame_info["player_detections"].append({
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "confidence": conf
                })
        
        all_frames_data.append(current_frame_info)
        # <--- 資訊儲存結束 --->


        # 4) 绘制、保存帧与标签
        draw_and_save(frame, ball_boxes, player_boxes, w, h, frame_idx, frame_dir, label_dir)

        # 5) 写入输出视频
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"Finished! 结果保存在: {out_root}")

if __name__ == "__main__":
    main()
