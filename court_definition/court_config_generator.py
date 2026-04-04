# court_definition/court_config_generator.py
# -*- coding: utf-8 -*-
"""
互動式設定沙灘排球場地邊界、排除區等。

如果出現 GUI 錯誤，請執行：
    pip uninstall opencv-python-headless
    pip install opencv-python
"""

import cv2
import numpy as np
import json
import argparse
import os
import sys

# 檢查 OpenCV GUI 支援
def check_opencv_gui():
    """檢查 OpenCV 是否支援 GUI"""
    try:
        # 嘗試建立一個小視窗
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imshow("test", test_img)
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        return True
    except cv2.error:
        return False

# --- 全域變數 ---
g_points = []
g_window_name = "Court Definition Tool"
g_final_config = {
    "court_boundary_polygon": [],
    "exclusion_zones": [],
    "net_y": None,
    "background_ball_zones": []
}

def mouse_callback(event, x, y, flags, param):
    """滑鼠回呼函數，用於記錄點擊座標"""
    global g_points
    if event == cv2.EVENT_LBUTTONDOWN:
        g_points.append((x, y))
        # 在影像上繪製一個點來標示
        frame_to_draw_on = param['frame']
        cv2.circle(frame_to_draw_on, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(frame_to_draw_on, str(len(g_points)), (x+10, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow(g_window_name, frame_to_draw_on)

def get_polygon_from_user(base_frame, poly_name, min_points, max_points):
    """
    讓使用者在影像上透過點擊定義一個多邊形。
    按 'q' 確認並結束，按 'c' 清除重來。
    """
    global g_points
    g_points = []
    
    clone = base_frame.copy()
    cv2.putText(clone, f"Define '{poly_name}': Click {min_points}-{max_points} points.", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.putText(clone, "Press 'q' to confirm, 'c' to clear.", (20, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.setMouseCallback(g_window_name, mouse_callback, {'frame': clone})
    
    while True:
        cv2.imshow(g_window_name, clone)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            if len(g_points) >= min_points:
                break
            else:
                print(f"錯誤：'{poly_name}' 至少需要 {min_points} 個點。")
        elif key == ord('c'):
            print("清除所有點，請重新標示。")
            g_points = []
            clone = base_frame.copy() # 重置畫面
            cv2.putText(clone, f"Define '{poly_name}': Click {min_points}-{max_points} points.", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.putText(clone, "Press 'q' to confirm, 'c' to clear.", (20, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)


    cv2.setMouseCallback(g_window_name, lambda *args: None) # 解除綁定
    return g_points

def get_point_from_user(base_frame, prompt):
    """讓使用者在影像上點擊一個單點"""
    global g_points
    g_points = []
    clone = base_frame.copy()
    cv2.putText(clone, prompt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.setMouseCallback(g_window_name, mouse_callback, {'frame': clone})
    
    while len(g_points) < 1:
        cv2.imshow(g_window_name, clone)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("使用者取消操作。")
            return None
            
    cv2.setMouseCallback(g_window_name, lambda *args: None)
    return g_points[0]

def main(video_path, config_save_path):
    global g_final_config

    # 檢查 GUI 支援
    if not check_opencv_gui():
        print("="*60)
        print("錯誤：OpenCV 沒有 GUI 支援！")
        print("="*60)
        print()
        print("請執行以下指令修復：")
        print()
        print("  pip uninstall opencv-python-headless -y")
        print("  pip install opencv-python")
        print()
        print("="*60)
        sys.exit(1)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"錯誤：無法開啟影片 '{video_path}'")
        return
    ret, first_frame = cap.read()
    cap.release()
    if not ret:
        print("錯誤：無法讀取影片的第一幀。")
        return

    cv2.namedWindow(g_window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(g_window_name, 1280, 720)

    # 1. 定義場地邊界
    print("步驟 1/4: 定義場地邊界 (左上 -> 左下 -> 右下 -> 右上)")
    boundary_points = get_polygon_from_user(first_frame.copy(), 'Court Boundary', 4, 4)
    if not boundary_points: print("使用者取消操作，定義終止。"); cv2.destroyAllWindows(); return
    g_final_config["court_boundary_polygon"] = boundary_points
    base_frame_with_boundary = first_frame.copy()
    cv2.polylines(base_frame_with_boundary, [np.array(boundary_points)], True, (0, 255, 0), 2)

    # 2. 定義排除區域
    current_drawing_frame = base_frame_with_boundary.copy()
    while True:
        print("\n步驟 2/4: 定義球員排除區 (可選) - 框選裁判站的位置")
        cv2.putText(current_drawing_frame, "Add exclusion zone? 'a': Add, 'n': Next, 'q': Quit", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 255), 2)
        cv2.imshow(g_window_name, current_drawing_frame)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('a'):
            exclusion_points = get_polygon_from_user(current_drawing_frame, 'Exclusion Zone', 3, 20)
            if exclusion_points:
                 g_final_config["exclusion_zones"].append({"polygon": exclusion_points})
                 cv2.polylines(current_drawing_frame, [np.array(exclusion_points)], True, (0, 0, 255), 2)
                 print(f"  [OK] 已新增排除區域 #{len(g_final_config['exclusion_zones'])}")
        elif key == ord('n'): break
        elif key == ord('q'): print("使用者取消操作，定義終止。"); cv2.destroyAllWindows(); return
    
    # 3. 定義網子Y座標
    print("\n步驟 3/4: 定義網子高度")
    net_point = get_point_from_user(current_drawing_frame, "Click on the net line")
    if not net_point: print("使用者取消操作，定義終止。"); cv2.destroyAllWindows(); return
    g_final_config["net_y"] = net_point[1]
    cv2.line(current_drawing_frame, (0, net_point[1]), (first_frame.shape[1], net_point[1]), (255, 255, 0), 2)

    # 4. 定義背景球過濾區
    while True:
        print("\n步驟 4/4: 定義背景球過濾區 (可選)")
        cv2.putText(current_drawing_frame, "Add background ball zone? 'a': Add, 'n': Finish", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        cv2.imshow(g_window_name, current_drawing_frame)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('a'):
            zone_points = get_polygon_from_user(current_drawing_frame, "Background Ball Zone (2 points)", 2, 2)
            if zone_points:
                x1, y1 = zone_points[0]; x2, y2 = zone_points[1]
                g_final_config["background_ball_zones"].append({"x1": min(x1, x2), "y1": min(y1, y2), "x2": max(x1, x2), "y2": max(y1, y2)})
                cv2.rectangle(current_drawing_frame, (min(x1,x2), min(y1,y2)), (max(x1,x2), max(y1,y2)), (255,0,255), 2)
        elif key == ord('n'): break
        elif key == ord('q'): print("使用者取消操作，定義終止。"); cv2.destroyAllWindows(); return
            
    cv2.destroyAllWindows()
    print("\n--- [OK] 定義完成，最終設定如下 ---")
    print(json.dumps(g_final_config, indent=2))

    try:
        with open(config_save_path, 'w') as f:
            json.dump(g_final_config, f, indent=4)
        print(f"\n設定已成功儲存至：{os.path.abspath(config_save_path)}")
    except Exception as e:
        print(f"\n錯誤：儲存設定檔失敗：{e}")

# --- 執行入口 ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="互動式設定沙灘排球場地邊界、排除區等。")
    parser.add_argument("--video_path", type=str, required=True, help="用於標示的範例影片路徑。")
    parser.add_argument("--output_path", type=str, default="court_config.json", help="儲存設定的 JSON 檔案路徑。")
    
    # 檢查是否提供了 video_path 參數
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
        
    args = parser.parse_args()

    # 檢查影片檔案是否存在
    if not os.path.exists(args.video_path):
        print(f"錯誤: 找不到指定的影片檔案 '{args.video_path}'")
        sys.exit(1)
        
    main(args.video_path, args.output_path)