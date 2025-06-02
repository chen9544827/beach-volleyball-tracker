import cv2
import json
import numpy as np

points_clicked = []
current_frame_for_drawing = None

def court_definition_mouse_callback(event, x, y, flags, param):
    global points_clicked, current_frame_for_drawing
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points_clicked) < 4:
            points_clicked.append((x, y))
            cv2.circle(current_frame_for_drawing, (x, y), 5, (0, 255, 0), -1)
            if len(points_clicked) > 1:
                cv2.line(current_frame_for_drawing, points_clicked[-2], points_clicked[-1], (0, 255, 0), 2)
            if len(points_clicked) == 4:
                cv2.line(current_frame_for_drawing, points_clicked[3], points_clicked[0], (0, 255, 0), 2)
            cv2.imshow("Define Court Boundary - Click 4 points", current_frame_for_drawing)
        else:
            print("已經選擇4個角點。請按 's' 保存或 'r' 重設。")

def define_court_boundaries_manually(video_path, config_save_path="court_config.json"):
    global points_clicked, current_frame_for_drawing
    points_clicked = []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"錯誤：無法開啟影片檔案 {video_path}")
        return False
    
    ret, first_frame = cap.read()
    cap.release()

    if not ret:
        print("錯誤：無法從影片讀取第一幀。")
        return False

    current_frame_for_drawing = first_frame.copy()
    window_name = "Define Court Boundary - Click 4 points"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, court_definition_mouse_callback)
    
    print("請在影像上依序點擊球場的4個角點（順時針或逆時針）。")
    print("完成4個點的選擇後，按 's' 鍵保存設定。")
    print("按 'r' 鍵可以清除已選的點並重新開始。")
    print("按 'q' 鍵退出而不保存。")

    while True:
        cv2.imshow(window_name, current_frame_for_drawing)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            cv2.destroyAllWindows()
            print("場地定義已取消。")
            return False
        elif key == ord('r'):
            points_clicked = []
            current_frame_for_drawing = first_frame.copy()
            print("點已重設，請重新選擇4個角點。")
        elif key == ord('s'):
            if len(points_clicked) == 4:
                court_boundary_polygon = list(points_clicked)
                
                # --- 移除了關於 ground_y_level 的輸入 ---

                config_data = {
                    "court_boundary_polygon": court_boundary_polygon,
                    # --- "ground_y_level" 不再儲存 ---
                }
                
                try:
                    with open(config_save_path, 'w') as f:
                        json.dump(config_data, f, indent=4)
                    print(f"場地邊界資訊已成功儲存到 {config_save_path}")
                    cv2.destroyAllWindows()
                    return True
                except IOError:
                    print(f"錯誤：無法寫入設定檔 {config_save_path}")
                    cv2.destroyAllWindows()
                    return False
            else:
                print("請先完整選擇4個角點才能保存。")
    
    cv2.destroyAllWindows()
    return False

def load_court_geometry(config_load_path="court_config.json"):
    try:
        with open(config_load_path, 'r') as f:
            geometry = json.load(f)
        # --- 修改驗證邏輯 ---
        if "court_boundary_polygon" not in geometry:
            print(f"警告：設定檔 {config_load_path} 中缺少 'court_boundary_polygon'。")
            return None
        # "ground_y_level" 不再是必要項
        return geometry
    except FileNotFoundError:
        print(f"提示：場地設定檔 {config_load_path} 未找到。")
        return None
    except json.JSONDecodeError:
        print(f"錯誤：場地設定檔 {config_load_path} 格式錯誤，無法解析。")
        return None
    except Exception as e:
        print(f"載入場地設定時發生未知錯誤：{e}")
        return None

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        video_file_for_test = sys.argv[1]
        print(f"使用影片 {video_file_for_test} 進行場地定義測試...")
        define_court_boundaries_manually(video_file_for_test, "test_court_config.json")
    else:
        print("請提供影片檔案路徑作為參數來進行測試。")
        print("用法: python court_config_generator.py <影片路徑>")