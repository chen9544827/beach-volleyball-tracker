import cv2
import json
import numpy as np
import os # 確保導入 os

# 全域變數來儲存點擊的點和暫存的幀
points_clicked = []
current_frame_for_drawing = None

def court_definition_mouse_callback(event, x, y, flags, param):
    """滑鼠回呼函數，用於在影像上選擇點"""
    global points_clicked, current_frame_for_drawing
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points_clicked) < 4: # 我們需要4個點來定義邊界多邊形
            points_clicked.append((x, y))
            # 在影像上畫出點和線
            cv2.circle(current_frame_for_drawing, (x, y), 5, (0, 255, 0), -1)
            if len(points_clicked) > 1:
                cv2.line(current_frame_for_drawing, points_clicked[-2], points_clicked[-1], (0, 255, 0), 2)
            if len(points_clicked) == 4: # 如果4個點都選了，自動連接首尾形成封閉多邊形
                cv2.line(current_frame_for_drawing, points_clicked[3], points_clicked[0], (0, 255, 0), 2)
            cv2.imshow("Define Court Boundary - Click 4 points", current_frame_for_drawing)
        else:
            print("已經選擇4個角點。請按 's' 保存或 'r' 重設。")

def define_court_boundaries_manually(video_path, 
                                     # 預設儲存路徑改到專案根目錄 (court_definition/ 的上一層)
                                     config_save_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "court_config.json")):
    """
    讀取影片第一幀，讓使用者手動點擊定義場地邊界，並儲存到 JSON 檔案。
    """
    global points_clicked, current_frame_for_drawing
    points_clicked = [] # 每次調用時重設點

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

        if key == ord('q'): # 按 'q' 退出
            cv2.destroyAllWindows()
            print("場地定義已取消。")
            return False
        elif key == ord('r'): # 按 'r' 重設
            points_clicked = []
            current_frame_for_drawing = first_frame.copy() # 恢復到原始第一幀
            print("點已重設，請重新選擇4個角點。")
        elif key == ord('s'): # 按 's' 保存
            if len(points_clicked) == 4:
                court_boundary_polygon = list(points_clicked) # 轉換為普通列表
                
                config_data = {
                    "court_boundary_polygon": court_boundary_polygon,
                    # "ground_y_level" 已移除
                }
                
                # 確保儲存路徑的目錄存在 (雖然這裡是根目錄，通常已存在)
                save_dir = os.path.dirname(config_save_path)
                if save_dir and not os.path.exists(save_dir): # 檢查 save_dir 是否為空 (根目錄時可能為空)
                    os.makedirs(save_dir, exist_ok=True)

                try:
                    with open(config_save_path, 'w') as f:
                        json.dump(config_data, f, indent=4) 
                    print(f"場地邊界資訊已成功儲存到 {os.path.abspath(config_save_path)}") # 顯示絕對路徑
                    cv2.destroyAllWindows()
                    return True
                except IOError as e:
                    print(f"錯誤：無法寫入設定檔 {os.path.abspath(config_save_path)}: {e}")
                    cv2.destroyAllWindows()
                    return False
            else:
                print("請先完整選擇4個角點才能保存。")
    
    cv2.destroyAllWindows() # 確保在任何情況下退出迴圈時都關閉視窗
    return False

def load_court_geometry(
    # 預設讀取路徑也改到專案根目錄
    config_load_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "court_config.json")):
    """從 JSON 檔案載入場地幾何資訊"""
    abs_config_load_path = os.path.abspath(config_load_path) # 獲取絕對路徑用於提示
    try:
        with open(abs_config_load_path, 'r') as f:
            geometry = json.load(f)
        
        if "court_boundary_polygon" not in geometry:
            print(f"警告：設定檔 {abs_config_load_path} 中缺少 'court_boundary_polygon'。")
            return None
        # "ground_y_level" 不再是必要項
        return geometry
    except FileNotFoundError:
        print(f"提示：場地設定檔 {abs_config_load_path} 未找到。")
        return None
    except json.JSONDecodeError:
        print(f"錯誤：場地設定檔 {abs_config_load_path} 格式錯誤，無法解析。")
        return None
    except Exception as e:
        print(f"載入場地設定 '{abs_config_load_path}' 時發生未知錯誤：{e}")
        return None

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        video_file_for_test = sys.argv[1]
        # 測試時，也讓它存到專案根目錄
        test_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "court_config.json")
        print(f"使用影片 '{video_file_for_test}' 進行場地定義測試...")
        print(f"測試設定將儲存到: {os.path.abspath(test_config_path)}")
        define_court_boundaries_manually(video_file_for_test, test_config_path)
    else:
        print("請提供影片檔案路徑作為參數來進行測試。")
        print(f"用法: python {os.path.basename(__file__)} <影片路徑>")