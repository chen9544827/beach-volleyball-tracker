import cv2
import json
import numpy as np
import os

current_polygon_points = []
current_frame_for_drawing = None
current_definition_mode = "boundary"
current_exclusion_zone_count = 0
active_window_name_for_callback = "" # 新增一個全域變數來儲存當前活動視窗的準確名稱

def court_definition_mouse_callback(event, x, y, flags, param):
    global current_polygon_points, current_frame_for_drawing, current_definition_mode, active_window_name_for_callback
    
    if event == cv2.EVENT_LBUTTONDOWN:
        max_points = 4 if current_definition_mode == "boundary" else 10
        
        if len(current_polygon_points) < max_points:
            current_polygon_points.append((x, y))
            color = (0, 255, 0) if current_definition_mode == "boundary" else (0, 0, 255)
            
            cv2.circle(current_frame_for_drawing, (x, y), 5, color, -1)
            if len(current_polygon_points) > 1:
                cv2.line(current_frame_for_drawing, current_polygon_points[-2], current_polygon_points[-1], color, 2)
            
            if current_definition_mode == "boundary" and len(current_polygon_points) == 4:
                cv2.line(current_frame_for_drawing, current_polygon_points[3], current_polygon_points[0], color, 2)
            elif current_definition_mode == "exclusion" and len(current_polygon_points) >= 3:
                 if len(current_polygon_points) > 1 :
                    cv2.polylines(current_frame_for_drawing, [np.array(current_polygon_points)], False, (255,100,0), 1)

            # 在回呼函數中，我們應該使用由 define_court_boundaries_manually 創建和維護的視窗名稱
            # 我們透過 active_window_name_for_callback 這個全域變數來確保名稱一致
            if active_window_name_for_callback: # 確保這個變數有值
                cv2.imshow(active_window_name_for_callback, current_frame_for_drawing) # <--- 使用固定的、正確的視窗名稱
            else:
                print("錯誤：active_window_name_for_callback 未設定！")
        else:
            print(f"已達到 '{current_definition_mode}' 模式的點數上限 ({max_points}點)。按 'n' 完成此區域，或 'r' 重設此區域的點。")

def display_defined_areas(image, boundary_polygon_points, exclusion_zones_points_list):
    """
    在給定的影像上繪製已定義的場地邊界和排除區域。
    - image: 要在其上繪製的影像 (通常是影片的第一幀副本)
    - boundary_polygon_points: 場地邊界的多邊形頂點列表 [[x,y], ...]
    - exclusion_zones_points_list: 排除區域的多邊形頂點列表的列表 [[poly1_points], [poly2_points], ...]
    """
    display_img = image.copy() # 複製一份影像以免影響原始影像

    # 繪製場地邊界 (綠色)
    if boundary_polygon_points and len(boundary_polygon_points) >= 3:
        cv2.polylines(display_img, [np.array(boundary_polygon_points, dtype=np.int32)], 
                      isClosed=True, color=(0, 255, 0), thickness=2)
        # (可選) 填充顏色以更明顯地區分
        # cv2.fillPoly(display_img, [np.array(boundary_polygon_points, dtype=np.int32)], (0, 255, 0, 50)) # 半透明綠色填充
        # 為了半透明填充，你需要確保 display_img 支持alpha通道，或者用 cv2.addWeighted
        
        # 在邊界中心或第一個點附近標註文字
        try:
            M = cv2.moments(np.array(boundary_polygon_points, dtype=np.int32))
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(display_img, "Court Boundary", (cx - 50, cy), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        except: # 如果計算中心點出錯，就不標註文字
            pass


    # 繪製排除區域 (紅色)
    if exclusion_zones_points_list:
        for i, zone_points in enumerate(exclusion_zones_points_list):
            if zone_points and len(zone_points) >= 3:
                polygon_np = np.array(zone_points, dtype=np.int32)
                cv2.polylines(display_img, [polygon_np], isClosed=True, color=(0, 0, 255), thickness=2)
                # (可選) 填充顏色
                # overlay = display_img.copy()
                # cv2.fillPoly(overlay, [polygon_np], (0, 0, 255))
                # cv2.addWeighted(overlay, 0.3, display_img, 0.7, 0, display_img) # 半透明紅色填充

                # 在排除區中心或第一個點附近標註文字
                try:
                    M = cv2.moments(polygon_np)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        cv2.putText(display_img, f"Exclusion Zone {i+1}", (cx - 60, cy), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                except:
                    pass


    cv2.imshow("Defined Areas Overview", display_img)
    print("\n已顯示所有定義的區域。按任意鍵退出此預覽。")
    cv2.waitKey(0)
    cv2.destroyWindow("Defined Areas Overview")


def define_court_boundaries_manually(video_path, 
                                     config_save_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "court_config.json")):
    global current_polygon_points, current_frame_for_drawing, current_definition_mode, all_defined_polygons, current_exclusion_zone_count, active_window_name_for_callback
    
    current_polygon_points = []
    all_defined_polygons = []
    current_definition_mode = "boundary"
    current_exclusion_zone_count = 0
    active_window_name_for_callback = "" # 初始化

    cap = cv2.VideoCapture(video_path)
    # ... (省略影片讀取和錯誤檢查) ...
    ret, first_frame_orig = cap.read()
    cap.release()
    if not ret: return False # 簡化錯誤處理

    current_frame_for_drawing = first_frame_orig.copy()
    
    defining_completed = False
    while not defining_completed:
        # --- 在這裡計算並設定當前活動視窗的準確名稱 ---
        window_title_str = f"Define {current_definition_mode.capitalize()} Zone "
        if current_definition_mode == "exclusion":
            window_title_str += f"#{current_exclusion_zone_count + 1}"
        window_title_str += " - Click points"
        active_window_name_for_callback = window_title_str # 將準確的名稱賦給全域變數
        # -------------------------------------------------
        
        cv2.namedWindow(active_window_name_for_callback) # 使用計算好的名稱創建視窗
        cv2.setMouseCallback(active_window_name_for_callback, court_definition_mouse_callback)

        current_polygon_points = [] 
        current_frame_for_drawing = first_frame_orig.copy()
        # 如果需要重繪已定義的區域，可以在這裡遍歷 all_defined_polygons 並在 current_frame_for_drawing 上繪製它們

        # ... (打印提示訊息) ...
        if current_definition_mode == "boundary":
            print("\n--- 步驟 1: 定義場地邊界 (4 個點) ---") # ...
        elif current_definition_mode == "exclusion":
            print(f"\n--- 步驟 2: 定義排除區域 #{current_exclusion_zone_count + 1} (至少3個點) ---") # ...-
        print("通用操作：按 'r' 鍵重設當前區域的點。按 'q' 鍵放棄所有定義並退出。按'n'繼續。當定義完排除區時按'f'")


        # 內部迴圈
        while True:
            cv2.imshow(active_window_name_for_callback, current_frame_for_drawing) # 使用一致的視窗名稱
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                cv2.destroyAllWindows()
                print("場地定義已取消。")
                return False
            elif key == ord('r'):
                current_polygon_points = []
                current_frame_for_drawing = first_frame_orig.copy()
                # 如果需要重繪已定義的區域，在這裡加入邏輯
                print("當前區域的點已重設，請重新選擇。")
            
            elif key == ord('n'):
                destroy_current_window = False
                if current_definition_mode == "boundary":
                    if len(current_polygon_points) == 4:
                        all_defined_polygons.append({"type": "boundary", "points": list(current_polygon_points)})
                        print("場地邊界已定義。")
                        current_definition_mode = "exclusion"
                        destroy_current_window = True 
                    else: print("場地邊界需要正好4個點。")
                elif current_definition_mode == "exclusion":
                    if len(current_polygon_points) >= 3:
                        all_defined_polygons.append({"type": "exclusion", 
                                                     "name": f"exclusion_zone_{current_exclusion_zone_count + 1}",
                                                     "points": list(current_polygon_points)})
                        current_exclusion_zone_count += 1
                        print(f"排除區域 #{current_exclusion_zone_count} 已定義。")
                        destroy_current_window = True
                    else: print("排除區域至少需要3個點。")
                
                if destroy_current_window:
                    cv2.destroyWindow(active_window_name_for_callback) # 銷毀當前視窗
                    active_window_name_for_callback = "" # 清空，避免回呼錯誤使用
                    break # 跳出內部迴圈，進入外部迴圈的下一輪
            
            elif key == ord('f') and current_definition_mode == "exclusion":
                print("所有排除區域定義完畢。")
                defining_completed = True
                if active_window_name_for_callback : cv2.destroyWindow(active_window_name_for_callback) # 確保最後一個視窗也關閉
                active_window_name_for_callback = ""
                break 
            
            elif key == 27: # ESC 鍵
                 cv2.destroyAllWindows()
                 print("場地定義已由ESC鍵取消。")
                 return False

        if defining_completed:
            break 

    # --- 所有定義完成後，整理並儲存資料 ---
    # ... (儲存邏輯保持不變) ...
    final_config_data = {}
    boundary_polygons_pts = None # 用於傳給 display_defined_areas
    exclusion_polygons_list_pts = [] # 用於傳給 display_defined_areas

    boundary_polygons = [p["points"] for p in all_defined_polygons if p["type"] == "boundary"]
    if boundary_polygons:
        final_config_data["court_boundary_polygon"] = boundary_polygons[0]
        boundary_polygons_pts = boundary_polygons[0] # 獲取點列表
    else:
        print("錯誤：未定義場地邊界！無法儲存。")
        cv2.destroyAllWindows() # 確保關閉可能存在的視窗
        return False

    exclusion_polygons = [p["points"] for p in all_defined_polygons if p["type"] == "exclusion"]
    if exclusion_polygons:
        final_config_data["exclusion_zones"] = exclusion_polygons
        exclusion_polygons_list_pts = exclusion_polygons # 獲取點列表的列表
    
    if not final_config_data.get("court_boundary_polygon"):
        print("最終配置中缺少場地邊界，不進行儲存。")
        cv2.destroyAllWindows()
        return False

    save_successful = False
    try:
        save_dir = os.path.dirname(config_save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        with open(config_save_path, 'w') as f:
            json.dump(final_config_data, f, indent=4) 
        print(f"場地設定已成功儲存到 {os.path.abspath(config_save_path)}")
        save_successful = True # 標記儲存成功
    except Exception as e:
        print(f"儲存設定檔 {os.path.abspath(config_save_path)} 時發生錯誤: {e}")
        save_successful = False
    finally:
        cv2.destroyAllWindows() # 確保所有定義過程中的視窗都被關閉

    # --- 新增：如果儲存成功，則顯示定義的區域 ---
    if save_successful and first_frame_orig is not None:
        print("\n正在顯示已定義的場地區域預覽...")
        # 從剛儲存的 final_config_data 中獲取點，或者直接使用上面提取的 pts 變數
        display_defined_areas(first_frame_orig, 
                              final_config_data.get("court_boundary_polygon"), 
                              final_config_data.get("exclusion_zones", [])) # 如果沒有排除區，傳空列表
    
    return save_successful

# ... (load_court_geometry 和 if __name__ == '__main__' 保持不變) ...

# load_court_geometry 函數需要能處理新的 exclusion_zones (之前的版本應該已經可以了)
def load_court_geometry(
    config_load_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "court_config.json")):
    abs_config_load_path = os.path.abspath(config_load_path)
    try:
        with open(abs_config_load_path, 'r') as f:
            geometry = json.load(f)
        
        if "court_boundary_polygon" not in geometry:
            print(f"警告：設定檔 {abs_config_load_path} 中缺少 'court_boundary_polygon'。")
            # 可以考慮返回 None 或部分有效的 geometry
            # return None
        
        # exclusion_zones 是可選的
        if "exclusion_zones" not in geometry:
            print(f"提示：設定檔 {abs_config_load_path} 中未找到 'exclusion_zones'。")
            geometry["exclusion_zones"] = [] # 如果沒有，給一個空列表

        return geometry
    except FileNotFoundError:
        print(f"提示：場地設定檔 {abs_config_load_path} 未找到。")
        return None
    # ... (其他 except 塊保持不變) ...



if __name__ == '__main__':
    # ... (測試程式碼保持不變，它會調用新的 define_court_boundaries_manually) ...
    import sys
    if len(sys.argv) > 1:
        video_file_for_test = sys.argv[1]
        test_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "test_court_config_with_exclusion.json") #改名以區分
        print(f"使用影片 '{video_file_for_test}' 進行場地定義測試...")
        print(f"測試設定將儲存到: {os.path.abspath(test_config_path)}")
        define_court_boundaries_manually(video_file_for_test, test_config_path)
    else:
        print("請提供影片檔案路徑作為參數來進行測試。")
        print(f"用法: python {os.path.basename(__file__)} <影片路徑>")