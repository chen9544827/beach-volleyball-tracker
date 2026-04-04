# test/test_roi_diff_analyzer.py
import cv2
import numpy as np
import os
import json
import sys

# --- ★★★★★ 在這裡修改您的測試參數 ★★★★★ ---

# 1. 指定要比較的兩張圖片的路徑 (一張分數變化前，一張變化後)
#    建議從影片中截圖來獲得
IMAGE_PATH_1 = "C:/Users/Aa954/Downloads/1.png"  # <<--- 修改這裡
IMAGE_PATH_2 = "C:/Users/Aa954/Downloads/2.png"  # <<--- 修改這裡

# 2. 精確定義兩個分數的ROI座標 (x, y, width, height)
#    ★★ 這裡的座標必須和你 video_slicer_by_score.py 裡設定的完全一樣 ★★
#    或者使用 --roi-config 參數載入 ROI 配置檔案
SCORE_ROI_TEAM1 = (280, 29, 59, 51)  # 隊伍1/上方分數
SCORE_ROI_TEAM2 = (287, 92, 59, 50)  # 隊伍2/下方分數

# 3. (可選) 是否對ROI進行預處理 (建議保持 True)
#    True: 灰階 + 輕微模糊 (可降低影像雜訊干擾)
#    False: 僅灰階
APPLY_PREPROCESSING = True
GAUSSIAN_BLUR_KERNEL_SIZE = (5, 5) # 模糊核心大小，必須是正奇數

# --- ★★★★★ 參數修改結束 ★★★★★ ---


def calculate_sad(image1_gray, image2_gray):
    """計算兩張灰階圖的 Sum of Absolute Differences (SAD)"""
    if image1_gray is None or image2_gray is None or image1_gray.shape != image2_gray.shape:
        return float('inf')
    return np.sum(cv2.absdiff(image1_gray, image2_gray))

def get_roi_from_image(image, roi_coords, image_name_for_error=""):
    """安全地從影像中擷取ROI"""
    x, y, w, h = roi_coords
    if image is None: return None
    frame_h, frame_w = image.shape[:2]
    if not (0 <= x < frame_w and 0 <= y < frame_h and x + w <= frame_w and y + h <= frame_h and w > 0 and h > 0):
        print(f"警告：影像 '{image_name_for_error}' 的 ROI {roi_coords} 超出其邊界 ({frame_w}x{frame_h})。")
        return None
    return image[y:y+h, x:x+w]

def preprocess_roi(roi_image):
    """對ROI影像進行預處理"""
    if roi_image is None: return None
    gray_roi = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
    if APPLY_PREPROCESSING:
        return cv2.GaussianBlur(gray_roi, GAUSSIAN_BLUR_KERNEL_SIZE, 0)
    return gray_roi


def load_roi_config(config_path):
    """載入 ROI 配置檔案"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        roi1 = config['score_roi_team1']
        roi2 = config['score_roi_team2']

        team1_roi = (roi1['x'], roi1['y'], roi1['width'], roi1['height'])
        team2_roi = (roi2['x'], roi2['y'], roi2['width'], roi2['height'])

        print(f"[OK] 已載入 ROI 配置: {config_path}")
        return team1_roi, team2_roi

    except Exception as e:
        print(f"[ERROR] 載入 ROI 配置失敗: {e}")
        return None


def main():
    print("--- ROI 差異分析工具 ---")

    # 檢查命令列參數
    roi_team1 = SCORE_ROI_TEAM1
    roi_team2 = SCORE_ROI_TEAM2

    if len(sys.argv) > 1 and sys.argv[1] == "--roi-config":
        if len(sys.argv) < 3:
            print("[ERROR] 使用方式: python test_roi_diff_analyzer.py --roi-config <config_path>")
            return
        roi_result = load_roi_config(sys.argv[2])
        if roi_result:
            roi_team1, roi_team2 = roi_result
        else:
            print("[INFO] 使用預設 ROI 座標")

    image1 = cv2.imread(IMAGE_PATH_1)
    image2 = cv2.imread(IMAGE_PATH_2)

    if image1 is None: print(f"錯誤: 無法讀取圖片 '{IMAGE_PATH_1}'"); return
    if image2 is None: print(f"錯誤: 無法讀取圖片 '{IMAGE_PATH_2}'"); return

    print(f"比較圖片 1: '{os.path.basename(IMAGE_PATH_1)}'")
    print(f"比較圖片 2: '{os.path.basename(IMAGE_PATH_2)}'")
    print(f"使用 ROI: Team1={roi_team1}, Team2={roi_team2}\n")

    # --- 處理隊伍1 ---
    roi1_img1 = get_roi_from_image(image1, roi_team1, IMAGE_PATH_1)
    roi1_img2 = get_roi_from_image(image2, roi_team1, IMAGE_PATH_2)
    processed_roi1_img1 = preprocess_roi(roi1_img1)
    processed_roi1_img2 = preprocess_roi(roi1_img2)
    sad_roi1 = calculate_sad(processed_roi1_img1, processed_roi1_img2)

    # --- 處理隊伍2 ---
    roi2_img1 = get_roi_from_image(image1, roi_team2, IMAGE_PATH_1)
    roi2_img2 = get_roi_from_image(image2, roi_team2, IMAGE_PATH_2)
    processed_roi2_img1 = preprocess_roi(roi2_img1)
    processed_roi2_img2 = preprocess_roi(roi2_img2)
    sad_roi2 = calculate_sad(processed_roi2_img1, processed_roi2_img2)

    print("--- 差異計算結果 ---")
    print(f"ROI 1 (Team1) 的 SAD 值: {sad_roi1}")
    print(f"ROI 2 (Team2) 的 SAD 值: {sad_roi2}\n")
    print("SAD 值越高，代表影像差異越大。")

    # --- 視覺化顯示 ---
    print("正在顯示視覺化結果... 按任意鍵關閉所有視窗。")
    if processed_roi1_img1 is not None: cv2.imshow("Team1 ROI - Before", processed_roi1_img1)
    if processed_roi1_img2 is not None: cv2.imshow("Team1 ROI - After", processed_roi1_img2)
    if processed_roi2_img1 is not None: cv2.imshow("Team2 ROI - Before", processed_roi2_img1)
    if processed_roi2_img2 is not None: cv2.imshow("Team2 ROI - After", processed_roi2_img2)
    
    # 顯示差異圖
    if sad_roi1 != float('inf'):
        diff_img1 = cv2.absdiff(processed_roi1_img1, processed_roi1_img2)
        cv2.imshow("Team1 ROI - DIFFERENCE", diff_img1)
        
    if sad_roi2 != float('inf'):
        diff_img2 = cv2.absdiff(processed_roi2_img1, processed_roi2_img2)
        cv2.imshow("Team2 ROI - DIFFERENCE", diff_img2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()