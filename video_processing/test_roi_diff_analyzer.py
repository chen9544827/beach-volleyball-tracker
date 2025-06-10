# roi_diff_analyzer_internal_config.py
import cv2
import numpy as np
# import os # 如果圖片路徑是相對於腳本的，可能需要os

# --- ★★★★★ 在這裡修改您的測試參數 ★★★★★ ---

# 1. 指定要比較的兩張圖片的路徑
IMAGE_PATH_1 = "D:/Github/beach-volleyball-tracker/output_data/tracking_output/segment_069/frames/frame_000100.jpg"  # <<--- 修改這裡
IMAGE_PATH_2 = "D:/Github/beach-volleyball-tracker/output_data/tracking_output/segment_170/frames/frame_000001.jpg" # <<--- 修改這裡

# 2. 精確定義兩個分數的ROI座標 (x, y, width, height)
#    這些值應該是針對您圖片中的分數板
SCORE_ROI_TEAM1 = (280, 29, 59, 51)  # 範例：隊伍1/上方分數 <<--- 修改這裡
SCORE_ROI_TEAM2 = (287, 92, 59, 50)  # 範例：隊伍2/下方分數 <<--- 修改這裡

# 3. 是否對ROI進行預處理 (True 或 False)
APPLY_PREPROCESSING = True # <<--- 修改這裡 (True: 進行灰階和模糊, False: 直接比較灰階)

# 4. (如果 APPLY_PREPROCESSING 為 True) 預處理中的高斯模糊核心大小 (必須是正奇數)
GAUSSIAN_BLUR_KERNEL_SIZE = (5, 5) # <<--- 修改這裡 (例如 (3,3) 或 (5,5))

# --- ★★★★★ 參數修改結束 ★★★★★ ---


def calculate_sad(image1_gray, image2_gray):
    """計算兩張灰階圖的 Sum of Absolute Differences (SAD)"""
    if image1_gray is None or image2_gray is None:
        return float('inf')
    if image1_gray.shape != image2_gray.shape:
        print("警告: 兩張ROI影像的尺寸不匹配，無法計算SAD。")
        return float('inf')
    diff = cv2.absdiff(image1_gray, image2_gray)
    sad = np.sum(diff)
    return sad

def calculate_mse(image1_gray, image2_gray):
    """計算兩張灰階圖的 Mean Squared Error (MSE)"""
    if image1_gray is None or image2_gray is None:
        return float('inf')
    if image1_gray.shape != image2_gray.shape:
        print("警告: 兩張ROI影像的尺寸不匹配，無法計算MSE。")
        return float('inf')
    err = np.sum((image1_gray.astype("float") - image2_gray.astype("float")) ** 2)
    mse = err / float(image1_gray.shape[0] * image1_gray.shape[1])
    return mse

def get_roi_from_image(image, roi_coords, image_name_for_error=""):
    """安全地從影像中擷取ROI"""
    x, y, w, h = roi_coords
    if image is None: 
        print(f"錯誤: 傳入 get_roi_from_image 的影像 ({image_name_for_error}) 為 None。")
        return None
    frame_height, frame_width = image.shape[:2]
    if not (0 <= x < frame_width and 0 <= y < frame_height and x + w <= frame_width and y + h <= frame_height and w > 0 and h > 0):
        print(f"警告：影像 '{image_name_for_error}' 的 ROI {roi_coords} 超出其邊界 ({frame_width}x{frame_height}) 或尺寸無效。")
        return None
    return image[y:y+h, x:x+w]

def preprocess_roi(roi_image):
    """對ROI影像進行預處理 (灰階, 可選模糊)"""
    if roi_image is None: return None
    gray_roi = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
    if APPLY_PREPROCESSING: # 根據全域變數決定是否模糊
        # print("對ROI進行預處理 (高斯模糊)...")
        blurred_roi = cv2.GaussianBlur(gray_roi, GAUSSIAN_BLUR_KERNEL_SIZE, 0)
        return blurred_roi
    return gray_roi


def main():
    # 載入兩張圖片
    image1 = cv2.imread(IMAGE_PATH_1)
    image2 = cv2.imread(IMAGE_PATH_2)

    if image1 is None:
        print(f"錯誤: 無法讀取第一張圖片 '{IMAGE_PATH_1}'")
        return
    if image2 is None:
        print(f"錯誤: 無法讀取第二張圖片 '{IMAGE_PATH_2}'")
        return

    print(f"比較圖片: '{IMAGE_PATH_1}' 與 '{IMAGE_PATH_2}'")
    print(f"使用 ROI Team 1: {SCORE_ROI_TEAM1}")
    print(f"使用 ROI Team 2: {SCORE_ROI_TEAM2}")
    if APPLY_PREPROCESSING:
        print(f"將對ROI進行預處理: 灰階 + 高斯模糊 (核心: {GAUSSIAN_BLUR_KERNEL_SIZE})")
    else:
        print(f"將對ROI進行預處理: 僅灰階")


    # 擷取並預處理ROI
    roi1_img1_orig = get_roi_from_image(image1, SCORE_ROI_TEAM1, IMAGE_PATH_1)
    roi1_img2_orig = get_roi_from_image(image2, SCORE_ROI_TEAM1, IMAGE_PATH_2)
    roi2_img1_orig = get_roi_from_image(image1, SCORE_ROI_TEAM2, IMAGE_PATH_1)
    roi2_img2_orig = get_roi_from_image(image2, SCORE_ROI_TEAM2, IMAGE_PATH_2)

    roi1_img1_processed = preprocess_roi(roi1_img1_orig)
    roi1_img2_processed = preprocess_roi(roi1_img2_orig)
    roi2_img1_processed = preprocess_roi(roi2_img1_orig)
    roi2_img2_processed = preprocess_roi(roi2_img2_orig)


    if roi1_img1_processed is None or roi1_img2_processed is None:
        print("錯誤: 無法處理隊伍1的ROI，請檢查ROI座標和圖片。")
        # return # 即使一個ROI失敗，我們仍然可以嘗試計算另一個
    if roi2_img1_processed is None or roi2_img2_processed is None:
        print("錯誤: 無法處理隊伍2的ROI，請檢查ROI座標和圖片。")
        # return

    # 計算差異
    sad_roi1 = calculate_sad(roi1_img1_processed, roi1_img2_processed)
    mse_roi1 = calculate_mse(roi1_img1_processed, roi1_img2_processed)

    sad_roi2 = calculate_sad(roi2_img1_processed, roi2_img2_processed)
    mse_roi2 = calculate_mse(roi2_img1_processed, roi2_img2_processed)

    print("\n--- 差異計算結果 ---")
    if sad_roi1 != float('inf'):
        print(f"ROI 1 (隊伍1分數區域):")
        print(f"  Sum of Absolute Differences (SAD): {sad_roi1}")
        print(f"  Mean Squared Error (MSE):        {mse_roi1:.2f}")
    else:
        print(f"ROI 1 (隊伍1分數區域): 無法計算差異 (可能ROI擷取失敗)")

    if sad_roi2 != float('inf'):
        print(f"\nROI 2 (隊伍2分數區域):")
        print(f"  Sum of Absolute Differences (SAD): {sad_roi2}")
        print(f"  Mean Squared Error (MSE):        {mse_roi2:.2f}")
    else:
        print(f"\nROI 2 (隊伍2分數區域): 無法計算差異 (可能ROI擷取失敗)")


    # 顯示用於比較的ROI圖像 (可選)
    if roi1_img1_processed is not None: cv2.imshow("Img1 - ROI1 (Processed)", roi1_img1_processed)
    if roi1_img2_processed is not None: cv2.imshow("Img2 - ROI1 (Processed)", roi1_img2_processed)
    if roi2_img1_processed is not None: cv2.imshow("Img1 - ROI2 (Processed)", roi2_img1_processed)
    if roi2_img2_processed is not None: cv2.imshow("Img2 - ROI2 (Processed)", roi2_img2_processed)
    
    # 顯示差異圖 (可選)
    if roi1_img1_processed is not None and roi1_img2_processed is not None and roi1_img1_processed.shape == roi1_img2_processed.shape:
        diff_display_roi1 = cv2.absdiff(roi1_img1_processed, roi1_img2_processed)
        cv2.imshow("Diff ROI1 (Img1 vs Img2)", diff_display_roi1)
    if roi2_img1_processed is not None and roi2_img2_processed is not None and roi2_img1_processed.shape == roi2_img2_processed.shape:
        diff_display_roi2 = cv2.absdiff(roi2_img1_processed, roi2_img2_processed)
        cv2.imshow("Diff ROI2 (Img1 vs Img2)", diff_display_roi2)

    print("\n顯示影像中... 按任意鍵退出。")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()