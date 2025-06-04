import cv2
import pytesseract
import re
import numpy as np

# --- 設定 ---
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# !! 請根據你的影像，重新精確定義這兩個只包含單個分數數字的ROI !!
# 範例值 (你需要根據 image_741f77.png 或 image_750832.jpg 實際調整)
# 假設 Team1 是上面的分數 (GER), Team2 是下面的分數 (CAN)
SCORE_ROI_TEAM1 = (280, 29, 59, 51)  # 估計的上半部分ROI (x, y, width, height_half)
SCORE_ROI_TEAM2 = (287, 92, 59, 50) # 估計的下半部分ROI (x, y + height_half, width, height_half)
                                       # 這些值你需要非常精確地從影像中量測和調整！

IMAGE_PATH = 'D:/Github/beach-volleyball-tracker/output_data/tracking_output/segment_004/frames/frame_000001.jpg' # 或者用你最新的截圖

# OCR 設定：強制只辨識數字，並嘗試更適合單個數字的 psm 模式
# --psm 7: Treat the image as a single text line.
# --psm 8: Treat the image as a single word.
# --psm 10: Treat the image as a single character. (如果分數只有一位數，這個可能很好)
OCR_CONFIG_SINGLE_DIGIT = r'--psm 10 -c tessedit_char_whitelist=0123456789'
OCR_CONFIG_MULTI_DIGIT = r'--psm 8 -c tessedit_char_whitelist=0123456789' # 如果分數可能有多位數

def preprocess_image_for_ocr_single_score(image_roi, window_title="Preprocessed ROI"):
    """對單個分數的ROI進行預處理"""
    gray = cv2.cvtColor(image_roi, cv2.COLOR_BGR2GRAY)

    scale_factor = 2.5 
    width = int(gray.shape[1] * scale_factor)
    height = int(gray.shape[0] * scale_factor)
    resized = cv2.resize(gray, (width, height), interpolation=cv2.INTER_CUBIC)
    resized_gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_CUBIC)
    blurred_gray = cv2.GaussianBlur(resized_gray, (3, 3), 0) # 核心大小(3,3)或(5,5)，標準差為0

    # 調整二值化閾值，目標是讓數字清晰且完整
    _, thresh_img = cv2.threshold(resized, 195, 255, cv2.THRESH_BINARY_INV) # 嘗試不同的閾值 150, 160, 170...
    # 或者自適應閾值
    # thresh_img = cv2.adaptiveThreshold(resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
    #                                      cv2.THRESH_BINARY_INV, blockSize=15, C=7)


    # (可選) 形態學操作，例如輕微的膨脹來連接斷裂的筆劃
    kernel = np.ones((2,2),np.uint8)
    opened_img = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel)
    thresh_img = opened_img # 更新 thresh_img

    kernel = np.ones((2,2),np.uint8)
    closed_img = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, kernel)
    thresh_img = closed_img

    cv2.imshow(window_title, thresh_img)
    # cv2.waitKey(0) # 在調試時取消註解以逐個查看
    return thresh_img

def extract_single_score_from_ocr(ocr_text, expected_digits=None):
    """從OCR文字中提取單個分數數字"""
    print(f"  原始OCR輸出 (單個分數): '{ocr_text.strip()}'")
    numbers = re.findall(r'\d+', ocr_text)
    if numbers:
        # 如果預期只有一個數字 (例如個位數分數)
        if expected_digits == 1 and len(numbers[0]) == 1:
             return int(numbers[0])
        # 如果分數可能是多位數，或者不確定
        try:
            # 通常第一個找到的數字串就是我們要的
            return int(numbers[0])
        except ValueError:
            return None
    return None

# --- 主測試邏輯 ---
frame = cv2.imread(IMAGE_PATH)
if frame is None:
    print(f"錯誤: 無法讀取影像 '{IMAGE_PATH}'")
else:
    scores = {}
    rois_to_process = {
        "team1": SCORE_ROI_TEAM1,
        "team2": SCORE_ROI_TEAM2
    }

    for team_name, roi_coords in rois_to_process.items():
        x, y, w, h = roi_coords
        if w <= 0 or h <= 0:
            print(f"警告: {team_name} 的 ROI 尺寸無效 (w={w}, h={h})，跳過處理。")
            continue

        single_score_roi = frame[y:y+h, x:x+w]

        # 放大顯示原始擷取的單個分數ROI，確保擷取正確
        cv2.imshow(f"Original ROI - {team_name}", cv2.resize(single_score_roi, (w*3,h*3), interpolation=cv2.INTER_NEAREST))
        # cv2.waitKey(0)

        preprocessed_single_roi = preprocess_image_for_ocr_single_score(single_score_roi.copy(), 
                                                                     f"Preprocessed ROI - {team_name}")

        # 根據分數是一位數還是多位數，選擇合適的OCR配置
        # 假設個位數用psm 10，多位數用psm 8 (或7)
        # 這裡先用一個通用的嘗試，你可以針對team1和team2用不同的config
        # 例如，如果 "8" 總是一位數，"12" 總是兩位數
        current_ocr_config = OCR_CONFIG_MULTI_DIGIT # 先假設可能有多位數
        # if team_name == "team1": # 假設team1的分數通常是個位數
        # current_ocr_config = OCR_CONFIG_SINGLE_DIGIT


        try:
            ocr_text = pytesseract.image_to_string(preprocessed_single_roi, config=current_ocr_config, lang='eng')
            score = extract_single_score_from_ocr(ocr_text)
            scores[team_name + "_score"] = score
        except pytesseract.TesseractNotFoundError:
            print("錯誤: Tesseract OCR 未安裝或未在系統 PATH 中找到。")
            break
        except Exception as e:
            print(f"處理 {team_name} 時發生OCR或解析錯誤: {e}")
            scores[team_name + "_score"] = None

    if cv2.waitKey(0) & 0xFF == ord('q'): # 顯示完所有預處理影像後等待按鍵
         pass
    cv2.destroyAllWindows()

    print("\n最終解析的分數:")
    if "team1_score" in scores:
        print(f"  隊伍1 分數: {scores['team1_score']}")
    if "team2_score" in scores:
        print(f"  隊伍2 分數: {scores['team2_score']}")