"""
錯誤訊息範本模組

提供統一的繁體中文錯誤訊息，用於整個專案的錯誤報告。
"""

# 檔案相關錯誤
ERROR_MESSAGES = {
    # 檔案系統錯誤
    'file_not_found': '找不到檔案: {path}',
    'file_read_error': '讀取檔案時發生錯誤: {path}\n詳細資訊: {error}',
    'file_write_error': '寫入檔案時發生錯誤: {path}\n詳細資訊: {error}',
    'directory_not_found': '找不到目錄: {path}',

    # JSON 相關錯誤
    'json_decode_error': 'JSON 解析失敗: {path}\n詳細資訊: {error}',
    'json_invalid_structure': 'JSON 結構無效: {path}\n缺少必要欄位: {missing_fields}',
    'json_empty_file': 'JSON 檔案為空: {path}',

    # 資料驗證錯誤
    'missing_frames': "JSON 中沒有 'frames' 欄位或 frames 為空",
    'missing_metadata': "JSON 中沒有 'metadata' 欄位",
    'missing_video_path': "metadata 中沒有 'video_path' 欄位",
    'missing_fps': "metadata 中沒有 'fps' 欄位",
    'invalid_frame_id': '幀 {index} 缺少 frame_id 欄位',
    'invalid_frame_structure': '幀 {frame_id} 的資料結構無效',

    # 追蹤資料錯誤
    'insufficient_ankle_data': '腳踝資料不足 ({count} 幀)，需要至少 {required} 幀',
    'no_valid_ankle_trajectory': '無有效腳踝軌跡資料',
    'no_valid_ankle_y_values': '無有效腳踝 Y 座標資料',
    'no_valid_frame_data': '無有效幀資料（所有幀都缺少 frame_id）',

    # 場地設定錯誤
    'invalid_court_config': '場地設定檔無效: {path}',
    'missing_court_boundary': "場地設定中沒有 'court_boundary_polygon' 欄位",
    'missing_exclusion_zones': "場地設定中沒有 'exclusion_zones' 欄位",

    # 偵測錯誤
    'no_ball_detections': '幀 {frame_id} 中沒有球偵測資料',
    'no_player_detections': '幀 {frame_id} 中沒有球員偵測資料',
    'no_pose_keypoints': '球員 {player_index} 在幀 {frame_id} 中沒有姿態關鍵點',

    # 影片相關錯誤
    'video_open_error': '無法開啟影片檔案: {path}',
    'video_read_error': '讀取影片幀時發生錯誤: {path}',
    'video_info_error': '無法取得影片資訊: {path}',
}

# 警告訊息
WARNING_MESSAGES = {
    # 資料品質警告
    'partial_ball_detections': '警告: 幀 {frame_id} 的球偵測資料不完整',
    'partial_player_detections': '警告: 幀 {frame_id} 的球員偵測資料不完整',
    'low_confidence_detection': '警告: 幀 {frame_id} 的偵測信心度過低 ({confidence:.2f})',
    'missing_keypoint': '警告: 球員在幀 {frame_id} 中缺少關鍵點 {keypoint_name}',

    # 處理警告
    'skipping_invalid_frame': '跳過無效幀: {frame_id}',
    'using_default_value': '使用預設值: {field}={value}',
    'occlusion_detected': '偵測到遮擋: 幀 {frame_id}',

    # 配置警告
    'no_court_config': '未提供場地設定檔，將使用預設值',
    'no_exclusion_zones': '場地設定中沒有排除區域，將不進行區域過濾',
}

# 資訊訊息
INFO_MESSAGES = {
    'processing_video': '正在處理影片: {video_name}',
    'loading_tracking_data': '正在載入追蹤資料: {json_path}',
    'serve_detected': '偵測到發球事件: 類型={serve_type}, 信心度={confidence:.2f}',
    'batch_progress': '批次處理進度: {current}/{total} ({percentage:.1f}%)',
    'validation_passed': '資料驗證通過: {file_name}',
}


def format_error(error_key: str, **kwargs) -> str:
    """
    格式化錯誤訊息

    Args:
        error_key: 錯誤訊息的鍵值
        **kwargs: 訊息中的佔位符參數

    Returns:
        格式化後的錯誤訊息
    """
    template = ERROR_MESSAGES.get(error_key, f"未知錯誤: {error_key}")
    try:
        return template.format(**kwargs)
    except KeyError as e:
        return f"{template} (格式化錯誤: 缺少參數 {e})"


def format_warning(warning_key: str, **kwargs) -> str:
    """
    格式化警告訊息

    Args:
        warning_key: 警告訊息的鍵值
        **kwargs: 訊息中的佔位符參數

    Returns:
        格式化後的警告訊息
    """
    template = WARNING_MESSAGES.get(warning_key, f"未知警告: {warning_key}")
    try:
        return template.format(**kwargs)
    except KeyError as e:
        return f"{template} (格式化錯誤: 缺少參數 {e})"


def format_info(info_key: str, **kwargs) -> str:
    """
    格式化資訊訊息

    Args:
        info_key: 資訊訊息的鍵值
        **kwargs: 訊息中的佔位符參數

    Returns:
        格式化後的資訊訊息
    """
    template = INFO_MESSAGES.get(info_key, f"資訊: {info_key}")
    try:
        return template.format(**kwargs)
    except KeyError as e:
        return f"{template} (格式化錯誤: 缺少參數 {e})"


class ValidationError(Exception):
    """自定義驗證錯誤異常"""

    def __init__(self, message: str, error_key: str = None, **kwargs):
        """
        初始化驗證錯誤

        Args:
            message: 錯誤訊息（或使用 error_key 自動格式化）
            error_key: 錯誤訊息的鍵值（可選）
            **kwargs: 格式化參數
        """
        if error_key:
            message = format_error(error_key, **kwargs)
        super().__init__(message)
        self.error_key = error_key
        self.format_kwargs = kwargs
