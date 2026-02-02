"""
測試資料驗證器

測試 DataValidator 對各種有效/無效資料的處理
"""

import sys
import os
import json
import tempfile

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.data_validator import DataValidator, safe_load_json, get_keypoint, validate_center_point


def test_valid_json_passes():
    """測試 1: 有效的 JSON 通過驗證"""
    print("\n測試 1: 有效的 JSON 通過驗證")

    valid_data = {
        "metadata": {
            "video_path": "test.mp4",
            "fps": 25.0,
            "total_frames": 100
        },
        "frames": [
            {
                "frame_id": 0,
                "ball_detections": [{"box_coords": [1, 2, 3, 4], "confidence": 0.9}],
                "player_detections": [{"box_coords": [10, 20, 30, 40], "confidence": 0.85}]
            },
            {
                "frame_id": 1,
                "ball_detections": [],
                "player_detections": []
            }
        ]
    }

    validator = DataValidator(verbose=False)
    is_valid, errors = validator.validate_tracking_json(valid_data)

    print(f"  結果: is_valid={is_valid}")
    print(f"  錯誤: {errors}")

    assert is_valid, f"驗證失敗: {errors}"
    assert len(errors) == 0, "不應有錯誤"
    print("  [OK] 通過")


def test_missing_frames_fails():
    """測試 2: 缺少 'frames' 時失敗"""
    print("\n測試 2: 缺少 'frames' 時失敗")

    invalid_data = {
        "metadata": {
            "video_path": "test.mp4",
            "fps": 25.0
        }
        # 缺少 'frames'
    }

    validator = DataValidator(verbose=False)
    is_valid, errors = validator.validate_tracking_json(invalid_data)

    print(f"  結果: is_valid={is_valid}")
    print(f"  錯誤: {errors}")

    assert not is_valid, "應該驗證失敗"
    assert len(errors) > 0, "應該有錯誤訊息"
    assert "frames" in errors[0].lower(), "錯誤訊息應提到 frames"
    print("  [OK] 通過")


def test_missing_metadata_fails():
    """測試 3: 缺少 'metadata' 時失敗"""
    print("\n測試 3: 缺少 'metadata' 時失敗")

    invalid_data = {
        "frames": [{"frame_id": 0}]
        # 缺少 'metadata'
    }

    validator = DataValidator(verbose=False)
    is_valid, errors = validator.validate_tracking_json(invalid_data)

    print(f"  結果: is_valid={is_valid}")
    print(f"  錯誤: {errors}")

    assert not is_valid, "應該驗證失敗"
    assert len(errors) > 0, "應該有錯誤訊息"
    assert "metadata" in errors[0].lower(), "錯誤訊息應提到 metadata"
    print("  [OK] 通過")


def test_missing_ball_detections_warns():
    """測試 4: 缺少 'ball_detections' 時警告但通過"""
    print("\n測試 4: 缺少 'ball_detections' 時警告但通過")

    data_with_missing_ball = {
        "metadata": {
            "video_path": "test.mp4",
            "fps": 25.0
        },
        "frames": [
            {
                "frame_id": 0,
                # 缺少 ball_detections（模擬遮擋情況）
                "player_detections": [{"box_coords": [10, 20, 30, 40]}]
            }
        ]
    }

    validator = DataValidator(verbose=False)
    is_valid, errors = validator.validate_tracking_json(data_with_missing_ball)

    print(f"  結果: is_valid={is_valid}")
    print(f"  錯誤: {errors}")
    print(f"  警告數量: {len(validator.warnings)}")

    assert is_valid, "應該通過驗證（寬鬆模式）"
    assert len(errors) == 0, "不應有嚴重錯誤"
    assert len(validator.warnings) > 0, "應該有警告訊息"
    print("  [OK] 通過")


def test_corrupt_json_fails():
    """測試 5: 損壞的 JSON 優雅失敗"""
    print("\n測試 5: 損壞的 JSON 優雅失敗")

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write('{"invalid": json syntax}')  # 故意寫入無效 JSON
        temp_path = f.name

    try:
        data, error = safe_load_json(temp_path)

        print(f"  結果: data={data}")
        print(f"  錯誤: {error}")

        assert data is None, "應該返回 None"
        assert error is not None, "應該有錯誤訊息"
        assert "JSON" in error or "解析" in error, "錯誤訊息應提到 JSON 解析"
        print("  [OK] 通過")

    finally:
        os.unlink(temp_path)


def test_file_not_found():
    """測試 6: 檔案不存在時優雅失敗"""
    print("\n測試 6: 檔案不存在時優雅失敗")

    data, error = safe_load_json("nonexistent_file.json")

    print(f"  結果: data={data}")
    print(f"  錯誤: {error}")

    assert data is None, "應該返回 None"
    assert error is not None, "應該有錯誤訊息"
    assert "找不到" in error or "not" in error.lower(), "錯誤訊息應提到檔案不存在"
    print("  [OK] 通過")


def test_get_keypoint_valid():
    """測試 7: get_keypoint 處理有效關鍵點"""
    print("\n測試 7: get_keypoint 處理有效關鍵點")

    pose_keypoints = [
        [100, 200, 0.9],  # 關鍵點 0
        [150, 250, 0.8],  # 關鍵點 1
        [0, 0, 0.1]       # 關鍵點 2（信心度過低）
    ]

    # 測試有效關鍵點
    point = get_keypoint(pose_keypoints, 0, confidence_threshold=0.5)
    assert point == (100, 200), f"關鍵點 0 錯誤: {point}"

    # 測試信心度過低的關鍵點
    point = get_keypoint(pose_keypoints, 2, confidence_threshold=0.5)
    assert point is None, f"信心度過低應返回 None: {point}"

    # 測試不存在的索引
    point = get_keypoint(pose_keypoints, 99)
    assert point is None, f"不存在的索引應返回 None: {point}"

    print("  [OK] 通過")


def test_validate_center_point():
    """測試 8: validate_center_point 處理各種輸入"""
    print("\n測試 8: validate_center_point 處理各種輸入")

    # 有效座標
    assert validate_center_point([100, 200]) == (100.0, 200.0)
    assert validate_center_point((150, 250)) == (150.0, 250.0)

    # 無效輸入
    assert validate_center_point(None) is None
    assert validate_center_point([]) is None
    assert validate_center_point([100]) is None  # 只有一個值
    assert validate_center_point("invalid") is None
    assert validate_center_point([None, None]) is None

    print("  [OK] 通過")


def test_invalid_fps():
    """測試 9: 無效的 fps 值"""
    print("\n測試 9: 無效的 fps 值")

    invalid_data = {
        "metadata": {
            "video_path": "test.mp4",
            "fps": -25.0  # 負數 fps
        },
        "frames": [{"frame_id": 0}]
    }

    validator = DataValidator(verbose=False)
    is_valid, errors = validator.validate_tracking_json(invalid_data)

    print(f"  結果: is_valid={is_valid}")
    print(f"  錯誤: {errors}")

    assert not is_valid, "負數 fps 應該驗證失敗"
    assert any("fps" in err.lower() for err in errors), "錯誤訊息應提到 fps"
    print("  [OK] 通過")


def test_court_config_validation():
    """測試 10: 場地設定驗證"""
    print("\n測試 10: 場地設定驗證")

    valid_config = {
        "court_boundary_polygon": [[0, 0], [100, 0], [100, 100], [0, 100]],
        "exclusion_zones": [
            {"name": "裁判區", "polygon": [[10, 10], [20, 10], [20, 20], [10, 20]]}
        ]
    }

    validator = DataValidator(verbose=False)
    is_valid, errors = validator.validate_court_config(valid_config)

    print(f"  結果: is_valid={is_valid}")
    print(f"  錯誤: {errors}")

    assert is_valid, f"有效的場地設定應該通過: {errors}"

    # 測試缺少 court_boundary_polygon
    invalid_config = {"exclusion_zones": []}
    is_valid, errors = validator.validate_court_config(invalid_config)
    assert not is_valid, "缺少 court_boundary_polygon 應該失敗"

    print("  [OK] 通過")


if __name__ == '__main__':
    print("="*70)
    print("資料驗證器單元測試")
    print("="*70)

    try:
        test_valid_json_passes()
        test_missing_frames_fails()
        test_missing_metadata_fails()
        test_missing_ball_detections_warns()
        test_corrupt_json_fails()
        test_file_not_found()
        test_get_keypoint_valid()
        test_validate_center_point()
        test_invalid_fps()
        test_court_config_validation()

        print("\n" + "="*70)
        print("[SUCCESS] 所有測試通過！")
        print("="*70)

    except AssertionError as e:
        print("\n" + "="*70)
        print(f"[FAILED] 測試失敗: {e}")
        print("="*70)
        import traceback
        traceback.print_exc()
        sys.exit(1)
