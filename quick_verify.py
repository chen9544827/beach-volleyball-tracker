# quick_verify.py
# 快速驗證新增功能

import os

print("\n" + "="*60)
print("階段1+2 功能驗證")
print("="*60 + "\n")

# 檢查新增的工具
tools = [
    ("tools/scoreboard_roi_marker.py", "ROI 標定工具"),
    ("tools/tracking_quality_checker.py", "追蹤品質檢查工具")
]

# 檢查配置檔
configs = [
    ("configs/bytetrack_ball.yaml", "球體追蹤配置"),
    ("configs/bytetrack_player.yaml", "球員追蹤配置")
]

# 檢查主腳本
scripts = [
    ("batch_slice_videos.py", "批次切片腳本"),
    ("STAGE1_2_GUIDE.md", "使用指南")
]

all_files = tools + configs + scripts

print("檢查新增檔案:\n")
passed = 0
total = len(all_files)

for filepath, description in all_files:
    exists = os.path.exists(filepath)
    status = "✓" if exists else "✗"
    print(f"  [{status}] {description}")
    print(f"      {filepath}")
    if exists:
        passed += 1

print(f"\n結果: {passed}/{total} 檔案存在")

if passed == total:
    print("\n✅ 所有檔案已成功建立！")
    print("\n下一步:")
    print("  請參考 STAGE1_2_GUIDE.md 開始使用")
else:
    print("\n⚠️  部分檔案缺失")

print("\n" + "="*60)
