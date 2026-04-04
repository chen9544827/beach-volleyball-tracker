# core/filename_parser.py
# -*- coding: utf-8 -*-
"""
FIVB 沙灘排球影片檔名解析模組

解析 FIVB Beach Volleyball World Tour 影片檔名。
支援底線和連字號兩種格式：
  底線: FIVB_BVB_WT19_Edmonton_3Star_1718_C4_QT_W_007_...
  連字號: FIVB-BVB-WT18-Chetumal-4Star-251018-C1-MD-W-003-...

欄位定義（依 FIVB 世界巡迴賽 / VIS 系統慣例）：
  [0] FIVB           - 國際排球總會 (International Volleyball Federation)
  [1] BVB            - Beach VolleyBall (沙灘排球)
  [2] WT19           - World Tour 2019 賽季
  [3] Edmonton       - 比賽站點名稱
  [4] 3Star          - 站點星級 (3Star/4Star/5Star)
  [5] 1718           - 日期範圍（比賽天數，如 17-18 日），格式不固定
  [6] C4             - Court 4，第 4 號場地（不是攝影機）
  [7] QT             - 輪次 (QT=資格賽, MD=主賽, SF=準決賽, F=決賽)
  [8] W              - 性別 (W=女子, M=男子)
  [9] 007            - 該輪/該場地的第 N 場比賽
  [10+] 球員名_國家   - 兩隊球員姓名與國家代碼
  最後可能有 clip/set index (如 _2) 或品質標記 (如 _high)
"""

import os
import re
import logging
from typing import Dict, List, Optional
from collections import defaultdict


# 有效的輪次代碼
VALID_ROUNDS = {'QT', 'MD', 'SF', 'F', 'QF', 'R16', 'R32', 'R64', 'Pool'}

# 有效的性別代碼
VALID_GENDERS = {'M', 'W'}

# 場地編號格式 (Court number)
COURT_PATTERN = re.compile(r'^C(\d+)$')

# 星級格式
STAR_PATTERN = re.compile(r'^(\d+)Star$')

# 年份格式 (WT18, WT19, WT20, etc.)
YEAR_PATTERN = re.compile(r'^WT(\d{2})$')


def _detect_separator(filename: str) -> str:
    """
    偵測檔名使用的分隔符

    Args:
        filename: 不含路徑和副檔名的檔名

    Returns:
        '_' 或 '-'
    """
    underscores = filename.count('_')
    hyphens = filename.count('-')
    return '_' if underscores >= hyphens else '-'


def parse_filename(filename: str) -> Optional[Dict]:
    """
    解析 FIVB 影片檔名，支援底線和連字號兩種格式

    Args:
        filename: 檔名（可含路徑和副檔名）

    Returns:
        解析結果字典，解析失敗返回 None
        {
            'original_filename': str,
            'venue': str,           # 比賽站點名（如 Edmonton）
            'year': str,            # 賽季（如 WT19 = World Tour 2019）
            'court': str,           # 場地編號（如 C4 = Court 4）
            'gender': str,          # 性別 (M=男子, W=女子)
            'round': str,           # 輪次 (QT=資格賽, MD=主賽, SF=準決賽, F=決賽)
            'match_number': str,    # 場次編號 (001-999)
            'star_level': int,      # 站點星級 (3/4/5)
            'date_raw': str,        # 原始日期字串（日期範圍，格式不固定）
            'separator': str,       # 使用的分隔符 ('_' 或 '-')
            'group_key': str,       # 分組鍵 venue_year_court
            'suffix': str,          # 後綴 (clip/set index 或 'high' 等品質標記)
            'players_raw': str,     # 球員名和國家的原始字串
        }
    """
    # 去除路徑和副檔名
    basename = os.path.splitext(os.path.basename(filename))[0]

    # 偵測分隔符
    sep = _detect_separator(basename)
    parts = basename.split(sep)

    if len(parts) < 10:
        logging.warning(f"Filename too short to parse: {basename}")
        return None

    result = {
        'original_filename': os.path.basename(filename),
        'separator': sep,
    }

    # [0] FIVB
    if parts[0].upper() != 'FIVB':
        logging.warning(f"Expected 'FIVB' prefix, got: {parts[0]}")
        return None

    # [1] BVB (Beach VolleyBall)
    if parts[1].upper() != 'BVB':
        logging.warning(f"Expected 'BVB' as second part, got: {parts[1]}")
        return None

    # [2] 賽季年份 (e.g., WT19 = World Tour 2019)
    year_match = YEAR_PATTERN.match(parts[2])
    if not year_match:
        logging.warning(f"Invalid year format: {parts[2]}")
        return None
    result['year'] = parts[2]

    # [3] 比賽站點名
    result['venue'] = parts[3]

    # [4] 站點星級 (3Star/4Star/5Star)
    star_match = STAR_PATTERN.match(parts[4])
    if star_match:
        result['star_level'] = int(star_match.group(1))
    else:
        logging.warning(f"Invalid star level format: {parts[4]}")
        result['star_level'] = None

    # [5] 日期範圍（格式不固定，可能是 DDDD 如 1718，或 DDMMYY 如 251018）
    result['date_raw'] = parts[5]

    # [6] 場地編號 (Court number, e.g., C4 = Court 4)
    court_match = COURT_PATTERN.match(parts[6])
    if court_match:
        result['court'] = parts[6]
    else:
        logging.warning(f"Invalid court format: {parts[6]}")
        result['court'] = parts[6]

    # 保留 'camera' 作為向後相容的別名
    result['camera'] = result['court']

    # [7] 輪次 (QT=Qualification, MD=Main Draw, SF=Semi-Final, F=Final)
    if parts[7] in VALID_ROUNDS:
        result['round'] = parts[7]
    else:
        logging.warning(f"Unknown round code: {parts[7]}")
        result['round'] = parts[7]

    # [8] 性別 (W=Women, M=Men)
    if parts[8] in VALID_GENDERS:
        result['gender'] = parts[8]
    else:
        logging.warning(f"Unknown gender code: {parts[8]}")
        result['gender'] = parts[8]

    # [9] 場次編號
    result['match_number'] = parts[9]

    # [10+] 球員名、國家、後綴
    remaining = parts[10:]
    result['players_raw'] = sep.join(remaining) if remaining else ''

    # 嘗試偵測後綴 (clip/set index 或品質標記)
    suffix = ''
    if remaining:
        last = remaining[-1].lower()
        if last in ('high', 'low', 'mid'):
            suffix = last
        elif last.isdigit() and int(last) <= 9:
            # clip/set index (如 _2 代表同場比賽的第 2 段影片或第 2 局)
            suffix = last
    result['suffix'] = suffix

    # 分組鍵：venue + year + court（同場地同場號共享 court_config）
    result['group_key'] = f"{result['venue']}_{result['year']}_{result['court']}"

    return result


def group_videos(video_paths: List[str]) -> Dict[str, List[Dict]]:
    """
    將影片按 venue + year + court 分組

    同一組的影片來自同一站點、同一賽季、同一場地，
    共享 court_config 和 ROI config。

    Args:
        video_paths: 影片路徑列表

    Returns:
        分組字典: {"Edmonton_WT19_C4": [{"path": ..., "parsed": ...}, ...], ...}
        無法解析的影片放入 "_unparsed" 組
    """
    groups = defaultdict(list)

    for path in video_paths:
        parsed = parse_filename(path)
        entry = {
            'path': path,
            'parsed': parsed,
        }

        if parsed:
            groups[parsed['group_key']].append(entry)
        else:
            groups['_unparsed'].append(entry)

    return dict(groups)


def extract_group_key_from_path(file_path: str) -> Optional[str]:
    """
    從檔案路徑中提取 group_key（fallback 方法）

    當檔名本身無法解析時（例如 segment_010_Team1.mp4），
    嘗試從父目錄路徑中找到符合 group_key 格式的目錄名，
    或解析父目錄中的 FIVB 檔名。

    目錄結構範例：
        video_segments/Chetumal_WT18_C1/FIVB-BVB-.../normal_segments/segment_010.mp4
                       ^^^^^^^^^^^^^^^^ ← group_key 格式的目錄名

    Args:
        file_path: 影片檔案的完整路徑

    Returns:
        group_key 字串（如 'Chetumal_WT18_C1'），找不到時回傳 None
    """
    # 模式：{Venue}_{WTxx}_{Cx}
    group_key_pattern = re.compile(r'^[A-Za-z][A-Za-z0-9_-]+_WT\d{2}_C\d+$')

    parts = os.path.normpath(file_path).split(os.sep)

    # 往上走每一層目錄，檢查是否符合 group_key 格式
    for part in reversed(parts[:-1]):  # 排除檔名本身
        if group_key_pattern.match(part):
            return part

    # 第二策略：嘗試解析路徑中的 FIVB 目錄名
    for part in reversed(parts[:-1]):
        parsed = parse_filename(part)
        if parsed and parsed.get('group_key'):
            return parsed['group_key']

    return None


def scan_video_directory(directory: str, extensions: tuple = ('.mp4', '.avi', '.mkv', '.mov')) -> List[str]:
    """
    掃描目錄中的影片檔案

    Args:
        directory: 影片目錄路徑
        extensions: 支援的副檔名

    Returns:
        影片檔案路徑列表
    """
    video_paths = []
    if not os.path.isdir(directory):
        logging.error(f"Directory not found: {directory}")
        return video_paths

    for fname in sorted(os.listdir(directory)):
        if any(fname.lower().endswith(ext) for ext in extensions):
            video_paths.append(os.path.join(directory, fname))

    return video_paths


def print_group_summary(groups: Dict[str, List[Dict]]) -> None:
    """
    印出分組摘要

    Args:
        groups: group_videos() 的回傳結果
    """
    print(f"\n=== Video Group Summary ===")
    print(f"Total groups: {len(groups)}")
    print()

    for group_key in sorted(groups.keys()):
        entries = groups[group_key]
        if group_key == '_unparsed':
            print(f"  [UNPARSED] ({len(entries)} videos)")
        else:
            print(f"  {group_key}: {len(entries)} videos")
            if entries and entries[0]['parsed']:
                p = entries[0]['parsed']
                print(f"    Venue={p['venue']}, Year={p['year']}, "
                      f"Court={p['court']}, Star={p['star_level']}")

    total = sum(len(v) for v in groups.values())
    unparsed = len(groups.get('_unparsed', []))
    print(f"\nTotal videos: {total} (parsed: {total - unparsed}, unparsed: {unparsed})")
