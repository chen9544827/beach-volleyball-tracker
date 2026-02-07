# Beach Volleyball Tracker

FIVB Beach Volleyball video analysis system - automated ball tracking, serve detection, reception analysis, and statistical export.

## Features

| Feature | Description | Status |
|---------|-------------|--------|
| Ball Tracking | YOLO model for volleyball position tracking | Done |
| Player Detection | Player position and pose estimation | Done |
| Court Config | Interactive court boundary and exclusion zone setup | Done |
| Serve Detection | Identify serve events (toss -> apex -> hit) | Done |
| Server Identification | Lookback method to identify serving player | Done |
| Jump Serve Detection | Classify jump vs standing serve via ankle tracking | Done |
| Data Validation | Input validation system (mixed strategy) | Done |
| Filename Parsing | FIVB filename parsing + auto video grouping | Done |
| Court Zones | 3 serve zones + 6 reception zones per side | Done |
| Reception Detection | Ball tracking after hit to detect reception | Done |
| Result Export | CSV/Excel export with structured columns | Done |
| Video Context | Lightweight video metadata recording | Done |
| Batch Segmentation | Auto ROI matching + video slicing pipeline | Done |

---

## Project Structure

```
beach-volleyball-tracker/
├── core/                              # Core modules
│   ├── __init__.py
│   ├── ball_tracker.py                # Ball tracker (YOLO + trajectory prediction)
│   ├── serve_detector.py              # Serve detector (toss -> apex -> hit)
│   ├── server_identifier.py           # Server identifier (Lookback method)
│   ├── jump_serve_detector.py         # Jump serve detector
│   ├── data_validator.py              # Data validator
│   ├── error_messages.py              # Error message system
│   ├── filename_parser.py             # FIVB filename parser + video grouping
│   ├── court_zones.py                 # Court zone system (serve 3 + reception 6)
│   ├── reception_detector.py          # Reception detector
│   ├── result_exporter.py             # CSV/Excel result exporter
│   └── video_context.py              # Lightweight video metadata
│
├── video_processing/                  # Video processing
│   ├── track_ball_and_player_v2.py    # Main tracking pipeline
│   ├── video_slicer_by_score.py       # Score-based video segmentation
│   ├── roi_config_generator_v2.py     # ROI config GUI tool
│   └── batch_assign_venues.py         # Batch venue assignment
│
├── court_definition/                  # Court definition tools
│   └── court_config_generator.py      # Interactive court config tool
│
├── batch_tracking.py                  # Batch tracking script
├── batch_test_serve.py                # Batch serve analysis (main pipeline)
├── batch_segment_pipeline.py          # Batch video segmentation pipeline
├── batch_video_slicing.py             # Batch video slicing
│
├── roi_configs/                       # ROI configurations
│   ├── venues/                        # Venue templates (reusable)
│   │   ├── Edmonton.json
│   │   └── Gstaad.json
│   └── videos/                        # Per-video configs
│
├── court_config.json                  # Court config (boundary + exclusion zones)
│
├── test/                              # Tests
│   ├── test_filename_parser.py        # Filename parser tests (12 cases)
│   ├── test_court_zones.py            # Court zones tests (15 cases)
│   ├── test_jump_serve_logic.py       # Jump serve logic tests (7 cases)
│   └── test_data_validator.py         # Data validator tests (10 cases)
│
├── models/                            # YOLO models (not in repo)
│   ├── ball_best.pt                   # Volleyball detection model
│   └── yolov8m-pose.pt               # Player pose estimation model
│
├── input_video/                       # Input videos
│   ├── original_video/                # Full match videos
│   └── analyze_serve/                 # Segmented clips for analysis
│
├── CLAUDE.md                          # Claude Code project instructions
└── README.md
```

---

## Requirements

### Python Environment
```bash
Python 3.8+
# Anaconda base environment recommended
```

### Packages
```bash
pip install torch torchvision
pip install ultralytics
pip install opencv-python   # NOT opencv-python-headless
pip install numpy
pip install openpyxl         # For Excel export (optional, falls back to CSV)
```

---

## Quick Start

### Full Pipeline

```bash
# Step 1: Parse filenames and group videos
python batch_segment_pipeline.py --video-dir input_video/original_video --dry-run

# Step 2: Set up ROI configs (interactive GUI for missing venues)
python batch_segment_pipeline.py --video-dir input_video/original_video --roi-only

# Step 3: Segment videos by score changes
python batch_segment_pipeline.py --video-dir input_video/original_video

# Step 4: Set up court config (once per venue group)
python court_definition/court_config_generator.py \
    --video_path input_video/segment_001.mp4 \
    --output_path court_config.json

# Step 5: Batch tracking
python batch_tracking.py \
    --video-dir input_video/analyze_serve \
    --output-dir test_output \
    --court-config court_config.json

# Step 6: Batch serve analysis + reception detection + export
python batch_test_serve.py \
    --video-dir input_video/analyze_serve \
    --json-dir test_output \
    --output batch_test_output \
    --court-config court_config.json \
    --export-excel
```

---

## FIVB Filename Format

Videos follow the FIVB World Tour / VIS naming convention:

```
FIVB_BVB_WT19_Edmonton_3Star_1718_C4_QT_W_007_Strauss_T_...
|    |   |    |         |      |    |  |  | |
|    |   |    |         |      |    |  |  | +-- Match number (007)
|    |   |    |         |      |    |  |  +---- Gender (W=Women, M=Men)
|    |   |    |         |      |    |  +------- Round (QT/MD/SF/F)
|    |   |    |         |      |    +---------- Court number (C1-C4)
|    |   |    |         |      +--------------- Date range (match days)
|    |   |    |         +---------------------- Star level (3Star/4Star/5Star)
|    |   |    +-------------------------------- Venue name
|    |   +------------------------------------- Season (WT18/WT19)
|    +----------------------------------------- Beach VolleyBall
+---------------------------------------------- FIVB
```

Both underscore (`_`) and hyphen (`-`) separators are supported.

**Grouping key:** `venue_year_court` (e.g., `Edmonton_WT19_C4`)
- Videos in the same group share court_config and ROI config

---

## Batch Segmentation Pipeline

`batch_segment_pipeline.py` automates the full segmentation workflow:

1. Scan video directory and parse FIVB filenames
2. Group videos by venue + year + court
3. Match each group to ROI venue templates
4. For missing ROIs: launch interactive GUI for setup
5. Segment all videos by score changes

```bash
# Preview grouping (no actions)
python batch_segment_pipeline.py --video-dir input_video/original_video --dry-run

# Only set up missing ROIs
python batch_segment_pipeline.py --video-dir input_video/original_video --roi-only

# Only slice (skip groups without ROI)
python batch_segment_pipeline.py --video-dir input_video/original_video --slice-only

# Full pipeline
python batch_segment_pipeline.py --video-dir input_video/original_video
```

---

## Court Zone System

Standard beach volleyball court zones for serve and reception analysis:

```
              Net
  +-----+-----+-----+
  |  1  |  2  |  3  |  Far side (top of screen)
  | Left| Mid |Right|  Reception zones = front row 1-3
  +-----+-----+-----+
  |  4  |  5  |  6  |  Reception zones = back row 4-6
  | Left| Mid |Right|
  +-----+-----+-----+
              Net
  +-----+-----+-----+
  |  1  |  2  |  3  |  Near side (bottom of screen)
  | Left| Mid |Right|  Reception zones = front row 1-3
  +-----+-----+-----+
  |  4  |  5  |  6  |  Reception zones = back row 4-6
  | Left| Mid |Right|
  +-----+-----+-----+

Serve zones (behind end line): Left / Center / Right = 3 zones
```

Zones are computed from `court_boundary_polygon` (4 points) + `net_y` with perspective correction.

---

## Result Export

`batch_test_serve.py --export-excel` outputs structured data:

**Columns per serve event:**
- Basic info: video_name, venue, year, court, gender, round, match_number, star_level, group_key
- Serve analysis: serve_detected, toss_frame, hit_frame, hit_speed, server_index, confidence, serve_type, is_jump_serve, jump_height
- Serve zone: serve_zone (1-3), serving_side (near/far)
- Reception: reception_detected, reception_frame, reception_zone (1-6), receiver_index, time_to_reception
- Quality: quality_grade (A/B/C/F), ball_detection_rate, status

**Quality grades:**
- A: detection rate > 70%
- B: detection rate 50-70%
- C: detection rate 30-50%
- F: detection rate < 30% (suggest exclude)

---

## Core Algorithms

### Server Identification - Lookback Method

From toss frame, search backward up to 90 frames to find the first frame where a player overlaps with the ball (distance < threshold), excluding players in exclusion zones.

### Serve Detection State Machine

```
SEARCHING_TOSS -> CONFIRMING_TOSS -> AWAITING_APEX -> AWAITING_HIT -> [Event] -> COOLDOWN
```

### Reception Detection

After serve hit, track ball trajectory until it crosses the net and a receiver player is found nearby. Uses ball-player proximity + speed/direction change as combined criteria.

### Jump Serve Detection

Analyzes ankle keypoint Y-trajectory from FOUND frame to hit frame. Jump threshold: min 30px height (720p), 3+ consecutive frames.

---

## Tests

```bash
# Unit tests (44 total)
python test/test_filename_parser.py      # Filename parser (12 cases)
python test/test_court_zones.py          # Court zones (15 cases)
python test/test_jump_serve_logic.py     # Jump serve logic (7 cases)
python test/test_data_validator.py       # Data validator (10 cases)

# Regression test
python batch_test_serve.py \
    --video-dir input_video/analyze_serve \
    --json-dir test_output \
    --output verification_output \
    --court-config court_config.json
```

---

## Known Limitations

1. **Resolution dependency**: All pixel thresholds calibrated for 720p
2. **Net occlusion**: Far-side player detection rate drops due to net
3. **Camera angle**: Different angles require separate court_config
4. **Max tracking gap**: 15 frames (adjustable via `--max-occlusion`)
5. **Player ID instability**: Player indices may change between frames
6. **Windows encoding**: Output avoids Unicode special chars (uses [OK], [ERROR])

---

## Changelog

### v3.0 (2026-02-07) - Analysis Pipeline Expansion

**New modules:**
- `core/filename_parser.py` - FIVB filename parsing + auto video grouping
- `core/court_zones.py` - 3 serve zones + 6 reception zones per side (perspective-aware)
- `core/reception_detector.py` - Ball tracking after serve hit to detect reception
- `core/result_exporter.py` - CSV/Excel export with 30+ structured columns
- `core/video_context.py` - Lightweight video metadata (resolution, FPS)
- `batch_segment_pipeline.py` - Automated ROI matching + video segmentation

**Integrations:**
- `batch_test_serve.py` - Serve zone, reception detection, CSV/Excel export
- `track_ball_and_player_v2.py` - video_context in metadata output

**Tests:** 44 unit tests (12 + 15 + 7 + 10), 100% pass rate

### v2.1 (2026-02-02) - Stability and Validation

- Fixed jump serve consecutive sequence algorithm
- Fixed Windows cp950 encoding issues
- Added data validation system (`core/data_validator.py`)
- 17 unit tests, 100% pass rate

### v2.0 (2025-01-26)

- Lookback method for server identification
- Court exclusion zones
- Batch tracking and analysis scripts

### v1.0 (2025-01-23)

- Basic ball tracking, player detection, serve detection

---

## License

MIT License
