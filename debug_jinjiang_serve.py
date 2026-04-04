# debug_jinjiang_serve.py
# -*- coding: utf-8 -*-
"""
Jinjiang 發球偵測狀態機追蹤腳本
"""
import os
import sys
import json
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.serve_detector import ServeDetector, ServeState

JSON_PATH = "output/multi_venue_analysis/tracking_merged/Jinjiang_WT19_C1/segment_005_Team1_all_frames_data_with_pose.json"
NET_Y = 297
IMAGE_HEIGHT = 720

def get_ball_center(frame_data):
    if not frame_data or not frame_data.get('ball_detections'):
        return None
    valid_balls = [b for b in frame_data['ball_detections']
                   if not b.get('is_in_background_zone', False)]
    if not valid_balls:
        return None
    best_ball = max(valid_balls, key=lambda b: b.get('confidence', 0))
    box = best_ball.get('box_coords')
    if box:
        return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
    center = best_ball.get('center_point')
    if center:
        return np.array(center)
    return None

print("Loading JSON...")
with open(JSON_PATH, 'r') as f:
    data = json.load(f)

frames = data.get('frames', data)
if isinstance(frames, dict):
    frames = list(frames.values())

print(f"Total frames: {len(frames)}")

# Count ball detection rate
ball_count = sum(1 for f in frames if f and f.get('ball_detections'))
ball_rate = ball_count / len(frames)
print(f"Ball detection rate: {ball_count}/{len(frames)} = {ball_rate:.1%}")

# Build patched ServeDetector that prints state transitions
cfg = {
    'net_y': NET_Y,
    'image_height': IMAGE_HEIGHT,
    'ball_detection_rate': ball_rate,
}
detector = ServeDetector(cfg)
print(f"  net_y={NET_Y}, net_y_toss_margin={detector.net_y_toss_margin:.1f}")
print(f"  threshold for Method C = {NET_Y} + {detector.net_y_toss_margin:.1f} = {NET_Y + detector.net_y_toss_margin:.1f}")
print(f"  min_toss_height={detector.min_toss_height:.1f} (relaxed: {detector._relaxed_mode})")
print(f"  toss_initial_vy_thresh={detector.toss_initial_vy_thresh}")
print()

# Patch process_frame to add logging
original_process = detector.process_frame
prev_state = detector.state
transition_log = []

# Manual trace
prev_state_val = ServeState.SEARCHING_TOSS

def traced_process(prev_ball_pos, curr_ball_pos, frame_id, use_dynamic_threshold=True, player_detections=None):
    global prev_state_val
    result = original_process(prev_ball_pos, curr_ball_pos, frame_id,
                               use_dynamic_threshold=use_dynamic_threshold,
                               player_detections=player_detections)
    new_state = detector.state
    if new_state != prev_state_val:
        cand = dict(detector.event_candidate) if detector.event_candidate else {}
        ball_y = curr_ball_pos[1] if curr_ball_pos is not None else 'N/A'
        prev_y = prev_ball_pos[1] if prev_ball_pos is not None else 'N/A'
        print(f"  [Frame {frame_id:4d}] {prev_state_val.value} -> {new_state.value}  "
              f"ball_y={ball_y}, prev_y={prev_y}")
        if 'toss_position' in cand:
            print(f"             toss_pos_y={cand['toss_position'][1]:.1f}")
        prev_state_val = new_state
    return result

# Also intercept to log Method C/D rejections in SEARCHING_TOSS
import types

def verbose_process(self, prev_ball_pos, curr_ball_pos, frame_id,
                    use_dynamic_threshold=True, player_detections=None):
    global prev_state_val

    if self.state == ServeState.COOLDOWN:
        self.cooldown_frames += 1
        if self.cooldown_frames >= self.cooldown_duration:
            self.state = ServeState.SEARCHING_TOSS
            self.cooldown_frames = 0
            print(f"  [Frame {frame_id:4d}] COOLDOWN -> SEARCHING_TOSS")
            prev_state_val = ServeState.SEARCHING_TOSS
        return None

    if prev_ball_pos is None or curr_ball_pos is None:
        if self.state in [ServeState.AWAITING_APEX, ServeState.AWAITING_HIT,
                          ServeState.CONFIRMING_TOSS]:
            self.event_candidate['lost_frames_count'] = \
                self.event_candidate.get('lost_frames_count', 0) + 1
            if self.state == ServeState.AWAITING_APEX and \
               self.event_candidate.get('lost_frames_count', 0) > self.max_lost_frames_apex:
                print(f"  [Frame {frame_id:4d}] AWAITING_APEX -> SEARCHING_TOSS (lost_frames exceeded)")
                self.state = ServeState.SEARCHING_TOSS
                prev_state_val = ServeState.SEARCHING_TOSS
            elif self.state == ServeState.AWAITING_HIT and \
                 self.event_candidate.get('lost_frames_count', 0) > 15:
                print(f"  [Frame {frame_id:4d}] AWAITING_HIT -> SEARCHING_TOSS (lost_frames exceeded)")
                self.state = ServeState.SEARCHING_TOSS
                prev_state_val = ServeState.SEARCHING_TOSS
            elif self.state == ServeState.CONFIRMING_TOSS:
                lost = self.event_candidate.get('lost_frames_count', 0)
                if lost > self.confirming_toss_gap_tolerance:
                    print(f"  [Frame {frame_id:4d}] CONFIRMING_TOSS -> SEARCHING_TOSS (gap exceeded {lost})")
                    self.state = ServeState.SEARCHING_TOSS
                    prev_state_val = ServeState.SEARCHING_TOSS
                else:
                    self.event_candidate['confirm_start_frame'] = \
                        self.event_candidate.get('confirm_start_frame', frame_id) + 1
        return None

    velocity = curr_ball_pos - prev_ball_pos
    vx, vy = velocity[0], velocity[1]
    speed = np.linalg.norm(velocity)

    self.position_history.append({
        'frame_id': frame_id,
        'position': curr_ball_pos.copy(),
        'speed': speed, 'vx': vx, 'vy': vy
    })
    self.update_statistics(speed, -vy)

    if speed > self.max_plausible_speed:
        return None

    if 'lost_frames_count' in self.event_candidate:
        self.event_candidate['lost_frames_count'] = 0

    if use_dynamic_threshold:
        thresholds = self.get_dynamic_thresholds()
        hit_v_thresh = thresholds['hit_v']
        toss_vy_thresh = thresholds['toss_vy']
    else:
        hit_v_thresh = self.hit_v_thresh
        toss_vy_thresh = self.toss_initial_vy_thresh

    if self.state == ServeState.SEARCHING_TOSS:
        upward_vy = -vy
        if upward_vy > toss_vy_thresh:
            vertical_horizontal_ratio = abs(upward_vy) / (abs(vx) + 1e-6)
            if vertical_horizontal_ratio > self.vertical_ratio_thresh:
                # Method D
                if self._was_ball_descending_recently(frame_id):
                    print(f"  [Frame {frame_id:4d}] BLOCKED by Method D (ball descending recently) "
                          f"upward_vy={upward_vy:.1f} curr_y={curr_ball_pos[1]:.1f}")
                    return None
                # Method C
                if self.net_y is not None:
                    recent_15 = [e for e in self.position_history
                                 if frame_id - 15 <= e['frame_id'] < frame_id]
                    if recent_15:
                        max_hist_y = max(e['position'][1] for e in recent_15)
                        ref_y = max(max_hist_y, prev_ball_pos[1])
                    else:
                        ref_y = prev_ball_pos[1]
                    threshold_c = self.net_y + self.net_y_toss_margin
                    if ref_y > threshold_c:
                        print(f"  [Frame {frame_id:4d}] BLOCKED by Method C "
                              f"ref_y={ref_y:.1f} > threshold={threshold_c:.1f} "
                              f"(net_y={self.net_y}+margin={self.net_y_toss_margin:.1f}) "
                              f"upward_vy={upward_vy:.1f}")
                        return None

                # Method A
                if self._is_near_player(prev_ball_pos, player_detections):
                    self.state = ServeState.CONFIRMING_TOSS
                    self.event_candidate = {
                        'confirm_start_frame': frame_id,
                        'upward_frames_count': 1,
                        'lost_frames_count': 0,
                        'last_pos': curr_ball_pos.copy(),
                        'toss_position': prev_ball_pos.copy()
                    }
                    print(f"  [Frame {frame_id:4d}] SEARCHING_TOSS -> CONFIRMING_TOSS "
                          f"upward_vy={upward_vy:.1f} toss_pos_y={prev_ball_pos[1]:.1f} curr_y={curr_ball_pos[1]:.1f}")
                    prev_state_val = ServeState.CONFIRMING_TOSS
                else:
                    print(f"  [Frame {frame_id:4d}] BLOCKED by Method A (no nearby player) "
                          f"upward_vy={upward_vy:.1f}")
            # else: vertical ratio too small, ignore

    elif self.state == ServeState.CONFIRMING_TOSS:
        if (frame_id - self.event_candidate['confirm_start_frame']) > self.frames_to_validate_toss:
            print(f"  [Frame {frame_id:4d}] CONFIRMING_TOSS -> SEARCHING_TOSS (timeout, "
                  f"upward_count={self.event_candidate.get('upward_frames_count',0)})")
            self.state = ServeState.SEARCHING_TOSS
            prev_state_val = ServeState.SEARCHING_TOSS
            return None

        upward_vy = self.event_candidate['last_pos'][1] - curr_ball_pos[1]
        if upward_vy > 0:
            self.event_candidate['upward_frames_count'] += 1
        self.event_candidate['last_pos'] = curr_ball_pos.copy()

        if self.event_candidate['upward_frames_count'] >= self.min_upward_confirms:
            self.state = ServeState.AWAITING_APEX
            self.event_candidate['toss_start_frame'] = self.event_candidate['confirm_start_frame']
            self.event_candidate['lost_frames_count'] = 0
            print(f"  [Frame {frame_id:4d}] CONFIRMING_TOSS -> AWAITING_APEX "
                  f"(upward_count={self.event_candidate['upward_frames_count']})")
            prev_state_val = ServeState.AWAITING_APEX

    elif self.state == ServeState.AWAITING_APEX:
        if (frame_id - self.event_candidate.get('toss_start_frame', frame_id)) > self.max_frames_to_apex:
            print(f"  [Frame {frame_id:4d}] AWAITING_APEX -> SEARCHING_TOSS (max_frames_to_apex exceeded)")
            self.state = ServeState.SEARCHING_TOSS
            prev_state_val = ServeState.SEARCHING_TOSS
            return None

        if vy > 1:
            toss_start_y = self.event_candidate.get('toss_position', curr_ball_pos)[1]
            apex_y = prev_ball_pos[1]
            toss_height = toss_start_y - apex_y
            if toss_height < self.min_toss_height:
                print(f"  [Frame {frame_id:4d}] AWAITING_APEX -> SEARCHING_TOSS "
                      f"(toss_height={toss_height:.1f} < min={self.min_toss_height:.1f})")
                self.state = ServeState.SEARCHING_TOSS
                self.event_candidate = {}
                prev_state_val = ServeState.SEARCHING_TOSS
                return None

            self.state = ServeState.AWAITING_HIT
            self.event_candidate['apex_frame'] = frame_id
            self.event_candidate['apex_position'] = prev_ball_pos.copy()
            self.event_candidate['lost_frames_count'] = 0
            self.event_candidate['toss_height_px'] = float(toss_height)
            print(f"  [Frame {frame_id:4d}] AWAITING_APEX -> AWAITING_HIT "
                  f"apex_y={prev_ball_pos[1]:.1f} toss_height={toss_height:.1f} speed={speed:.1f}")
            prev_state_val = ServeState.AWAITING_HIT

    elif self.state == ServeState.AWAITING_HIT:
        if (frame_id - self.event_candidate.get('apex_frame', frame_id)) > self.max_frames_to_hit:
            print(f"  [Frame {frame_id:4d}] AWAITING_HIT -> SEARCHING_TOSS "
                  f"(max_frames_to_hit exceeded, apex={self.event_candidate.get('apex_frame')})")
            self.state = ServeState.SEARCHING_TOSS
            prev_state_val = ServeState.SEARCHING_TOSS
            return None

        if speed > hit_v_thresh:
            is_valid, reason = self.validate_hit(speed, vx, vy)
            if is_valid:
                est_frame, est_pos, orig_frame = self._estimate_contact_frame(
                    frame_id, curr_ball_pos, hit_v_thresh)
                event = {
                    'hit_frame_id': est_frame,
                    'hit_speed': speed,
                    'toss_start_frame': self.event_candidate.get('toss_start_frame'),
                    'apex_frame': self.event_candidate.get('apex_frame'),
                    'toss_height_px': self.event_candidate.get('toss_height_px'),
                    'hit_frame_id_detected': orig_frame,
                }
                self.detected_events.append(event)
                self.state = ServeState.COOLDOWN
                self.cooldown_frames = 0
                print(f"  [Frame {frame_id:4d}] *** SERVE DETECTED! *** "
                      f"speed={speed:.1f} hit_v_thresh={hit_v_thresh:.1f} vx={vx:.1f} vy={vy:.1f}")
                prev_state_val = ServeState.COOLDOWN
                return event
            else:
                print(f"  [Frame {frame_id:4d}] Speed {speed:.1f} > {hit_v_thresh:.1f} but REJECTED: {reason} "
                      f"(vx={vx:.1f}, vy={vy:.1f})")

    return None

detector.process_frame = types.MethodType(verbose_process, detector)

print("=== State Machine Trace ===")
for i in range(1, len(frames)):
    prev_ball_pos = get_ball_center(frames[i - 1])
    curr_ball_pos = get_ball_center(frames[i])
    curr_players = frames[i].get('player_detections', []) if frames[i] else []
    detector.process_frame(prev_ball_pos, curr_ball_pos, i,
                           use_dynamic_threshold=True,
                           player_detections=curr_players)

print(f"\n=== Total detected events: {len(detector.detected_events)} ===")
