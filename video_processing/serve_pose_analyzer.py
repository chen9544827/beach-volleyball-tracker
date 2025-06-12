import os
import json
import argparse

# COCO keypoint indices
LEFT_HIP_INDEX = 11
RIGHT_HIP_INDEX = 12

def calculate_jump_height(keypoints):
    # Calculate jump height based on hip keypoints
    try:
        left_hip = keypoints[LEFT_HIP_INDEX]
        right_hip = keypoints[RIGHT_HIP_INDEX]
        left_hip_y = left_hip[1]  # y coordinate
        right_hip_y = right_hip[1]  # y coordinate
        
        # Use the average of left and right hip y coordinates
        hip_y = (left_hip_y + right_hip_y) / 2
        # For simplicity, we'll use the maximum hip y as initial height
        initial_hip_y = max(left_hip_y, right_hip_y)
        jump_height = initial_hip_y - hip_y
        return jump_height
    except (IndexError, TypeError):
        return None

def analyze_pose_data(all_frames_data_path, jump_threshold):
    # Load the all_frames_data_with_pose.json file
    try:
        with open(all_frames_data_path, 'r') as f:
            all_frames_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {all_frames_data_path} not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {all_frames_data_path}.")
        return

    player_jump_events = []
    for frame in all_frames_data:
        frame_number = frame['frame_id']
        for player in frame['player_detections']:
            keypoints_list = player['pose_keypoints']['keypoints_xyc_list']
            # Extract left and right hip keypoints by index
            try:
                keypoints = {
                    LEFT_HIP_INDEX: keypoints_list[LEFT_HIP_INDEX],
                    RIGHT_HIP_INDEX: keypoints_list[RIGHT_HIP_INDEX]
                }
            except IndexError:
                continue
                
            jump_height = calculate_jump_height(keypoints)
            if jump_height is not None and jump_height > jump_threshold:
                player_jump_events.append({
                    'frame_number': frame_number,
                    'jump_height': jump_height,
                    'keypoints': {
                        'left_hip': keypoints_list[LEFT_HIP_INDEX],
                        'right_hip': keypoints_list[RIGHT_HIP_INDEX]
                    }
                })

    return player_jump_events

def save_jump_events(player_jump_events, output_path):
    # Save the jump events to a JSON file
    try:
        with open(output_path, 'w') as f:
            json.dump(player_jump_events, f, indent=2)
        print(f"Jump events saved to {output_path}")
    except Exception as e:
        print(f"Error saving jump events: {e}")

def main():
    parser = argparse.ArgumentParser(description="Analyze pose data and extract jump events.")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to the _all_frames_data_with_pose.json file.")
    parser.add_argument("--output", type=str, default="player_jump_events.json",
                        help="Path to save the player_jump_events.json file.")
    parser.add_argument("--jump_threshold", type=float, default=50.0,
                        help="Minimum jump height (in pixels) to be considered a jump.")

    args = parser.parse_args()

    player_jump_events = analyze_pose_data(args.input, args.jump_threshold)
    if player_jump_events:
        save_jump_events(player_jump_events, args.output)

if __name__ == "__main__":
    main()
