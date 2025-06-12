import json
import argparse
import os

def find_serve_events_file(search_dir):
    """在指定目錄下尋找 serve_events_analysis.json 檔案"""
    print(f"Searching for serve_events_analysis.json in {search_dir}")
    for root, _, files in os.walk(search_dir):
        print(f"  Currently at root: {root}")
        for file in files:
            print(f"    Found file: {file}")
            if file == "serve_events_analysis.json":
                print(f"    Found serve_events_analysis.json at {os.path.join(root, file)}")
                return os.path.join(root, file)
    print(f"serve_events_analysis.json not found in {search_dir}")
    return None

def integrate_jump_events(player_jump_events_path, serve_events_path, output_path):
    # Load jump events
    with open(player_jump_events_path, 'r') as f:
        jump_events = json.load(f)
    
    # Load serve events (with auto-detected serving players)
    with open(serve_events_path, 'r') as f:
        serve_events = json.load(f)
    
    # Create mapping from hit_frame to serving_player_id
    serve_event_map = {}
    for event in serve_events:
        hit_frame = event.get('hit_frame')
        if hit_frame is not None:
            serve_event_map[hit_frame] = event.get('serving_player_id', None)
    
    # Create jump serve events
    jump_serve_events = []
    for event in jump_events:
        frame_num = event['frame_number']
        # Find closest serve event within ±20 frames
        closest_serve = None
        min_diff = float('inf')
        for hit_frame, player_id in serve_event_map.items():
            diff = abs(frame_num - hit_frame)
            if diff < min_diff and diff <= 20:  # Only consider nearby events
                min_diff = diff
                closest_serve = player_id
        
        jump_serve_events.append({
            'frame_number': frame_num,
            'jump_height': event['jump_height'],
            'is_jump_serve': closest_serve is not None,
            'serving_player_id': closest_serve,
            'keypoints': event['keypoints']
        })
    
    # Save jump serve events
    with open(output_path, 'w') as f:
        json.dump(jump_serve_events, f, indent=2)
    print(f"Jump serve events saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Integrate jump events with serving player information.")
    parser.add_argument("--jump_events", type=str, required=True,
                        help="Path to player_jump_events.json file.")
    parser.add_argument("--search_dir", type=str, required=True,
                        help="Directory to search for serve_events_analysis.json.")
    parser.add_argument("--output", type=str, default="jump_serve_events.json",
                        help="Path to save the jump_serve_events.json file.")
    
    args = parser.parse_args()
    
    # 尋找 serve_events_analysis.json 檔案
    serve_events_path = find_serve_events_file(args.search_dir)
    if not serve_events_path:
        print("錯誤: 找不到 serve_events_analysis.json 檔案")
        return
    
    integrate_jump_events(args.jump_events, serve_events_path, args.output)

if __name__ == "__main__":
    main()
