import json
import csv
import argparse
import os

def generate_summary(jump_serve_events_path, trajectory_analysis_path, output_path):
    # Load jump serve events
    with open(jump_serve_events_path, 'r') as f:
        jump_serve_events = json.load(f)
    
    # Load trajectory analysis
    with open(trajectory_analysis_path, 'r') as f:
        trajectory_data = json.load(f)
    
    # Create mapping from frame number to trajectory data
    trajectory_map = {item['hit_frame']: item for item in trajectory_data}
    
    # Create summary data
    summary = []
    for event in jump_serve_events:
        frame_num = event['frame_number']
        trajectory = trajectory_map.get(frame_num, {})
        
        summary.append({
            'frame_number': frame_num,
            'is_jump_serve': event.get('is_jump_serve', None),
            'jump_height': event.get('jump_height', None),
            'max_height': trajectory.get('max_height', None),
            'predicted_landing_x': trajectory.get('predicted_landing_x', None),
            'parameters': str(trajectory.get('parameters', []))
        })
    
    # Save summary CSV
    with open(output_path, 'w', newline='') as f:
        fieldnames = ['frame_number', 'is_jump_serve', 'jump_height', 
                      'max_height', 'predicted_landing_x', 'parameters']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)
    print(f"Summary saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate summary CSV from jump serve events and trajectory analysis.")
    parser.add_argument("--jump_serve", type=str, required=True,
                        help="Path to jump_serve_events.json file.")
    parser.add_argument("--trajectory", type=str, required=True,
                        help="Path to ball_trajectory_analysis.json file.")
    parser.add_argument("--output", type=str, default="jump_serve_summary.csv",
                        help="Path to save the summary CSV file.")
    
    args = parser.parse_args()
    generate_summary(args.jump_serve, args.trajectory, args.output)

if __name__ == "__main__":
    main()
