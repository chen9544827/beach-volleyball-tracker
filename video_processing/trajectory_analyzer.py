import json
import argparse
import numpy as np
from scipy.optimize import curve_fit

def parabola(x, a, b, c):
    return a*x**2 + b*x + c

def analyze_trajectory(ball_events_path, output_path):
    # Load ball events
    with open(ball_events_path, 'r') as f:
        ball_events = json.load(f)
    
    trajectory_data = []
    for event in ball_events:
        toss_pos = event['toss_pos']
        peak_pos = event['peak_pos']
        hit_pos = event['hit_pos']
        
        # Collect points
        x = [toss_pos[0], peak_pos[0], hit_pos[0]]
        y = [toss_pos[1], peak_pos[1], hit_pos[1]]
        
        # Fit parabola
        try:
            popt, _ = curve_fit(parabola, x, y)
            a, b, c = popt
            
            # Calculate max height
            max_height = c - (b**2)/(4*a)
            max_height_x = -b/(2*a)
            
            # Predict landing point (where y = 0)
            roots = np.roots([a, b, c])
            landing_x = max([r for r in roots if r > hit_pos[0]], default=hit_pos[0])
            
            trajectory_data.append({
                'toss_frame': event['toss_frame'],
                'peak_frame': event['peak_frame'],
                'hit_frame': event['hit_frame'],
                'max_height': max_height,
                'max_height_x': max_height_x,
                'predicted_landing_x': landing_x,
                'parameters': [float(a), float(b), float(c)]
            })
        except Exception as e:
            print(f"Error fitting trajectory for event {event}: {e}")
            trajectory_data.append({
                'toss_frame': event['toss_frame'],
                'peak_frame': event['peak_frame'],
                'hit_frame': event['hit_frame'],
                'error': str(e)
            })
    
    # Save trajectory analysis
    with open(output_path, 'w') as f:
        json.dump(trajectory_data, f, indent=2)
    print(f"Trajectory analysis saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Analyze ball trajectory from event data.")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to ball_event_timestamps.json file.")
    parser.add_argument("--output", type=str, default="ball_trajectory_analysis.json",
                        help="Path to save the ball_trajectory_analysis.json file.")
    
    args = parser.parse_args()
    analyze_trajectory(args.input, args.output)

if __name__ == "__main__":
    main()
