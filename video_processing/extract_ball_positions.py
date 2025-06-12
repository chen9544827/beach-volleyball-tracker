import json
import sys

if len(sys.argv) != 5:
    print("用法: python extract_ball_positions.py <json_path> <start_frame> <end_frame> <output_txt>")
    sys.exit(1)

json_path = sys.argv[1]
start_frame = int(sys.argv[2])
end_frame = int(sys.argv[3])
output_txt = sys.argv[4]

with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

with open(output_txt, 'w', encoding='utf-8') as out:
    for frame_data in data:
        frame_id = frame_data.get('frame_id')
        if frame_id is None or not (start_frame <= frame_id <= end_frame):
            continue
        ball_list = frame_data.get('ball_detections', [])
        if not ball_list:
            continue
        # 取信心分數最高的球
        ball = max(ball_list, key=lambda x: x['confidence'])
        x, y = ball['center_x'], ball['center_y']
        out.write(f"frame {frame_id}: x={x:.1f}, y={y:.1f}\n")
print(f"已輸出 {start_frame}~{end_frame} 幀的球座標到 {output_txt}") 