import pandas as pd

# 讀取 CSV 檔案
df = pd.read_csv("D:/Github/beach-volleyball-tracker/output_data/video_segments_with_score/slicing_summary.csv")

# 統計得分次數
score_count = df["Scoring_Team"].value_counts()

# 確保缺少的隊伍補 0
team1_score = score_count.get("Team1", 0)
team2_score = score_count.get("Team2", 0)

print(f"比分：Team1 {team1_score} 比 Team2 {team2_score}")