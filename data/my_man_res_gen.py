import json
from collections import defaultdict

# 载入 GT_dict（字符串键格式）
with open("gt_dict.json", "r") as f:
    raw_dict = json.load(f)

# 转换为 {(frame:int, cell_id:int): track_id:int}
gt_dict = {tuple(map(int, k.split(","))): v for k, v in raw_dict.items()}

# 构建 track_id -> list of frames
track_frames = defaultdict(list)
for (frame, cell_id), track_id in gt_dict.items():
    track_frames[track_id].append(frame)

# 写入 txt 文件
with open("track_summary.txt", "w") as f:
    f.write("track_id,start_frame,end_frame,parent_id\n")
    for track_id, frames in track_frames.items():
        start = min(frames)
        end = max(frames)
        f.write(f"{track_id},{start},{end},-1\n")  # parent_id 默认写 -1
