#!/usr/bin/env python3
# build_edge_labels_from_gt.py

import pandas as pd
import os
import itertools

def build_edges(gt_csv, max_frame_skip=1):
    """
    输入: gt_dict.csv
    输出: 边列表 (frame_i, cell_i) -> (frame_j, cell_j) 及标签（是否为真实连接）
    """
    df = pd.read_csv(gt_csv)
    df = df.sort_values(by=["track_id", "frame"]).reset_index(drop=True)

    edges = []
    grouped = df.groupby("track_id")

    for track_id, group in grouped:
        group = group.sort_values(by="frame").reset_index(drop=True)
        for i in range(len(group) - 1):
            f1, c1 = group.loc[i, ["frame", "cell_id"]]
            f2, c2 = group.loc[i+1, ["frame", "cell_id"]]
            if 0 < (f2 - f1) <= max_frame_skip:
                edges.append((int(f1), int(c1), int(f2), int(c2), 1))  # 正样本

    return pd.DataFrame(edges, columns=["source_frame", "source_cell", "target_frame", "target_cell", "label"])


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_dict_csv', type=str, required=True, help='输入 GT 字典 CSV')
    parser.add_argument('--out_edge_csv', type=str, default='edge_labels.csv', help='输出边标签 CSV')
    parser.add_argument('--max_frame_skip', type=int, default=1, help='允许连接的最大帧间隔')
    args = parser.parse_args()

    edge_df = build_edges(args.gt_dict_csv, args.max_frame_skip)
    edge_df.to_csv(args.out_edge_csv, index=False)
    print(f"已保存边标签到: {args.out_edge_csv}")

if __name__ == '__main__':
    main()
