#!/usr/bin/env python3
# build_edge_labels.py
# 用 GT_dict.json 构建跨帧节点对之间的边标签

import os
import json
import argparse
import itertools
import csv

def load_gt_dict(gt_json_path):
    with open(gt_json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    # 将 "frame,cell" 转换为元组 (int, int)
    return {tuple(map(int, k.split(","))): v for k, v in raw.items()}

def build_edge_labels(gt_dict, output_csv_path, frame_diff=1):
    """
    对于每对相邻帧(frame, frame+1)，判断所有cell_id两两组合之间是否属于同一track_id，生成边标签。
    gt_dict: {(frame, cell_id): track_id}
    保存为CSV，包含: source_frame, source_cell, target_frame, target_cell, label
    """
    edge_labels = []

    # 提取所有帧号
    frames = sorted(set(f for f, _ in gt_dict.keys()))
    for f in frames:
        source_nodes = [(f, cid) for (f0, cid) in gt_dict if f0 == f]
        target_nodes = [(f + frame_diff, cid) for (f0, cid) in gt_dict if f0 == f + frame_diff]
        if not source_nodes or not target_nodes:
            continue

        for (sf, scid), (tf, tcid) in itertools.product(source_nodes, target_nodes):
            tid1 = gt_dict.get((sf, scid))
            tid2 = gt_dict.get((tf, tcid))
            label = 1 if tid1 == tid2 else 0
            edge_labels.append((sf, scid, tf, tcid, label))

    print(f"共构建 {len(edge_labels)} 条边标签。")
    # 保存为CSV
    with open(output_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source_frame", "source_cell", "target_frame", "target_cell", "label"])
        writer.writerows(edge_labels)
    print(f"边标签已保存至 {output_csv_path}")

'''
def main():
    parser = argparse.ArgumentParser(description="使用 GT 字典构建图神经网络边标签")
    parser.add_argument("--gt_json", required=True, help="路径：gt_dict.json")
    parser.add_argument("--output", default="edge_labels.csv", help="输出CSV路径")
    parser.add_argument("--frame_diff", type=int, default=1, help="跨帧间隔（默认1，即 f 和 f+1）")
    args = parser.parse_args()

    gt_dict = load_gt_dict(args.gt_json)
    build_edge_labels(gt_dict, args.output, frame_diff=args.frame_diff)
'''
import json
# import argparse  # 不再需要
# from your_module import load_gt_dict, build_edge_labels  # 确保你有这两个函数

def main():
    # ===== 手动设置参数 =====
    #dataset_name= "PhC-C2DH-U373"  # dataset name
    dataset_name = 'Fluo-N2DH-SIM+'
    seq_num = '02'  #

    gt_json_path = f"{dataset_name}-{seq_num}-gt_dict.json"            # todo update to  GT dict JSON path

    output_path = f"{dataset_name}-{seq_num}-edge_labels.csv"          # 替换为输出CSV路径
    frame_diff = 1                           # between frames, default is 1 (i.e., f and f+1)

    # ===== 执行主逻辑 =====
    gt_dict = load_gt_dict(gt_json_path)
    build_edge_labels(gt_dict, output_path, frame_diff=frame_diff)

if __name__ == "__main__":
    main()

