#!/usr/bin/env python3
# build_edge_labels.py
# Build labeled edges for GNN training using GT dictionary

import os
import json
import csv
import itertools
from tqdm import tqdm  # ✅ for progress bar

def load_gt_dict(gt_json_path):
    """Load GT dictionary from JSON: { (frame, cell): track_id }"""
    with open(gt_json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {tuple(map(int, k.split(","))): v for k, v in raw.items()}

def build_edge_labels(gt_dict, output_csv_path, frame_diff=1, include_track_ids=True):
    """
    Build edge labels from GT dictionary.
    For each pair of frames (f, f+1), generate edges between all cell pairs.
    Label = 1 if both cells have the same track_id, else 0.
    """
    edge_labels = []

    # Get all unique frame IDs
    frames = sorted(set(f for f, _ in gt_dict.keys()))

    print(f"🧠 Building edges with frame_diff={frame_diff} across {len(frames)} frames...")

    for f in tqdm(frames, desc="🔄 Processing frames"):
        source_nodes = [(f, cid) for (f0, cid) in gt_dict if f0 == f]
        target_nodes = [(f + frame_diff, cid) for (f0, cid) in gt_dict if f0 == f + frame_diff]

        if not source_nodes or not target_nodes:
            continue

        for (sf, scid), (tf, tcid) in itertools.product(source_nodes, target_nodes):
            tid1 = gt_dict.get((sf, scid))
            tid2 = gt_dict.get((tf, tcid))
            label = 1 if tid1 == tid2 else 0

            if include_track_ids:
                edge_labels.append((sf, scid, tid1, tf, tcid, tid2, label))
            else:
                edge_labels.append((sf, scid, tf, tcid, label))

    print(f"✅ Total edge labels created: {len(edge_labels)}")

    # Write to CSV
    with open(output_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        if include_track_ids:
            writer.writerow(["source_frame", "source_cell", "source_track_id",
                             "target_frame", "target_cell", "target_track_id", "label"])
        else:
            writer.writerow(["source_frame", "source_cell", "target_frame", "target_cell", "label"])

        writer.writerows(edge_labels)
    print(f"📁 Saved edge labels to: {output_csv_path}")

def main():
    # ===== Manual config =====
    #dataset_name = 'Fluo-N2DH-SIM+'
    dataset_name = 'Fluo-C2DL-Huh7'
    seq_num = '02'
    gt_json_path = f"{dataset_name}-{seq_num}-gt_dict.json"
    output_path = f"{dataset_name}-{seq_num}-edge_labels.csv"
    frame_diff = 1
    include_track_ids = True

    gt_dict = load_gt_dict(gt_json_path)
    build_edge_labels(gt_dict, output_path, frame_diff, include_track_ids)

if __name__ == "__main__":
    main()

