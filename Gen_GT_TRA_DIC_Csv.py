#!/usr/bin/env python3
# build_graph_pyg.py

import os
import torch
import pandas as pd
from torch_geometric.data import Data
import numpy as np
import argparse

def load_features_from_folder(folder):
    """加载每帧特征 CSV 文件，返回节点特征矩阵和映射字典"""
    all_feats = []
    node_map = {}  # (frame, seg_label) -> node_id
    node_id = 0
    for fname in sorted(os.listdir(folder)):
        if not fname.endswith('.csv'):
            continue
        frame = int(fname.split('_')[-1].split('.')[0])
        df = pd.read_csv(os.path.join(folder, fname))
        features = df[[col for col in df.columns if col.startswith("feat_")]].values
        all_feats.append(features)
        for row in df.itertuples():
            node_map[(int(row.frame_num), int(row.seg_label))] = node_id
            node_id += 1
    return np.vstack(all_feats), node_map

def load_edges(edge_csv_path, node_map):
    df = pd.read_csv(edge_csv_path)
    edge_index = []
    edge_label = []
    for row in df.itertuples():
        src = node_map.get((row.source_frame, row.source_cell))
        tgt = node_map.get((row.target_frame, row.target_cell))
        if src is not None and tgt is not None:
            edge_index.append([src, tgt])
            edge_label.append(row.label)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_label = torch.tensor(edge_label, dtype=torch.float)
    return edge_index, edge_label

def build_graph_dataset(feat_dir, edge_csv_path):
    x_np, node_map = load_features_from_folder(feat_dir)
    x = torch.tensor(x_np, dtype=torch.float)
    edge_index, edge_label = load_edges(edge_csv_path, node_map)
    return Data(x=x, edge_index=edge_index, edge_label=edge_label)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feat_dir', type=str, required=True, help='路径：每帧CSV特征文件夹')
    parser.add_argument('--edge_csv', type=str, required=True, help='路径：边标签CSV')
    parser.add_argument('--save_path', type=str, default="graph_data.pt", help='保存路径')
    args = parser.parse_args()

    data = build_graph_dataset(args.feat_dir, args.edge_csv)
    torch.save(data, args.save_path)
    print(f"图数据已保存至 {args.save_path}")

if __name__ == '__main__':
    main()
