#!/usr/bin/env python3
# generate_cell_linked_lists_from_edges.py

import pandas as pd
import scipy.io as sio

def build_cell_linked_list_from_edges(edge_csv_path):
    """
    Load edge labels from a CSV and construct a linked list of positive edges.
    Each entry: [target_frame, target_cell, source_frame, source_cell]
    """
    df = pd.read_csv(edge_csv_path)
    df_positive = df[df["label"] == 1]  # Keep only matching links
    linked_list = df_positive[["target_frame", "target_cell", "source_frame", "source_cell"]].values.tolist()
    return linked_list

def save_to_mat(data, output_path):
    """
    Save the linked list to a MATLAB .mat file.
    """
    sio.savemat(output_path, {"Cell_Linked_Lists": data})
    print(f"Saved Cell_Linked_Lists to {output_path}")

def main():
    # ======== Configurable Parameters ========
    #edge_csv_path = "edge_labels.csv"  # Input CSV file path

    #dataset_name= "PhC-C2DH-U373"  # dataset name
    #dataset_name = 'Fluo-N2DH-SIM+'
    dataset_name =  'Fluo-C2DL-Huh7'
    seq_num = '02'  #


    edge_csv_path = f"{dataset_name}-{seq_num}-edge_labels.csv"          # 替换为输出CSV路径
    output_mat_path = f"{dataset_name}-{seq_num}-Cell_Linked_Lists_from_edges.mat"  # Output .mat file path
    # =========================================

    linked_list = build_cell_linked_list_from_edges(edge_csv_path)
    save_to_mat(linked_list, output_mat_path)

if __name__ == "__main__":
    main()
