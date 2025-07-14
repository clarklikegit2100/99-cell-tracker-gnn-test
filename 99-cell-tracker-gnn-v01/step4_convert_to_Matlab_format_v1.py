#!/usr/bin/env python3
# convert_cell_linked_lists_format.py

import scipy.io as sio

def load_cell_linked_lists(mat_path):
    """
    Load the Cell_Linked_Lists variable from a .mat file.
    """
    data = sio.loadmat(mat_path)
    return data["Cell_Linked_Lists"]

def convert_to_simplified_format(linked_list):
    """
    Convert each entry from [target_frame, target_cell, source_frame, source_cell]
    to [target_frame, target_cell, source_cell]
    """
    return [[cur_frame, cur_cell, prev_cell] for cur_frame, cur_cell, prev_frame, prev_cell in linked_list]

def save_to_mat(data, output_path):
    """
    Save the simplified list to a .mat file.
    """
    sio.savemat(output_path, {"Cell_Linked_Lists": data})
    print(f"Simplified format saved to {output_path}")

def main():
    # Input and output paths
    #source_mat_path = "Cell_Linked_Lists_from_edges.mat"
    #dataset_name = 'Fluo-N2DH-SIM+'
    dataset_name =  'Fluo-C2DL-Huh7'
    seq_num = '02'  #


    #edge_csv_path = f"{dataset_name}-{seq_num}-edge_labels.csv"          # 替换为输出CSV路径
    source_mat_path = f"{dataset_name}-{seq_num}-Cell_Linked_Lists_from_edges.mat"  # Output .mat file path

    output_mat_path =  f"{dataset_name}-{seq_num}-Cell_Linked_Lists_New_Format_Converted.mat"

    # Process
    original_list = load_cell_linked_lists(source_mat_path)
    simplified_list = convert_to_simplified_format(original_list)
    save_to_mat(simplified_list, output_mat_path)

if __name__ == "__main__":
    main()
