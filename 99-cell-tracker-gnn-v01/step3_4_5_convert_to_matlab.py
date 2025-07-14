#!/usr/bin/env python3
# Combined Step 3–5: CSV → simplified → MATLAB structured .mat

import os
import pandas as pd
import numpy as np
import scipy.io as sio
from collections import defaultdict

def load_edge_labels_csv(edge_csv_path):
    """
    Load edge labels CSV and extract only positive links (label==1)
    Output: list of [target_frame, target_cell, source_frame, source_cell]
    """
    df = pd.read_csv(edge_csv_path)
    df_positive = df[df["label"] == 1]
    return df_positive[["target_frame", "target_cell", "source_frame", "source_cell"]].values.tolist()

def convert_to_simplified_format(linked_list):
    """
    Convert [t, c, t-1, c_prev] → [t, c, c_prev]
    """
    return [[cur_frame, cur_cell, prev_cell] for cur_frame, cur_cell, _, prev_cell in linked_list]

def convert_to_matlab_style(flat_links):
    """
    Convert flat links into a MATLAB-style 1xN cell array.
    Each cell holds an (M x 1) array where index = cell ID, value = parent ID.
    """
    num_frames = int(max(row[0] for row in flat_links)) + 1
    structured_cell_array = [np.array([], dtype=np.uint8) for _ in range(num_frames)]

    frame_to_cells = defaultdict(list)
    for cur_frame, cur_cell, prev_cell in flat_links:
        frame_to_cells[int(cur_frame)].append((int(cur_cell), int(prev_cell)))

    for frame_idx in range(num_frames):
        pairs = frame_to_cells.get(frame_idx, [])
        if pairs:
            max_cell_id = max(c[0] for c in pairs)
            vec = np.zeros((max_cell_id + 1, 1), dtype=np.uint8)
            for cur_cell, parent_cell in pairs:
                vec[cur_cell] = parent_cell
            structured_cell_array[frame_idx] = vec

    return structured_cell_array

def save_matlab_format(data, output_path):
    """
    Save data to .mat file using MATLAB-compatible structure.
    """
    sio.savemat(output_path, {"Cell_Linked_Lists": np.array(data, dtype=object)})
    print(f"✅ Saved MATLAB-style file: {output_path}")

def main():
    # ======= Configurable Dataset Info =======
    dataset_name = 'Fluo-N2DH-SIM+'
    #dataset_name = 'Fluo-C2DL-Huh7'
    seq_num = '02'

    edge_csv = f"{dataset_name}-{seq_num}-edge_labels.csv"
    simplified_mat = f"{dataset_name}-{seq_num}-Cell_Linked_Lists_New_Format_Converted.mat"
    matlab_final_mat = f"{dataset_name}-{seq_num}-Cell_Linked_Lists_MATLAB_Format.mat"
    # =========================================

    if not os.path.exists(edge_csv):
        print(f"❌ Edge CSV not found: {edge_csv}")
        return

    # Step 3: Load edges and build initial linked list
    raw_links = load_edge_labels_csv(edge_csv)

    # Step 4: Simplify format to [frame, cell, parent_cell]
    flat_links = convert_to_simplified_format(raw_links)
    sio.savemat(simplified_mat, {"Cell_Linked_Lists": flat_links})
    print(f"✅ Saved simplified .mat: {simplified_mat}")

    # Step 5: Convert to MATLAB-style cell array
    structured = convert_to_matlab_style(flat_links)
    save_matlab_format(structured, matlab_final_mat)

if __name__ == "__main__":
    main()
