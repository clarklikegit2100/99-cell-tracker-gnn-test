import scipy.io as sio
import numpy as np
from collections import defaultdict
import os

def load_flat_links(flat_mat_path):
    """
    Load the Cell_Linked_Lists from a .mat file with support for both
    2D numeric arrays and MATLAB-style object arrays (cells).
    """
    data = sio.loadmat(flat_mat_path)
    raw_links = data["Cell_Linked_Lists"]

    # Case 1: 2D array of shape (N, 3)
    if isinstance(raw_links, np.ndarray) and raw_links.ndim == 2 and raw_links.shape[1] == 3:
        return raw_links

    # Case 2: Object array (MATLAB-style cell)
    elif raw_links.dtype == 'O':
        try:
            flat_links = np.vstack([
                row for row in raw_links.flat
                if isinstance(row, np.ndarray) and row.size > 0
            ])
            if flat_links.shape[1] != 3:
                raise ValueError("Each entry must be a row of 3 values (target_frame, target_cell, source_cell).")
            return flat_links
        except Exception as e:
            raise ValueError(f"Failed to parse MATLAB object cell array: {e}")

    else:
        raise ValueError("Unrecognized or unsupported format in Cell_Linked_Lists.")

def convert_to_matlab_style(flat_links):
    """
    Convert flat format [target_frame, target_cell, source_cell] into:
    - 1xN MATLAB-style cell array
    - Each cell holds (M x 1) array where index = cell ID and value = parent ID
    """
    num_frames = int(flat_links[:, 0].max()) + 1
    structured_cell_array = [np.array([], dtype=np.uint8) for _ in range(num_frames)]

    # Group by frame
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

def save_to_matlab_format(structured_cell_array, output_path):
    """
    Save the result as a MATLAB .mat file using object type for cells.
    """
    sio.savemat(output_path, {"Cell_Linked_Lists": np.array(structured_cell_array, dtype=object)})
    print(f"✅ Saved to: {output_path}")

def main():


    #dataset_name= "PhC-C2DH-U373"  # dataset name
    #dataset_name = 'Fluo-N2DH-SIM+'
    dataset_name =  'Fluo-C2DL-Huh7'
    seq_num = '02'  #



    #input_path = f"{dataset_name}-{seq_num}-Cell_Linked_Lists_from_edges.mat"  # Output .mat file path
    input_path =  f"{dataset_name}-{seq_num}-Cell_Linked_Lists_New_Format_Converted.mat"

    output_path = "Cell_Linked_Lists_MATLAB_Format.mat"

    # Check if file exists
    if not os.path.exists(input_path):
        print(f"❌ Input file not found: {input_path}")
        return

    # Load and convert
    try:
        flat_links = load_flat_links(input_path)
        if flat_links.size == 0:
            print("❌ No links found in input file.")
            return
        structured = convert_to_matlab_style(flat_links)
        save_to_matlab_format(structured, output_path)
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
