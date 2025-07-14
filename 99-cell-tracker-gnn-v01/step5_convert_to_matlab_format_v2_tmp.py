import scipy.io as sio
import numpy as np
from collections import defaultdict

def load_flat_links(mat_path):
    """
    Load the flat format Cell_Linked_Lists: [target_frame, target_cell, source_cell]
    """
    data = sio.loadmat(mat_path)
    return np.array(data["Cell_Linked_Lists"])

def build_structured_cell_array(flat_links):
    """
    Convert flat format to structured MATLAB-style cell array:
    Cell_Linked_Lists[frame] = numpy array where
    index = current cell ID, value = parent cell ID
    """
    num_frames = int(flat_links[:, 0].max()) + 1
    cell_linked_lists = [np.array([], dtype=np.uint16) for _ in range(num_frames)]
    frame_to_cells = defaultdict(list)

    # Group links by target frame
    for cur_frame, cur_cell, prev_cell in flat_links:
        frame_to_cells[int(cur_frame)].append((int(cur_cell), int(prev_cell)))

    # Build vector for each frame
    for frame_idx in range(num_frames):
        cells = frame_to_cells.get(frame_idx, [])
        if cells:
            max_cell_id = max(c[0] for c in cells)
            vec = np.zeros(max_cell_id + 1, dtype=np.uint16)
            for cur_cell, parent_cell in cells:
                vec[cur_cell] = parent_cell
            cell_linked_lists[frame_idx] = vec

    return cell_linked_lists

def save_as_structured_mat(cell_linked_lists, output_path):
    """
    Save the structured cell array to .mat
    """
    sio.savemat(output_path, {"Cell_Linked_Lists": np.array(cell_linked_lists, dtype=object)})
    print(f"Saved structured Cell_Linked_Lists to: {output_path}")

def main():
    # ===== Input and Output Paths =====
    #input_mat_path = "Fluo-N2DH-SIM+-02-Cell_Linked_Lists_New_Format_Converted.mat"
    #dataset_name = 'Fluo-N2DH-SIM+'
    dataset_name =  'Fluo-C2DL-Huh7'
    seq_num = '02'  #


    input_mat_path =  f"{dataset_name}-{seq_num}-Cell_Linked_Lists_New_Format_Converted.mat"

    output_mat_path = f"{dataset_name}-{seq_num}-Cell_Linked_Lists_Matlab.mat"

    # ===== Process =====
    flat_links = load_flat_links(input_mat_path)
    structured_data = build_structured_cell_array(flat_links)
    save_as_structured_mat(structured_data, output_mat_path)

if __name__ == "__main__":
    main()
