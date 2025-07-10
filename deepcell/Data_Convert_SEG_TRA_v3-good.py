import os
import numpy as np
import tifffile
from pathlib import Path
from collections import defaultdict

def load_track_file(track_file):
    """
    Parse CTC man_track.txt
    Returns: dict mapping (frame, seg_label) -> global_cell_id
    """
    mapping = defaultdict(dict)  # frame -> {local_id: global_id}
    with open(track_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cell_id, t, z, y, x = map(int, parts)
            mapping[int(t)][(int(y), int(x))] = cell_id
    return mapping

def match_ids_from_centroids(seg, centroids_dict):
    """
    Assign global cell IDs to segmented regions by matching centroid coordinates
    """
    output = np.zeros_like(seg, dtype=np.uint16)
    for seg_id in np.unique(seg):
        if seg_id == 0:
            continue
        mask = (seg == seg_id)
        yx = np.argwhere(mask).mean(axis=0).astype(int)
        key = (yx[0], yx[1])
        global_id = centroids_dict.get(key, 0)
        if global_id > 0:
            output[mask] = global_id
    return output

def convert_seg_to_track(seg_dir, track_txt, output_dir):
    seg_dir = Path(seg_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    track_map = load_track_file(track_txt)

    for seg_path in sorted(seg_dir.glob('man_seg*.tif')):
        frame_idx = int(seg_path.stem.replace('man_seg', ''))
        seg = tifffile.imread(str(seg_path))

        centroids_dict = track_map.get(frame_idx, {})
        track_mask = match_ids_from_centroids(seg, centroids_dict)

        out_path = output_dir / f"man_track{frame_idx:03d}.tif"
        tifffile.imwrite(out_path, track_mask)
        print(f"[✓] Saved {out_path}")

# === Modify these paths as needed ===

from pathlib import Path

def process_train_set(base_dir):
    base_dir = Path(base_dir)
    for i in range(90):  # 0 to 64 inclusive
        batch_name = f"{i:02d}_GT"
        seg_dir = base_dir / batch_name / "SEG"
        tra_dir = base_dir / batch_name / "TRA"
        track_txt = tra_dir / "man_track.txt"

        if seg_dir.exists() and track_txt.exists():
            print(f"[→] Processing batch {i:02d}")
            convert_seg_to_track(seg_dir, track_txt, tra_dir)
        else:
            print(f"[!] Skipping {batch_name} (missing SEG or man_track.txt)")

# Run it
process_train_set("CTC/train")



def process_train_set(base_dir):
    base_dir = Path(base_dir)
    for i in range(90):  # 0 to 64 inclusive
        batch_name = f"{i:02d}_GT"
        seg_dir = base_dir / batch_name / "SEG"
        tra_dir = base_dir / batch_name / "TRA"
        track_txt = tra_dir / "man_track.txt"

        if seg_dir.exists() and track_txt.exists():
            print(f"[→] Processing batch {i:02d}")
            convert_seg_to_track(seg_dir, track_txt, tra_dir)
        else:
            print(f"[!] Skipping {batch_name} (missing SEG or man_track.txt)")

# Run it

def process_val_set(base_dir):
    base_dir = Path(base_dir)
    for i in range(26):  # 0 to 64 inclusive
        batch_name = f"{i:02d}_GT"
        seg_dir = base_dir / batch_name / "SEG"
        tra_dir = base_dir / batch_name / "TRA"
        track_txt = tra_dir / "man_track.txt"

        if seg_dir.exists() and track_txt.exists():
            print(f"[→] Processing batch {i:02d}")
            convert_seg_to_track(seg_dir, track_txt, tra_dir)
        else:
            print(f"[!] Skipping {batch_name} (missing SEG or man_track.txt)")

# Run it



def process_test_set(base_dir):
    base_dir = Path(base_dir)
    for i in range(11):  # 0 to 64 inclusive
        batch_name = f"{i:02d}_GT"
        seg_dir = base_dir / batch_name / "SEG"
        tra_dir = base_dir / batch_name / "TRA"
        track_txt = tra_dir / "man_track.txt"

        if seg_dir.exists() and track_txt.exists():
            print(f"[→] Processing batch {i:02d}")
            convert_seg_to_track(seg_dir, track_txt, tra_dir)
        else:
            print(f"[!] Skipping {batch_name} (missing SEG or man_track.txt)")

# Run it
process_train_set("CTC/train")
process_train_set("CTC/val")
process_train_set("CTC/test")



