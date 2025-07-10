import os
import numpy as np
import tifffile
from pathlib import Path
from collections import defaultdict

def load_track_file(track_file):
    mapping = defaultdict(dict)
    with open(track_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cell_id, t, z, y, x = map(int, parts)
            mapping[int(t)][(y, x)] = cell_id
    return mapping

def match_ids_from_centroids(seg, centroids_dict):
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

def process_all_ctc_datasets(base_dir):
    base_dir = Path(base_dir)
    for gt_dir in sorted(base_dir.glob("*_GT")):
        seg_dir = gt_dir / "SEG"
        tra_dir = gt_dir / "TRA"
        track_txt = tra_dir / "man_track.txt"

        if not seg_dir.exists() or not track_txt.exists():
            print(f"[!] Skipping {gt_dir} (missing SEG or man_track.txt)")
            continue

        print(f"[→] Processing {gt_dir}")
        convert_seg_to_track(seg_dir, track_txt, tra_dir)

# === Run it on your full CTC dataset directory
ctc_base_dir = "CTC"
process_all_ctc_datasets(ctc_base_dir)
