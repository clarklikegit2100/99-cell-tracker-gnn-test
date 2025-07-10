# pip install deepcell-tracking tifffile

import os
import numpy as np
import tifffile
from deepcell_tracking.trk_io import load_trks
from tqdm import tqdm


def convert_trks_to_ctc(trk_file, split_name, output_root='CTC_DynamicNuclear'):
    # Load the .trks file
    print(f"Processing: {trk_file}")
    data = load_trks(trk_file)
    X_all = data['X']
    y_all = data['y']
    lineages_all = data['lineages']

    num_batches = X_all.shape[0]

    # Create split subdirectory
    split_root = os.path.join(output_root, split_name)
    os.makedirs(split_root, exist_ok=True)

    for batch in range(num_batches):
        out_prefix = f"{batch:02d}"
        image_dir = os.path.join(split_root, out_prefix)
        seg_dir = os.path.join(split_root, f"{out_prefix}_GT", "SEG")
        tra_dir = os.path.join(split_root, f"{out_prefix}_GT", "TRA")

        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(seg_dir, exist_ok=True)
        os.makedirs(tra_dir, exist_ok=True)

        frames = X_all[batch]
        masks = y_all[batch]
        lineage = lineages_all[batch]

        # Save raw images and segmentation masks
        for t in range(frames.shape[0]):
            im = (frames[t, ..., 0] * 255).astype(np.uint8)
            tifffile.imwrite(os.path.join(image_dir, f't{t:03d}.tif'), im)
            seg = masks[t].astype(np.uint16)
            tifffile.imwrite(os.path.join(seg_dir, f'man_seg{t:03d}.tif'), seg)

        # Save tracking info
        track_lines = []
        for cell_id, cell_data in lineage.items():
            for t in cell_data['frames']:
                mask = masks[t]
                coords = np.argwhere(mask == cell_id)
                if coords.size == 0:
                    continue
                mean_coords = coords.mean(axis=0)
                y_coord, x_coord = mean_coords[-2], mean_coords[-1]
                track_lines.append(f"{cell_id} {t} 0 {int(y_coord)} {int(x_coord)}")

        with open(os.path.join(tra_dir, 'man_track.txt'), 'w') as f:
            f.write('\n'.join(track_lines))

        print(f"Saved {split_name} batch {batch} → {out_prefix}")


# === Main processing for all splits ===
splits = {
    'train': 'train.trks',
    'val': 'val.trks',
    'test': 'test.trks'
}

for split_name, trk_path in splits.items():
    if os.path.exists(trk_path):
        convert_trks_to_ctc(trk_path, split_name)
    else:
        print(f"File not found: {trk_path}")
