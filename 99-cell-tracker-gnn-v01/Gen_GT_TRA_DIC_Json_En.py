#!/usr/bin/env python3
# build_gt_dict.py
# Dependencies: tifffile, imagecodecs

import os
import argparse
import json
import numpy as np
import tifffile

def build_gt_dict(gt_dir, seg_dir, num_frames=100):
    """
    Generate a mapping dictionary from (frame, cell_id) to track_id using GT and SEG folders.
    gt_dir: Path to the GT image folder, should contain images named like man_trackXXX.tif.
    seg_dir: Path to the SEG image folder, should contain images named like man_segXXX.tif.
    num_frames: Number of frames to process.
    Returns: Dictionary {(frame, cell_id): track_id, ...}.
    """
    gt_dict = {}  # Store the mapping results
    total_matches = 0  # Count total number of matched pairs

    # Iterate through each frame
    for frame in range(1, num_frames + 1):
        # Build the filename for the current frame in 3-digit format
        gt_filename = os.path.join(gt_dir, f"man_track{frame:03d}.tif")
        seg_filename = os.path.join(seg_dir, f"man_seg{frame:03d}.tif")

        # Skip the frame if either file is missing
        if not os.path.isfile(gt_filename) or not os.path.isfile(seg_filename):
            print(f"Frame {frame:03d} is missing GT or SEG file, skipping.")
            continue

        # Read GT and SEG images into numpy arrays
        gt_image = tifffile.imread(gt_filename)
        seg_image = tifffile.imread(seg_filename)

        # Get all non-zero track_ids from GT image
        track_ids = np.unique(gt_image)
        track_ids = track_ids[track_ids != 0]  # Remove 0 (background)

        match_count = 0  # Match count for current frame
        for tid in track_ids:
            # Create a mask for the current track_id
            mask = (gt_image == tid)
            if not np.any(mask):
                continue  # Skip if mask has no pixels

            # Find overlapping cell_ids in SEG image within the mask region
            overlapping_cells = seg_image[mask]
            overlapping_cells = overlapping_cells[overlapping_cells != 0]  # Exclude background
            if overlapping_cells.size == 0:
                continue

            # Count frequency of each cell_id and select the most frequent one
            cell_ids, counts = np.unique(overlapping_cells, return_counts=True)
            main_cell = cell_ids[np.argmax(counts)]  # Cell with the highest overlap

            # Add the mapping to the dictionary
            gt_dict[(frame, int(main_cell))] = int(tid)
            match_count += 1
            total_matches += 1

        # Print log for current frame
        print(f"Processed frame {frame:03d}: matched {match_count} track_ids")

    print(f"All frames processed, total {total_matches} mappings found.")
    return gt_dict

def main():
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(
        description="Generate cell tracking label mapping dictionary from GT and SEG folders and save as JSON")
    parser.add_argument("--gt_path", required=True, help="Path to GT folder containing man_trackXXX.tif files")
    parser.add_argument("--seg_path", required=True, help="Path to SEG folder containing man_segXXX.tif files")
    parser.add_argument("--num_frames", type=int, default=100,
                        help="Number of frames to process (default 100)")
    parser.add_argument("--output", default="gt_dict.json",
                        help="Output JSON file path (default gt_dict.json)")
    args = parser.parse_args()

    gt_dir = args.gt_path
    seg_dir = args.seg_path
    num_frames = args.num_frames
    output_path = args.output

    # Check path validity
    if not os.path.isdir(gt_dir):
        raise FileNotFoundError(f"Provided GT path does not exist or is not a directory: {gt_dir}")
    if not os.path.isdir(seg_dir):
        raise FileNotFoundError(f"Provided SEG path does not exist or is not a directory: {seg_dir}")

    # Build the GT_dict mapping
    gt_dict = build_gt_dict(gt_dir, seg_dir, num_frames)

    # Save as JSON file
    try:
        # Convert tuple keys to string keys for JSON serialization
        dict_to_save = {f"{frame},{cell}": track for (frame, cell), track in gt_dict.items()}
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(dict_to_save, f, ensure_ascii=False, indent=4)
        print(f"GT dictionary saved to {output_path}")
    except Exception as e:
        print(f"Failed to save JSON file: {e}")

if __name__ == "__main__":
    main()
