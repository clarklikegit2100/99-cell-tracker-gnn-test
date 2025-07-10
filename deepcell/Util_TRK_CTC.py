import os
import numpy as np
from deepcell_tracking.trk_io import load_trks

# Import functions from your files
from utils import save_ctc_raw, save_ctc_gt, convert_to_contiguous
# Note: deepcell_tracking.isbi_utils.trk_to_isbi is used internally

# === SETTINGS ===
trk_file = 'example.trk'  # your .trk file path
output_dir = 'CTC_Output'  # where to save CTC-style output
batch_id = 0  # ID to name output folders (e.g. 000, 001)

# === LOAD ===
data = load_trks(trk_file)
X_all = data['X']  # raw images
y_all = data['y']  # label masks
lineages_all = data['lineages']  # tracking info

# Each batch corresponds to 1 video/experiment
for batch in range(X_all.shape[0]):
    X = X_all[batch]        # shape: [T, H, W, 1]
    y = y_all[batch]        # shape: [T, H, W]
    lineage = lineages_all[batch]

    # Remove channel dimension if present
    if X.ndim == 4 and X.shape[-1] == 1:
        X = np.squeeze(X, axis=-1)

    # Optional: make label IDs contiguous across time
    y, lineage = convert_to_contiguous(y, lineage)

    # Save raw image
    save_ctc_raw(output_dir, batch, X)

    # Save GT (SEG and TRA)
    save_ctc_gt(output_dir, batch, y, lineage)

print(f"✅ Conversion completed! Output saved to: {output_dir}")
