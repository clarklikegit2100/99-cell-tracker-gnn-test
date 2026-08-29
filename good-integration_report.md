# Integration Report: Deep_CSTQ_Datasets → CellTracktor

*Analysis of `99-cell-tracker-gnn-v02-ubu2004` and `99-CellTracktor`*

---

## Part 1 — What "No Overlap Mode" Does

### Location in the codebase

The feature is implemented in two symmetric files:

- `src_metric_learning/Data/dataset_2D.py` — `ImgDataset.__init__()`, lines ~103–124
- `src_metric_learning/Data/dataset_3D.py` — same class, same logic

It is activated via the config key `deviation` in:

```
configs/metric_learning/dataset/dataset_2D.yaml
```

```yaml
deviation: 'no_overlap'   # or 'with_overlap'
```

### What the two modes do

**`with_overlap` (frame-based split):**
The sequence is divided by frame range. For an 80/20/0 split, the first 80 % of frames go to train, the next 20 % to validation. Because cells can live across many frames, the same cell often appears in both the train and val sets.

**`no_overlap` (cell-ID-based split):**
```python
# dataset_2D.py, lines 112–124
un_lables, un_counts = np.unique(curr_df_cells.id, return_counts=True)
un_counts = 100 * np.cumsum(un_counts) / un_counts.sum()
np_split = np.array(split).cumsum()          # e.g. [80, 100, 100]
train_val_test_split = []
for ind, d_type in enumerate(['train', 'valid', 'test']):
    curr_precent = np_split[ind]
    train_val_test_split.append(
        np.argmin(np.abs(un_counts - curr_precent)) + 1
    )
```

It sorts all unique cell IDs by cumulative frame-occurrence count, then finds the index where that cumulative percentage crosses 80 % (train/val boundary). Every frame of a given cell ends up in exactly one split. No cell is shared between train and val.

### Why it matters for metric learning

The GNN pipeline uses a ResNet18 + MLP metric learning step (stage 2, `run_train_metric_learning.py`) whose job is to learn a 128-dimensional embedding space where the same cell at different frames is close, and different cells are far apart. If the same cell ID appears in both train and val, the metric learner has already "seen" that cell during training when it evaluates on val — classic data leakage. No-overlap mode eliminates that.

### Should you apply this concept to Deep_CSTQ_Datasets?

**Context matters.** No-overlap mode is a property of the *metric learning* training split inside the GNN pipeline, not of the dataset generation itself. Deep_CSTQ_Datasets generates modified image sequences (deletions, etc.); it does not train a metric learner itself.

However, if you use the GNN pipeline's metric learning stage on your Deep_CSTQ_Datasets output (which is likely, since the GNN pipeline is the intended consumer), then you should verify that `deviation: 'no_overlap'` is active in `configs/metric_learning/dataset/dataset_2D.yaml`. It already defaults to `no_overlap`, so no change is required unless it was overridden.

**One scenario where it matters more:** If Deep_CSTQ_Datasets' deletion strategy systematically removes cells in early frames but keeps those same cell IDs in later frames, a frame-based split would put the "intact" version of a cell in val and the "deleted" version in train, creating asymmetric leakage. No-overlap mode avoids this automatically by splitting on cell IDs rather than frame positions.

**Concrete recommendation:** Leave `deviation: 'no_overlap'` as-is (it is the default). If you ever switch to a deletion strategy that removes cells entirely for whole sequences, the mode still works correctly because those cells simply never appear in the held-out split.

---

## Part 2 — Feeding Deep_CSTQ_Datasets Output into CellTracktor

### How the pipeline connects

```
Deep_CSTQ_Datasets output (CTC format)
        ↓
data/<your_dataset>/CTC/train/ and CTC/val/
        ↓  (run once)
scripts/create_coco_dataset_from_CTC.py
        ↓
data/<your_dataset>/COCO/   (COCO format)
        ↓  (add config file)
cfgs/train_<your_dataset>.yaml
        ↓
python src/train.py with dataset=<your_dataset>
```

### Step 1 — Structure your CTC output correctly

The conversion script (`scripts/create_coco_dataset_from_CTC.py`, line 38–39) scans for directories whose names end in two digits:

```python
train_sets = sorted([x for x in (datapath.parent / 'CTC' / 'train').iterdir()
                     if x.is_dir() and re.findall('\d\d$', x.name)])
```

Your Deep_CSTQ_Datasets output must be arranged as follows **before** running the conversion:

```
data/
└── Deep_CSTQ/                        ← choose any name; must match config
    └── CTC/
        ├── train/
        │   ├── 01/                   ← raw image frames (*.tif)
        │   │   ├── t000.tif
        │   │   ├── t001.tif
        │   │   └── ...
        │   ├── 01_GT/
        │   │   └── TRA/
        │   │       ├── man_track.txt
        │   │       ├── man_track000.tif   ← uint16 labeled masks
        │   │       ├── man_track001.tif
        │   │       └── ...
        │   ├── 02/
        │   ├── 02_GT/TRA/
        │   └── ...  (add more sequences as needed)
        └── val/
            ├── 01/
            ├── 01_GT/TRA/
            └── ...
```

**Critical details from the conversion script:**

- Image files: any name, TIFF format, sorted alphabetically = frame order.
- Mask directory: must be `<seq_id>_GT/TRA/` (not `_GT/SEG/`). The script reads `man_track.txt` and `man_track*.tif` from here.
- `man_track.txt` format: one row per cell, four space-separated integers — `cell_id  start_frame  end_frame  parent_id` (0 if no parent).
- **Minimum frames:** sequences with fewer than 4 frames are silently skipped (line 53: `if len(fps) < 4: continue`). Make sure each generated sequence has ≥ 4 frames.
- Masks must be `uint16`, with pixel value = cell label (0 = background).

### Step 2 — Run the conversion script

The script currently has hardcoded paths at lines 11–14:

```python
dataset = '2D'
datapath = Path('/home/clark/Documents/GitHub/99-CellTracktor/code-ubu2004/data') / dataset / 'COCO'
```

Edit those two lines for your dataset:

```python
dataset = 'Deep_CSTQ'
datapath = Path('/home/jack/Documents/GitHub/99-CellTracktor/code-ubu2004/data') / dataset / 'COCO'
```

Also check lines 16–23. If your images are not from a mother-machine setup, the `else` branch already handles you correctly (`resize = False`, `target_size = None`).

Then run from the `scripts/` directory:

```bash
cd /path/to/99-CellTracktor/code-ubu2004/scripts
python create_coco_dataset_from_CTC.py
```

This produces:

```
data/Deep_CSTQ/COCO/
├── train/
│   ├── img/          ← renamed TIFs: CTC_01_frame_000.tif, CTC_01_frame_001.tif, ...
│   └── gt/           ← matching uint16 mask TIFs with same names
├── val/
│   ├── img/
│   └── gt/
├── annotations/
│   ├── train/
│   │   └── anno.json
│   └── val/
│       └── anno.json
└── man_track/
    ├── train/
    │   └── 01.txt
    └── val/
        └── 01.txt
```

### Step 3 — Create a training config file

Copy the closest existing config as a starting point:

```bash
cp cfgs/train_2D.yaml cfgs/train_Deep_CSTQ.yaml
```

Edit these keys in `cfgs/train_Deep_CSTQ.yaml`:

| Key | Change to | Reason |
|-----|-----------|--------|
| `dataset` | `Deep_CSTQ` | Must match the directory name under `data/` |
| `data_dir` | `/absolute/path/to/99-CellTracktor/code-ubu2004/data` | Your actual path |
| `output_dir` | `/absolute/path/to/99-CellTracktor/code-ubu2004/results/Deep_CSTQ` | Where checkpoints are saved |
| `num_queries` | Set to ~1.5× max cells per frame | Run the conversion script first; it prints `Max number of cells in all frames is N` at the end |
| `target_size` | `(H, W)` matching your image dimensions | If images are not 584×600 |
| `epochs` | Start with 10–20 | The 2D config has only 3, which is minimal |
| `backbone` | `resnet18` if GPU memory is tight | Reduces VRAM requirement |

How `data_dir` and `dataset` are used (from `src/trackformer/datasets/mot.py`, line 208):

```python
root = Path(args.data_dir) / args.dataset / 'COCO'
# → /your/data_dir/Deep_CSTQ/COCO/
```

### Step 4 — Train

```bash
cd /path/to/99-CellTracktor/code-ubu2004
python src/train.py with dataset=Deep_CSTQ
```

Sacred automatically loads `cfgs/train_Deep_CSTQ.yaml` based on the dataset name.

---

## Part 3 — Mismatches and Gaps to Be Aware Of

### Gap 1: TRA vs SEG masks

`create_coco_dataset_from_CTC.py` reads GT exclusively from `_GT/TRA/` (line 58). If Deep_CSTQ_Datasets produces segmentation masks in `_GT/SEG/` format only, you have two options:

- Configure Deep_CSTQ_Datasets to also write TRA-format output (tracking markers), or
- Symlink/copy `_GT/SEG/` to `_GT/TRA/` if the mask format is identical (same uint16 labeled TIFs).

The GNN pipeline's feature extractor in contrast supports both via `seg_dir` in `configs/feat_extract/params/params.yaml` (`'_GT/TRA'`, `'_GT/SEG'`, or `'_ST/SEG'`), so the GNN side is flexible. CellTracktor is not.

### Gap 2: Hardcoded paths in the conversion script

`create_coco_dataset_from_CTC.py` has no CLI argument parsing — the `dataset` name and `datapath` are hardcoded at lines 11–14. You must edit the file each time you run it for a new dataset, or add `argparse` yourself.

### Gap 3: CellTracktor requires a CUDA GPU

The `MultiScaleDeformableAttention` CUDA extension must be compiled from source (step 7 of the README). Training and inference both fail on CPU. The GNN pipeline has no such hard requirement.

### Gap 4: Image normalization differences

The GNN pipeline normalises images cell-wise (`normalize_type: 'MinMaxCell'` in the metric learning config). CellTracktor normalises to ImageNet statistics per-channel (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`) after converting the single-channel TIF to a 3-channel RGB image. Both pipelines handle raw uint16 TIFs, but their internal normalisation is completely different. You do not need to pre-normalise your images before passing them to either pipeline — just give them raw TIFs.

### Gap 5: Frame count minimum

CellTracktor skips sequences with < 4 frames. The GNN pipeline imposes no such minimum. If Deep_CSTQ_Datasets generates short test sequences (e.g., 2–3 frames for ablation studies), CellTracktor will silently drop them. Check the script's printed output for `folders skipped!`.

### Gap 6: `dataset` key vs config file name mismatch

In `cfgs/train_2D.yaml` (line 150), the `dataset` key is set to `DynamicNuclearNet-tracking-v1_0`, not `2D`. This is the name used to look up data under `data_dir`. When you create `cfgs/train_Deep_CSTQ.yaml`, make sure the `dataset` key inside the file matches the directory name under `data/` — they are independent of the config file's own filename.

### Gap 7: Cell divisions

The GNN pipeline models divisions through its graph construction. CellTracktor models divisions through a separate division prediction head and optionally a future frame (`flex_div: false` by default). If Deep_CSTQ_Datasets' modifications involve cell divisions (splitting one cell into two), both pipelines will handle them, but you must verify that `man_track.txt` contains non-zero parent IDs for daughter cells, as both pipelines use this field to identify division events.

### Gap 8: No overlap mode does not apply to CellTracktor training

CellTracktor splits data at the **sequence level**: entire sequence folders (e.g., `01/`, `02/`) go into `CTC/train/` or `CTC/val/`. There is no cell-level splitting. The no-overlap concept from the GNN pipeline has no equivalent here and does not need to be ported.

---

## Quick-Reference Checklist

```
[ ] Deep_CSTQ_Datasets output uses _GT/TRA/ format (not _GT/SEG/ only)
[ ] Each generated sequence has ≥ 4 frames
[ ] Sequence folders are named with 2-digit suffixes (01, 02, ...)
[ ] man_track.txt exists for each sequence with correct format
[ ] Masks are uint16 labeled TIFs
[ ] Edit lines 11–14 of create_coco_dataset_from_CTC.py with your dataset name and path
[ ] Run create_coco_dataset_from_CTC.py and note the reported max cell count
[ ] Copy cfgs/train_2D.yaml → cfgs/train_Deep_CSTQ.yaml
[ ] Update: dataset, data_dir, output_dir, num_queries, target_size in the new config
[ ] Compile MultiScaleDeformableAttention CUDA extension (required)
[ ] Run: python src/train.py with dataset=Deep_CSTQ
[ ] Verify deviation: 'no_overlap' is set in configs/metric_learning/dataset/dataset_2D.yaml (GNN pipeline only)
```
