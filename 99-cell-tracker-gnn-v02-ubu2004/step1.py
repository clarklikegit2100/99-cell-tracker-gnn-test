import os
import sys, subprocess, shlex
# Environment variables
os.environ["HYDRA_FULL_ERROR"] = "1"

# Build paths
folder_path = "data/gnntest/CTC"
data_name = "PhC-C2DH-U373"

# Command as list (safer than string)
cmd = [
    sys.executable, "run_feat_extract.py",
    f"params.input_images={folder_path}/{data_name}",
    f"params.input_masks={folder_path}/{data_name}",
    f"params.input_seg={folder_path}/{data_name}",
    f"params.output_csv={folder_path}/basic_features",
    "params.sequences=['01','02']",
    "params.seg_dir=_GT/TRA",
    "params.basic=True",
]

# Run
subprocess.run(cmd, check=True)

# cmd = [
#     sys.executable, args.inference_py,
#     "--checkpoint_load", epoch_ckpt_path,
#     "--dataset_name", args.dataset_name, "--seq_num", vseq,
#     "--local_dir", args.local_dir, "--basicfeatures", args.basicfeatures,
#     "--seg_dir", os.path.join(args.local_dir, args.dataset_name, f"{vseq}_{args.seg_dir}"),
#     "--out_res", out_res,
#     "--k_div_try", str(args.k_div_try), "--max_children", str(args.max_children),
#     "--eval_bin_dir", args.eval_bin_dir,
#     "--select_thresholds", "ckpt",
#     "--sr_min", str(args.sr_min),
#     "--sr_max", str(args.sr_max),
#     "--log_ctc_csv", os.path.join("logs", f"{args.dataset_name}_{vseq}_ctc_train.csv"),
# ]
print("[Step 1 : Run Extrac Features]", " ".join(shlex.quote(x) for x in cmd))
p = subprocess.run(cmd, text=True)
if p.returncode != 0:
    raise RuntimeError(f"Command failed with exit code {p.returncode}")