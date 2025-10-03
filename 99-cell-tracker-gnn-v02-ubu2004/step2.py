# step2.py — run metric learning cleanly
import os, sys, subprocess, shlex, logging

os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["PILLOW_LOG_LEVEL"] = "WARNING"
logging.getLogger("PIL.TiffImagePlugin").setLevel(logging.WARNING)

root = os.path.abspath("data/gnntest/CTC")
dataset_name = "PhC-C2DH-U373"
data_dir = os.path.join(root, dataset_name)
dir_csv  = os.path.join(root, "basic_features", dataset_name)  # <- NO $ vars

def posix(p):  # normalize for Windows
    return p.replace("\\", "/")

cmd = [
    sys.executable, "run_train_metric_learning.py",
    f"dataset.kwargs.data_dir_img={posix(data_dir)}",
    f"dataset.kwargs.data_dir_mask={posix(data_dir)}",
    f"dataset.kwargs.dir_csv={posix(dir_csv)}",
    "dataset.kwargs.subdir_mask=GT/TRA",
    "hydra.verbose=true",
    # no 'params.basic=true' here
]

print("[Step 2 : Run Metric Learning]", " ".join(shlex.quote(x) for x in cmd))
subprocess.run(cmd, check=True, text=True)
