# step1.py  (launcher)  — normalize to POSIX
import os, sys, subprocess, shlex
import logging
os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["PILLOW_LOG_LEVEL"] = "WARNING"
# OR
logging.getLogger("PIL.TiffImagePlugin").setLevel(logging.WARNING)


root = os.path.abspath("data/gnntest/CTC")
data = os.path.join(root, "PhC-C2DH-U373")
out  = os.path.join(root, "basic_features")

def posix(p):  # <- key trick
    return p.replace("\\", "/")


METRIC_PATH= "data/gnntest/software/parameters/Features_Models/PhC-C2DH-U373/all_params.pth"  # TODO: update this path
abs_metric_path = os.path.abspath(METRIC_PATH)
seg_dir= '_GT/TRA'

cmd = [
    sys.executable, "run_feat_extract.py",
    f"params.input_images={posix(data)}",
    f"params.input_masks={posix(data)}",
    f"params.input_seg={posix(data)}",
    f"params.output_csv={posix(out)}",
    "params.sequences=['01','02']",
    f"params.seg_dir={posix(seg_dir)}",
    "params.basic=False",
    f"params.input_model={abs_metric_path}",
    "hydra.verbose=true",
]

print("[Step 1 : Run Extract Features]", " ".join(shlex.quote(x) for x in cmd))
subprocess.run(cmd, check=True, text=True)




#python run_train_metric_learning.py dataset.kwargs.data_dir_img="${FOLDER_PATH}/${DATA_NAME}" dataset.kwargs.data_dir_mask="${FOLDER_PATH}/${DATA_NAME}" dataset.kwargs.dir_csv="${FOLDER_PATH}/basic_features/${DATA_NAME}" dataset.kwargs.subdir_mask='GT/TRA'
