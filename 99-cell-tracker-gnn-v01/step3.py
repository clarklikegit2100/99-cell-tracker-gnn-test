import os
import subprocess

FOLDER_PATH = r"C:\Users\jzhao16\Documents\mydata\gnntest"
METRIC_PATH = r"C:\Users\jzhao16\Documents\mydata\gnntest\software\parameters\Features_Models\PhC-C2DH-U373\all_params.pth"
DATA_NAME = "PhC-C2DH-U373"

env = os.environ.copy()
env["HYDRA_FULL_ERROR"] = "1"

cmd = [
    "python", "run_feat_extract.py",
    f"params.input_images={FOLDER_PATH}\\{DATA_NAME}",
    f"params.input_masks={FOLDER_PATH}\\{DATA_NAME}",
    f"params.input_seg={FOLDER_PATH}\\{DATA_NAME}",
    f"params.output_csv={FOLDER_PATH}\\ct_features",
    "params.sequences=['01','02']",
    "params.seg_dir=_GT/TRA",
    "params.basic=False",
    f"params.input_model={METRIC_PATH}"
]

subprocess.run(cmd, env=env)
