# mystep4.py — Python replacement for the %%bash pipeline
import os, sys, subprocess, shlex


# Env (match your Hydra debugging preferences)
os.environ["HYDRA_FULL_ERROR"] = "1"

def posix(p: str) -> str:
    # Keep arguments clean and portable (matches your step3.py approach)
    return p.replace("\\", "/")




# ---------------- CONFIG ----------------
SEQUENCE = "02"  # sequence number
FOLDER_PATH = "data/gnntest"  # dataset root
DATA_NAME = "PhC-C2DH-U373"                  # dataset name
MODALITY = "2D"                              # dataset modality


root_abs = os.path.abspath(FOLDER_PATH)
log_abs = os.path.abspath('logs')
src_abs = os.path.abspath('src')


MODEL_METRIC_LEARNING = (
    posix(f"{root_abs}/software/parameters/Features_Models/{DATA_NAME}/all_params.pth")
)

MODEL_PYTORCH_LIGHTNING = (
    posix(f"{log_abs}/runs/2025-10-09/12-59-18/checkpoints/epoch=57.ckpt")
)

CODE_TRA = posix(f"{src_abs}/inference")
#PYTHONPATH = "/home/tony/code/cell-tracker-gnn"
PYTHONPATH=os.getcwd()
# -----------------------------------------

# Set env
os.environ["PYTHONPATH"] = PYTHONPATH

dataset_abs = os.path.abspath(os.path.join(FOLDER_PATH,'CTC',DATA_NAME))
dataset_relative = os.path.join(FOLDER_PATH,'CTC',DATA_NAME)

def run_cmd(args):
    print("[Run]", " ".join(shlex.quote(a) for a in args))
    subprocess.run(args, check=True, text=True)

# ---- Step 1: Preprocess (create CSVs from images/segmentation)
run_cmd([
    sys.executable, os.path.join(CODE_TRA, "preprocess_seq2graph_clean.py"),
    "-cs", "20",
    "-ii", posix(f"{dataset_relative}/{SEQUENCE}"),
    "-iseg", posix(f"{dataset_relative}/{SEQUENCE}_GT/TRA"),
    "-im", MODEL_METRIC_LEARNING,
    "-oc", posix(f"{dataset_relative}/{SEQUENCE}_CSV"),
])

# ---- Step 2: Inference
run_cmd([
    sys.executable, os.path.join(CODE_TRA, "inference_clean.py"),
    "-mp", MODEL_PYTORCH_LIGHTNING,
    "-ns", SEQUENCE,
    "-oc", posix(dataset_abs),
])

# ---- Step 3: Postprocess (create label maps)
run_cmd([
    sys.executable, os.path.join(CODE_TRA, "postprocess_clean.py"),
    "-modality", MODALITY,
    "-iseg", posix(f"{dataset_relative}/{SEQUENCE}_GT/TRA"),
    "-oi", posix(f"{dataset_relative}/{SEQUENCE}_RES_inference"),
])
