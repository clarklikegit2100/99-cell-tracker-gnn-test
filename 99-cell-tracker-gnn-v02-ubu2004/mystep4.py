# step_train.py — Python replacement for the %%bash cell
import os, sys, subprocess, shlex
import torch
# ---- Config (edit these) ----
FOLDER_PATH = "data/gnntest/CTC"       # TODO: update this path
DATA_NAME   = "PhC-C2DH-U373"                     # TODO: update this path
EXP_NAME    = "2D_SIM"                            # e.g., "2D_SIM"
DROP_FEAT   = []                                  # e.g., ["feat_a", "feat_b"]
# -----------------------------

# Env (match your Hydra debugging preferences)
os.environ["HYDRA_FULL_ERROR"] = "1"

def posix(p: str) -> str:
    # Keep arguments clean and portable (matches your step3.py approach)
    return p.replace("\\", "/")


root = os.path.abspath("data/gnntest/CTC")
ct_features_path = os.path.join(root, "ct_features", DATA_NAME)
ct_features_path_posix = posix(ct_features_path)

# Build the command as a list (no shell=True)
cmd = [
    sys.executable, "run.py",
    f"datamodule.dataset_params.main_path={ct_features_path_posix}",
    f"datamodule.dataset_params.exp_name={EXP_NAME}",
    # Represent [] as [] and lists as [a,b] without spaces (Hydra-friendly)
    "datamodule.dataset_params.drop_feat=" + (
        "[]" if not DROP_FEAT else "[" + ",".join(DROP_FEAT) + "]"
    ),
    "hydra.verbose=true",
]

print("[Train] ", " ".join(shlex.quote(x) for x in cmd))




print("torch:", torch.__version__)        # e.g. 2.6.0
print("torch CUDA tag:", torch.version.cuda)  # e.g. 12.1, 12.8, or None



subprocess.run(cmd, check=True, text=True)
