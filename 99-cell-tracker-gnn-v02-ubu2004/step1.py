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

cmd = [
    sys.executable, "run_feat_extract.py",
    f"params.input_images={posix(data)}",
    f"params.input_masks={posix(data)}",
    f"params.input_seg={posix(data)}",
    f"params.output_csv={posix(out)}",
    "params.sequences=['01','02']",
    "params.seg_dir=_GT/TRA",
    "params.basic=true",
    "hydra.verbose=true",
]

print("[Step 1 : Run Extract Features]", " ".join(shlex.quote(x) for x in cmd))
subprocess.run(cmd, check=True, text=True)
