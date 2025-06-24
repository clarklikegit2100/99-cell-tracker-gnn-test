#!/bin/bash
export HYDRA_FULL_ERROR=1

# weights_only=False only if weights_only was not passed as an argument.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

#FOLDER_PATH="/Users/my505/Documents/Data/gnntest"

FOLDER_PATH="C:\Users\jzhao16\Documents\mydata\gnntest"

DATA_NAME="PhC-C2DH-U373"
FULL_PATH="${FOLDER_PATH}/ct_features/${DATA_NAME}"

python run.py \
  trainer.gpus=0 \
  datamodule.dataset_params.main_path="${FULL_PATH}" \
  datamodule.dataset_params.exp_name="2D_SIM" \
  datamodule.dataset_params.drop_feat=[]

