%%bash
source activate cell-tracking-challenge
set HYDRA_FULL_ERROR=1

#export FOLDER_PATH=/content/drive/MyDrive/CellTracking/nature_method_ctc
set FOLDER_PATH=C:\Users\jzhao16\Documents\mydata\gnntest

#export METRIC_PATH=/content/drive/MyDrive/CellTracking/nature_method_ctc/software/parameters/Features_Models/PhC-C2DH-U373/all_params.pth
set METRIC_PATH=C:\Users\jzhao16\Documents\mydata\gnntest\software\parameters\Features_Models\PhC-C2DH-U373\all_params.pth

set DATA_NAME=PhC-C2DH-U373

python run_feat_extract.py params.input_images="${FOLDER_PATH}/${DATA_NAME}" params.input_masks="${FOLDER_PATH}/${DATA_NAME}" params.input_seg="${FOLDER_PATH}/${DATA_NAME}" params.output_csv="${FOLDER_PATH}/ct_features/" params.sequences=['01','02']  params.seg_dir='_GT/TRA' params.basic=False params.input_model="${METRIC_PATH}"
