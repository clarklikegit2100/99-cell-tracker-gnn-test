#%%bash
#source activate cell-tracking-challenge
export HYDRA_FULL_ERROR=1

#export FOLDER_PATH=/content/drive/MyDrive/CellTracking/nature_method_ctc
export FOLDER_PATH=/home/tony/data/gnntest/CTC

export METRIC_PATH=/home/tony/data/gnntest/software/parameters/Features_Models/PhC-C2DH-U373/all_params.pth
#export METRIC_PATH=C:\Users\jzhao16\Documents\mydata\gnntest\software\parameters\Features_Models\PhC-C2DH-U373\all_params.pth

export DATA_NAME=PhC-C2DH-U373

python run_feat_extract.py params.input_images="${FOLDER_PATH}/${DATA_NAME}" params.input_masks="${FOLDER_PATH}/${DATA_NAME}" params.input_seg="${FOLDER_PATH}/${DATA_NAME}" params.output_csv="${FOLDER_PATH}/ct_features/" params.sequences=['01','02']  params.seg_dir='_GT/TRA' params.basic=False params.input_model="${METRIC_PATH}"
