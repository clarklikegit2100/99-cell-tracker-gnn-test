#%%bash
export HYDRA_FULL_ERROR=1
#export FOLDER_PATH=/content/drive/MyDrive/CellTracking/nature_method_ctc # TODO: update this path
export FOLDER_PATH=/home/tony/data/gnntest/CTC # TODO: update this path
export DATA_NAME=PhC-C2DH-U373 # TODO: update this path
# cell tracking training run
python run.py datamodule.dataset_params.main_path="${FOLDER_PATH}/ct_features/${DATA_NAME}" datamodule.dataset_params.exp_name="2D_SIM" datamodule.dataset_params.drop_feat=[]
