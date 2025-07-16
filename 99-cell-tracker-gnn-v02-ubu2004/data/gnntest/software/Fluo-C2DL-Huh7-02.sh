#!/bin/bash

SEQUENCE="02"
SPACING="999 0.65 0.65"
FOV="50"
MIN_SIZE="1000"
SIZES="1000 1000000"
DATASET="${PWD}/../Fluo-C2DL-Huh7"
CODE_SEG="${PWD}/seg_code/new_version"
CODE_TRA="${PWD}"
SEG_MODEL="${PWD}/parameters/Seg_Models/Fluo-C2DL-Huh7/"
MODEL_METRIC_LEARNING="${PWD}/parameters/Features_Models/Fluo-C2DL-Huh7/all_params.pth"
MODEL_PYTORCH_LIGHTNING="${PWD}/parameters/Tracking_Models/Fluo-C2DL-Huh7/checkpoints/epoch=136.ckpt"
MODALITY="2D"
        
# seg prediction
python ${CODE_SEG}/Inference2D.py --gpu_id 0 --model_path ${SEG_MODEL} --sequence_path "${DATASET}/${SEQUENCE}" --output_path "${DATASET}/${SEQUENCE}_SEG_RES" --edge_dist 3 --edge_thresh 0.3 --min_cell_size 1000 --max_cell_size 100000 --fov 50 --centers_sigmoid_threshold 0.1 --min_center_size 10 --pre_sequence_frames 4 --data_format NCHW --save_intermediate --save_intermediate_path ${DATASET}/${SEQUENCE}_SEG_intermediate

# cleanup
rm -r "${DATASET}/${SEQUENCE}_SEG_intermediate"

# Finish segmentation - start tracking

# our model needs CSVs, so let's create from image and segmentation.
python ${CODE_TRA}/preprocess_seq2graph_patch_based.py -cs ${MIN_SIZE} -ii "${DATASET}/${SEQUENCE}" -iseg "${DATASET}/${SEQUENCE}_SEG_RES" -im "${MODEL_METRIC_LEARNING}" -oc "${DATASET}/${SEQUENCE}_CSV"

# run the prediction
python ${CODE_TRA}/inference_clean.py -mp "${MODEL_PYTORCH_LIGHTNING}" -ns "${SEQUENCE}" -oc "${DATASET}"

# postprocess
python ${CODE_TRA}/postprocess_clean.py -modality "${MODALITY}" -iseg "${DATASET}/${SEQUENCE}_SEG_RES" -oi "${DATASET}/${SEQUENCE}_RES_inference"

rm -r "${DATASET}/${SEQUENCE}_CSV" "${DATASET}/${SEQUENCE}_RES_inference" "${DATASET}/${SEQUENCE}_SEG_RES"