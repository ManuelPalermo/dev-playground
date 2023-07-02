#!/use/bin//bash

DATASET_PATH=$1
FILTERED_PATH="${DATASET_PATH}/filtered/"
SCORED_PATH="${DATASET_PATH}/top_k_scored/"

# filter similar images
python filter_similar.py  \
    --dataset_path ${DATASET_PATH} \
    --output_path ${FILTERED_PATH} \
    --clear_old

# from the filtered, select the ones which are most different
python select_top_k.py \
    --dataset_path ${FILTERED_PATH} \
    --output_path ${SCORED_PATH} \
    --top_k 5 \
    --clear_old
