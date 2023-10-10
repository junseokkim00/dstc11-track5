#!/bin/bash

# Prepare directories for intermediate results of each subtask
eval_dataset=val
mkdir -p pred/${eval_dataset}
model_name_exp=deberta-v3-base


# track entities
em_output_file=pred/${eval_dataset}/baseline.em.${model_name_exp}.json


# eval for knowledge selection
model_name=microsoft/deberta-v3-base
model_name_exp=deberta-v3-base
checkpoint=runs/ks-review-${model_name_exp}-oracle-baseline
ks_output_file=pred/${eval_dataset}/baseline.ks.${model_name_exp}_analysis.json
cuda_id=3

CUDA_VISIBLE_DEVICES=${cuda_id} python3 analysis.py \
        --eval_only \
        --task selection \
        --model_name_or_path ${model_name_exp} \
        --checkpoint ${checkpoint} \
        --dataroot data \
        --eval_dataset ${eval_dataset} \
        --labels_file ${em_output_file} \
        --output_file ${ks_output_file} \
        --knowledge_file knowledge.json \
        --pred_file /home/jihyunlee/dstc11-track5/pred/val/baseline.ks.deberta-v3-base.json
        


