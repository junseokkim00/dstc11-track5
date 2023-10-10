#!/bin/bash

# Prepare directories for intermediate results of each subtask
eval_dataset=val
mkdir -p pred/${eval_dataset}
model_name_exp=deberta-v3-base


# track entities
em_output_file=pred/${eval_dataset}/baseline.em.${model_name_exp}.json


# eval for knowledge selection
model_name=t5-small
model_name_exp=t5-small
checkpoint=runs/ks-${model_name_exp}-aug
ks_output_file=pred/${eval_dataset}/baseline.ks.${model_name_exp}_aug.json
cuda_id=0,1

CUDA_VISIBLE_DEVICES=${cuda_id} python3 augmentation.py \
        --task selection_aug \
        --params_file /home/jihyunlee/dstc11-track5/augmentation/configs/selection_aug/params.json \
        --model_name_or_path ${model_name_exp} \
        --dataroot data \
        --eval_dataset ${eval_dataset} \
        --labels_file ${em_output_file} \
        --output_file ${ks_output_file} \
        --knowledge_file knowledge.json \
        --exp_name aug-${model_name_exp}


# --checkpoint ${checkpoint} \

# --eval_only \

        


