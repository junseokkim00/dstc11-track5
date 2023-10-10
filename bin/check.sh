#!/bin/bash

# Prepare directories for intermediate results of each subtask
eval_dataset=test


model_name_exp=bart-base
rg_output_file=pred/${eval_dataset}/baseline.rg.${model_name_exp}-BT_short.json

# verify and evaluate the output
rg_output_score_file=pred/${eval_dataset}/baseline.rg.${model_name_exp}-BT_short.score.json
rg_output_score_file=pred/submit/typo_long.json
python -m scripts.check_results --dataset ${eval_dataset} --dataroot data --outfile ${rg_output_file}
