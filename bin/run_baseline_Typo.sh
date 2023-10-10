model_name=microsoft/deberta-v3-base
model_name_exp=deberta-v3-base-Typo
cuda_id=0,1

CUDA_VISIBLE_DEVICES=${cuda_id} python baseline.py \
        --task selection \
        --dataroot baseline_Typo \
        --model_name_or_path ${model_name} \
        --negative_sample_method "oracle" \
        --knowledge_file knowledge.json \
        --params_file baseline/configs/selection/params.json \
        --exp_name ks-review-${model_name_exp}-oracle-baseline


eval_dataset=val
mkdir -p pred/${eval_dataset}

cuda_id=0,1
em_output_file=pred/${eval_dataset}/baseline.em.deberta-v3-base.json

model_name=microsoft/deberta-v3-base
model_name_exp=deberta-v3-base-Typo
checkpoint=runs/ks-review-${model_name_exp}-oracle-baseline
ks_output_file=pred/${eval_dataset}/baseline.ks.${model_name_exp}.json


CUDA_VISIBLE_DEVICES=${cuda_id} python3 baseline.py \
        --eval_only \
        --task selection \
        --model_name_or_path ${model_name_exp} \
        --checkpoint ${checkpoint} \
        --dataroot baseline_Typo \
        --eval_dataset ${eval_dataset} \
        --labels_file ${em_output_file} \
        --output_file ${ks_output_file} \
        --knowledge_file knowledge.json

model_name=facebook/bart-base
model_name_exp=bart-base
checkpoint=runs/rg-review-${model_name_exp}-oracle-baseline
rg_output_file=pred/${eval_dataset}/baseline.rg.Typo.json

CUDA_VISIBLE_DEVICES=${cuda_id} python3 baseline.py \
        --task generation \
        --generate runs/rg-review-${model_name_exp}-baseline \
        --generation_params_file baseline/configs/generation/generation_params.json \
        --eval_dataset ${eval_dataset} \
        --dataroot data \
        --labels_file ${ks_output_file} \
        --knowledge_file knowledge.json \
        --output_file ${rg_output_file}


# verify and evaluate the output
rg_output_score_file=pred/${eval_dataset}/baseline.rg.Typo.json
python -m scripts.check_results --dataset ${eval_dataset} --dataroot data --outfile ${rg_output_file}
python -m scripts.scores --dataset ${eval_dataset} --dataroot data --outfile ${rg_output_file} --scorefile ${rg_output_score_file}

