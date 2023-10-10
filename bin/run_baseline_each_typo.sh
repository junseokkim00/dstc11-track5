typos=("randDelete" "swapAdjacent" "swapChar" "randInsert" "randSubstitute")

for typo_type in ${typos[@]}
do
    model_name=microsoft/deberta-v3-base
    model_name_exp=deberta-v3-base-${typo_type}
    cuda_id=0,1

    CUDA_VISIBLE_DEVICES=${cuda_id} python baseline.py \
            --task selection \
            --dataroot Typo/each_typo/${typo_type} \
            --model_name_or_path ${model_name} \
            --negative_sample_method "oracle" \
            --knowledge_file knowledge.json \
            --params_file baseline/configs/selection/params.json \
            --exp_name ks-review-${model_name_exp}-oracle-baseline
done