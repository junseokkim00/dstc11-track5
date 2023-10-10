typos=("randDelete" "swapAdjacent" "swapChar" "randInsert" "randSubstitute")
for typo in ${typos[@]}
do
    echo "$typo"
    cat pred/val/baseline.rg.bart-base-${typo}.json | jq
    echo -e "\n"
done