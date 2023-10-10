typology=('fus' 'agg' 'iso')
typos=("randDelete" "swapAdjacent" "swapChar" "randInsert" "randSubstitute")

for typol in ${typology[@]}
do
    for typo in ${typos[@]}
    do
        echo "$typol-$typo"
        cat pred/val/baseline.rg.bart-base-${typol}-${typo}.json | jq
        echo -e "\n"
    done
done