repeats=('repeat2' 'repeat3' 'repeat4')
for repeat in ${repeats[@]}
do
    echo "$repeat"
    cat pred/val/baseline.rg.bart-base-${repeat}.json | jq
    echo -e "\n"
done