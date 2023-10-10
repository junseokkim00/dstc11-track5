typologys=("fus" "agg" "iso")
for i in ${typologys[@]}
do
    echo "$i"
    cat pred/val/baseline.rg.bart-base-$i.json | jq
    echo -e '\n'
done