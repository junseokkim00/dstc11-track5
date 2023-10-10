repeats=('one' 'two' 'three' 'four' 'five')

for i in ${repeats[@]}
do
    echo "$i"
    cat pred/val/baseline.rg.bart-base-typo-$i.json | jq
    echo -e '\n'
done