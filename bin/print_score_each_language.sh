langarr=('da' 'de' 'es' 'fi' 'hi' 'hu' 'id' 'ine' 'ko' 'tl' 'tr' 'zh')
for i in ${langarr[@]}
do
    echo "$i"
    cat pred/val/baseline.rg.bart-base-$i.json | jq
    echo -e '\n'
done

# cat pred/val/baseline.rg.bart-base-da.json | jq