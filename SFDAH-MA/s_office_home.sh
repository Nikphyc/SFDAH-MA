#!/bin/bash
set -e

function map_domain_to_index {
    case $1 in
        'Art') echo 0 ;;
        'Clipart') echo 1 ;;
        'Product') echo 2 ;;
        'Real_World') echo 3 ;;
        *) echo "Invalid domain: $1"; exit 1 ;;
    esac
}

for i in 16 32 64 128
#for i in 64
do
    for domain in 'Art'  'Clipart'  'Product' 'Real_World'
#    for domain in 'Real_World'
    do
        domain_index=$(map_domain_to_index $domain)
        CUDA_VISIBLE_DEVICES=0 python -u train_source.py --nbit $i --dset office-home --s $domain_index
#        cd matlab &&
#        matlab -nojvm -nodesktop -r "demo_eval_PWCF($i, '$domain', 'office-home', 'prefix'); quit;" &&
#        cd ..
    done
done
