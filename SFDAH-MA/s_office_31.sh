#!/bin/bash
set -e

function map_domain_to_index {
    case $1 in
        'Amazon') echo 0 ;;
        'Dslr') echo 1 ;;
        'Webcam') echo 2 ;;
        *) echo "Invalid domain: $1"; exit 1 ;;
    esac
}

for i in 16 32 64 128
#for i in 64
do
    for domain in  'Amazon' 'Dslr' 'Webcam'
#    for domain in 'Real_World'
#    for domain in 'Amazon'
    do
        domain_index=$(map_domain_to_index $domain)
        CUDA_VISIBLE_DEVICES=0 python -u train_source.py --nbit $i --dset office-31 --s $domain_index
#        cd matlab &&
#        matlab -nojvm -nodesktop -r "demo_eval_PWCF($i, '$domain', 'office-31', 'prefix'); quit;" &&
#        cd ..
    done
done
