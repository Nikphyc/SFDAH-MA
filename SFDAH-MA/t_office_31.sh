#!/bin/bash
set -e

function map_domain_to_index {
    case $1 in
        'A') echo 0 ;;
        'D') echo 1 ;;
        'W') echo 2 ;;
        *) echo "Invalid domain: $1"; exit 1 ;;
    esac
}

for i in 16 32 64 128
#for i in 64
do
#    for domain in 'ArtToReal_World' 'Real_WorldToArt' 'ClipartToReal_World' 'Real_WorldToClipart' 'ProductToReal_World' 'Real_WorldToProduct'
  for domain in 'AD' 'AW' 'DA' 'DW' 'WA' 'WD'
#  for domain in 'AD'
  do
    source_domain_idx=$(map_domain_to_index ${domain:0:1})
    target_domain_idx=$(map_domain_to_index ${domain:1:1})

    CUDA_VISIBLE_DEVICES=0 python -u train_target.py --nbit $i --dset office-31 --s $source_domain_idx --t $target_domain_idx
    cd matlab &&
    matlab -nojvm -nodesktop -r "demo_eval_PWCF($i, '$domain', 'office-31', 'prefix'); quit;" &&
    cd ..
  done
done
