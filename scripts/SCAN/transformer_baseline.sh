#!/bin/bash

lr=2.0
warmup_steps=5000
max_steps=50000
home="/your_root_path/met-primaug"
savefolder=$home/save/

for split in jump around_right mcd1; do
expname=scan_${split}_transformer_baseline
mkdir -p $savefolder/$expname
for i in `seq 0 4`
  do
      cd $savefolder/$expname
      python -u  $home/main.py \
      --train \
      --model_type transformer \
      --seed $i \
      --n_batch 128 \
      --lr ${lr} \
      --temp 1.0 \
      --beam_size 5 \
      --gclip 5.0 \
      --accum_count 1 \
      --transformer_config 3layer \
      --valid_steps 500 \
      --warmup_steps ${warmup_steps} \
      --max_step ${max_steps} \
      --tolarance 50 \
      --exp_name ${expname} \
      --scan_split ${split} \
      --SCAN  > eval.$i.out 2> eval.$i.err
  done
done