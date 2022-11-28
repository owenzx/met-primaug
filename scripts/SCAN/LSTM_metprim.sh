#!/bin/bash

lr=1.0
warmup_steps=4000
max_steps=40000
batch_size=128
home="/your_root_path/met-primaug"
savefolder=$home/save/
for split in jump around_right ; do
	expname=scan_${split}_LSTM_met_prim2primx
	mkdir -p $savefolder/$expname
	for i in `seq 0 4`
	do
     cd $savefolder/$expname
     python -u  $home/main.py \
     --train \
     --unlikelihood \
     --meta_loss_type unlikelihood \
     --special_train_data train_primx5 \
     --seed $i \
     --n_batch ${batch_size} \
     --n_layers 2 \
     --noregularize \
     --dim 512 \
     --lr ${lr} \
     --temp 1.0 \
     --dropout 0.4 \
     --beam_size 5 \
     --gclip 5.0 \
     --accum_count 4 \
     --valid_steps 500 \
     --warmup_steps ${warmup_steps} \
     --max_step ${max_steps} \
     --tolarance 50 \
     --exp_name ${expname} \
     --scan_split ${split} \
     --scan_perturbation \
     --SCAN > eval.$i.out 2> eval.$i.err
	done
done
