
lr=2.0
warmup_steps=5000
max_steps=50000
expname=toy_transformer_met
home="/your_root_path/met-primaug"
savefolder=$home/save/
mkdir -p $savefolder/$expname
for i in `seq 0 9`
do
    cd $savefolder/$expname
    python -u  $home/main.py \
    --train \
    --unlikelihood \
    --meta_loss_type unlikelihood \
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
    --TOY  > eval.$i.out 2> eval.$i.err
done
