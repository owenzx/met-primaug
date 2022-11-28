
lr=2.0
warmup_steps=5000
max_steps=50000
expname=cogs_transformer_metmeta
home="/your_root_path/met-primaug"
savefolder=$home/save/
mkdir -p $savefolder/$expname
for i in `seq 0 4`
do
    cd $savefolder/$expname
    python -u  $home/main.py \
    --train \
    --meta \
    --meta_loss_type unmaml \
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
    --tolarance 10 \
    --exp_name ${expname} \
    --cogs_perturbation \
    --COGS  > eval.$i.out 2> eval.$i.err
done
