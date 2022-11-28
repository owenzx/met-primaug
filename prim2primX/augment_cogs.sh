export DATA=COGS

python align_srctgt_vocab.py \
--dataset_name ${DATA} \
--eps 3 \
--threshold 0.5

python prim2primX.py \
--dataset_name ${DATA} \
--aligner_file aligned_vocab.json \
--num_extra_actions 2 \
--num_mutation_per_example 2