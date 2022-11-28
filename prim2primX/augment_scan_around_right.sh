export DATA=SCAN
export SPLIT=around_right

python align_srctgt_vocab.py \
--dataset_name ${DATA} \
--split ${SPLIT} \
--eps 3 \
--threshold 0.9

python prim2primX.py \
--dataset_name ${DATA} \
--split ${SPLIT} \
--aligner_file aligned_vocab_around_right.json \
--num_extra_actions 5 \
--num_mutation_per_example 10