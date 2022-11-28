export DATA=SCAN
export SPLIT=addprim_jump

python align_srctgt_vocab.py \
--dataset_name ${DATA} \
--split ${SPLIT} \
--eps 3 \
--threshold 0.5

python prim2primX.py \
--dataset_name ${DATA} \
--split ${SPLIT} \
--aligner_file aligned_vocab_addprim_jump.json \
--num_extra_actions 5