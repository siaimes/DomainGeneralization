#!/usr/bin/env bash
dataset=$1
for i in `seq 0 3`
do
python ./main_intra_single.py \
--target_idx=$i \
--batch_size=512 \
--learning_rate=0.008 \
--epochs=150 \
--suffix=$1 \
--intra_adr=1 \
--warm_epochs 5
done