#!/usr/bin/env bash
dataset=$1
for i in `seq 0 3`
do
python ./main_intra.py \
--target_idx=$i \
--batch_size=64 \
--learning_rate=0.008 \
--epochs=150 \
--suffix=$1 \
--intra_adr=1 \
--warm_epochs 5
done
