#!/usr/bin/env bash
dataset=$1
for i in `seq 0 3`
do
python ./main_i2.py \
--target_idx=$i \
--batch_size=64 \
--learning_rate=0.005 \
--epochs=100 \
--suffix=$1 

done
