#!/bin/bash


python ./src/eval.py \
  --dataset=VKITTI \
  --data_path=./data/VKITTI \
  --image_set=val \
  --eval_dir=./logs/SSD/eval_val \
  --checkpoint_path=$2 \
  --net=$1 \
  --gpu=0

