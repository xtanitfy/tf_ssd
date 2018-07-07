#!/bin/bash
rm -rf logs/SSD/eval_val/*

python ./src/ssd_eval.py \
  --dataset=VKITTI \
  --data_path=./data/VKITTI \
  --image_set=val \
  --eval_dir=./logs/SSD/eval_val \
  --checkpoint=logs/SSD/model_save/model.ckpt-53548 \
  --net=SSD \
  --gpu=0

