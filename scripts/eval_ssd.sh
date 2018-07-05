#!/bin/bash


python ./src/ssd_eval.py \
  --dataset=VKITTI \
  --data_path=./data/VKITTI \
  --image_set=val \
  --eval_dir=./logs/SSD/eval_val \
  --checkpoint=logs/SSD/model_save/model.ckpt-6527 \
  --net=SSD \
  --gpu=0

