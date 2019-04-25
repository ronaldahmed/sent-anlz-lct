#!/bin/bash

## profiling
# mem: 200

# 1 float = 297MB
# batch_size 150 -> 64
#  with bs=150
# n.params = 607877
# with bs = 64

# bs=64, gpu=12 -> funciona
# test: bs=64, gpu=8

# cpu = 89min

conda activate sopa

python soft_patterns.py \
-e ../emb/glove.840B.300d.txt \
--batch_size 64 -i 100 \
--gpu \
--td data/train.data \
--tl data/train.labels \
--vd data/dev.data \
--vl data/dev.labels \
-p 5-50_4-50_3-50_2-50 \
--model_save_dir models/movies-src