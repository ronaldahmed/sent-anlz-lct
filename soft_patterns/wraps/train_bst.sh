#!/bin/bash


conda activate sopa

python bootstrap.py \
-e ../emb/glove.840B.300d.txt \
--batch_size 10 \
-i 5 -bi 5 \
-p 5-50_4-50_3-50_2-50 \
--td amazon_reviews/train.data \
--tl amazon_reviews/train.labels \
--tud amazon_reviews/test_unlb.data \
--vd amazon_reviews/dev_tgt.data \
--vl amazon_reviews/dev_tgt.labels \
--model_save_dir models/bootstrap \
--gpu