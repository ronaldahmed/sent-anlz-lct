#!/bin/bash


conda activate sopa

python bootstrap.py \
-e ../emb/glove.840B.300d.txt \
--batch_size 20 \
-i 5 -bi 5 \
-p 6-10_5-10_4-10 \
--dropout 0.0016912395027359473 \
--learning_rate 0.0001475142447988253 \
--mlp_hidden_dim 150 \
--td amazon_reviews/train.data \
--tl amazon_reviews/train.labels \
--tud amazon_reviews/test_unlb.data \
--vd amazon_reviews/dev_tgt.data \
--vl amazon_reviews/dev_tgt.labels \
--model_save_dir models/bootstrap \
--input_model models/movies-src/model_9.pth \
--gpu