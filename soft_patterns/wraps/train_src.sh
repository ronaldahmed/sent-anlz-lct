#!/bin/bash

# train source, eval on source

conda activate sopa

cd $HOME/sent-anlz-lct/soft_patterns

python soft_patterns.py \
-e ../emb/glove.840B.300d.txt \
--batch_size 20 -i 10 \
-p 6-10_5-10_4-10 \
--dropout 0.0016912395027359473 \
--learning_rate 0.0001475142447988253 \
--mlp_hidden_dim 150 \
--td amazon_reviews/train.data \
--tl amazon_reviews/train.labels \
--vd amazon_reviews/dev_src.data \
--vl amazon_reviews/dev_src.labels \
--gpu \
--model_save_dir models/movies-src