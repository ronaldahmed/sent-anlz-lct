#!/bin/bash

python random_search.py \
-e $HOME/sent-anlz-lct/emb/glove.840B.300d.txt \
--patience 3 \
--td amazon_reviews/train.data.subs.10k \
--tl amazon_reviews/train.labels.subs.10k \
--vd amazon_reviews/dev_src.data.subs.5k \
--vl amazon_reviews/dev_src.labels.subs.5k \
--model_save_dir $HOME/sent-anlz-lct/soft_patterns/models/rnd_search \
--gpu