#!/bin/bash
#SBATCH --job-name=random_search
#SBATCH --output=/users/cborg/rcardenas/sent-anlz-lct/soft_patterns/models/rnd_search/dante.log
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --mem=50GB
#SBATCH --gres=gpu:1
#SBATCH --time=100:00:00

source /users/cborg/.bashrc
conda init bash
conda activate sopa

cd /users/cborg/rcardenas/sent-anlz-lct/soft_patterns

python random_search.py \
-e $HOME/rcardenas/sent-anlz-lct/emb/glove.840B.300d.txt \
--patience 3 \
--td amazon_reviews/train.data.subs.10k \
--tl amazon_reviews/train.labels.subs.10k \
--vd amazon_reviews/dev_src.data.subs.5k \
--vl amazon_reviews/dev_src.labels.subs.5k \
--model_save_dir $HOME/rcardenas/sent-anlz-lct/soft_patterns/models/rnd_search \
--gpu