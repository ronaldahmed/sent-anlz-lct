#!/bin/bash

model_dir="models/movies-src"

qsub -q 'gpu*' -cwd -l gpu=1,gpu_cc_min3.5=1,gpu_ram=4G,mem_free=20G,act_mem_free=20G,h_data=30G -p -50 \
-o $model_dir/log.out \
-e $model_dir/log.err \
wraps/train_standalone.sh


# random search of hyperparam
qsub -q 'gpu*' -cwd -l gpu=1,gpu_cc_min3.5=1,gpu_ram=8G,mem_free=80G,act_mem_free=80G,h_data=100G -p -50 \
-o $HOME/sent-anlz-lct/soft_patterns/models/rnd_search/log.out \
-e $HOME/sent-anlz-lct/soft_patterns/models/rnd_search/log.err \
sge_rnd_search.sh


# training on src data
qsub -q 'gpu*' -cwd -l gpu=1,gpu_cc_min3.5=1,gpu_ram=8G,mem_free=80G,act_mem_free=80G,h_data=100G -p -100 \
-o $HOME/sent-anlz-lct/soft_patterns/models/movies-src/log.out \
-e $HOME/sent-anlz-lct/soft_patterns/models/movies-src/log.err \
wraps/train_src.sh


# training boostrap
qsub -q 'gpu*' -cwd -l gpu=1,gpu_cc_min3.5=1,gpu_ram=2G,mem_free=10G,act_mem_free=10G,h_data=15G -p -60 \
-o $HOME/sent-anlz-lct/soft_patterns/models/boostrap/log.out \
-e $HOME/sent-anlz-lct/soft_patterns/models/boostrap/log.err \
wraps/train_bst.sh
