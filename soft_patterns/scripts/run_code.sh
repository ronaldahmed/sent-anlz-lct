#!/usr/bin/env bash 

set -e

r=0
mp=0
rf=''
mpf=''
rs=''
mps='_pt'
b=1
clip=''
clips=''
gpu=''
file_type=0
w=0
mtf=''
mts=''
glove_index=0
gloves=(6B.100d 6B.300d 840B.300d 6B.50d)
dirs=(stanford_sentiment_binary amazon_reviews ROC_stories stanford_sentiment_binary_100 stanford_sentiment_binary_500 stanford_sentiment_binary_1000 stanford_sentiment_binary_2500 amazon_reviews_100 amazon_reviews_500 amazon_reviews_1000 amazon_reviews_2500 amazon_reviews_5000 amazon_reviews_10000 politifact Bills)

n_dirs=${#dirs[@]}
let n_dirs--
datadir_index=0
seed=100

bilstm=0
bilstms=''

base_dir="$HOME/sent-anlz-lct/soft_patterns"
model_dir="$HOME/sent-anlz-lct/soft_patterns/models"


# if [ -z ${WORK+x} ]; then
#     WORK=$HOME/
#     if [ -z ${model_dir+x} ]; then
#         model_dir=$HOME/work/soft_patterns
#     fi
# else
#     if [ -z ${model_dir+x} ]; then
#         model_dir=${WORK}/soft_patterns
#     fi
# fi

# if [ -z ${resource_dir+x} ]; then
#     resource_dir=${WORK}/resources
# fi

train_subs=".subs.10k"
val_subs=".subs.5k"

# train_subs=""
# val_subs=""

suffix=''


if [ "$#" -lt 4 ]; then
	echo "Usage: $0"
	echo "<Pattern specification>"
	echo "<MLP dim>"
	echo "<Learning rate>"
	echo "<dropout>"
	echo "<reschedule=$r>"
	echo "<maxplus=$mp (1 for maxplus, 2 for maxtimes, 0 for prob)>"
	echo " <batch size=$b>"
	echo "<gradient clipping (optional)>"
	echo "<gpu (optional)>" 
	echo "<glove index=$glove_index (${gloves[@]})>"
	echo "<file type=$file_type (0 -- lower case, 1 -- case sensitive, 2 -- train with phrases, 3 -- fine grained categories, 4 -- fine grained categories with phrases)>"
	echo "<word_dropout=$w>"
	echo " <data dir: 0 -- stanford (default), 1 -- amazon, 2 -- ROC stories>"
	echo "<seed=$seed>"
	echo "<bilstm=$bilstm>"

	echo "Dirs:"
        for i in $(seq 0 $n_dirs); do
                echo ${i}: ${dirs[$i]}
        done

	exit -1
elif [ "$#" -gt 4 ]; then
	r=$5

	if [ ${r} -eq 1 ]; then
		rf="-r"
		rs='_r'
	fi
	if [ "$#" -gt 5 ]; then
		mp=$6
		if [ ${mp} -eq 1 ]; then
			mpf="--maxplus"
			mps='_mp'
		elif [ ${mp} -eq 2 ]; then
			mpf="--maxtimes"
			mps='_mt'
		fi
		if [ "$#" -gt 6 ]; then
			b=$7
			if [ "$#" -gt 7 ]; then
				clip="--clip $8"
				clips="_clip$8"
				if [ "$#" -gt 8 ]; then
					if [ $9 -eq 1 ]; then
						gpu='-g'
					fi
					if [ "$#" -gt 9 ]; then
						glove_index=${10}
						if [ "$#" -gt 10 ]; then
							if [ "${11}" -eq 1 ]; then
								suffix='_case_sensitive'
							elif [ "${11}" -eq 2 ]; then
								suffix='_phrases'
							elif [ "${11}" -eq 3 ]; then
								suffix='_fine'
							elif [ "${11}" -eq 4 ]; then
								suffix='_phrases_fine'
							elif [ "${11}" -ne 0 ]; then
								echo "Expected a number between 0-4 for file type, got ${11}"
								exit -2
							fi
							if [ "$#" -gt 11 ]; then
                               					if [ "$#" -gt 12 ]; then
				                                    datadir_index=${13}
                               					    if [ "$#" -gt 13 ]; then
				                                        seed=${14}
                                                        if [ "$#" -gt 14 ]; then
                                                            bilstm=${15}

                                                            if [ $bilstm -gt 0 ]; then
                                                                bilstms="--use_rnn --hidden_dim $bilstm"
                                                            fi
                                                        fi
                                				    fi
                                				fi
                                				w=${12}
                            				fi
						fi
					fi
				fi
			fi
		fi
	fi
fi


p=$1

p2=`echo ${p} | tr ',' '_' | tr ':' '-'`
dim=$2
lr=$3
t=$4

glove=${gloves[$glove_index]}

git_tag=$(git log | head -1 | awk '{print $2}' | cut -b-7)

s=p${p2}_d${dim}_l${lr}_t${t}${rs}${mps}_b${7}${clips}_${glove}${suffix}_w${w}_${dirs[$datadir_index]}_seed${seed}_bh${bilstm}_${git_tag}
odir=${model_dir}/output_${s}

# data_dir="${resource_dir}/text_cat/${dirs[$datadir_index]}"
#glove_dir="${resource_dir}/glove"
data_dir=$base_dir/${dirs[$datadir_index]}
glove_dir=$base_dir/../emb

mkdir -p ${odir}

if [ $dim -eq 0 ]; then
    dim="0 -y 1"
fi

## changed itearation 250 -> 1

com="python -u soft_patterns.py        \
         -e ${glove_dir}/glove.${glove}.txt         \
        --td ${data_dir}/train$suffix.data${train_subs}    \
        --tl ${data_dir}/train$suffix.labels${train_subs}  \
        --vd ${data_dir}/dev$suffix.data${dev_subs}      \
        --vl ${data_dir}/dev.labels${dev_subs}           \
        --model_save_dir $odir \
        -i 1 \
        -p $p \
        -t $t \
        -d $dim \
        $bilstms \
        -l $lr $rf $mpf $clip $gpu\
        -b $b \
	--max_doc_len 100 \
	--seed $seed \
	-w $w"

# ${com}

function gen_cluster_file {
    local s=$1

    f=$HOME/sent-anlz-lct/soft_patterns/wraps/${s}

    echo "#!/bin/bash" > ${f}
    echo "" >> ${f}
    echo "conda activate sopa" >> ${f}
    echo "" >> ${f}
    echo "${com}" >> ${f}

    echo ${f}
}

if [[ "$HOSTNAME" == dll* ]]||[[ "$HOSTNAME" == kronos ]]||[[ "$HOSTNAME" == titan* ]]||[[ "$HOSTNAME" == tdll* ]]||[[ "$HOSTNAME" == sol* ]]; then
    echo "entra!!"
    f=$(gen_cluster_file ${s})

    qsub -q 'gpu*' -cwd -l gpu=1,gpu_cc_min3.5=1,gpu_ram=8G,mem_free=50G,act_mem_free=50G,h_data=70G -p -50 \
    -o $base_dir/logs/$s.out \
    -e $base_dir/logs/$s.err \
    ${f}
elif [[ $HOME == "/home/ronald" ]]; then
    ${com} 2&>1 | tee ${odir}/output.dat
else
	echo "entra last!!"
    ${com} |& tee ${odir}/output.dat
fi
