#!/usr/bin/env bash

. ./cmd.sh
. ./path.sh

stage=1
ngr=3
train_text=data/train_l2/text
dev_text=data/test_l2/text
#lm_dev=texts/${lmid}dev.trn.${prefix}.txt
lm_dir=data/local/l2_lm
lm_train=data/local/l2_lm/train.txt
lm_dev=data/local/l2_lm/dev.txt
oov_symbol="sil"
. utils/parse_options.sh

mkdir -p $lm_dir

if [ $stage -le 1 ]; then

    echo "$0:  ... preparing language model"
    if ! test -d ${lm_dir}; then mkdir -p ${lm_dir}; fi
	
    lm=${lm_dir}/${ngr}gram.me.gz
    cut -d" " -f2- $train_text > $lm_train
    cut -d" " -f2- $dev_text > $lm_dev

    ngram-count \
	-lm - -order ${ngr} -text ${lm_train} \
	-unk -sort -maxent -maxent-convert-to-arpa|\
    ngram -lm - -order ${ngr} -unk -map-unk "$oov_symbol" -prune-lowprobs -write-lm - |\
    sed 's/<unk>/'${oov_symbol}'/g' | gzip -c > $lm

    echo    ngram -order ${ngr} -lm $lm -unk -map-unk "$oov_symbol" -prune-lowprobs -ppl ${lm_dev}
    ngram -order ${ngr} -lm $lm -unk -map-unk "$oov_symbol" -prune-lowprobs -ppl ${lm_dev}

fi

if [ $stage -le 2 ]; then
    lm=${lm_dir}/${ngr}gram.me.gz
    utils/format_lm.sh \
        data/lang $lm data/local/dict/lexicon.txt data/lang_${ngr}grl2_test
fi

if [ $stage -le 3 ]; then
    mainlm=${lm_dir}/${ngr}gram.me.gz
    nistlm=data/local/nist_lm/lm_phone_bg.arpa.gz
    mergelm=data/local/l2_lm/${ngr}gram.mix05.me.gz
    # https://blog.csdn.net/xmdxcsj/article/details/50353689
    # data/local/l2_lm/${ngr}gram.mix05.me.gz
    ngram -lm ${mainlm} -order ${ngr} -mix-lm ${nistlm} -lambda 0.5 -write-lm ${mergelm}
    utils/format_lm.sh \
        data/lang ${mergelm} data/local/dict/lexicon.txt data/lang_${ngr}grl2mix05_test
fi
