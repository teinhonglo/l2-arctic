#!/bin/bash

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch # chainer or pytorch
stage=0         # start from 0 if you need to start from data preparation
stop_stage=100
ngpu=1          # number of gpus during training ("0" uses cpu, otherwise use gpu)
dec_ngpu=0      # number of gpus during decoding ("0" uses cpu, otherwise use gpu)
nj=4            # number of parallel jobs for decoding
debugmode=1
dumpdir=dump_mt    # directory to dump full features
N=0             # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0       # verbose option
resume=         # Resume the training from snapshot
seed=1          # seed to generate random number

train_config=conf/mt_train.yaml
decode_config=conf/mt_decode.yaml

# decoding parameter
trans_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'

# model average realted (only for transformer)
n_average=5                  # the number of MT models to be averaged
use_valbest_average=true     # if true, the validation `n_average`-best MT models will be averaged.
                             # if false, the last `n_average` MT models will be averaged.
metric=acc                  # loss/acc/bleu

# cascaded-ST related
asr_model=
decode_config_asr=
dict_asr=

# preprocessing related
# if true, reverse source and target languages: **->English
reverse_direction=false

# use the same dict as in the ST task
use_st_dict=true

# exp tag
tag="" # tag for managing experiments.

# data
trans_type=phn

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_tr.asr
train_dev=train_cv.asr
trans_set="test_timit.asr test_l2.asr"

feat_tr_dir=${dumpdir}/${train_set}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}; mkdir -p ${feat_dt_dir}
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"
	
    for x in train_tr train_cv test_l2 test_timit; do
        local/divide_lang.sh ${x}
    done

    # remove long and short utterances
    for x in train_tr train_cv test_timit test_l2; do
        clean_corpus.sh --no_feat true --maxchars 400 --utt_extra_files "text.asr" data/${x} "asr as"
    done
fi


dict=data/lang_1char/train_tr_units.txt
nlsyms=data/lang_1char/non_lang_syms.txt

echo "dictionary: ${dict}"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"

    echo "make json files"
    data2json.sh --nj 16 --text data/${train_set}/text --trans_type ${trans_type} --lang asr \
        data/${train_set} ${dict} > ${feat_tr_dir}/data.json
    for x in ${train_dev} ${trans_set}; do
        feat_trans_dir=${dumpdir}/${x}; mkdir -p ${feat_trans_dir}
        data2json.sh --text data/${x}/text --trans_type ${trans_type} --lang asr \
            data/${x} ${dict} > ${feat_trans_dir}/data.json
    done

    # update json (add source references)
    for x in ${train_set} ${train_dev} ${trans_set}; do
        feat_dir=${dumpdir}/${x}
        data_dir=data/$(echo ${x} | cut -f 1 -d ".").as
        update_json.sh --text ${data_dir}/text --trans_type ${trans_type} \
            ${feat_dir}/data.json ${data_dir} ${dict}
    done
fi

# NOTE: skip stage 3: LM Preparation

if [ -z ${tag} ]; then
    expname=${train_set}_mt_asr_as_${backend}_$(basename ${train_config%.*})
else
    expname=${train_set}_mt_asr_as_${backend}_${tag}
fi
expdir=exp_mt/${expname}
mkdir -p ${expdir}

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Network Training"

    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        mt_train.py \
        --config ${train_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --tensorboard-dir tensorboard/${expname} \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --seed ${seed} \
        --verbose ${verbose} \
        --resume ${resume} \
        --train-json ${feat_tr_dir}/data.json \
        --valid-json ${feat_dt_dir}/data.json
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Decoding"
    if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]]; then
        # Average MT models
        if ${use_valbest_average}; then
            trans_model=model.val${n_average}.avg.best
            opt="--log ${expdir}/results/log --metric ${metric}"
        else
            trans_model=model.last${n_average}.avg.best
            opt="--log"
        fi
        average_checkpoints.py \
            ${opt} \
            --backend ${backend} \
            --snapshots ${expdir}/results/snapshot.ep.* \
            --out ${expdir}/results/${trans_model} \
            --num ${n_average}
    fi

    if [ ${dec_ngpu} = 1 ]; then
        nj=1
    fi

    pids=() # initialize pids
    for x in ${trans_set}; do
    (
        decode_dir=decode_${x}_$(basename ${decode_config%.*})
        feat_trans_dir=${dumpdir}/${x}

        # reset log for RTF calculation
        if [ -d ${expdir}/${decode_dir}/log/decode.1.log ]; then
            rm ${expdir}/${decode_dir}/log/decode.*.log
        fi

        # split data
        splitjson.py --parts ${nj} ${feat_trans_dir}/data.json

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            mt_trans.py \
            --config ${decode_config} \
            --ngpu ${dec_ngpu} \
            --backend ${backend} \
            --batchsize 0 \
            --trans-json ${feat_trans_dir}/split${nj}utt/data.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${trans_model}

        #if [ ${reverse_direction} = true ]; then
        #    score_bleu.sh --case ${tgt_case} --bpe ${nbpe} \
        #        ${expdir}/${decode_dir} "as" ${dict}
        #else
        #    local/score_bleu.sh --case ${tgt_case} --set ${x} --bpe ${nbpe} \
        #        ${expdir}/${decode_dir} ${dict}
        #fi

        # calculate_rtf.py --log-dir ${expdir}/${decode_dir}/log
        asr_x=$(echo $x | cut -d "." -f1)
        local/score_sclite_mt.sh ${expdir}/${decode_dir} exp/train_tr_sp_pytorch_train_conformer_no_preprocess/decode_${asr_x}_decode_pytorch_transformer ${dict}
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
fi

exit 0;

