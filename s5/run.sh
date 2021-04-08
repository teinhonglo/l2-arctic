#!/usr/bin/env bash

# Change this location to somewhere where you want to put the data.
. ./cmd.sh
. ./path.sh

stage=0
dnn_stage=0
train_stage=-10
. utils/parse_options.sh

set -euo pipefail

if [ $stage -le 1 ]; then
  # format the data as Kaldi data directories

  utils/prepare_lang.sh --sil-prob 0.0 --position-dependent-phones false --num-sil-states 3 \
    data/local/dict "sil" data/local/lang_tmp data/lang
  
  local/timit_format_lm.sh
fi

if [ $stage -le 2 ]; then
  mfccdir=mfcc
  # spread the mfccs over various machines, as this data-set is quite large.

  for part in train_timit test_timit train_l2 test_l2; do
    steps/make_mfcc.sh --cmd "$train_cmd" --nj 10 data/$part exp/make_mfcc/$part $mfccdir
    steps/compute_cmvn_stats.sh data/$part exp/make_mfcc/$part $mfccdir
  done
fi

# train a monophone system
if [ $stage -le 3 ]; then
  # TODO(galv): Is this too many jobs for a smaller dataset?
  steps/train_mono.sh --boost-silence 1.25 --nj 5 --cmd "$train_cmd" \
    data/train_timit data/lang exp/mono

  steps/align_si.sh --boost-silence 1.25 --nj 5 --cmd "$train_cmd" \
    data/train_timit data/lang exp/mono exp/mono_ali_train_timit
fi

# train a first delta + delta-delta triphone system on all utterances
if [ $stage -le 4 ]; then
  steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" \
    2000 10000 data/train_timit data/lang exp/mono_ali_train_timit exp/tri1

  steps/align_si.sh --nj 5 --cmd "$train_cmd" \
    data/train_timit data/lang exp/tri1 exp/tri1_ali_train_timit
fi

# train an LDA+MLLT system.
if [ $stage -le 5 ]; then
  steps/train_lda_mllt.sh --cmd "$train_cmd" \
    --splice-opts "--left-context=3 --right-context=3" 2500 15000 \
    data/train_timit data/lang exp/tri1_ali_train_timit exp/tri2b

  # Align utts using the tri2b model
  steps/align_si.sh  --nj 5 --cmd "$train_cmd" --use-graphs true \
    data/train_timit data/lang exp/tri2b exp/tri2b_ali_train_timit
fi

# Train tri3b, which is LDA+MLLT+SAT
if [ $stage -le 6 ]; then
  steps/train_sat.sh --cmd "$train_cmd" 2500 15000 \
    data/train_timit data/lang exp/tri2b_ali_train_timit exp/tri3b
  
  utils/combine_data.sh data/train data/train_l2 data/train_timit

  steps/align_fmllr.sh --nj 16 --cmd "$train_cmd" \
    data/train data/lang exp/tri3b exp/tri3b_ali_train || exit 1
fi

# Train tri3b, which is LDA+MLLT+SAT (l2-arctic)
if [ $stage -le 7 ]; then
  steps/train_sat.sh --cmd "$train_cmd" 2500 15000 \
    data/train data/lang exp/tri3b_ali_train exp/tri4b
fi

# Train a chain model
if [ $stage -le 9 ]; then
  local/chain/run_tdnn.sh --stage $dnn_stage --train-stage $train_stage
fi

# local/grammar/simple_demo.sh
