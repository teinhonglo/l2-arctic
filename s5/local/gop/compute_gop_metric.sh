#!/bin/bash

# Copyright 2017   Author: Ming Tu
# Arguments:
# audio-dir: where audio files are stored
# data-dir: where extracted features are stored
# result-dir: where results are stored                               

set -e
#set -x
stage=0
split_per_speaker=true # split by speaker (true) or sentence (false)
dataset="train_l2_should_say test_l2_should_say"
actual_dataset="train_l2 test_l2"
dev_set="train_l2_should_say"
ivectors_dir=exp/nnet3
lang=data/lang
data_root=data
exp_root=exp/chain
reduce="false"
models="$exp_root/cnn_tdnn1c_sp"
phonemap="conf/phones.60-48-39.map"

# Enviroment preparation
. ./cmd.sh
. ./path.sh

. parse_options.sh || exit 1;


if [ $stage -le 0 ]; then
    for dset in $dataset; do
        utils/copy_data_dir.sh  $data_root/${dset} $data_root/${dset}_capt_hires
        nspk=$(wc -l <$data_root/$dset/spk2utt)
        if [ $nspk -ge 20 ]; then
            nspk=20;
        fi
        steps/make_mfcc.sh --nj $nspk --mfcc-config conf/mfcc_hires.conf \
          --cmd "$train_cmd" $data_root/${dset}_capt_hires || exit 1;
        steps/compute_cmvn_stats.sh $data_root/${dset}_capt_hires || exit 1;
        utils/fix_data_dir.sh $data_root/${dset}_capt_hires
    done
fi

if [ $stage -le 1 ]; then
    for dset in $dataset; do
        nspk=$(wc -l <$data_root/$dset/spk2utt)
        if [ $nspk -ge 20 ]; then
            nspk=20;
        fi
        steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd --num-threads 5" --nj $nspk \
          $data_root/${dset}_capt_hires $ivectors_dir/extractor \
          $ivectors_dir/ivectors_${dset}_capt_hires || exit 1;
   done
fi

if [ $reduce == "true" ]; then
	echo "Recduce 48 to 39"
    for act_dset in $actual_dataset; do
		dset=${act_dset}_should_say
		if [ -f $data_root/${dset}_capt_hires/text.backup ]; then
			mv $data_root/${dset}_capt_hires/text.backup $data_root/${dset}_capt_hires/text
		fi
		cat $data_root/${dset}_capt_hires/text | local/timit_norm_trans.pl -i - -m $phonemap -from 48 -to 39 > $data_root/${dset}_capt_hires/text.tmp
		
		cp $data_root/${dset}_capt_hires/text $data_root/${dset}_capt_hires/text.backup
		mv $data_root/${dset}_capt_hires/text.tmp $data_root/${dset}_capt_hires/text
		sed -i "s/sil//g" $data_root/${dset}_capt_hires/text
		for model in $models; do
            result_dir=${model}/gop_${dset}_capt_hires
			mkdir -p $result_dir/capt/capt
			cp $data_root/${dset}_capt_hires/text $result_dir/capt/should_say.txt
			cat $data_root/${act_dset}/text | local/timit_norm_trans.pl -i - -m $phonemap -from 48 -to 39 > $result_dir/capt/actual_say.txt
			sed -i "s/sil//g" $result_dir/capt/actual_say.txt
		done
    done
else
	echo "Remain original transcript"
    for act_dset in $actual_dataset; do
		dset=${act_dset}_should_say
		if [ -f $data_root/${dset}_capt_hires/text.backup ]; then
			mv $data_root/${dset}_capt_hires/text.backup $data_root/${dset}_capt_hires/text
		fi
		cp $data_root/${dset}_capt_hires/text $data_root/${dset}_capt_hires/text.backup
		sed -i "s/sil//g" $data_root/${dset}_capt_hires/text
		for model in $models; do
            result_dir=${model}/gop_${dset}_capt_hires
			mkdir -p $result_dir/capt
			cp $data_root/${dset}_capt_hires/text $result_dir/capt/should_say.txt
			
			cat $data_root/${act_dset}/text > $result_dir/capt/actual_say.txt
			sed -i "s/sil//g" $result_dir/capt/actual_say.txt
		done
    done
fi


if [ $stage -le 3 ]; then
    for dset in $dataset; do
        nspk=$(wc -l <$data_root/$dset/spk2utt)
        if [ $nspk -ge 20 ]; then
            nspk=20;
        fi
        datadir=$data_root/${dset}_capt_hires
        ivectors_data_dir=$ivectors_dir/ivectors_${dset}_capt_hires
        for model in $models; do
            echo "Using DNN model!"
            result_dir=${model}/gop_${dset}_capt_hires
            local/gop/compute-dnn-bi-gop.sh --nj "$nspk" --cmd "queue.pl" --split_per_speaker "$split_per_speaker" $datadir $ivectors_data_dir \
              $lang $model $result_dir    ### dnn model
        done
   done
fi

if [ $stage -le 4 ]; then
	for act_dset in $actual_dataset; do
        nspk=$(wc -l <$data_root/$dset/spk2utt)
        if [ $nspk -ge 20 ]; then
            nspk=20;
        fi
		dset=${act_dset}_should_say
        for model in $models; do
            echo "Using DNN model!"
            capt_dir=${model}/gop_${dset}_capt_hires/capt
			# Annotation
			align-text --special-symbol="'***'" ark:$capt_dir/should_say.txt ark:$capt_dir/actual_say.txt ark,t:- | \
			utils/scoring/wer_per_utt_details.pl --special-symbol "'***'" > $capt_dir/annotation.txt
        done
   done
fi
eval "$(/share/homes/teinhonglo/anaconda3/bin/conda shell.bash hook)"
if [ $stage -le 5 ]; then
    for dset in $dataset; do
        nspk=$(wc -l <$data_root/$dset/spk2utt)
        if [ $nspk -ge 20 ]; then
            nspk=20;
        fi
        for model in $models; do
            echo "Compute GOP-score $model"
            python local/gop/compute_gop_metric_dev.py --test_set $model/gop_${dset}_capt_hires --dev_set $model/gop_${dev_set}_capt_hires
        done
   done
fi
