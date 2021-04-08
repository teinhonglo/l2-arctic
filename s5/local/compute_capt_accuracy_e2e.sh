#!/usr/bin/env bash

. ./cmd.sh
. ./path.sh

stage=-1

phonemap="conf/phones.60-48-39.map"
should_say="data/test_l2_should_say/text"
actual_say="data/test_l2/text"
predict_say=
capt_dir=

. utils/parse_options.sh || exit 1;

if [ -z $predict_say ] || [ -z $capt_dir ]; then
	echo "local/compute_capt_accuracy_e2e.sh --should_say data/test_l2_should_say/text --actual_say data/test_l2/text --predict_say exp/chain/tdnn1k_sp/decode_test_l2/scoring/4.tra --capt-dir exp/chain/tdnn1k_sp/decode_test_l2/capt"
	exit 0;
fi

if [ ! -d $capt_dir ]; then
    mkdir -p $capt_dir;
fi

if [ $stage -le -1 ]; then
    python local/preprocess_e2e_results.py --src $predict_say --dest $capt_dir/espnet_predict_say.txt;
    #python local/preprocess_e2e_results.py --src $actual_say --dest $capt_dir/espnet_actual_say.txt;
fi

# Map reference to 39 phone classes
if [ $stage -le 0 ]; then
    local/compute_capt_accuracy.sh  --stage $stage \
                                        --should_say data/test_l2_should_say/text \
                                        --actual_say data/test_l2/text \
                                        --predict_say $capt_dir/espnet_predict_say.txt \
                                        --capt-dir $capt_dir
fi
