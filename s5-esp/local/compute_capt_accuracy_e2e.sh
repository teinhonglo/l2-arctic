#!/usr/bin/env bash

. ./cmd.sh
. ./path.sh

stage=-1

phonemap="conf/phones.60-48-39.map"
should_say="data/test_l2_should_say/text"
actual_say="data/test_l2/text"
predict_say=
capt_dir=
phone_table=
ignored_phones="<eps>,<space>"

. utils/parse_options.sh || exit 1;

if [ -z $predict_say ] || [ -z $capt_dir ] || [ -z $phone_table ]; then
    echo "data_set=test_l2; dict=data/lang_1char/train_tr_units.txt; dir=exp/train_tr_sp_pytorch_train_transformer_no_preprocess/decode_test_l2_decode_pytorch_transformer"
    echo 'local/compute_capt_accuracy_e2e.sh --should_say data/${data_set}_should_say/text --actual_say data/${data_set}/text --predict_say $dir/hyp.trn --capt-dir $dir/capt --phone_table ${dict}'
	exit 0
fi

if [ ! -d $capt_dir ]; then
    mkdir -p $capt_dir;
fi

if [ $stage -le -1 ]; then
    python local/preprocess_e2e_results.py --src $predict_say --dest $capt_dir/espnet_predict_say.txt
fi

# Map reference to 39 phone classes
if [ $stage -le 0 ]; then
    local/compute_capt_accuracy.sh  --stage $stage \
                                        --should_say data/test_l2_should_say/text \
                                        --actual_say data/test_l2/text \
                                        --predict_say $capt_dir/espnet_predict_say.txt \
                                        --capt-dir $capt_dir \
                                        --phone_table $phone_table \
                                        --ignored_phones $ignored_phones
fi
