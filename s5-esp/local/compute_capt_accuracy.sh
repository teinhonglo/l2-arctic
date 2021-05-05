#!/usr/bin/env bash

. ./cmd.sh
. ./path.sh

stage=0

phonemap="conf/phones.60-48-39.map"
should_say="data/test_l2_should_say/text"
actual_say="data/test_l2/text"
predict_say=
capt_dir=
phone_table=
ignored_phones="<eps>"

. parse_options.sh || exit 1;

if [ -z $predict_say ] || [ -z $capt_dir ] || [ -z $phone_table ]; then
	echo "local/compute_capt_accuracy.sh --should_say data/test_l2_should_say/text --actual_say data/test_l2/text --predict_say exp/chain/tdnn1k_sp/decode_test_l2/scoring/4.tra --capt-dir exp/chain/tdnn1k_sp/decode_test_l2/capt --phone_table data/local/lang/phones.txt"
	exit 0;
fi

if [ ! -d $capt_dir ]; then
    mkdir -p $capt_dir;
fi

# Map reference to 39 phone classes
if [ $stage -le 0 ]; then
    cat $should_say | local/timit_norm_trans.pl -i - -m $phonemap -from 48 -to 39 > $capt_dir/should_say.txt
    cat $actual_say | local/timit_norm_trans.pl -i - -m $phonemap -from 48 -to 39 > $capt_dir/actual_say.txt
    cat $predict_say | local/timit_norm_trans.pl -i - -m $phonemap -from 48 -to 39 > $capt_dir/predict_say.txt
	# remove sil or not?
	for fn in $capt_dir/should_say.txt $capt_dir/actual_say.txt $capt_dir/predict_say.txt; do
		sed -i "s/sil//g" $fn
	done
fi

# Alignment
if [ $stage -le 1 ]; then
    # Annotation
    align-text --special-symbol="'***'" ark:$capt_dir/should_say.txt ark:$capt_dir/actual_say.txt ark,t:- | \
	utils/scoring/wer_per_utt_details.pl --special-symbol "'***'" > $capt_dir/annotation.txt
    # Prediction
    align-text --special-symbol="'***'" ark:$capt_dir/should_say.txt ark:$capt_dir/predict_say.txt ark,t:- | \
	utils/scoring/wer_per_utt_details.pl --special-symbol "'***'" > $capt_dir/prediction.txt
fi

eval "$(/share/homes/teinhonglo/anaconda3/bin/conda shell.bash hook)"
# Performance (recall, precision, and f1)
if [ $stage -le 2 ]; then
    # python
	python local/compute_capt_accuracy.py --anno $capt_dir/annotation.txt \
                                          --pred $capt_dir/prediction.txt \
                                          --capt_dir $capt_dir \
                                          --phone_table $phone_table \
                                          --ignored_phones $ignored_phones > $capt_dir/results.log
fi

cat $capt_dir/results.log
