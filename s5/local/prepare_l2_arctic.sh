
eval "$(/share/homes/teinhonglo/anaconda3/bin/conda shell.bash hook)"

ignore_chars="sp,sil,,spn,SIL"
train_set=train_l2
test_set=test_l2

. utils/parse_options.sh

echo "$ignore_chars"
python local/prepare_l2_arctic.py --ignore_chars "$ignore_chars";

utils/subset_data_dir.sh --utt-list data/l2_arctic/train_uttid data/l2_arctic/actual_say_phone data/$train_set
utils/subset_data_dir.sh --utt-list data/l2_arctic/test_uttid data/l2_arctic/actual_say_phone data/$test_set

utils/data/resample_data_dir.sh 16000 data/$train_set
utils/data/resample_data_dir.sh 16000 data/$test_set

utils/fix_data_dir.sh data/${train_set}
utils/fix_data_dir.sh data/${test_set}
