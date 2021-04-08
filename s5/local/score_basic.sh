#!/bin/bash
# Copyright 2012  Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0

# begin configuration section.
cmd=run.pl
stage=0
min_lmwt=1
max_lmwt=10
#end configuration section.

[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

if [ $# -ne 3 ]; then
  echo "Usage: local/score.sh [--cmd (run.pl|queue.pl...)] <data-dir> <lang-dir|graph-dir> <decode-dir>"
  echo " Options:"
  echo "    --cmd (run.pl|queue.pl...)      # specify how to run the sub-processes."
  echo "    --stage (0|1|2)                 # start scoring script from part-way through."
  echo "    --min_lmwt <int>                # minumum LM-weight for lattice rescoring "
  echo "    --max_lmwt <int>                # maximum LM-weight for lattice rescoring "
  exit 1;
fi

data=$1
lang_or_graph=$2
dir=$3

phonemap="conf/phones.60-48-39.map"
nj=$(cat $dir/num_jobs)

symtab=$lang_or_graph/words.txt

for f in $symtab $dir/lat.1.gz $data/text; do
  [ ! -f $f ] && echo "score.sh: no such file $f" && exit 1;
done

mkdir -p $dir/scoring/log

# Map reference to 39 phone classes:
cat $data/text | local/timit_norm_trans.pl -i - -m $phonemap -from 48 -to 39 > $dir/scoring/test_filt.txt

# Get the phone-sequence on the best-path:
for LMWT in $(seq $min_lmwt $max_lmwt); do
  $cmd JOB=1:$nj $dir/scoring/log/best_path_basic.$LMWT.JOB.log \
    lattice-best-path --lm-scale=$LMWT --word-symbol-table=$symtab --verbose=2 \
      "ark:gunzip -c $dir/lat.JOB.gz|" ark,t:- \| \
    utils/int2sym.pl -f 2- $symtab ">" $dir/scoring/$LMWT.JOB.tra || exit 1;
  cat $dir/scoring/$LMWT.*.tra | sort > $dir/scoring/$LMWT.tra
  rm $dir/scoring/$LMWT.*.tra
done

# Map hypothesis to 39 phone classes:
$cmd LMWT=$min_lmwt:$max_lmwt $dir/scoring/log/score_basic.LMWT.log \
   cat $dir/scoring/LMWT.tra \| \
    local/timit_norm_trans.pl -i - -m $phonemap -from 48 -to 39 \| \
    compute-wer --text --mode=all \
     ark:$dir/scoring/test_filt.txt ark,p:- ">&" $dir/wer_LMWT || exit 1;

grep WER $dir/wer_* | awk '{print $2,$i}' | sort -n|head -1 | awk '{print $2,$3}' | tee ${dir}/best_wer

#exp/chain/tdnn1k_sp/decode_test_l2/wer_4:%WER 30.71 
best_lmwt=`awk -F":" '{print $1}' ${dir}/best_wer | awk -F"_" '{print $NF}'`
echo "Best LMWT $best_lmwt"
mkdir -p $dir/scoring/wer_details

$cmd $dir/scoring/log/stats1.log \
    cat $dir/scoring/$best_lmwt.tra \| \
      align-text --special-symbol="'***'" ark:$dir/scoring/test_filt.txt ark:- ark,t:- \|  \
      utils/scoring/wer_per_utt_details.pl --special-symbol "'***'" \| tee $dir/scoring/wer_details/per_utt \|\
       utils/scoring/wer_per_spk_details.pl $data/utt2spk \> $dir/scoring/wer_details/per_spk || exit 1;
exit 0;
