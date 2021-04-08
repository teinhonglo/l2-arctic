#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train_tr"
valid_set="train_cv"
test_sets="test_l2 test_timit"
skip_data_prep=true

asr_config=conf/train_asr_conformer.yaml
lm_config=conf/train_lm_transformer.yaml
inference_config=conf/decode_asr.yaml

./asr.sh \
    --audio_format wav \
    --feats-type raw \
    --token-type word \
    --ngpu 1 \
    --nbpe 5000 \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --max_wav_duration 30 \
    --asr_config "${asr_config}" \
    --lm_config "${lm_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --skip_data_prep $skip_data_prep \
    "$@"
