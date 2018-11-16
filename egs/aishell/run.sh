#!/bin/bash

stage=-1

ngpu=1         # number of gpus ("0" uses cpu, otherwise use gpu)
nj=40

dumpdir=dump   # directory to dump full features

# Feature configuration
do_delta=true

# Network architecture
# Encoder
einput=240
ehidden=256
elayer=3
edropout=0.2
ebidirectional=1
etype=lstm
# Attention
atype=dot
# Decoder
dembed=512
dhidden=512
dlayer=1

# Training config
epochs=20
half_lr=1
early_stop=0
max_norm=5
batch_size=32
maxlen_in=800
maxlen_out=150
optimizer=adam
lr=1e-3
momentum=0
l2=1e-5
checkpoint=1
print_freq=10
visdom=0
visdom_id="LAS Training"

# Decode config
beam_size=30
nbest=1
decode_max_len=100

# exp tag
tag="" # tag for managing experiments.

data=/home/work_nfs/common/data

. utils/parse_options.sh || exit 1;
. ./cmd.sh
. ./path.sh

if [ $stage -le 0 ]; then
    echo "stage 0: Data Preparation"
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    # Generate wav.scp, text, utt2spk, spk2utt (segments)
    local/aishell_data_prep.sh $data/data_aishell/wav $data/data_aishell/transcript || exit 1;
    # remove space in text
    for x in train test dev; do
        cp data/${x}/text data/${x}/text.org
        paste -d " " <(cut -f 1 -d" " data/${x}/text.org) <(cut -f 2- -d" " data/${x}/text.org | tr -d " ") \
            > data/${x}/text
    done
fi

feat_train_dir=${dumpdir}/train/delta${do_delta}; mkdir -p ${feat_train_dir}
feat_test_dir=${dumpdir}/test/delta${do_delta}; mkdir -p ${feat_test_dir}
feat_dev_dir=${dumpdir}/dev/delta${do_delta}; mkdir -p ${feat_dev_dir}
if [ $stage -le 1 ]; then
    echo "stage 1: Feature Generation"
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    fbankdir=fbank
    for data in train test dev; do
        steps/make_fbank.sh --cmd "$train_cmd" --nj $nj --write_utt2num_frames true \
            data/$data exp/make_fbank/$data $fbankdir/$data || exit 1;
    done
    # compute global CMVN
    compute-cmvn-stats scp:data/train/feats.scp data/train/cmvn.ark
    # dump features for training
    for data in train test dev; do
        feat_dir=`eval echo '$feat_'${data}'_dir'`
        dump.sh --cmd "$train_cmd" --nj $nj --do_delta $do_delta \
            data/$data/feats.scp data/train/cmvn.ark exp/dump_feats/$data $feat_dir
    done
fi

dict=data/lang_1char/train_chars.txt
echo "dictionary: ${dict}"
nlsyms=data/lang_1char/non_lang_syms.txt
if [ $stage -le 2 ]; then
    echo "stage 2: Dictionary and Json Data Preparation"
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    mkdir -p data/lang_1char/

    echo "make a non-linguistic symbol list"
    # It's empty in AISHELL-1
    cut -f 2- data/train/text | grep -o -P '\[.*?\]' | sort | uniq > ${nlsyms}
    cat ${nlsyms}

    echo "make a dictionary"
    echo "<unk> 0" >  ${dict}
    echo "<sos> 1" >> ${dict}
    echo "<eos> 2" >> ${dict}
    text2token.py -s 1 -n 1 -l ${nlsyms} data/train/text | cut -f 2- -d" " | tr " " "\n" \
    | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+2}' >> ${dict}
    wc -l ${dict}

    echo "make json files"
    for data in train test dev; do
        feat_dir=`eval echo '$feat_'${data}'_dir'`
        data2json.sh --feat ${feat_dir}/feats.scp --nlsyms ${nlsyms} \
             data/$data ${dict} > ${feat_dir}/data.json
    done
fi

if [ -z ${tag} ]; then
    expdir=exp/train_in${einput}_hidden${ehidden}_e${elayer}_${etype}_drop${edropout}_${atype}_emb${dembed}_hidden${dhidden}_d${dlayer}_epoch${epochs}_norm${max_norm}_bs${batch_size}_mli${maxlen_in}_mlo${maxlen_out}_${optimizer}_lr${lr}_mmt${momentum}_l2${l2}
    if ${do_delta}; then
        expdir=${expdir}_delta
    fi
else
    expdir=exp/train_${tag}
fi
mkdir -p ${expdir}

if [ ${stage} -le 3 ]; then
    echo "stage 4: Network Training"
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        train.py \
        --train-json ${feat_train_dir}/data.json \
        --valid-json ${feat_dev_dir}/data.json \
        --dict ${dict} \
        --einput $einput \
        --ehidden $ehidden \
        --elayer $elayer \
        --edropout $edropout \
        --ebidirectional $ebidirectional \
        --etype $etype \
        --atype $atype \
        --dembed $dembed \
        --dhidden $dhidden \
        --dlayer $dlayer \
        --epochs $epochs \
        --half-lr $half_lr \
        --early-stop $early_stop \
        --max-norm $max_norm \
        --batch-size $batch_size \
        --maxlen-in $maxlen_in \
        --maxlen-out $maxlen_out \
        --optimizer $optimizer \
        --lr $lr \
        --momentum $momentum \
        --l2 $l2 \
        --save-folder ${expdir} \
        --checkpoint $checkpoint \
        --print-freq ${print_freq} \
        --visdom $visdom \
        --visdom-id "$visdom_id"
fi

if [ ${stage} -le 4 ]; then
    echo "stage 5: Decoding"
    decode_dir=${expdir}/decode_test_beam${beam_size}_nbest${nbest}_ml${decode_max_len}
    mkdir -p ${decode_dir}
    ${cuda_cmd} --gpu ${ngpu} ${decode_dir}/decode.log \
        recognize.py \
        --recog-json ${feat_test_dir}/data.json \
        --dict $dict \
        --result-label ${decode_dir}/data.json \
        --model-path ${expdir}/final.pth.tar \
        --beam-size $beam_size \
        --nbest $nbest \
        --decode-max-len $decode_max_len

    # Compute CER
    local/score.sh --nlsyms ${nlsyms} ${decode_dir} ${dict}
fi