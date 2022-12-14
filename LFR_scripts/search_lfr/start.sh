#!/bin/bash
#export PATH="/{data_path}/miniconda3/bin:$PATH"
code_path=/apdcephfs/share_1157273/users/gushuhao/1_reasearch/fairseq-flores
data_path=/apdcephfs/share_1157273/users/gushuhao/1_reasearch/fairseq-flores
env_path=/apdcephfs/share_1157273/users/gushuhao/envs/py3-tc1.10-yu/bin
DATA=${data_path}/data_bin/flores_mbart50spm_en
TOOL=${code_path}/par_range/train_kd.py
dir=baseline_model_per1_1017_fix0._lr2e-4
lang_pairs=${data_path}/par_range/lang_pairs.txt
#sleep 10000000h
num=0
for file in ${data_path}/checkpoints/$dir/*
do
    num=`expr $num + 1`
done

if [ $num -gt 2 ]
then
    reset=''
    ckt='checkpoint_last.pt'
else
    reset='--reset-optimizer --reset-dataloader --reset-meters'
    ckt='/apdcephfs/share_1157273/users/gushuhao/1_reasearch/fairseq-flores/checkpoints/baseline_model/tmp/checkpoint_append_emb_flores.pt.bak'
fi


#:<<BLOCK
CUDA_VISIBLE_DEVICES=7  ${env_path}/python  $TOOL \
    $DATA $reset  --restore-file $ckt \
    --user-dir ${code_path}/par_range  \
    --seed 1234  --kd-lambda 1 \
    --encoder-attention-heads 16 --decoder-attention-heads 16 \
    --attention-dropout 0.1 --layernorm-embedding \
    --encoder-learned-pos --decoder-learned-pos \
    --dataset-impl mmap  \
    --arch transformer_big_adapter \
    --dropout 0.3  \
    --encoder-layers 12 --decoder-layers 12 \
    --encoder-normalize-before --decoder-normalize-before \
    --share-all-embeddings \
    --save-dir ${data_path}/checkpoints/$dir \
    --task translation_multi_simple_epoch_with_kd \
    --encoder-langtok "src" --decoder-langtok \
    --lang-pairs $lang_pairs \
    --criterion label_smoothed_cross_entropy_with_kd --label-smoothing 0.1 \
    --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
    --lr-scheduler inverse_sqrt --lr 2e-4 --warmup-init-lr 1e-07 --warmup-updates 4000 \
    --max-tokens 1024  --update-freq 2 --max-epoch 200 --max-update 5000 \
    --save-interval 1 --disable-validation --no-epoch-checkpoints \
    --save-interval-updates 1000 --keep-interval-updates 30 \
    --no-progress-bar --log-format json --log-interval 25 2>&1 | tee ${data_path}/out_log/out.$dir && 
#BLOCK
#exit


ref=/apdcephfs/share_1157273/users/gushuhao/2_data/flores/flores101_dataset/devtest
zh=$ref/zho.devtest
en=$ref/eng.devtest
fr=$ref/fra.devtest
de=$ref/deu.devtest
el=$ref/ell.devtest
sk=$ref/slk.devtest
lang_pairs=${DATA}/lang_pairs.txt

gen_sen(){
    CUDA_VISIBLE_DEVICES=$1 ${env_path}/python ${code_path}/fairseq_cli/generate.py ${data_path}/data_bin/mBart50-test-file/$2 --path $3 --gen-subset test --beam 5 --batch-size 400 --remove-bpe 'sentencepiece'  --lenpen 1  -s $4 -t $5  --task translation_multi_simple_epoch --lang-pairs ${data_path}/data_bin/lang_pairs.txt --decoder-langtok --encoder-langtok src --fp16 --dataset-impl mmap --fixed-dictionary  ${data_path}/data_bin/flores_mbart50spm_en/dict.af_ZA.txt --user-dir ${code_path}/par_range  | tee ${data_path}/tmp/$6/gen.out.$(basename $3).$4-$5
    ${env_path}/python ${data_path}/choose-translation.py ${data_path}/tmp/$6/gen.out.$(basename $3).$4-$5 ${data_path}/tmp/$6/$4-$5.$(basename $3)
}

subset=test
#gpu=1
#src_l=4
#tgt_l=5
#data_bin=2
#model=3
#model file=6
time=wmt14/full

#model=$1
model=$dir
mkdir -p ${data_path}/tmp/$model
mkdir -p ${data_path}/BLEU_per
out_file=${data_path}/BLEU_per/$model


for file in ${data_path}/checkpoints/$model/*
do
    echo $(basename $file) >> $out_file
    gen_sen 0 data_bin_zh_CN_en_XX $file zh_CN en_XX $model &
    gen_sen 2 data_bin_en_XX_zh_CN $file en_XX zh_CN $model &
    gen_sen 1 data_bin_en_XX_fr_XX $file en_XX fr_XX $model &
    gen_sen 3 data_bin_en_XX_de_DE $file en_XX de_DE $model &
    #wait
    gen_sen 5 data_bin_fr_XX_en_XX $file fr_XX en_XX $model &
    gen_sen 4 data_bin_de_DE_fr_XX $file de_DE fr_XX $model &
    #wait
    gen_sen 6 data_bin_fr_XX_zh_CN $file fr_XX zh_CN $model &
    gen_sen 7 data_bin_de_DE_en_XX $file de_DE en_XX $model &
    #wait
    #gen_sen 0 data_bin_el_EL_en_XX $file el_EL en_XX $model &
    #gen_sen 1 data_bin_en_XX_el_EL $file en_XX el_EL $model &
    #gen_sen 2 data_bin_en_XX_sk_SK $file en_XX sk_SK $model &
    #gen_sen 3 data_bin_sk_SK_en_XX $file sk_SK en_XX $model &
    wait
    declare -a bleu 
    sum=0.0
    bleu[0]=zh-en`cat ${data_path}/tmp/$model/zh_CN-en_XX.$(basename $file) | ${env_path}/sacrebleu $en -w 2 -tok spm`
    bleu[1]=fr-en`cat ${data_path}/tmp/$model/fr_XX-en_XX.$(basename $file) | ${env_path}/sacrebleu $en -w 2 -tok spm`
    bleu[2]=de-en`cat ${data_path}/tmp/$model/de_DE-en_XX.$(basename $file) | ${env_path}/sacrebleu $en -w 2 -tok spm`
    bleu[3]=en-zh`cat ${data_path}/tmp/$model/en_XX-zh_CN.$(basename $file) | ${env_path}/sacrebleu $zh -w 2 -tok spm`
    bleu[4]=en-fr`cat ${data_path}/tmp/$model/en_XX-fr_XX.$(basename $file) | ${env_path}/sacrebleu $fr -w 2 -tok spm`
    bleu[5]=en-de`cat ${data_path}/tmp/$model/en_XX-de_DE.$(basename $file) | ${env_path}/sacrebleu $de -w 2 -tok spm`
    bleu[6]=de-fr`cat ${data_path}/tmp/$model/de_DE-fr_XX.$(basename $file) | ${env_path}/sacrebleu $fr -w 2 -tok spm`
    bleu[7]=fr-zh`cat ${data_path}/tmp/$model/fr_XX-zh_CN.$(basename $file) | ${env_path}/sacrebleu $zh -w 2 -tok spm`
    #bleu[8]=el-en`cat ${data_path}/tmp/$model/el_EL-en_XX.$(basename $file) | ${env_path}/sacrebleu $en -w 2 -tok spm`
    #bleu[9]=en-el`cat ${data_path}/tmp/$model/en_XX-el_EL.$(basename $file) | ${env_path}/sacrebleu $el -w 2 -tok spm`
    #bleu[10]=sk-en`cat ${data_path}/tmp/$model/sk_SK-en_XX.$(basename $file) | ${env_path}/sacrebleu $en -w 2 -tok spm`
    #bleu[11]=en-sk`cat ${data_path}/tmp/$model/en_XX-sk_SK.$(basename $file) | ${env_path}/sacrebleu $sk -w 2 -tok spm`
    declare -a b
    for ((i=0;i<${#bleu[*]};i++))
    do
        echo "${bleu[$i]}" >> $out_file
        b[$i]=`echo ${bleu[$i]}|sed "s/.*1.5.1\ =\ \([0-9.]\{1,\}\).*/\1/"`
        sum=`echo "scale=2;$sum+${b[$i]}"|bc`
    done
    avg=`echo "scale=2;$sum/${#b[*]}"|bc`
    echo  "AVG  $avg" >> $out_file
done
