#!/bin/bash
#export PATH="/{data_path}/miniconda3/bin:$PATH"
code_path=/apdcephfs/share_1157273/users/gushuhao/1_reasearch/fairseq-flores
data_path=/apdcephfs/share_1157273/users/gushuhao/1_reasearch/fairseq-flores
env_path=/apdcephfs/share_1157273/users/gushuhao/envs/py3-tc1.10-yu/bin
DATA=${data_path}/data_bin/ende_5domain/data_bin_combine_de_DE_en_XX
TOOL=${code_path}/lfr/train_control.py
dir=mBart50_5domain_1017_lr2e-4
lang_pairs='de_DE-en_XX'
#sleep 10000000h
num=0
for file in ${data_path}/checkpoints/$dir/*
do
    num=`expr $num + 1`
done

if [ $num -gt 1 ]
then
    reset='' 
    ckt='checkpoint_last.pt'
else
    ckt=${data_path}/checkpoints/baseline_model/tmp/mBart50-nn.pt
    reset='--reset-optimizer --reset-dataloader --reset-meters'
fi


#    --encoder-embedding-adapter --decoder-embedding-adapter \
#    --freeze-specific-module  --need-new-adapter-embed-layer \
#    --fp16   --memory-efficient-fp16 --fp16-init-scale 128  --fp16-scale-tolerance 0.0 \
#:<<BLOCK
#CUDA_VISIBLE_DEVICES=0,1,2,3  
${env_path}/python  $TOOL \
    $DATA $reset --fp16 --ddp-backend=legacy_ddp \
    --user-dir ${code_path}/lfr \
    --control-type 'output'  --seed 9527   \
    --par-fixed-ratio 0.75 --par-change-range 0.2  \
    --ref-model-path ${data_path}/checkpoints/baseline_model_per1_1017_fix0._lr2e-4/checkpoint_3_5000.pt \
    --freeze-specific-module \
    --restore-file $ckt \
    --encoder-adapter-langs el_EL,sk_SK --decoder-adapter-langs el_EL,sk_SK \
    --through-adapter 'none' \
    --fim-path ${data_path}/checkpoints/baseline_model/fim.pt \
    --encoder-attention-heads 16 --decoder-attention-heads 16 \
    --layernorm-embedding \
    --encoder-learned-pos --decoder-learned-pos \
    --dataset-impl mmap  \
    --arch transformer_big_adapter \
    --dropout 0.3 --attention-dropout 0.1 \
    --encoder-layers 12 --decoder-layers 12 \
    --encoder-normalize-before --decoder-normalize-before \
    --share-all-embeddings \
    --save-dir ${data_path}/checkpoints/$dir \
    --task translation_multi_simple_epoch_with_adapter \
    --encoder-langtok "src" --decoder-langtok \
    --lang-pairs $lang_pairs \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
    --lr-scheduler inverse_sqrt --lr 5e-4 --warmup-init-lr 1e-07 --warmup-updates 4000 \
    --max-tokens 1024  --update-freq 2 --max-epoch 30 --max-update 30000 \
    --save-interval 1 --disable-validation   --no-epoch-checkpoints \
    --save-interval-updates 2000 --keep-interval-updates 10 \
    --no-progress-bar --log-format json --log-interval 25 2>&1 | tee ${data_path}/out_log/out.$dir && 
#BLOCK
#exit


ref=/apdcephfs/share_1157273/users/gushuhao/2_data/5domain
it=$ref/it/test.en
law=$ref/law/test.en
medical=$ref/medical/test.en
subtitle=$ref/subtitles/test.en
koran=$ref/koran/test.en

ref1=/apdcephfs/share_1157273/users/gushuhao/2_data/flores/flores101_dataset/devtest
zh=$ref1/zho.devtest
en=$ref1/eng.devtest
fr=$ref1/fra.devtest
de=$ref1/deu.devtest
el=$ref1/ell.devtest
sk=$ref1/slk.devtest

gen_sen(){
    #CUDA_VISIBLE_DEVICES=$1 ${env_path}/python ${code_path}/fairseq_cli/generate.py ${data_path}/data_bin/$2 --path $3 --gen-subset test --beam 5 --batch-size 200 --remove-bpe 'sentencepiece'  --lenpen 1  -s $4 -t $5  --task translation_multi_simple_epoch --lang-pairs ${data_path}/data_bin/lang_pairs.txt --decoder-langtok --encoder-langtok src --fp16 --dataset-impl mmap --fixed-dictionary  ${data_path}/data_bin/flores_mbart50spm_en/dict.af_ZA.txt   | tee ${data_path}/tmp/$6/gen.out.$(basename $3).$4-$5.$7
    CUDA_VISIBLE_DEVICES=$1 ${env_path}/python ${code_path}/fairseq_cli/generate.py ${data_path}/data_bin/$2 --path $3 --gen-subset test --beam 5 --batch-size 200 --remove-bpe 'sentencepiece'  --lenpen 1  -s $4 -t $5  --task translation_multi_simple_epoch_with_adapter --lang-pairs ${data_path}/data_bin/lang_pairs.txt --decoder-langtok --encoder-langtok src --fp16 --dataset-impl mmap --fixed-dictionary  ${data_path}/data_bin/flores_mbart50spm_en/dict.af_ZA.txt --user-dir ${code_path}/lfr --encoder-adapter-langs el_EL,sk_SK --decoder-adapter-langs el_EL,sk_SK  | tee ${data_path}/tmp/$6/gen.out.$(basename $3).$4-$5.$7
    ${env_path}/python ${data_path}/choose-translation.py ${data_path}/tmp/$6/gen.out.$(basename $3).$4-$5.$7 ${data_path}/tmp/$6/$4-$5.$(basename $3).$7 
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
mkdir -p ${data_path}/BLEU_domain
out_file=${data_path}/BLEU_domain/$model


for file in ${data_path}/checkpoints/$model/*
do
    echo $(basename $file) >> $out_file
    gen_sen 0 mBart50-test-file/data_bin_zh_CN_en_XX $file zh_CN en_XX $model lang &
    gen_sen 2 mBart50-test-file/data_bin_en_XX_zh_CN $file en_XX zh_CN $model lang &
    gen_sen 1 mBart50-test-file/data_bin_en_XX_fr_XX $file en_XX fr_XX $model lang &
    gen_sen 3 mBart50-test-file/data_bin_en_XX_de_DE $file en_XX de_DE $model lang &
    gen_sen 4 mBart50-test-file/data_bin_fr_XX_en_XX $file fr_XX en_XX $model lang &
    gen_sen 5 mBart50-test-file/data_bin_de_DE_en_XX $file de_DE en_XX $model lang &
    wait
    gen_sen 0 ende_5domain/data_bin_it_de_DE_en_XX $file  de_DE en_XX $model it &
    gen_sen 1 ende_5domain/data_bin_law_de_DE_en_XX $file  de_DE en_XX $model law &
    gen_sen 2 ende_5domain/data_bin_medical_de_DE_en_XX $file  de_DE en_XX $model medical &
    gen_sen 3 ende_5domain/data_bin_subtitles_de_DE_en_XX $file  de_DE en_XX $model subtitles &
    gen_sen 4 ende_5domain/data_bin_koran_de_DE_en_XX $file  de_DE en_XX $model koran &
    wait
    declare -a bleu 
    sum=0.0
    bleu[0]=it`cat ${data_path}/tmp/$model/de_DE-en_XX.$(basename $file).it | ${env_path}/sacrebleu $it -w 2 -tok spm`
    bleu[1]=law`cat ${data_path}/tmp/$model/de_DE-en_XX.$(basename $file).law | ${env_path}/sacrebleu $law -w 2 -tok spm`
    bleu[2]=medical`cat ${data_path}/tmp/$model/de_DE-en_XX.$(basename $file).medical | ${env_path}/sacrebleu $medical -w 2 -tok spm`
    bleu[3]=subtitle`cat ${data_path}/tmp/$model/de_DE-en_XX.$(basename $file).subtitles | ${env_path}/sacrebleu $subtitle -w 2 -tok spm`
    bleu[4]=koran`cat ${data_path}/tmp/$model/de_DE-en_XX.$(basename $file).koran | ${env_path}/sacrebleu $koran -w 2 -tok spm`
    bleu[5]=zh-en`cat ${data_path}/tmp/$model/zh_CN-en_XX.$(basename $file).lang | ${env_path}/sacrebleu $en -w 2 -tok spm`
    bleu[6]=fr-en`cat ${data_path}/tmp/$model/fr_XX-en_XX.$(basename $file).lang | ${env_path}/sacrebleu $en -w 2 -tok spm`
    bleu[7]=de-en`cat ${data_path}/tmp/$model/de_DE-en_XX.$(basename $file).lang | ${env_path}/sacrebleu $en -w 2 -tok spm`
    bleu[8]=en-zh`cat ${data_path}/tmp/$model/en_XX-zh_CN.$(basename $file).lang | ${env_path}/sacrebleu $zh -w 2 -tok spm`
    bleu[9]=en-fr`cat ${data_path}/tmp/$model/en_XX-fr_XX.$(basename $file).lang | ${env_path}/sacrebleu $fr -w 2 -tok spm`
    bleu[10]=en-de`cat ${data_path}/tmp/$model/en_XX-de_DE.$(basename $file).lang | ${env_path}/sacrebleu $de -w 2 -tok spm`
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
