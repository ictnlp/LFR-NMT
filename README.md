# LFR-NMT

Source code for the EMNLP 2022 main conference long paper ["Continual Learning of Neural Machine Translation within Low Forgetting Risk Regions"](https://arxiv.org/abs/2211.01542)

In this work, we focus on the continual learning of neural machine translation, wehre the model should learn the new knowledge constanly without forgetting the old knowledg.  Like many continual learning work, we assume that we have no access to the previous training data, so catastrophic forgetting problem is the biggest challenge for continual learning. 

To achieve this, we propose a two-stage training method. 
+ In the first stage, we search a low forgetting riks (LFR) region around the parameters of the pre-trained model, where we can retain the performance of the model on the previous task as the parameters are updated within this region.
+ In the second stage, the parameters are updated completely guided by the gradients produced by the new training data within this region.


## Code  

This code is based on the open source toolkit [fairseq-py](https://github.com/facebookresearch/fairseq).

All the core codes of our method are put in the folders "./par_range" and "./lfr".

Codes in "./par_range" are mainly related to the fisrt stage training, which searches the LFR regions.

Codes in "./lfr" are mainly related to the second stage training.


## Get Started 

This system has been tested in the following environment.
+ Python version \== 3.7
+ Pytorch version \== 1.10

### Build 
```
pip install --editable ./
```

### Pre-trained Model & Data
+ mBART50-nn: https://github.com/facebookresearch/fairseq/tree/main/examples/multilingual#mbart50-models
+ Multi-domain data: https://github.com/roeeaharoni/unsupervised-domain-clusters
+ Mulilingual data: https://opus.nlpl.eu/opus-100.php


### Training
We take the domain adaptation task as an example to show how to use the two methods.

**Curvature-based Method** 

First, we need to use the flroes dev data to computer the emprical fisher information matrix:

```
# pretrained mBART model
ckt=
CUDA_VISIBLE_DEVICES=0  python  par_range/fisher_information.py data_bin/flores_mbart50spm_en --reset-optimizer  --restore-file $ckt 
```

Then, train the model within the LFR regions:

```
TOOL=lfr/train_control.py
DATA=data_bin/ende_5domain/data_bin_combine_de_DE_en_XX
lang_pairs='de_DE-en_XX'
# pretrained model mBart50-nn
ckt=
# directory for saving checkpoints 
dir=
python  $TOOL \
    $DATA --fp16 --ddp-backend=legacy_ddp \
    --reset-optimizer --reset-dataloader --reset-meters \
    --user-dir lfr \
    --control-type 'curvature'  --seed 9527   \
    --par-fixed-ratio 0.75 --par-change-range 0.2  \
    --freeze-specific-module \
    --restore-file $ckt \
    --through-adapter 'none' \
    --fim-path par_range/fim.pt \
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
    --no-progress-bar --log-format json --log-interval 25 2>&1 
```

**Output-based Method** 

First, we need to search the LFR model by the model iteslf.

```
DATA=data_bin/flores_mbart50spm_en
TOOL=par_range/train_kd.py
lang_pairs=par_range/lang_pairs.txt
# pretrained model mBART-50nn
ckt=
# directory for saving checkpoints 
dir=checkpoints/lfr_ckt

CUDA_VISIBLE_DEVICES=0  python  $TOOL \
    $DATA  --restore-file $ckt \
    --reset-optimizer --reset-dataloader --reset-meters \
    --user-dir par_range  \
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
    --save-dir checkpoints/$dir \
    --task translation_multi_simple_epoch_with_kd \
    --encoder-langtok "src" --decoder-langtok \
    --lang-pairs $lang_pairs \
    --criterion label_smoothed_cross_entropy_with_kd --label-smoothing 0.1 \
    --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
    --lr-scheduler inverse_sqrt --lr 2e-4 --warmup-init-lr 1e-07 --warmup-updates 4000 \
    --max-tokens 1024  --update-freq 2 --max-epoch 200 --max-update 5000 \
    --save-interval 1 --disable-validation --no-epoch-checkpoints \
    --save-interval-updates 1000 --keep-interval-updates 30 \
    --no-progress-bar --log-format json --log-interval 25 2>&1 
```

Then, train the model with LFR regions:

```
DATA=data_bin/ende_5domain/data_bin_combine_de_DE_en_XX
TOOL=${code_path}/lfr/train_control.py
# directory for saving checkpoints 
dir=
lang_pairs='de_DE-en_XX'
# pretrained model mBART-50nn
ckt=

python  $TOOL \
    $DATA --fp16 --ddp-backend=legacy_ddp \
    --reset-optimizer --reset-dataloader --reset-meters \
    --user-dir lfr \
    --control-type 'output'  --seed 9527   \
    --ref-model-path checkpoints/lfr_ckt/checkpoint_last.pt \
    --freeze-specific-module \
    --restore-file $ckt \
    --through-adapter 'none' \
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
    --no-progress-bar --log-format json --log-interval 25 2>&1

```

**Decoding and Computing spmBLEU**

Taking the IT domain as an example

```
# model file
ckt=
CUDA_VISIBLE_DEVICES=0 python fairseq_cli/generate.py data_bin/ende_5domain/data_bin_it_de_DE_en_XX \
--path $ckt --gen-subset test --beam 5 --batch-size 200 --remove-bpe 'sentencepiece'  \
--lenpen 1  -s de_DE -t en_XX  --task translation_multi_simple_epoch_with_adapter \
--lang-pairs data_bin/lang_pairs.txt --decoder-langtok --encoder-langtok src \
--fp16 --dataset-impl mmap --fixed-dictionary  data_bin/flores_mbart50spm_en/dict.af_ZA.txt \
--user-dir lfr | tee it.out

python choose-translation.py it.out it.translation

#reference
it=
cat it.out | sacrebleu $it -w 2 -tok spm
```

I put all my scripts in the folder 'LFR_scripts'. You can refer to it for more instructions.

## Citation

```
@inproceedings{GuHF22,
  author    = {Shuhao Gu and
               Bojie Hu and
               Yang Feng},
  title     = {Continual Learning of Neural Machine Translation within Low Forgetting
               Risk Regions},
  booktitle = {Proceedings of the EMNLP 2022 Main Conference},
  year      = {2022},
  url       = {https://doi.org/10.48550/arXiv.2211.01542},
}
```

