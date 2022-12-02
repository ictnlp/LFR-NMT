# LFR-NMT

Source code for the EMNLP 2022 main conference long paper ["Continual Learning of Neural Machine Translation within Low Forgetting Risk Regions"](https://arxiv.org/abs/2211.01542)

In this work, we focus on the continual learning of neural machine translation, wehre the model should learn the new knowledge constanly without forgetting the old knowledg.  Like many continual learning work, we assume that we have no access to the previous training data, so catastrophic forgetting problem is the biggest challenge for continual learning. 

To achieve this, we propose a two-stage training method. 
+ In the first stage, we search a low forgetting riks (LFR) region around the parameters of the pre-trained model, where we can retain the performance of the model on the previous task as the parameters are updated within this region.
+ In the second stage, the parameters are updated completely guided by the gradients produced by the new training data within this region.


## Code  

This code is based on the open source toolkit [fairseq-py](https://github.com/facebookresearch/fairseq).

All the core codes of our method are put in the folders "./k_d" and "./lfr".

Codes in "./k_d" are mainly related to the fisrt stage training, which searches the LFR regions.

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


