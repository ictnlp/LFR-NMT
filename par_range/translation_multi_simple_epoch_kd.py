from fairseq.tasks.translation_multi_simple_epoch import TranslationMultiSimpleEpochTask

from fairseq.tasks import register_task
from fairseq.optim.amp_optimizer import AMPOptimizer
from fairseq import metrics, search, tokenizer, utils
from collections import defaultdict

import torch
import torch.nn as nn

def dot_product(g_i, g_j): # use double() if returns INF
    return torch.dot(g_i.float(), g_j.float()).item()

def l2_norm(g_i):
    return g_i.float().norm().item()

def pcgrad_proj(g_i, g_j):
    return (dot_product(g_i, g_j) / (l2_norm(g_j) ** 2)) * g_j



@register_task("translation_multi_simple_epoch_with_kd")
class TranslationMultiSimpleEpochTaskWithKD(TranslationMultiSimpleEpochTask):
    """docstring for TranslationMultiSimpleEpochTaskWithAdapter"""

    def __init__(self, args, langs, dicts, training):
        super().__init__(args, langs, dicts, training)
        self.args = args
        #self.oracle_manager = MultilingualDatasetManager.setup_data_manager(
        #    args, self.lang_pairs, langs, dicts, self.sampling_method
        #)

    @staticmethod
    def add_args(parser):
        TranslationMultiSimpleEpochTask.add_args(parser)
        parser.add_argument(
            "--kd",
            type=str,
            default='',
            help='place holder.'
        )

    
    def train_step(
        self, sample, model, model_teacher, criterion, optimizer, update_num, ignore_grad=False
    ):
        """
        Do forward and backward, and return the loss as computed by *criterion*
        for the given *model* and *sample*.

        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~fairseq.data.FairseqDataset`.
            model (~fairseq.models.BaseFairseqModel): the model
            criterion (~fairseq.criterions.FairseqCriterion): the criterion
            optimizer (~fairseq.optim.FairseqOptimizer): the optimizer
            update_num (int): the current update
            ignore_grad (bool): multiply loss by 0 if this is set to True

        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                  gradient
                - logging outputs to display while training
        """
        model.train()
        model.set_num_updates(update_num)
        #for n, p in model.named_parameters():
        #    print(n)
        #exit(-1)
        with torch.autograd.profiler.record_function("forward"):
            with torch.cuda.amp.autocast(enabled=(isinstance(optimizer, AMPOptimizer))):
                loss, sample_size, logging_output = criterion(model, model_teacher, sample)
                #loss, sample_size, logging_output = criterion(model, sample)
        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)
        return loss, sample_size, logging_output
