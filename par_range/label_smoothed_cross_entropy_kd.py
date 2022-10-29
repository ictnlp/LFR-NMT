from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import (
    LabelSmoothedCrossEntropyCriterion,
    LabelSmoothedCrossEntropyCriterionConfig,
)
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import metrics, utils


@register_criterion("label_smoothed_cross_entropy_with_kd")
class LabelSmoothedCrossEntropyCriterionWithKD(
    LabelSmoothedCrossEntropyCriterion
):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
        kd_lambda=0.0,
    ):
        super().__init__(task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy)
        self.kd_lambda = kd_lambda
        self.flag = True
        self.par_index = {}


    @staticmethod
    def add_args(parser):
        LabelSmoothedCrossEntropyCriterion.add_args(parser)
        parser.add_argument("--kd-lambda", type=float,
                            default=1.,
                            help="The kd loss weight")


    def compute_kl_loss(self, model, model_teacher, net_output, teacher_net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1)
        teacher_probs = model_teacher.get_normalized_probs(teacher_net_output, log_probs=False)
        teacher_probs = teacher_probs.view(-1, teacher_probs.size(-1))
        if target.dim() == lprobs.dim() - 1:
            target = target.unsqueeze(-1)
        #kl_loss = -(teacher_probs.detach().clone() * lprobs).sum(dim=-1, keepdim=True)
        kl_loss = F.kl_div(lprobs, teacher_probs.detach().clone(), reduction='none').sum(dim=-1, keepdim=True)
        pad_mask = target.eq(self.padding_idx)
        kl_loss.masked_fill_(pad_mask, 0.0)
        kl_loss = kl_loss.sum()
        return kl_loss

    def compute_par_loss(self, model, model_teacher):
        par_loss = 0
        par_num = 0

        for item in zip(model.parameters(), model_teacher.parameters()):
            assert item[0].numel() == item[1].numel()
            if getattr(item[0], 'requires_grad', None) != None:
                if item[0].requires_grad():
                    par_loss += (item[0] - item[1].detach().clone()).abs().sum()  # / item[0].numel()
                    par_num += item[0].numel()
            #par_num += 1

        par_loss = par_loss / par_num
        return par_loss.half()

    def par_l2_loss_mask(self, model, model_teacher, eps=100):
        par_loss = 0
        par_num = 0
        # for n, p in model.named_parameters():
        for item in zip(model.named_parameters(), model_teacher.named_parameters()):
            n, p = item[0][0], item[0][1]
            n = '.'.join(n.split('.')[2:])
            n_t, p_t = item[1][0], item[1][1]
            n_t = '.'.join(n_t.split('.')[2:])
            if p.requires_grad and 'embed' not in n and 'norm' not in n:
                assert p.numel() == p_t.numel()
                #par_change = 1 / (self.kd_lambda * (p - p_t.detach().clone())**2 + eps).view(-1)
                par_change = (-self.kd_lambda * (p - p_t.detach().clone())**2).flatten()
                par_loss += par_change.double().sum() #/ (p_t.numel() - num_p))
                par_num += p_t.numel()
        # print(par_loss)
        par_loss = par_loss / par_num
        self.flag = False
        return par_loss.half()

    def forward(self, model, model_teacher, sample, reduce=True):
        net_output = model(**sample["net_input"])
        teacher_net_output = model_teacher(**sample['net_input'])
        kl_loss = self.compute_kl_loss(model, model_teacher, net_output, teacher_net_output, sample)
        #par_loss = self.compute_par_loss(model, model_teacher)
        par_loss = self.par_l2_loss_mask(model, model_teacher)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        nsentences = sample["target"].size(0)
        ntokens = sample["ntokens"]

        loss = par_loss * sample_size + kl_loss
        #loss = -par_loss * sample_size + kl_loss * self.kd_lambda

        logging_output = {
            "loss": loss.data,
            "nll_loss": 0.0,
            "ntokens": ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
            "par_loss": par_loss.data,
            "kl_loss": kl_loss.data,
        }

        return loss, sample_size, logging_output

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        super().reduce_metrics(logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        par_loss = utils.item(
            sum(log.get("par_loss", 0) for log in logging_outputs)
        )
        kl_loss = utils.item(
            sum(log.get("kl_loss", 0) for log in logging_outputs)
        )
        metrics.log_scalar(
            "par_loss",
            par_loss / math.log(2) * 1000,
            sample_size,
            round=10,
        )
        metrics.log_scalar(
            "kl_loss",
            kl_loss / sample_size / math.log(2),
            sample_size,
            round=3,
        )
