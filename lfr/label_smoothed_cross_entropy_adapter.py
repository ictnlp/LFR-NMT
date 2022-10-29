from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import (
    LabelSmoothedCrossEntropyCriterion,
    LabelSmoothedCrossEntropyCriterionConfig,
)
import math
import torch
import torch.nn as nn

from fairseq import metrics, utils


def sentence_embedding(encoder_out, sample, padding_idx):
    encoder_output = encoder_out.transpose(0, 1)
    src_tokens = sample["net_input"]["src_tokens"]
    mask = (src_tokens != padding_idx)
    encoder_embedding = (encoder_output * mask.unsqueeze(-1)).sum(dim=1) / \
        mask.float().sum(dim=1).unsqueeze(-1)  # [batch, hidden_size]
    return encoder_embedding


def cost(x, y):
    len1 = x.size(-2)
    len2 = y.size(-2)
    dim = x.size(-1)
    bsz = x.size(0)
    tx = x.unsqueeze(dim=-2).expand(bsz, len1, len2, dim)
    ty = y.unsqueeze(dim=-3).expand(bsz, len1, len2, dim)

    # cosine
    #f_simi = torch.nn.CosineSimilarity(dim=-1)
    #res = 1. - f_simi(tx, ty)

    # L2
    res = torch.linalg.norm(tx - ty, dim=-1)
    return res


def compute_op_distance_min(x, y, x_mask, y_mask):
    C = cost(x, y)
    # approximate solution
    C.masked_fill_(x_mask.unsqueeze(dim=-1), 0).masked_fill_(y_mask.unsqueeze(dim=-2), 0)
    weight = torch.linalg.norm(x, dim=-1) / torch.linalg.norm(x, dim=-1).sum(dim=-1, keepdim=True)
    res = (C.min(dim=-1)[0] * weight.detach().clone()).sum()
    #res = C.min(dim=-1)[0].mean(dim=-1).sum()
    return res


@register_criterion("label_smoothed_cross_entropy_with_adapter")
class LabelSmoothedCrossEntropyCriterionWithAdapter(
    LabelSmoothedCrossEntropyCriterion
):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
        contrastive_lambda=0.0,
        contrastive_type='none',
    ):
        super().__init__(task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy)
        self.contrastive_lambda = contrastive_lambda
        self.contrastive_type = contrastive_type

    @staticmethod
    def add_args(parser):
        LabelSmoothedCrossEntropyCriterion.add_args(parser)
        parser.add_argument("--contrastive-lambda", type=float,
                            default=0.0,
                            help="The contrastive loss weight")
        parser.add_argument('--contrastive-type', type=str,
                            default='none',
                            choices=['none', 'cl', 'asl', 'ot'],
                            help='the type of contrastive loss.')

    def swap_sample(self, sample):
        target = sample["target"]
        prev_output_tokens = sample["net_input"]["prev_output_tokens"]
        src_tokens = torch.cat((prev_output_tokens[:, :1], sample["net_input"]['src_tokens']), dim=-1)
        return {
            "net_input": {
                "src_tokens": target.contiguous(),
                "src_lengths": (target != self.padding_idx).int().sum(dim=1),
                "prev_output_tokens": src_tokens[:, :-1].contiguous()
            },
            'nsentences': sample['nsentences'],
            'ntokens': utils.item((src_tokens[:, 1:] != self.padding_idx).int().sum().data),
            "target": src_tokens[:, 1:].contiguous(),
            "id": sample["id"],
        }

    def similarity_function(self, ):
        return nn.CosineSimilarity(dim=-1)

    def get_contrastive_loss(self, encoder_out1, encoder_out2, sample1, sample2):
        encoder_embedding1 = sentence_embedding(encoder_out1, sample1, self.padding_idx)  # [batch, hidden_size]
        encoder_embedding2 = sentence_embedding(encoder_out2, sample2, self.padding_idx)  # [batch, hidden_size]

        batch_size = encoder_embedding2.shape[0]
        feature_dim = encoder_embedding2.shape[1]
        anchor_feature = encoder_embedding1
        contrast_feature = encoder_embedding2

        similarity_function = self.similarity_function()
        anchor_dot_contrast = similarity_function(anchor_feature.expand((batch_size, batch_size, feature_dim)), torch.transpose(
            contrast_feature.expand((batch_size, batch_size, feature_dim)), 0, 1))

        loss = -nn.LogSoftmax(0)(torch.div(anchor_dot_contrast, 0.1)).diag().sum()

        return loss

    def average_sentence_loss(self, encoder_out1, encoder_out2, sample1, sample2):
        encoder_embedding1 = sentence_embedding(encoder_out1, sample1, self.padding_idx)  # [batch, hidden_size]
        encoder_embedding2 = sentence_embedding(encoder_out2, sample2, self.padding_idx)  # [batch, hidden_size]
        loss = torch.linalg.norm(encoder_embedding1 - encoder_embedding2, dim=-1)

        return loss.sum()

    def get_ot_loss(self, encoder_out, reversed_encoder_out, sample, reverse_sample):
        encoder_out = encoder_out.transpose(0, 1)
        reversed_encoder_out = reversed_encoder_out.transpose(0, 1)
        mask1 = (sample["net_input"]["src_tokens"] == self.padding_idx)
        mask2 = (reverse_sample["net_input"]["src_tokens"] == self.padding_idx)
        # for id in langugae_tag_id:
        #    mask1 |= sample["net_input"]["src_tokens"].eq(id)
        #    mask2 |= reverse_sample["net_input"]["src_tokens"].eq(id)
        #op_loss1 = compute_op_distance_ipot(reversed_encoder_out, encoder_out, mask2, mask1, time=5)
        #op_loss2 = compute_op_distance_ipot(encoder_out, reversed_encoder_out, mask1, mask2, time=5)
        op_loss1 = compute_op_distance_min(encoder_out, reversed_encoder_out, mask1, mask2)
        #op_loss2 = compute_op_distance_min(reversed_encoder_out, encoder_out, mask2, mask1)
        return op_loss1
        # return 0.5 * (op_loss1 + op_loss2)

    def forward(self, model, sample, reduce=True):
        net_output = model(**sample["net_input"])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        nsentences = sample["target"].size(0)
        ntokens = sample["ntokens"]
        if self.contrastive_type == 'none':
            contrastive_loss = 0
        else:
            encoder_out = model.encoder.forward(
                sample["net_input"]["src_tokens"],
                sample["net_input"]["src_lengths"],
                through_adapter=sample['net_input']['encoder_through_adapter'],
                adapter_mask=sample['net_input']['encoder_adapter_mask']
            )
            encoder_out = encoder_out['encoder_out'][-1]
            reverse_sample = self.swap_sample(sample)
            reversed_encoder_out = model.encoder.forward(
                reverse_sample["net_input"]["src_tokens"],
                reverse_sample["net_input"]["src_lengths"],
                through_adapter=sample['net_input']['decoder_through_adapter'],
                adapter_mask=sample['net_input']['decoder_adapter_mask']
            )
            reversed_encoder_out = reversed_encoder_out['encoder_out'][-1]
            if self.contrastive_type == 'cl':
                contrastive_loss = self.get_contrastive_loss(
                    encoder_out,
                    reversed_encoder_out,
                    sample,
                    reverse_sample,
                )
            elif self.contrastive_type == 'asl':
                contrastive_loss = self.average_sentence_loss(
                    encoder_out,
                    reversed_encoder_out,
                    sample,
                    reverse_sample,
                )
            elif self.contrastive_type == 'ot':
                contrastive_loss = self.get_ot_loss(
                    encoder_out,
                    reversed_encoder_out,
                    sample,
                    reverse_sample,
                )
        if isinstance(contrastive_loss, int):
            all_loss = loss
        else:
            all_loss = loss + contrastive_loss * self.contrastive_lambda * ntokens / nsentences
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
        }
        if isinstance(contrastive_loss, int):
            logging_output["contrastive_loss"] = 0
        else:
            logging_output["contrastive_loss"] = utils.item(contrastive_loss.data)

        return all_loss, sample_size, logging_output

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        super().reduce_metrics(logging_outputs)
        nsentences = utils.item(
            sum(log.get("nsentences", 0) for log in logging_outputs)
        )
        contrastive_loss = utils.item(
            sum(log.get("contrastive_loss", 0) for log in logging_outputs)
        )
        metrics.log_scalar(
            "contrastive_loss",
            contrastive_loss / nsentences / math.log(2),
            nsentences,
            round=3,
        )
