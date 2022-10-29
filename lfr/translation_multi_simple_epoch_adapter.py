from fairseq.tasks.translation_multi_simple_epoch import TranslationMultiSimpleEpochTask

from fairseq.tasks import register_task
from fairseq.optim.amp_optimizer import AMPOptimizer
from fairseq import metrics, search, tokenizer, utils

from.sequence_generator_adapter import SequenceGeneratorAdapter

import torch
import torch.nn as nn

from collections import defaultdict


def dot_product(g_i, g_j): # use double() if returns INF
    return torch.dot(g_i.float(), g_j.float()).item()

def l2_norm(g_i):
    return g_i.float().norm().item()

def pcgrad_proj(g_i, g_j):
    return (dot_product(g_i, g_j) / (l2_norm(g_j) ** 2)) * g_j


@register_task("translation_multi_simple_epoch_with_adapter")
class TranslationMultiSimpleEpochTaskWithAdapter(TranslationMultiSimpleEpochTask):
    """docstring for TranslationMultiSimpleEpochTaskWithAdapter"""

    def __init__(self, args, langs, dicts, training):
        super().__init__(args, langs, dicts, training)
        self.args = args
        # modify the gradient
        if args.fim_path != 'none':
            self.fisher_matrix = torch.load(args.fim_path)
            for key in self.fisher_matrix:
                self.fisher_matrix[key] = self.fisher_matrix[key] / (self.fisher_matrix[key].mean(dim=-1, keepdim=True) + 1e-8)
            
                if torch.isnan(self.fisher_matrix[key]).any():
                    print(key)
                    exit(-1)
        self.device = True


    @staticmethod
    def add_args(parser):
        TranslationMultiSimpleEpochTask.add_args(parser)
        parser.add_argument(
            "--encoder-adapter-langs",
            type=str,
            default='',
            help='The language needs to through the encoder adapter.'
        )
        parser.add_argument(
            "--decoder-adapter-langs",
            type=str,
            default='',
            help='The language needs to through the encoder adapter.'
        )
        parser.add_argument(
            '--through-adapter',
            type=str,
            default='both',
            choices=['encoder', 'decoder', 'both', 'none'],
            help='where the new langugaes?'
        )
        parser.add_argument('--fim-path', type=str, default='none', help='The fim file.')


    def get_langtok_index(self, lang_tok, dic):
        idx = dic.index(lang_tok)
        assert (
            idx != dic.unk_index
        ), "cannot find language token {} in the dictionary".format(lang_tok)
        return idx


    def move_device(self, model):
        for n, p in model.named_parameters():
            n = '.'.join(n.split('.')[2:])
            if p.requires_grad and n in self.fisher_matrix.keys():
                self.fisher_matrix[n] = self.fisher_matrix[n].to(p.device)



    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
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
        if self.args.through_adapter == 'encoder':
            sample['net_input']['encoder_through_adapter'] = True
            sample['net_input']['decoder_through_adapter'] = False
            sample['net_input']['encoder_adapter_mask'] = None
            sample['net_input']['decoder_adapter_mask'] = None
        elif self.args.through_adapter == 'decoder':
            sample['net_input']['encoder_through_adapter'] = False
            sample['net_input']['decoder_through_adapter'] = True
            sample['net_input']['encoder_adapter_mask'] = None
            sample['net_input']['decoder_adapter_mask'] = None
        elif self.args.through_adapter == 'both':
            sample['net_input']['encoder_through_adapter'] = True
            sample['net_input']['decoder_through_adapter'] = True
            src_tok = sample['net_input']['src_tokens']
            tgt_tok = sample['target']
            encoder_adapter_mask = src_tok.new_zeros(src_tok.size(0))
            for l in self.args.encoder_adapter_langs.split(','):
                encoder_adapter_mask |= (src_tok[:, 0] == self.get_langtok_index('__' + l + '__', self.dicts[l]))
            encoder_adapter_mask = encoder_adapter_mask.contiguous().view(1, -1, 1).bool()
            decoder_adapter_mask = tgt_tok.new_zeros(tgt_tok.size(0))
            for l in self.args.decoder_adapter_langs.split(','):
                decoder_adapter_mask |= (tgt_tok[:, 0] == self.get_langtok_index('__' + l + '__', self.dicts[l]))
            decoder_adapter_mask = decoder_adapter_mask.contiguous().view(1, -1, 1).bool()
            sample['net_input']['encoder_adapter_mask'] = encoder_adapter_mask
            sample['net_input']['decoder_adapter_mask'] = decoder_adapter_mask
        else:
            sample['net_input']['encoder_through_adapter'] = False
            sample['net_input']['decoder_through_adapter'] = False
            sample['net_input']['encoder_adapter_mask'] = None
            sample['net_input']['decoder_adapter_mask'] = None
        


        with torch.autograd.profiler.record_function("forward"):
            with torch.cuda.amp.autocast(enabled=(isinstance(optimizer, AMPOptimizer))):
                loss, sample_size, logging_output = criterion(model, sample)
        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)

        # modify the gradient
        if self.device:
            self.move_device(model)
            self.device = False
        return loss, sample_size, logging_output

    def inference_step(
        self, generator, models, sample, prefix_tokens=None, constraints=None
    ):
        #print(models)
        #print(models[0].args)
        #print(models[0].args.prompt_type)
        #exit(-1)
        with torch.no_grad():
            if self.args.source_lang in self.args.encoder_adapter_langs:
                sample['net_input']['encoder_through_adapter'] = True
            else:
                sample['net_input']['encoder_through_adapter'] = False
            
            if self.args.target_lang in self.args.decoder_adapter_langs:
                sample['net_input']['decoder_through_adapter'] = True
            else:
                sample['net_input']['decoder_through_adapter'] = False

            
            #print('args', self.args.prompt_type)
            #print('sample', sample)
            #exit(-1)
            _, tgt_langtok_spec = self.args.langtoks["main"]
            if not self.args.lang_tok_replacing_bos_eos:
                if prefix_tokens is None and tgt_langtok_spec:
                    tgt_lang_tok = self.data_manager.get_decoder_langtok(
                        self.args.target_lang, tgt_langtok_spec
                    )
                    src_tokens = sample["net_input"]["src_tokens"]
                    bsz = src_tokens.size(0)
                    prefix_tokens = (
                        torch.LongTensor([[tgt_lang_tok]]).expand(bsz, 1).to(src_tokens)
                    )
                return generator.generate(
                    models,
                    sample,
                    prefix_tokens=prefix_tokens,
                    constraints=constraints,
                )
            else:
                return generator.generate(
                    models,
                    sample,
                    prefix_tokens=prefix_tokens,
                    bos_token=self.data_manager.get_decoder_langtok(
                        self.args.target_lang, tgt_langtok_spec
                    )
                    if tgt_langtok_spec
                    else self.target_dictionary.eos(),
                )

    def build_generator(
        self,
        models,
        args,
        seq_gen_cls=None,
        extra_gen_cls_kwargs=None,
    ):
        if not getattr(args, "keep_inference_langtok", False):
            _, tgt_langtok_spec = self.args.langtoks["main"]
            if tgt_langtok_spec:
                tgt_lang_tok = self.data_manager.get_decoder_langtok(
                    self.args.target_lang, tgt_langtok_spec
                )
                extra_gen_cls_kwargs = extra_gen_cls_kwargs or {}
                extra_gen_cls_kwargs["symbols_to_strip_from_output"] = {tgt_lang_tok}

        # return super().build_generator(
        #    models, args, seq_gen_cls=None, extra_gen_cls_kwargs=extra_gen_cls_kwargs
        #)

        if getattr(args, "score_reference", False):
            from fairseq.sequence_scorer import SequenceScorer

            return SequenceScorer(
                self.target_dictionary,
                compute_alignment=getattr(args, "print_alignment", False),
            )

        from fairseq.sequence_generator import (
            SequenceGenerator,
            SequenceGeneratorWithAlignment,
        )

        # Choose search strategy. Defaults to Beam Search.
        sampling = getattr(args, "sampling", False)
        sampling_topk = getattr(args, "sampling_topk", -1)
        sampling_topp = getattr(args, "sampling_topp", -1.0)
        diverse_beam_groups = getattr(args, "diverse_beam_groups", -1)
        diverse_beam_strength = getattr(args, "diverse_beam_strength", 0.5)
        match_source_len = getattr(args, "match_source_len", False)
        diversity_rate = getattr(args, "diversity_rate", -1)
        constrained = getattr(args, "constraints", False)
        # if prefix_allowed_tokens_fn is None:
        prefix_allowed_tokens_fn = getattr(args, "prefix_allowed_tokens_fn", None)
        if (
            sum(
                int(cond)
                for cond in [
                    sampling,
                    diverse_beam_groups > 0,
                    match_source_len,
                    diversity_rate > 0,
                ]
            )
            > 1
        ):
            raise ValueError("Provided Search parameters are mutually exclusive.")
        assert sampling_topk < 0 or sampling, "--sampling-topk requires --sampling"
        assert sampling_topp < 0 or sampling, "--sampling-topp requires --sampling"

        if sampling:
            search_strategy = search.Sampling(
                self.target_dictionary, sampling_topk, sampling_topp
            )
        elif diverse_beam_groups > 0:
            search_strategy = search.DiverseBeamSearch(
                self.target_dictionary, diverse_beam_groups, diverse_beam_strength
            )
        elif match_source_len:
            # this is useful for tagging applications where the output
            # length should match the input length, so we hardcode the
            # length constraints for simplicity
            search_strategy = search.LengthConstrainedBeamSearch(
                self.target_dictionary,
                min_len_a=1,
                min_len_b=0,
                max_len_a=1,
                max_len_b=0,
            )
        elif diversity_rate > -1:
            search_strategy = search.DiverseSiblingsSearch(
                self.target_dictionary, diversity_rate
            )
        elif constrained:
            search_strategy = search.LexicallyConstrainedBeamSearch(
                self.target_dictionary, args.constraints
            )
        elif prefix_allowed_tokens_fn:
            search_strategy = search.PrefixConstrainedBeamSearch(
                self.target_dictionary, prefix_allowed_tokens_fn
            )
        else:
            search_strategy = search.BeamSearch(self.target_dictionary)

        extra_gen_cls_kwargs = extra_gen_cls_kwargs or {}
        if seq_gen_cls is None:
            if getattr(args, "print_alignment", False):
                seq_gen_cls = SequenceGeneratorWithAlignment
                extra_gen_cls_kwargs["print_alignment"] = args.print_alignment
            else:
                seq_gen_cls = SequenceGeneratorAdapter

        return seq_gen_cls(
            models,
            self.target_dictionary,
            beam_size=getattr(args, "beam", 5),
            max_len_a=getattr(args, "max_len_a", 0),
            max_len_b=getattr(args, "max_len_b", 200),
            min_len=getattr(args, "min_len", 1),
            normalize_scores=(not getattr(args, "unnormalized", False)),
            len_penalty=getattr(args, "lenpen", 1),
            unk_penalty=getattr(args, "unkpen", 0),
            temperature=getattr(args, "temperature", 1.0),
            match_source_len=getattr(args, "match_source_len", False),
            no_repeat_ngram_size=getattr(args, "no_repeat_ngram_size", 0),
            search_strategy=search_strategy,
            **extra_gen_cls_kwargs,
        )
