import torch
import torch.nn as nn

from fairseq.models.transformer import (
    TransformerModelBase,
    TransformerModel,
)
from fairseq.models import (
    register_model,
    register_model_architecture,
)
#from fairseq.models.transformer.transformer_base import Embedding
from fairseq.dataclass.utils import gen_parser_from_dataclass
from typing import Any, Dict, List, Optional
from torch import Tensor
from .transformer_decoder_adapter import TransformerDecoderAdapter
from .transformer_encoder_adapter import TransformerEncoderAdapter
from fairseq.models.transformer.transformer_config import (
    TransformerConfig,
    DEFAULT_MAX_SOURCE_POSITIONS,
    DEFAULT_MAX_TARGET_POSITIONS,
    DEFAULT_MIN_PARAMS_TO_WRAP,
)


@register_model("transformerwithadapter")
class TransformerWithAdapter(TransformerModelBase):
    """docstring for TransformerWithAdapter"""

    def __init__(self, args, encoder, decoder):
        cfg = TransformerConfig.from_namespace(args)
        super().__init__(cfg, encoder, decoder)
        self.args = args
        if args.fim_path != 'none':
            self.fisher_matrix = torch.load(args.fim_path)
            for key in self.fisher_matrix:
                self.fisher_matrix[key] = self.fisher_matrix[key] / (self.fisher_matrix[key].mean() + 1e-8)
                if torch.isnan(self.fisher_matrix[key]).any():
                    print(key)
                    exit(-1)

    @classmethod
    def add_args(cls, parser):
        """Add model-specific arguments to the parser."""
        gen_parser_from_dataclass(
            parser, TransformerConfig(), delete_default=True, with_prefix=""
        )
        parser.add_argument(
            "--need-new-adapter-embed-layer",
            action='store_true',
            default=False,
            help='add new embedding layer for the new language.'
        )

        parser.add_argument(
            "--control-type",
            type=str,
            default='none',
            choices=['curvature', 'output', 'none']
        )

        # parameter control based on the ratio and value
        parser.add_argument(
            "--par-fixed-ratio",
            type=float,
            default=0.,
            help="The ratio of parameters kept fixed"
        )
        parser.add_argument(
            "--par-change-range",
            type=float,
            default=-1.,
            help="The change range of pars."
        )
        parser.add_argument(
            "--freeze-specific-module",
            action='store_true',
            default=False,
            help='Freeze the old embed layer, position embed layer, norm layer.'
        )
        parser.add_argument(
            "--freeze-new-embed",
            action='store_true',
            default=False,
            help='Freeze the new embed layer. Used with --freeze-specific-module.'
        )
        # par control based on the computation
        parser.add_argument(
            "--ref-model-path",
            type=str,
            default='',
            help='The absolute path to the file saving the change range of par.'
        )


    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, "max_source_positions", None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if args.decoder_embed_path and (
                args.decoder_embed_path != args.encoder_embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            args.share_decoder_input_output_embed = True

        if getattr(args, "offload_activations", False):
            args.checkpoint_activations = True  # offloading implies checkpointing

        if not args.share_all_embeddings:
            args.min_params_to_wrap = getattr(
                args, "min_params_to_wrap", DEFAULT_MIN_PARAMS_TO_WRAP
            )
        cfg = TransformerConfig.from_namespace(args)

        cfg.decoder.input_dim = int(cfg.decoder.input_dim)
        cfg.decoder.output_dim = int(cfg.decoder.output_dim)
        if cfg.encoder.layers_to_keep:
            cfg.encoder.layers = len(cfg.encoder.layers_to_keep.split(","))
        if cfg.decoder.layers_to_keep:
            cfg.decoder.layers = len(cfg.decoder.layers_to_keep.split(","))

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        if cfg.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if cfg.encoder.embed_dim != cfg.decoder.embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if cfg.decoder.embed_path and (
                cfg.decoder.embed_path != cfg.encoder.embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            encoder_embed_tokens = cls.build_embedding(
                cfg, src_dict, cfg.encoder.embed_dim, cfg.encoder.embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            if args.need_new_adapter_embed_layer:
                adapter_encoder_embed_tokens = cls.build_embedding(
                    cfg, src_dict, cfg.encoder.embed_dim, cfg.encoder.embed_path
                )
                adapter_decoder_embed_tokens = adapter_encoder_embed_tokens
            else:
                adapter_encoder_embed_tokens = None
                adapter_decoder_embed_tokens = None

            cfg.share_decoder_input_output_embed = True

        else:
            encoder_embed_tokens = cls.build_embedding(
                cfg, src_dict, cfg.encoder.embed_dim, cfg.encoder.embed_path
            )
            decoder_embed_tokens = cls.build_embedding(
                cfg, tgt_dict, cfg.decoder.embed_dim, cfg.decoder.embed_path
            )
            if args.need_new_adapter_embed_layer:
                adapter_encoder_embed_tokens = cls.build_embedding(
                    cfg, src_dict, cfg.encoder.embed_dim, cfg.encoder.embed_path
                )
                adapter_decoder_embed_tokens = cls.build_embedding(
                    cfg, tgt_dict, cfg.decoder.embed_dim, cfg.decoder.embed_path
                )
            else:
                adapter_encoder_embed_tokens = None
                adapter_decoder_embed_tokens = None

        if cfg.offload_activations:
            cfg.checkpoint_activations = True  # offloading implies checkpointing
        encoder = cls.build_encoder(args, src_dict, encoder_embed_tokens, adapter_encoder_embed_tokens)
        decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens, adapter_decoder_embed_tokens)
        if not cfg.share_all_embeddings:
            # fsdp_wrap is a no-op when --ddp-backend != fully_sharded
            encoder = fsdp_wrap(encoder, min_num_params=cfg.min_params_to_wrap)
            decoder = fsdp_wrap(decoder, min_num_params=cfg.min_params_to_wrap)
        model = cls(args, encoder, decoder)

        if args.freeze_specific_module:
            for n, p in model.named_parameters():
                if 'norm' in n or 'embed_positions' in n:
                    p.requires_grad = False
                if args.need_new_adapter_embed_layer and '.embed_tokens.' in n:
                    p.requires_grad = False
                if args.freeze_new_embed and 'adapter' in n:
                    p.requires_grad = False

        return model

    def register_par_mask(self):
        self.par_min = {}
        self.par_max = {}
        
        if self.args.control_type == 'output':
            model_path = self.args.ref_model_path
            assert model_path != ''
            ref_model = torch.load(model_path)['model']
            for n, p in self.named_parameters():
                if p.requires_grad and 'adapter' not in n:
                    pp_min = p.data.detach().clone()
                    pp_max = ref_model[n].data.to(p.data).detach().clone()
                    p_min = torch.min(pp_min, pp_max)
                    p_max = torch.max(pp_min, pp_max)
                    self.par_min[n] = p_min
                    self.par_max[n] = p_max
            del ref_model

        elif self.args.control_type == 'curvature':
            for n, p in self.named_parameters():
                if p.requires_grad and 'adapter' not in n:
                    #tmp_n = '.'.join(n.split('.')[2:])
                    tmp_p = self.fisher_matrix[n].data.detach().clone().abs().view(-1).to(p.device)
                    num_p = int(tmp_p.numel() * (1 - self.args.par_fixed_ratio))
                    value, index = tmp_p.topk(num_p, largest=False)
                    p_min = p.data.detach().clone().view(-1)
                    p_max = p.data.detach().clone().view(-1)
                    p_tmp = p_min.index_select(0, index)
                    if self.args.par_change_range > 0:
                        pp_min = -self.args.par_change_range * (p_tmp.abs()) + p_tmp
                        pp_max = self.args.par_change_range * (p_tmp.abs()) + p_tmp
                    else:
                        pp_min = -p.data.new_ones(p_tmp.size()) * 1e4
                        pp_max = p.data.new_ones(p_tmp.size()) * 1e4
                    p_min.scatter_(0, index, torch.min(pp_min, pp_max))
                    p_max.scatter_(0, index, torch.max(pp_min, pp_max))
                    p_min = p_min.view(p.data.size())
                    p_max = p_max.view(p.data.size())
                    self.par_min[n] = p_min
                    self.par_max[n] = p_max

    def clip_par(self):
        with torch.no_grad():
            if self.args.control_type != 'none':
                for n, p in self.named_parameters():
                    if n in self.par_min.keys():
                        p.data = torch.max(torch.min(p.data, self.par_max[n]), self.par_min[n])
    @classmethod
    def get_langtok_index(cls, lang_tok, dic):
        idx = dic.index(lang_tok)
        assert (
            idx != dic.unk_index
        ), "cannot find language token {} in the dictionary".format(lang_tok)
        return idx

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens, adapter_embed_tokens):
        return TransformerEncoderAdapter(args, src_dict, embed_tokens, adapter_embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens, adapter_embed_tokens):
        return TransformerDecoderAdapter(
            args,
            tgt_dict,
            embed_tokens,
            adapter_embed_tokens,
        )

    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        encoder_through_adapter=False,
        decoder_through_adapter=False,
        encoder_adapter_mask=None,
        decoder_adapter_mask=None,
    ):
        """
        Run the forward pass for an encoder-decoder model.

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """
        encoder_out = self.encoder(
            src_tokens,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
            through_adapter=encoder_through_adapter,
            adapter_mask=encoder_adapter_mask,
        )
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
            through_adapter=decoder_through_adapter,
            adapter_mask=decoder_adapter_mask,
        )
        return decoder_out

def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    #nn.init.constant_(m.weight[padding_idx], 0)
    return m

@register_model_architecture("transformerwithadapter", "transformer_base_adapter")
def base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.no_cross_attention = getattr(args, "no_cross_attention", False)
    args.cross_self_attention = getattr(args, "cross_self_attention", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)
    args.checkpoint_activations = getattr(args, "checkpoint_activations", False)
    args.offload_activations = getattr(args, "offload_activations", False)
    if args.offload_activations:
        args.checkpoint_activations = True
    args.encoder_layers_to_keep = getattr(args, "encoder_layers_to_keep", None)
    args.decoder_layers_to_keep = getattr(args, "decoder_layers_to_keep", None)
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)
    args.quant_noise_pq_block_size = getattr(args, "quant_noise_pq_block_size", 8)
    args.quant_noise_scalar = getattr(args, "quant_noise_scalar", 0)


@register_model_architecture("transformerwithadapter", "transformer_big_adapter")
def transformer_vaswani_wmt_en_de_big(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.dropout = getattr(args, "dropout", 0.3)
    base_architecture(args)
