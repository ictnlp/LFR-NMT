import torch
import torch.nn as nn

from .transformer_layer_adapter import (
    TransformerEncoderLayerAdapter,
    TransformerDecoderLayerAdapter,
)

from fairseq.models.transformer import (
    TransformerConfig,
    TransformerEncoder,
    TransformerDecoder,
)

from typing import Any, Dict, List, Optional
from torch import Tensor
from fairseq.distributed import fsdp_wrap
from fairseq.modules.checkpoint_activations import checkpoint_wrapper


class TransformerEncoderAdapter(TransformerEncoder):
    """docstring for TransformerEncoderAdapter"""

    def __init__(self, args, dictionary, embed_tokens, adapter_embed_tokens):
        super().__init__(args, dictionary, embed_tokens)
        self.args = args
        self.need_new_adapter_embed_layer = args.need_new_adapter_embed_layer

        # new embedding layer for new laguages
        if self.need_new_adapter_embed_layer:
            self.adapter_embed_tokens = adapter_embed_tokens
        else:
            self.adapter_embed_tokens = None

        self.layers = nn.ModuleList([])
        for i in range(self.cfg.encoder.layers):
            self.layers.extend([self.build_encoder_layer(args, False)])

    def build_encoder_layer(self, args, adapter=False):
        cfg = TransformerConfig.from_namespace(args)
        layer = TransformerEncoderLayerAdapter(args, adapter)
        checkpoint = cfg.checkpoint_activations
        if checkpoint:
            offload_to_cpu = cfg.offload_activations
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = cfg.min_params_to_wrap if not checkpoint else 0
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer

    def adapter_forward_embedding(
        self, src_tokens, token_embedding: Optional[torch.Tensor] = None
    ):
        # embed tokens and positions
        if token_embedding is None:
            token_embedding = self.adapter_embed_tokens(src_tokens)
        x = embed = self.embed_scale * token_embedding
        if self.embed_positions is not None:
            x = embed + self.embed_positions(src_tokens)
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = self.dropout_module(x)
        if self.quant_noise is not None:
            x = self.quant_noise(x)
        return x, embed


    @torch.jit.unused
    def forward_non_torchscript(self, net_input: Dict[str, Tensor]):
        encoder_input = {}
        for k, v in net_input.items():
            if k != "prev_output_tokens" and k != "encoder_through_adapter" and k != "decoder_through_adapter":
                encoder_input[k] = v
            elif k == "encoder_through_adapter":
                encoder_input['through_adapter'] = v
        return self.forward(**encoder_input)

    def forward(
        self,
        src_tokens,
        src_lengths: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
        through_adapter=False,
        adapter_mask=None,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        return self.forward_scriptable(
            src_tokens, src_lengths, return_all_hiddens, token_embeddings, through_adapter, adapter_mask,
        )

    # TorchScript doesn't support super() method so that the scriptable Subclass
    # can't access the base class model in Torchscript.
    # Current workaround is to add a helper function with different name and
    # call the helper function from scriptable Subclass.
    def forward_scriptable(
        self,
        src_tokens,
        src_lengths: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
        through_adapter=False,
        adapter_mask=None,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings
            adapter_mask (bool): 1 * B * 1

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        has_pads = src_tokens.device.type == "xla" or encoder_padding_mask.any()

        if through_adapter and self.adapter_embed_tokens is not None:
            if adapter_mask is None:
                x, encoder_embedding = self.adapter_forward_embedding(src_tokens, token_embeddings)
            else:
                x, encoder_embedding = self.forward_embedding(src_tokens, token_embeddings)
                x_adapter, encoder_embedding_adapter = self.adapter_forward_embedding(src_tokens)
                x = x.masked_fill(adapter_mask.transpose(0, 1), 0.) + \
                    x_adapter.masked_fill(~adapter_mask.transpose(0, 1), 0.)
                x = x.contiguous()
                encoder_embedding = encoder_embedding.masked_fill(
                    adapter_mask.transpose(0, 1), 0.) + encoder_embedding_adapter.masked_fill(~adapter_mask.transpose(0, 1), 0.)
                encoder_embedding = encoder_embedding.contiguous()
        else:
            x, encoder_embedding = self.forward_embedding(src_tokens, token_embeddings)

        # account for padding while computing the representation
        if has_pads:
            x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_states = []

        if return_all_hiddens:
            encoder_states.append(x)

        # encoder layers
        for layer in self.layers:
            # for i in range(len(self.layers)):
            x = layer(
                x, encoder_padding_mask=encoder_padding_mask if has_pads else None,
            )
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # The Pytorch Mobile lite interpreter does not supports returning NamedTuple in
        # `forward` so we use a dictionary instead.
        # TorchScript does not support mixed values so the values are all lists.
        # The empty list is equivalent to None.
        src_lengths = (
            src_tokens.ne(self.padding_idx)
            .sum(dim=1, dtype=torch.int32)
            .reshape(-1, 1)
            .contiguous()
        )
        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [encoder_embedding],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [src_lengths],
        }
