from typing import List, Optional, Tuple, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.utils.checkpoint

from transformers import LlamaForCausalLM
from transformers.models.llama.modeling_llama import (
    _make_causal_mask,
    _expand_mask,
)
from transformers.utils import logging, ModelOutput


logger = logging.get_logger(__name__)


@dataclass
class TextDecoderOutputWithPast(ModelOutput):
    logits: torch.FloatTensor = None
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class TextDecoder(nn.Module):
    def __init__(
        self,
        config=None,
        txt_vocab_size=32002,
        orig_txt_vocab_size=-1,
        is_freeze=True,
        gradient_checkpointing=True,
    ):
        super().__init__()
        self.config = config
        self.is_freeze = is_freeze
        self.orig_txt_vocab_size = orig_txt_vocab_size
        assert orig_txt_vocab_size > 0 and orig_txt_vocab_size < txt_vocab_size
        
        self.head = nn.Linear(
            config.hidden_size, txt_vocab_size, bias=True
        )
        self.head_new = nn.Linear(config.hidden_size, txt_vocab_size - orig_txt_vocab_size, bias=True)

        self.gradient_checkpointing = gradient_checkpointing

        self.requires_grad_(not is_freeze)
        self.head_new.requires_grad_(True)

    def init_from_llm(self, llm_model: LlamaForCausalLM, orig_txt_vocab_size=-1):
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

        # init head weight from llm_model
        self.head.weight.data.copy_(
            llm_model.lm_head.weight.data[: self.head.weight.data.shape[0]]
        )
        if orig_txt_vocab_size > 0:
            if self.is_freeze:
                nn.init.constant_(self.head.weight[orig_txt_vocab_size:], 0.0)
            else:
                mean = llm_model.lm_head.weight[:orig_txt_vocab_size].mean()
                std = llm_model.lm_head.weight[:orig_txt_vocab_size].std()
                nn.init.trunc_normal_(
                    self.head.weight[orig_txt_vocab_size:], mean=mean, std=std
                )

        # init head bias from llm_model
        if llm_model.lm_head.bias is not None:
            self.head.bias.data.copy_(llm_model.lm_head.bias.data)
            if orig_txt_vocab_size > 0:
                if self.is_freeze:
                    nn.init.constant_(self.head.bias[orig_txt_vocab_size:], -100.0)
                else:
                    mean = llm_model.lm_head.bias[:orig_txt_vocab_size].mean()
                    std = llm_model.lm_head.bias[:orig_txt_vocab_size].std()
                    nn.init.trunc_normal_(
                        self.head.bias[orig_txt_vocab_size:], mean=mean, std=std
                    )
        elif self.head.bias is not None:
            nn.init.constant_(self.head.bias, 0)
            if self.is_freeze:
                nn.init.constant_(self.head.bias[orig_txt_vocab_size:], -100.0)
                self.head.bias.requires_grad_(False)

        nn.init.constant_(self.head_new.weight.data, 0.0)
        bias_min = -5.
        nn.init.constant_(self.head_new.bias, 100.0 + bias_min)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def print_parameters_stats(self, prefix=""):
        for name, module in self.named_children():
            print(
                f"# {prefix}{name} Total parameters: {sum(p.numel() for p in module.parameters()) / 1e6:.2f}M"
            )
            print(
                f"# {prefix}{name} Trainable parameters: {sum(p.numel() for p in module.parameters() if p.requires_grad) / 1e6:.2f}M"
            )

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(
        self, attention_mask, input_shape, inputs_embeds, past_key_values_length
    ):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(
                attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            ).to(inputs_embeds.device)
            combined_attention_mask = (
                expanded_attn_mask
                if combined_attention_mask is None
                else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def forward(
        self,
        inputs_embeds: Optional[torch.FloatTensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, TextDecoderOutputWithPast]:
        hidden_states = inputs_embeds
        outputs = TextDecoderOutputWithPast() if return_dict else ()

        logits = self.head(hidden_states)
        logits_new = self.head_new(hidden_states)
        logits[..., self.orig_txt_vocab_size:] = logits[..., self.orig_txt_vocab_size:] + logits_new

        if not return_dict:
            return (logits, *outputs)

        outputs.logits = logits
        return outputs
