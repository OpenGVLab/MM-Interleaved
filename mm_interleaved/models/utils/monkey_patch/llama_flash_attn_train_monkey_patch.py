from typing import List, Optional, Tuple
import warnings

import torch

from einops import rearrange
from flash_attn.flash_attn_interface import flash_attn_func

# ADAPTED from https://github.com/allenai/open-instruct/blob/main/open_instruct/llama_flash_attn_monkey_patch.py
# AND https://github.com/lm-sys/FastChat/blob/main/fastchat/train/llama_flash_attn_monkey_patch.py
# AND https://github.com/LAION-AI/Open-Assistant/blob/04fa9a24b2a58c8885b8aa6a2eb02b18de6b4961/model/model_training/models/patching_llama.py
# AND Sourabh https://github.com/huggingface/transformers/commit/ee81bf5aee0d65f005d157c013777e3d27d8d6bf


def _rotate_half_train(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# @torch.compile(mode="reduce-overhead")
def _apply_rotary_pos_emb_train(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    # q,k : [bsz, q_len, nh, hd]
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(2)  # [bs, seq_len, 1, dim]
    sin = sin[position_ids].unsqueeze(2)  # [bs, seq_len, 1, dim]
    q_embed = (q * cos) + (_rotate_half_train(q) * sin)
    k_embed = (k * cos) + (_rotate_half_train(k) * sin)
    return q_embed, k_embed


def _forward_train(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """Input shape: Batch x Time x Channel

    attention_mask: [bsz, q_len]
    """
    if output_attentions:
        warnings.warn(
            "Output attentions is not supported for patched `LlamaAttention`, returning `None` instead."
        )

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states).view(
        bsz, q_len, self.num_heads, self.head_dim
    )
    key_states = self.k_proj(hidden_states).view(
        bsz, q_len, self.num_heads, self.head_dim
    )
    value_states = self.v_proj(hidden_states).view(
        bsz, q_len, self.num_heads, self.head_dim
    )
    # [bsz, q_len, nh, hd] -> [bsz, nh, q_len, hd]

    cos, sin = self.rotary_emb(value_states, seq_len=q_len)
    query_states, key_states = _apply_rotary_pos_emb_train(
        query_states, key_states, cos, sin, position_ids
    )

    # Flash attention codes from
    # https://github.com/HazyResearch/flash-attention/blob/main/flash_attn/flash_attention.py
    # only work for training, not using key padding
    # q: (batch_size, seqlen, nheads, headdim)
    # k: (batch_size, seqlen, nheads_k, headdim)
    # v: (batch_size, seqlen, nheads_k, headdim)
    # out: (batch_size, seqlen, nheads, headdim)
    output = flash_attn_func(
        query_states, key_states, value_states, 0.0, softmax_scale=None, causal=True
    )

    return self.o_proj(rearrange(output, "b s h d -> b s (h d)")), None, None


# Disable the transformation of the attention mask in LlamaModel as the flash attention
# requires the attention mask to be the same as the key_padding_mask
def _prepare_decoder_attention_mask_train(
    self, attention_mask, input_shape, inputs_embeds, past_key_values_length
):
    # [bsz, seq_len]
    return attention_mask


def replace_old_func_with_new_func_only_for_train(old_func, new_func):
    def combined_func(
        self,
        *args,
        **kwargs,
    ):
        if self.training:
            return new_func(self, *args, **kwargs)
        else:
            return old_func(self, *args, **kwargs)

    return combined_func


def replace_llama_attn_with_flash_attn():
    cuda_major, cuda_minor = torch.cuda.get_device_capability()
    if cuda_major < 8:
        warnings.warn(
            "Flash attention is only supported on A100 or H100 GPU during training due to head dim > 64 backward."
            "ref: https://github.com/HazyResearch/flash-attention/issues/190#issuecomment-1523359593"
        )
    # for original llama in transformers
    import transformers.models.llama.modeling_llama as llama
    llama.LlamaModel._prepare_decoder_attention_mask = (
        replace_old_func_with_new_func_only_for_train(
            llama.LlamaModel._prepare_decoder_attention_mask,
            _prepare_decoder_attention_mask_train,
        )
    )
    llama.LlamaAttention.forward = replace_old_func_with_new_func_only_for_train(
        llama.LlamaAttention.forward, _forward_train
    )
    # for our text decoder
    import mm_interleaved.models.decoders.decoder_text as decoder_text
    decoder_text.TextDecoder._prepare_decoder_attention_mask = (
        replace_old_func_with_new_func_only_for_train(
            decoder_text.TextDecoder._prepare_decoder_attention_mask,
            _prepare_decoder_attention_mask_train,
        )
    )
    import mm_interleaved.models.decoders.modeling_llama_mmfs as llama
    llama.LlamaModel._prepare_decoder_attention_mask = (
        replace_old_func_with_new_func_only_for_train(
            llama.LlamaModel._prepare_decoder_attention_mask,
            _prepare_decoder_attention_mask_train,
        )
    )
    llama.LlamaAttention.forward = replace_old_func_with_new_func_only_for_train(
        llama.LlamaAttention.forward, _forward_train
    )
