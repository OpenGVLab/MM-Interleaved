from typing import Optional, Tuple

import torch
import torch.utils.checkpoint
from torch import nn

# the xformers lib allows less memory, faster training and inference
try:
    import xformers
    import xformers.ops

    XFORMERS_IS_AVAILBLE = True
    # print('xformers enabled')
except:
    XFORMERS_IS_AVAILBLE = False
    print("xformers disabled")

from transformers import CLIPVisionModel, CLIPTextModel


class CLIPXAttention(nn.Module):
    """Memory Efficient Attention layer for CLIP, support full & causal attn mask"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).contiguous()
        # return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        bsz, tgt_len, embed_dim = hidden_states.size()

        query_states = self._shape(self.q_proj(hidden_states), tgt_len, bsz)
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        # use xformers here
        assert (self.dropout == 0.0) and (attention_mask is None)
        attention_mask = (
            xformers.ops.LowerTriangularMask()
            if causal_attention_mask is not None
            else None
        )
        # q, k, v = query_states.transpose(1, 2), key_states.transpose(1, 2), value_states.transpose(1, 2)
        q, k, v = query_states, key_states, value_states
        attn_output = xformers.ops.memory_efficient_attention(
            q, k, v, attn_bias=attention_mask
        )
        attn_weights_reshaped = None

        # # get query proj
        # query_states = self.q_proj(hidden_states) * self.scale
        # key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        # value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        # proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        # query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        # key_states = key_states.view(*proj_shape)
        # value_states = value_states.view(*proj_shape)

        # src_len = key_states.size(1)
        # attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        # if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
        #     raise ValueError(
        #         f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
        #         f" {attn_weights.size()}"
        #     )

        # # apply the causal_attention_mask first
        # if causal_attention_mask is not None:
        #     if causal_attention_mask.size() != (bsz, 1, tgt_len, src_len):
        #         raise ValueError(
        #             f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is"
        #             f" {causal_attention_mask.size()}"
        #         )
        #     attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + causal_attention_mask
        #     attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        # if attention_mask is not None:
        #     if attention_mask.size() != (bsz, 1, tgt_len, src_len):
        #         raise ValueError(
        #             f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
        #         )
        #     attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        #     attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        # attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # if output_attentions:
        #     # this operation is a bit akward, but it's required to
        #     # make sure that attn_weights keeps its gradient.
        #     # In order to do so, attn_weights have to reshaped
        #     # twice and have to be reused in the following
        #     attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        #     attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        # else:
        #     attn_weights_reshaped = None

        # attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        # attn_output = torch.bmm(attn_probs, value_states)

        # if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
        #     raise ValueError(
        #         f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
        #         f" {attn_output.size()}"
        #     )

        # attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        # attn_output = attn_output.transpose(1, 2)

        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped


def convert_clip_visual_attn(model: CLIPVisionModel):
    for layer in model.vision_model.encoder.layers:
        attn_o = layer.self_attn
        attn_x = CLIPXAttention(config=attn_o.config)
        for module_name in ["q_proj", "v_proj", "k_proj", "out_proj"]:
            module_o: nn.Linear = getattr(attn_o, module_name)
            module_x: nn.Linear = getattr(attn_x, module_name)
            module_x.weight.data.copy_(module_o.weight.data)
            module_x.bias.data.copy_(module_o.bias.data)
        layer.self_attn = attn_x
        del attn_o
    print("convert clip visual self_attn to memory efficient mode successfully")


def convert_clip_text_attn(model: CLIPTextModel):
    for layer in model.text_model.encoder.layers:
        attn_o = layer.self_attn
        attn_x = CLIPXAttention(config=attn_o.config)
        for module_name in ["q_proj", "v_proj", "k_proj", "out_proj"]:
            module_o: nn.Linear = getattr(attn_o, module_name)
            module_x: nn.Linear = getattr(attn_x, module_name)
            module_x.weight.data.copy_(module_o.weight.data)
            module_x.bias.data.copy_(module_o.bias.data)
        layer.self_attn = attn_x
        del attn_o
    print("convert clip text self_attn to memory efficient mode successfully")
