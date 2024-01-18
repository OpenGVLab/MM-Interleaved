# ------------------------------------------------------------------------------------------------
# Multi-Image Multi-Scale Feature Synchronizer
# Modifed from Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

from __future__ import absolute_import, division, print_function

import os
import math
import warnings

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import constant_, xavier_uniform_

from einops import rearrange

from ..functions import MSDeformAttnFunction


class MMFS(nn.Module):
    def __init__(
        self,
        layer_idx=0,
        d_model=256,
        d_query=-1,
        d_value=256,
        d_out=-1,
        n_levels=4,
        n_heads=8,
        n_points=8,
        ratio=1.0,
        offset_init_magnitude=3,
        spatial_shapes=[16],
        base_spatial_shape=16,
        max_num_image_per_seq=50,
    ):
        """Multi-Image Multi-Scale Feature Synchronizer.

        :param d_model      hidden dimension
        :param d_query      query input dimension
        :param d_value      value input dimension
        :param d_out        output dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                "d_model must be divisible by n_heads, "
                "but got {} and {}".format(d_model, n_heads)
            )
        # _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2
        # which is more efficient in our CUDA implementation
        if d_query < 0:
            d_query = d_model
        if d_out < 0:
            d_out = d_model

        self.layer_idx = layer_idx
        self.im2col_step = 1

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.ratio = ratio
        self.offset_init_magnitude = offset_init_magnitude

        self.max_num_image_per_seq = max_num_image_per_seq

        assert len(spatial_shapes) == self.n_levels
        scale_ratios = [s / base_spatial_shape for s in spatial_shapes]
        self.register_buffer(
            "scale_ratios", torch.tensor(scale_ratios), persistent=False
        )
        print(f"MMFS {spatial_shapes=} {base_spatial_shape=} {self.scale_ratios=}")
        self.sampling_offsets = nn.Linear(d_query, n_heads * n_points * 2)

        self.ignore_token = nn.Parameter(
            torch.zeros(1, 1, 1, int(d_model * ratio)), requires_grad=False
        )
        self.dynamic_offset_mask = nn.Linear(d_query, d_query)

        self.attention_weights = nn.Linear(d_query, n_heads * n_levels * (n_points + 1))
        self.value_proj = nn.Linear(d_value, int(d_model * ratio))
        self.output_proj = nn.Linear(int(d_model * ratio), d_out)

        self.query_relpos = nn.Embedding(self.max_num_image_per_seq, d_query)

        self._reset_parameters()
        for p in self.sampling_offsets.parameters():
            p.requires_grad = True

    def _reset_parameters(self):
        grid_init = torch.zeros(self.n_heads, 1, self.n_points, 2)
        grid_init = nn.init.uniform_(
            grid_init, -self.offset_init_magnitude, self.offset_init_magnitude
        )

        constant_(self.sampling_offsets.weight.data, 0.0)
        with torch.no_grad():
            self.sampling_offsets.bias.data = grid_init.view(-1)

        constant_(self.attention_weights.bias.data, 0.0)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.0)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.0)
        constant_(self.dynamic_offset_mask.bias.data, 0.0)
        nn.init.trunc_normal_(self.query_relpos.weight, std=0.02)

    def forward(
        self,
        query,
        reference_points,
        input_flatten,
        input_spatial_shapes,
        input_level_start_index,
        input_padding_mask=None,
        attention_mask: torch.LongTensor = None,
    ):
        """
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        :param input_flatten               (N, n_images, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param input_spatial_shapes        (n_images * n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_images * n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements
        :param attention_mask              (N, Length_{query}, n_images) or (N, n_images) True for valid images, False for padding images

        :return output                     (N, Length_{query}, C)
        """

        N, Len_q, _ = query.shape
        N, n_images, hw, _ = input_flatten.shape

        Len_in = n_images * hw
        assert (
            input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]
        ).sum() == Len_in, f"{input_spatial_shapes},{(input_spatial_shapes[:, 0]*input_spatial_shapes[:, 1]).sum()},{Len_in}"
        assert attention_mask.ndim in [2, 3]
        assert attention_mask.shape[-1] == n_images
        nlevels = n_images * self.n_levels

        attention_mask_bool = attention_mask.long()
        image_num_tot = attention_mask_bool.sum(dim=-1, keepdim=True)
        image_num_prev = attention_mask_bool.cumsum(dim=-1)
        image_relpos = (image_num_tot + 1 - image_num_prev) * attention_mask_bool
        if attention_mask.ndim == 2:
            image_relpos = image_relpos.unsqueeze(-1).repeat(1, 1, Len_q)
        else:
            if attention_mask.shape[1] != Len_q:
                image_relpos = image_relpos[:, -1:, ...]
            image_relpos = rearrange(image_relpos, "b q n -> b n q")

        value = self.value_proj(input_flatten)

        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))

        value = value.reshape(
            N, Len_in, self.n_heads, int(self.ratio * self.d_model) // self.n_heads
        ).contiguous()

        query = query.unsqueeze(1).repeat(1, n_images, 1, 1)  # N x n x Len_q x d_model
        query = self.dynamic_offset_mask(query)

        assert image_relpos.max() < self.max_num_image_per_seq
        query_relpos = self.query_relpos(image_relpos)  # [N, n_images, q, d]
        query = query + query_relpos

        sampling_offsets = self.sampling_offsets(query).view(
            N, n_images, Len_q, self.n_heads, 1, self.n_points, 2
        )
        sampling_offsets = rearrange(
            sampling_offsets, "b n q h l p t -> b q h (n l) p t"
        )

        attention_weights = self.attention_weights(query).view(
            N, n_images, Len_q, self.n_heads, self.n_levels, self.n_points + 1
        )
        attention_weights = rearrange(attention_weights, "b n q h l p -> b q h (n l) p")

        sampling_offsets = sampling_offsets[:, :, :, :n_images, None].contiguous()
        scale_ratios = rearrange(self.scale_ratios, "l -> 1 1 1 1 l 1 1")
        sampling_offsets = sampling_offsets * scale_ratios
        sampling_offsets = rearrange(
            sampling_offsets, "b q h n l p t -> b q h (n l) p t"
        )

        attention_weights = attention_weights[:, :, :, :nlevels].contiguous()

        # add attention mask
        attention_mask = (1.0 - attention_mask.to(attention_weights.dtype)) * -10000.0
        if attention_mask.ndim == 2:
            attention_mask = rearrange(attention_mask, "b n -> b 1 1 n 1")
            attention_mask = attention_mask.repeat_interleave(self.n_levels, dim=3)
            attention_weights = attention_weights + attention_mask
            # check all ignore mask
            ignore_idx = torch.nonzero(
                ((attention_mask > -1000.0).sum(dim=-2) == 0), as_tuple=True
            )
            attention_weights[ignore_idx[0], :, :, :, -1] = 1
        else:
            if attention_mask.shape[1] != Len_q:
                attention_mask = attention_mask[:, -1:, ...]
            attention_mask = rearrange(attention_mask, "b q n -> b q 1 n 1")
            attention_mask = attention_mask.repeat_interleave(self.n_levels, dim=3)
            attention_weights = attention_weights + attention_mask
            # check all ignore mask
            ignore_idx = torch.nonzero(
                ((attention_mask > -1000.0).sum(dim=-2) == 0), as_tuple=True
            )
            attention_weights[ignore_idx[0], ignore_idx[1], :, :, -1] = 1

        attention_weights[..., -1] = -math.log(nlevels)
        attention_weights = attention_weights.view(
            N, Len_q, self.n_heads, nlevels * (self.n_points + 1)
        )
        attention_weights = F.softmax(attention_weights, -1).view(
            N, Len_q, self.n_heads, nlevels, self.n_points + 1
        )

        attention_weights_ignore = attention_weights[..., -1:].contiguous()
        attention_weights = attention_weights[..., :-1].contiguous()

        ignore_token = self.ignore_token.repeat(N, Len_q, nlevels, 1)
        ignore_token = rearrange(
            ignore_token, "b l n (h d) -> b l h n d", h=self.n_heads
        )
        ignore_token = (ignore_token * attention_weights_ignore).sum(dim=-2)
        ignore_token = rearrange(ignore_token, "b l h d -> b l (h d)")

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1
            )
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets
                / self.n_points
                * reference_points[:, :, None, :, None, 2:]
                * 0.5
            )
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, but get {} instead.".format(
                    reference_points.shape[-1]
                )
            )
        sampling_locations = sampling_locations.to(value.dtype)
        output = MSDeformAttnFunction.apply(
            value,
            input_spatial_shapes,
            input_level_start_index,
            sampling_locations,
            attention_weights,
            self.im2col_step,
        )
        output = output + ignore_token
        output = self.output_proj(output)
        return output
