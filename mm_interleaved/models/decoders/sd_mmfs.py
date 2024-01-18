from functools import partial
from typing import List

from einops import rearrange

import torch
from torch import nn
import torch.utils.checkpoint as cp


from ..utils.ops.modules import MMFS
from ..utils.pos_embed import get_abs_pos, get_2d_sincos_pos_embed


def get_reference_points(spatial_shapes, device):
    reference_points_list = []
    for lvl, (H_, W_) in enumerate(spatial_shapes):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
            torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device),
        )
        ref_y = ref_y.reshape(-1)[None] / H_
        ref_x = ref_x.reshape(-1)[None] / W_
        ref = torch.stack((ref_x, ref_y), -1)
        reference_points_list.append(ref)
    reference_points = torch.cat(reference_points_list, 1)
    reference_points = reference_points[:, :, None]
    return reference_points


def deform_inputs(sample, spatial_shapes=[(8, 8)]):
    bs, c, h, w = sample.shape
    spatial_shapes = torch.as_tensor(
        spatial_shapes, dtype=torch.long, device=sample.device
    )
    level_start_index = torch.cat(
        (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
    )
    reference_points = get_reference_points([(h, w)], sample.device)

    return reference_points, spatial_shapes, level_start_index


class MMFSBlock(nn.Module):
    def __init__(
        self,
        attn_dim=1024,
        query_dim=320,
        feat_dim=1024,
        num_heads=16,
        n_points=8,
        n_levels=1,
        deform_ratio=1.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        gradient_checkpointing=False,
        grid_size=64,
        offset_init_magnitude=1,
        max_num_image_per_seq=10,
        spatial_shapes=[16],
        base_spatial_shape=8,
        layer_idx=0,
    ):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing

        self.query_norm = norm_layer(query_dim)
        self.feat_norm = norm_layer(feat_dim)
        self.mmfs = MMFS(
            d_model=attn_dim,
            d_query=query_dim,
            d_value=feat_dim,
            d_out=query_dim,
            n_levels=n_levels,
            n_heads=num_heads,
            n_points=n_points,
            ratio=deform_ratio,
            offset_init_magnitude=offset_init_magnitude,
            spatial_shapes=spatial_shapes,
            base_spatial_shape=base_spatial_shape,
            max_num_image_per_seq=max_num_image_per_seq,
            layer_idx=layer_idx,
        )

        pos_embed = get_2d_sincos_pos_embed(query_dim, grid_size, cls_token=False)
        pos_embed = torch.from_numpy(pos_embed).float()
        self.pos_embed = nn.Parameter(pos_embed).requires_grad_(False)

        conv = nn.Conv2d(
            query_dim,
            query_dim,
            kernel_size=1,
            stride=1,
        )
        self.conv = zero_module(conv)

    def _reset_parameters(self):
        self.mmfs._reset_parameters()

    def forward(self, sample, ms_feat, ms_feat_mask, spatial_shapes):
        """
        sample: [B, C_q, H, W]
        ms_feat: [B, N, \sum_{l}(H_l * W_l), C_v]
        ms_mask: [B, N]
        spatial_shapes: shapes of each value feature map within one single image
        """

        B, C, H, W = sample.shape
        n_images = ms_feat_mask.shape[-1]

        def _inner_forward(sample, ms_feat, ms_feat_mask, spatial_shapes):
            reference_points, spatial_shapes, level_start_index = deform_inputs(
                sample, spatial_shapes=spatial_shapes * n_images
            )

            query = rearrange(sample, "b c h w -> b (h w) c")
            query = self.query_norm(query)

            pos_embed = get_abs_pos(self.pos_embed, H * W)
            query = query + pos_embed

            ms_feat = self.feat_norm(ms_feat)

            attn_output = self.mmfs(
                query,
                reference_points,
                ms_feat,
                spatial_shapes,
                level_start_index,
                input_padding_mask=None,
                attention_mask=ms_feat_mask,
            )

            deform_sample = rearrange(attn_output, "b (h w) c -> b c h w", h=H)
            deform_sample = self.conv(deform_sample)

            return deform_sample

        if self.gradient_checkpointing and self.training:
            sample = cp.checkpoint(
                _inner_forward, sample, ms_feat, ms_feat_mask, spatial_shapes
            )
        else:
            sample = _inner_forward(sample, ms_feat, ms_feat_mask, spatial_shapes)

        return sample


def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class MMFSNet(nn.Module):
    def __init__(
        self,
        input_channel,
        block_out_channels,
        layers_per_block,
        downsample_factor=1,
        n_levels=4,
        n_points=8,
        gradient_checkpointing=True,
        spatial_shapes=[64, 32, 16, 8],
    ) -> None:
        super().__init__()
        self.downsample_factor = downsample_factor

        sd_spatial_shapes = [s // downsample_factor for s in spatial_shapes]

        def _init_block(query_dim, spatial_shape_idx=0, layer_idx=0):
            return MMFSBlock(
                query_dim=query_dim,
                feat_dim=input_channel,
                n_points=n_points,
                n_levels=n_levels,
                gradient_checkpointing=gradient_checkpointing,
                grid_size=64 // downsample_factor,
                spatial_shapes=spatial_shapes,
                base_spatial_shape=sd_spatial_shapes[spatial_shape_idx],
                layer_idx=layer_idx,
            )

        # down
        self.mmfs_down_blocks = nn.ModuleList([])
        output_channel = block_out_channels[0]
        mmfs_block = _init_block(
            output_channel,
            spatial_shape_idx=len(self.mmfs_down_blocks) // 3,
            layer_idx=len(self.mmfs_down_blocks),
        )
        self.mmfs_down_blocks.append(mmfs_block)

        for i, output_channel in enumerate(block_out_channels):
            is_final_block = i == len(block_out_channels) - 1

            for _ in range(layers_per_block):
                mmfs_block = _init_block(
                    output_channel,
                    spatial_shape_idx=len(self.mmfs_down_blocks) // 3,
                    layer_idx=len(self.mmfs_down_blocks),
                )
                self.mmfs_down_blocks.append(mmfs_block)

            if not is_final_block:
                mmfs_block = _init_block(
                    output_channel,
                    spatial_shape_idx=len(self.mmfs_down_blocks) // 3,
                    layer_idx=len(self.mmfs_down_blocks),
                )
                self.mmfs_down_blocks.append(mmfs_block)

        # mid
        mid_block_channel = block_out_channels[-1]
        mmfs_block = _init_block(
            mid_block_channel,
            spatial_shape_idx=-1,
            layer_idx=len(self.mmfs_down_blocks),
        )
        self.mmfs_mid_block = mmfs_block

        self._reset_parameters()

    def _reset_parameters(self):
        for block in self.mmfs_down_blocks:
            block._reset_parameters()

        self.mmfs_mid_block._reset_parameters()

    def forward(
        self,
        sample: torch.Tensor,
        down_block_res_samples: List[torch.Tensor],
        mmfs_features: List[torch.Tensor],
        mmfs_mask: torch.Tensor,
    ):
        mmfs_down_blocks = self.mmfs_down_blocks
        assert len(down_block_res_samples) == len(mmfs_down_blocks)

        spatial_shapes = [(feat.shape[-2], feat.shape[-1]) for feat in mmfs_features]
        mmfs_features = [
            rearrange(feat, "b n c h w -> b n (h w) c") for feat in mmfs_features
        ]
        mmfs_features = torch.cat(mmfs_features, dim=2)

        new_down_block_res_samples = ()
        for sample_idx, (down_block_res_sample, mmfs_down_block) in enumerate(
            zip(down_block_res_samples, mmfs_down_blocks)
        ):
            down_block_additional_residual = mmfs_down_block(
                down_block_res_sample,
                mmfs_features,
                mmfs_mask,
                spatial_shapes,
            )
            down_block_res_sample = (
                down_block_res_sample + down_block_additional_residual
            )
            new_down_block_res_samples = new_down_block_res_samples + (
                down_block_res_sample,
            )
        down_block_res_samples = new_down_block_res_samples

        mid_block_additional_residual = self.mmfs_mid_block(
            sample,
            mmfs_features,
            mmfs_mask,
            spatial_shapes,
        )
        sample = sample + mid_block_additional_residual

        return sample, down_block_res_samples
