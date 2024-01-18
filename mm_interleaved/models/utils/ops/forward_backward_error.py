# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time
import torch
import torch.nn as nn
from torch.autograd import gradcheck
import torch.nn.functional as F

import copy
import pandas as pd
from collections import OrderedDict
import matplotlib.pyplot as plt

from functions.ms_deform_attn_func import MSDeformAttnFunction

torch.manual_seed(0)

def generate_inputs(dtype, dim=4):
    N, M, D = 1, 2, dim
    Lq, L, P = 2, 2, 2
    shapes = torch.as_tensor([(6, 4), (3, 2)], dtype=torch.long)
    level_start_index = torch.cat((shapes.new_zeros((1, )), shapes.prod(1).cumsum(0)[:-1]))
    S = sum([(H*W).item() for H, W in shapes])

    value = torch.rand(N, S, M, D)
    sampling_locations = torch.rand(N, Lq, M, L, P, 2)
    attention_weights = torch.rand(N, Lq, M, L, P).cuda() + 1e-5
    attention_weights /= attention_weights.sum(-1, keepdim=True).sum(-2, keepdim=True)
    im2col_step = 2
    return (
        value.cuda().to(dtype),
        shapes.cuda(),
        level_start_index.cuda(),
        sampling_locations.cuda().to(dtype),
        attention_weights.cuda().to(dtype),
        im2col_step
    )

def ms_deform_attn_core_pytorch(
        value, value_spatial_shapes, sampling_locations, attention_weights):
    # for debug and test only,
    # need to use cuda version instead    
    N_, S_, M_, D_ = value.shape
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_*M_, D_, H_, W_)
        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        # N_*M_, D_, Lq_, P_
        sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_,
                                          mode='bilinear', padding_mode='zeros', align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_, M_, 1, Lq_, L_*P_)
    attention_weights = attention_weights.transpose(1, 2).reshape(N_*M_, 1, Lq_, L_*P_)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(N_, M_*D_, Lq_)
    return output.transpose(1, 2).contiguous()


def pytorch_op(value, shapes, level_start_index, sampling_locations, attention_weights, im2col_step):
    return ms_deform_attn_core_pytorch(value, shapes, sampling_locations, attention_weights)

def test_op(value, shapes, level_start_index, sampling_locations, attention_weights, im2col_step):
    return MSDeformAttnFunction.apply(
            value, shapes, level_start_index, sampling_locations, attention_weights, im2col_step)

def to_dtype(inputs, dtype):
    new_inputs = list(copy.deepcopy(inputs))
    for i, x in enumerate(new_inputs):
        if isinstance(x, torch.Tensor) and torch.is_floating_point(x):
            new_inputs[i] = x.to(dtype).detach()
            new_inputs[i].requires_grad = x.requires_grad
    return tuple(new_inputs)


def check_equal(func, dim=4, loss_scaling=1):
    inputs_fp16 = generate_inputs(torch.float16, dim)
    inputs_fp16[0].requires_grad = True
    inputs_fp16[3].requires_grad = True
    inputs_fp16[4].requires_grad = True

    outs_fp16 = func(*inputs_fp16)
    (outs_fp16.sum() * loss_scaling).backward()

    inputs_fp32 = to_dtype(inputs_fp16, torch.float32)
    outs_fp32 = func(*inputs_fp32)
    outs_fp32.sum().backward()

    inputs_fp64 = to_dtype(inputs_fp16, torch.float64)
    outs_fp64 = func(*inputs_fp64)
    outs_fp64.sum().backward()

    def _forward_backward_error(inputs, outputs):
        result = OrderedDict()
        # backward error
        for i in range(len(inputs)):
            if (not isinstance(inputs[i], torch.Tensor)) or (not inputs[i].requires_grad):
                continue
            rel_mask = torch.abs(inputs_fp64[i].grad) > 1e-6
            abs_err, rel_err, mean_rel_err = calc_err(
                    inputs[i].grad.double() / loss_scaling,
                    inputs_fp64[i].grad,
                    rel_mask=rel_mask)
            name = f'input #{i},'
            print(f"{name: <10} max_abs_err: {abs_err:.8f}, max_rel_err: {rel_err:.8f}, mean_rel_err: {mean_rel_err:.8f}")
            result[f'inp{i}_max_ae'] = abs_err
            result[f'inp{i}_max_re'] = rel_err
            result[f'inp{i}_avg_re'] = mean_rel_err

        # forward error
        rel_mask = torch.abs(outs_fp64) > 1e-6
        abs_err, rel_err, mean_rel_err = calc_err(
                outputs.double(),
                outs_fp64,
                rel_mask=rel_mask)
        name = f'output'
        print(f"{name: <10} max_abs_err: {abs_err:.8f}, max_rel_err: {rel_err:.8f}, mean_rel_err: {mean_rel_err:.8f}")
        result[f'out_max_ae'] = abs_err
        result[f'out_max_re'] = rel_err
        result[f'out_avg_re'] = mean_rel_err
        return result

    fp16_error = _forward_backward_error(inputs_fp16, outs_fp16)
    fp32_error = _forward_backward_error(inputs_fp32, outs_fp32)

    # tensor_info
    tensors = OrderedDict({
        'inp0': inputs_fp64[0],
        'inp3': inputs_fp64[3],
        'inp4': inputs_fp64[4],
        'out': outs_fp64
    })
    tensor_info = OrderedDict()
    for name, tensor in tensors.items():
        for k, v in calc_statistic(tensor).items():
            tensor_info[f'{name}_value_{k}'] = v
        if tensor.grad is not None:
            for k, v in calc_statistic(tensor.grad).items():
                tensor_info[f'{name}_grad_{k}'] = v

    return fp16_error, fp32_error, tensor_info

def calc_statistic(x):
    return {
        'avg': torch.mean(x).item(),
        'std': torch.std(x, unbiased=False).item(),
        'min': torch.min(x).item(),
        'max': torch.max(x).item()
    }

def calc_err(x1, x2, rel_mask = None):
    err = (x1 - x2).abs()
    max_abs_err = err.max()
    max_rel_err = (err[rel_mask] / x2.abs()[rel_mask]).max()
    mean_rel_err = (err[rel_mask] / x2.abs()[rel_mask]).mean()
    return max_abs_err.item(), max_rel_err.item(), mean_rel_err.item()

def plot_hist(x, out_file='hist.png'):
    plt.hist(x.abs().cpu().numpy().reshape([-1]), bins=200)
    plt.savefig(out_file)

if __name__ == '__main__':
    print(">>> fp16")
    fp16_error_list = pd.DataFrame(None)
    fp32_error_list = pd.DataFrame(None)
    tensor_info_list = pd.DataFrame(None)
    for i in range(20):
        fp16_error, fp32_error, tensor_info = check_equal(test_op, dim=64)
        if i == 0:
            for k in fp16_error.keys():
                fp16_error_list[k] = []
            for k in fp32_error.keys():
                fp32_error_list[k] = []
            for k in tensor_info.keys():
                tensor_info_list[k] = []
        fp16_error = pd.DataFrame(fp16_error, index=[0])
        fp32_error = pd.DataFrame(fp32_error, index=[0])
        tensor_info = pd.DataFrame(tensor_info, index=[0])
        fp16_error_list = pd.concat([fp16_error_list, fp16_error], ignore_index=True)
        fp32_error_list = pd.concat([fp32_error_list, fp32_error], ignore_index=True)
        tensor_info_list = pd.concat([tensor_info_list, tensor_info], ignore_index=True)

    with pd.option_context(
            'display.max_rows', None, 'display.max_columns', None,
            'display.max_colwidth', None, 'display.width', None,
            'display.precision', 2, 
        ):
        print(">>> fp16")
        print(fp16_error_list)
        print(">>> fp32")
        print(fp32_error_list)
        print(">>> tensor_info")
        print(tensor_info_list)

