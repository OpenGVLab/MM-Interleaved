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

def test_op(value, shapes, level_start_index, sampling_locations, attention_weights, im2col_step):
    return MSDeformAttnFunction.apply(
            value, shapes, level_start_index, sampling_locations, attention_weights, im2col_step)

def backward_op(grad, value, shapes, level_start_index, sampling_locations, attention_weights, im2col_step):
    import MultiScaleDeformableAttention as MSDA
    return MSDA.ms_deform_attn_backward(value, shapes, level_start_index, sampling_locations, attention_weights, grad, im2col_step)

def to_dtype(inputs, dtype):
    new_inputs = list(copy.deepcopy(inputs))
    for i, x in enumerate(new_inputs):
        if isinstance(x, torch.Tensor) and torch.is_floating_point(x):
            new_inputs[i] = x.to(dtype).detach()
            new_inputs[i].requires_grad = x.requires_grad
    return tuple(new_inputs)


def check_equal_1(func, dim=4):
    """directly call MSDA.backward"""
    inputs_fp16 = generate_inputs(torch.float16, dim)
    inputs_fp16[0].requires_grad = True
    inputs_fp16[3].requires_grad = True
    inputs_fp16[4].requires_grad = True

    inputs_fp64 = to_dtype(inputs_fp16, torch.float64)
    outs_fp64 = func(*inputs_fp64)
    outs_fp64.retain_grad()
    (outs_fp64.sum()).backward()
    
    grads_fp64 = [x.grad.detach().clone() for x in inputs_fp64 if hasattr(x, 'grad') and x.grad is not None]

    with torch.no_grad():
        grads_fp16_by_fp64 = backward_op(outs_fp64.grad.half().detach().clone(), *inputs_fp16)
    
    outs_fp16 = func(*inputs_fp16)
    outs_fp16.retain_grad()
    (outs_fp16.sum()).backward()
    grads_fp16 = [x.grad.detach().clone() for x in inputs_fp16 if hasattr(x, 'grad') and x.grad is not None]
    
    result = OrderedDict()
    # backward error
    for i in range(len(grads_fp64)):
        rel_mask = torch.abs(grads_fp64[i]) > 1e-6
        abs_err, rel_err, mean_rel_err = calc_err(
                grads_fp16_by_fp64[i].double(),
                grads_fp64[i].double(),
                rel_mask=rel_mask)
        name = f'input #{i},'
        print(f"{name: <10} max_abs_err: {abs_err:.8f}, max_rel_err: {rel_err:.8f}, mean_rel_err: {mean_rel_err:.8f}")
        result[f'inp{i}_max_ae'] = abs_err
        result[f'inp{i}_max_re'] = rel_err
        result[f'inp{i}_avg_re'] = mean_rel_err

    return result


def check_equal_2(func, dim=4):
    """replace fp16 forward result by fp64"""
    inputs_fp16 = generate_inputs(torch.float16, dim)
    inputs_fp16[0].requires_grad = True
    inputs_fp16[3].requires_grad = True
    inputs_fp16[4].requires_grad = True

    inputs_fp64 = to_dtype(inputs_fp16, torch.float64)
    outs_fp64 = func(*inputs_fp64)
    outs_fp64 = torch.sum(outs_fp64)
    outs_fp64.backward()
    grads_fp64 = [x.grad.detach().clone() for x in inputs_fp64 if hasattr(x, 'grad') and x.grad is not None]


    outs_fp16 = func(*inputs_fp16)
    outs_fp16 = torch.sum(outs_fp16)
    with torch.no_grad():
        outs_fp16.data.copy_(outs_fp64.data.to(torch.float16))
    outs_fp16.backward()
    outs_fp16 = [x.grad.detach().clone() for x in inputs_fp16 if hasattr(x, 'grad') and x.grad is not None]
    
    result = OrderedDict()
    # backward error
    for i in range(len(grads_fp64)):
        rel_mask = torch.abs(grads_fp64[i]) > 1e-6
        abs_err, rel_err, mean_rel_err = calc_err(
                outs_fp16[i].double(),
                grads_fp64[i].double(),
                rel_mask=rel_mask)
        name = f'input #{i},'
        print(f"{name: <10} max_abs_err: {abs_err:.8f}, max_rel_err: {rel_err:.8f}, mean_rel_err: {mean_rel_err:.8f}")
        result[f'inp{i}_max_ae'] = abs_err
        result[f'inp{i}_max_re'] = rel_err
        result[f'inp{i}_avg_re'] = mean_rel_err

    return result


def check_equal_3(func, dim=4):
    """
    1. fp32 forward and backward
    2. convert feature and grad to fp16
    3. calc_error
    """
    inputs_fp16 = generate_inputs(torch.float16, dim)
    inputs_fp16[0].requires_grad = True
    inputs_fp16[3].requires_grad = True
    inputs_fp16[4].requires_grad = True

    inputs_fp32 = to_dtype(inputs_fp16, torch.float32)
    outs_fp32 = func(*inputs_fp32)
    sum_outs_fp32 = torch.sum(outs_fp32)
    sum_outs_fp32.backward()
    grads_fp32 = [x.grad.detach().clone() for x in inputs_fp32 if hasattr(x, 'grad') and x.grad is not None]


    outs_fp16 = outs_fp32.half()
    grads_fp16 = [x.half() for x in grads_fp32]

    results = OrderedDict()
    tensor_fp16 = OrderedDict(
        {'query_grad': grads_fp16[0], 'offset_grad': grads_fp16[1], 'attn_grad': grads_fp16[2], 'out': outs_fp16})
    tensor_fp32 = OrderedDict(
        {'query_grad': grads_fp32[0], 'offset_grad': grads_fp32[1], 'attn_grad': grads_fp32[2], 'out': outs_fp32})
    compare_tensor_dict(tensor_fp16, tensor_fp32, results, skip_t2_zero=True)

    return results

def compare_tensor_dict(tensors1, tensors2, result, skip_t2_zero=False):
    for name in tensors1.keys():
        t1 = tensors1[name]
        t2 = tensors2[name]
        if skip_t2_zero:
            rel_mask = torch.abs(t2) > 1e-6
        else:
            rel_mask = torch.ones(t2) > 1
        abs_err, rel_err, mean_rel_err = calc_err(
                t1.double(),
                t2.double(),
                rel_mask=rel_mask)
        result[f'{name}_max_ae'] = abs_err
        result[f'{name}_max_re'] = rel_err
        result[f'{name}_avg_re'] = mean_rel_err

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
    fp16_error_list = pd.DataFrame(None)
    for i in range(20):
        fp16_error = check_equal_3(test_op, dim=64)
        if i == 0:
            for k in fp16_error.keys():
                fp16_error_list[k] = []
        fp16_error = pd.DataFrame(fp16_error, index=[0])
        fp16_error_list = pd.concat([fp16_error_list, fp16_error], ignore_index=True)

    with pd.option_context(
            'display.max_rows', None, 'display.max_columns', None,
            'display.max_colwidth', None, 'display.width', None,
            'display.precision', 2, 
        ):
        print(">>> fp16")
        print(fp16_error_list)

