
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


def compare_tensor_dict(tensors1, tensors2, skip_t2_zero=False):
    result = OrderedDict()
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
    return result


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

def print_table(results):
    df = pd.DataFrame(None)
    for i, ret in enumerate(results):
        if i == 0:
            for k in ret.keys():
                df[k] = []
        new_row = pd.DataFrame(ret, index=[0])
        df = pd.concat([df, new_row], ignore_index=True)

    with pd.option_context(
            'display.max_rows', None, 'display.max_columns', None,
            'display.max_colwidth', None, 'display.width', None,
            'display.precision', 2, 
        ):
        print(df)


def internally_fp32():
    data_file = 'data/fp64_data.pkl'
    test_data_list = torch.load(data_file)

    results = []
    for test_data in test_data_list:
        inputs_fp64 = test_data['inputs']
        grads_fp64 = test_data['grads']
        grads_fp64 = [x for x in grads_fp64 if x is not None]
        outs_fp64 = test_data['outs']

        inputs_fp32 = to_dtype(inputs_fp64, torch.float32)
        outs_fp32 = test_op(*inputs_fp32)
        sum_outs_fp32 = torch.sum(outs_fp32)
        sum_outs_fp32.backward()
        grads_fp32 = [x.grad.detach().clone() for x in inputs_fp32 if hasattr(x, 'grad') and x.grad is not None]
        grads_fp16 = to_dtype(grads_fp32, torch.float16)
        outs_fp16 = outs_fp32.to(torch.float16)

        tensor_fp16 = OrderedDict(
            {'query_grad': grads_fp16[0], 'offset_grad': grads_fp16[1], 'attn_grad': grads_fp16[2], 'out': outs_fp16})
        tensor_fp64 = OrderedDict(
            {'query_grad': grads_fp64[0], 'offset_grad': grads_fp64[1], 'attn_grad': grads_fp64[2], 'out': outs_fp64})

        ret = compare_tensor_dict(tensor_fp16, tensor_fp64, skip_t2_zero=True)
        results.append(ret)

    print_table(results)


def compare_with_data():
    data_file = 'data/fp64_data.pkl'
    test_data_list = torch.load(data_file)

    results = []
    for test_data in test_data_list:
        inputs_fp64 = test_data['inputs']
        grads_fp64 = test_data['grads']
        grads_fp64 = [x for x in grads_fp64 if x is not None]
        outs_fp64 = test_data['outs']

        inputs_fp16 = to_dtype(inputs_fp64, torch.float16)
        outs_fp16 = test_op(*inputs_fp16)
        sum_outs_fp16 = torch.sum(outs_fp16)
        sum_outs_fp16.backward()
        grads_fp16 = [x.grad.detach().clone() for x in inputs_fp16 if hasattr(x, 'grad') and x.grad is not None]

        tensor_fp16 = OrderedDict(
            {'query_grad': grads_fp16[0], 'offset_grad': grads_fp16[1], 'attn_grad': grads_fp16[2], 'out': outs_fp16})
        tensor_fp64 = OrderedDict(
            {'query_grad': grads_fp64[0], 'offset_grad': grads_fp64[1], 'attn_grad': grads_fp64[2], 'out': outs_fp64})
        
        ret = compare_tensor_dict(tensor_fp16, tensor_fp64, skip_t2_zero=True)
        results.append(ret)
    
    print_table(results)



if __name__ == '__main__':
    compare_with_data()
    # internally_fp32()