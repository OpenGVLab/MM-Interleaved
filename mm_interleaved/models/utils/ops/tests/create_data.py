import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
import pickle as pkl
import copy

from functions.ms_deform_attn_func import MSDeformAttnFunction

def generate_inputs(dtype, bs=1, n_levels=2, shapes=((6,4), (3,2)), n_query=2, n_points=2, n_heads=2, head_dim=4):
    assert len(shapes) == n_levels
    shapes = torch.as_tensor(list(shapes), dtype=torch.long)
    assert shapes.shape[1] == 2
    level_start_index = torch.cat((shapes.new_zeros((1, )), shapes.prod(1).cumsum(0)[:-1]))
    spatial_size = sum([(H*W).item() for H, W in shapes])

    value = torch.rand(bs, spatial_size, n_heads, head_dim)
    sampling_locations = torch.rand(bs, n_query, n_heads, n_levels, n_points, 2)
    attention_weights = torch.rand(bs, n_query, n_heads, n_levels, n_points).cuda() + 1e-5
    attention_weights /= attention_weights.sum(-1, keepdim=True).sum(-2, keepdim=True)
    im2col_step = 2
    return (
        value.cuda().to(dtype).requires_grad_(True),
        shapes.cuda(),
        level_start_index.cuda(),
        sampling_locations.cuda().to(dtype).requires_grad_(True),
        attention_weights.cuda().to(dtype).requires_grad_(True),
        im2col_step
    )

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


if __name__ == '__main__':
    out_file = 'data/fp64_data.pkl'
    data_list = []
    for i in range(20):
        inputs = generate_inputs(torch.float16, 64)
        inputs = to_dtype(inputs, torch.float64)
        inputs[0].requires_grad = True
        inputs[3].requires_grad = True
        inputs[4].requires_grad = True

        outs = test_op(*inputs)
        outs.sum().backward()

        grads = [x.grad if hasattr(x, 'grad') else None for x in inputs]

        data_list.append(OrderedDict({
            'inputs': inputs,
            'grads': grads,
            'outs': outs
        }))
    
    torch.save(data_list, out_file)
    

