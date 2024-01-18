import sys
sys.path.append('./')
import time
import torch

import copy

from functions.ms_deform_attn_func import MSDeformAttnFunction

from tests.create_data import generate_inputs
from easydict import EasyDict as edict

torch.manual_seed(0)

def test_op(value, shapes, level_start_index, sampling_locations, attention_weights, im2col_step):
    return MSDeformAttnFunction.apply(
            value, shapes, level_start_index, sampling_locations, attention_weights, im2col_step
    )


def to_dtype(inputs, dtype):
    new_inputs = list(copy.deepcopy(inputs))
    for i, x in enumerate(new_inputs):
        if isinstance(x, torch.Tensor) and torch.is_floating_point(x):
            new_inputs[i] = x.to(dtype).detach()
            new_inputs[i].requires_grad = x.requires_grad
    return tuple(new_inputs)


def run(module, args, name='Unknown'):
    inputs = generate_inputs(args.dtype, **args.data_args)

    # cudnn warmup
    for _ in range(50):
        if args.backward:
            module(*inputs).sum().backward()
        else:
            module(*inputs)

    torch.cuda.synchronize()
    t0 = time.time()

    for _ in range(args.num_iter):
        if args.backward:
            module(*inputs).sum().backward()
        else:
            module(*inputs)

    torch.cuda.synchronize()
    t1 = time.time()

    avg_time = (t1 - t0) * 1000 / args.num_iter
    print(
        f'>>> {name} finished {args.num_iter} running, avg_time: {avg_time:.6f} ms')
    return avg_time

def info_memory(msg=None):
    if msg:
        print(msg)
    print(f"MA {round(torch.cuda.memory_allocated() / (1024 * 1024 * 1024),2 )} GB \
        Max_MA {round(torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024),2)} GB \
        CA {round(torch.cuda.memory_reserved() / (1024 * 1024 * 1024),2)} GB \
        Max_CA {round(torch.cuda.max_memory_reserved() / (1024 * 1024 * 1024))} GB ")



if __name__ == '__main__':

    data_args = edict()
    data_args.bs = 32
    data_args.n_levels = 2
    data_args.shapes=[(16,16), (8,8)]
    data_args.n_query = 128
    data_args.n_points = 64
    data_args.n_heads = 8
    data_args.head_dim = 128

    args = edict()
    args.num_iter = 200
    args.backward = True
    args.dtype = torch.float16
    args.data_args = data_args

    run(test_op, args, name='fp16')
    info_memory()
    args.dtype = torch.float32
    run(test_op, args, name='fp32')
    info_memory()