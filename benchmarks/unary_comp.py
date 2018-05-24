from __future__ import print_function
import torch
import time
import gc
import argparse
import math
import numpy as np
from torch import functional as F

import sys

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def make_size(dim, size_):
    if dim == 1:
        size = size_
    else:
        size = [0] * dim
        for i in range(dim):
            size[i] = int(math.pow(size_, 1.0 / float(dim)))
        size = tuple(size)
    return size

def make_tensor(size_, dtype, cont, dim, trans):
    size = make_size(dim, size_)
    if cont:
        tv = torch.rand(size).type(dtype)
    else:
        size = [size[0]] + list(size)
        size[dim] = 18
        size = tuple(size)
        tv = torch.rand(size).type(dtype)
        tv = tv.select(dim, 0)
    if trans:
        # tv = tv.transpose(dim -2, dim -1)
        tv = tv.transpose(0, 1)
    return tv

def start_stats(common_name, framework_name, fname, mag, count, tv):
    status = ""
    status += "tag: {:<15}".format(common_name)
    status += "fname: {:<15}".format(framework_name)
    status += "{:<15}".format(fname)
    status += " memory: {:<10}".format("O(10^" + str(mag) + ")KB")
    status += " count: {:<6}".format(count)
    status += " size: {:<20}".format(list(tv.size()))
    status += " stride: {:<60}".format(list(map(lambda x: "{:>7}".format(x), list(tv.stride()))))
    status += " numel: {:<9}".format(tv.numel())
    return status

def finish_stats(dtype, dim, elapsed):
    status = ""
    status += " type: {:<18}".format(dtype)
    status += " dim: {:<5}".format(dim)
    status += " elapsed: {:8.4f}".format(elapsed)
    return status

def run(tv, count, fname, mag, dtype, dim):
    status = start_stats(fname, mag, count, tv)
    f = getattr(tv, fname)
    gc.collect()
    gc.collect()
    tstart = time.time()
    for i in range(count):
        c = f()
    elapsed = time.time() - tstart
    print(status + finish_stats(dtype, dim, elapsed))
    gc.collect()

def run_reduce(tv, count, fname, mag, dtype, dim, opdim):
    status = start_stats(fname, mag, count, tv)
    gc.collect()
    gc.collect()
    tstart = time.time()
    for i in range(count):
        c = fname(tv, opdim)
    elapsed = time.time() - tstart
    print(status + finish_stats(dtype, dim, elapsed))
    gc.collect()

def run_all(fns, mags, dtypes, conts, dims, transs, goal_size=1000):
    onek = 1000
    goal = onek * 1000 * goal_size
    for dim_ in dims:
        for dtype in dtypes:
            for mag in mags:
                for fn in fns:
                    for cont in conts:
                        for trans in transs:
                            size_ = int(onek * math.pow(10, mag))
                            counts_ = goal / size_
                            tv = make_tensor(size_, dtype, cont, dim_, trans)
                            run(tv, counts_, fn, mag, dtype, dim_)

def run_reduce_all(fns, fns_opdims, mags, dtypes, conts, dims, transs, goal_size=1000):
    onek = 1000
    goal = onek * 1000 * goal_size
    for dim_ in dims:
        for dtype in dtypes:
            for mag in mags:
                for ii in range(len(fns)):
                    for cont in conts:
                        for trans in transs:
                            size_ = int(onek * math.pow(10, mag))
                            counts_ = goal / size_
                            tv = make_tensor(size_, dtype, cont, dim_, trans)
                            run_reduce(tv, counts_, fns[ii], mag, dtype, dim_, fns_opdims[ii])

float_types = ['torch.FloatTensor', 'torch.DoubleTensor']
int_types = ['torch.IntTensor', 'torch.LongTensor', 'torch.ShortTensor']

def sleef_benchmark():
    float_fns_nonvec = [
        "cos",
        "sin",
        "tan"
    ]
    float_fns_nonvec_touched = [
        "cosh",
        "sinh",
        "tanh"
    ]
    float_fns_vec_old = [
        "abs",
        "ceil",
        "floor",
        "round",
        "sqrt",
        "trunc"
    ]
    float_fns = [
        "acos",
        "asin",
        "atan",
        "erf",
        "exp",
        "expm1",
        "log",
        "log10",
        "log1p",
        "log2",
        "rsqrt",
    ]
    types = float_types
    # sleef benchmark

    float_fns = float_fns + float_fns_nonvec + float_fns_nonvec_touched + float_fns_vec_old
    float_fns = ["acos", "erf", "log", "log10", "log2", "rsqrt", "abs", "sqrt"]
    float_fns = ["acos", "log", "log10"]
    funcs = list(map(lambda x: x + "_", float_fns)) + float_fns
    # run_all(funcs, [4], types, [True], [3], [False], goal_size=250)
    # run_all(funcs, [4, 2], [torch.float32], [True, False], [3], [True, False], goal_size=250)
    run_all(funcs, [4], [torch.float32], [True, False], [3], [True, False], goal_size=250)
    # run_all(funcs, [4, 2, 1], types, [True, False], [5, 3], [True, False], goal_size=250)

    # # Check for regression
    # float_fns = float_fns_nonvec_touched
    # funcs = list(map(lambda x: x + "_", float_fns)) + float_fns
    # run_all(funcs, [4, 2, 1], types, [True, False], [5, 3], [True, False], goal_size=250)

def softmax_benchmark():
    types = float_types
    # Compare contiguous only
    funcs = [torch.nn.functional.log_softmax, torch.nn.functional.softmax]
    func_opdims = [2, 2]
    run_reduce_all(funcs, func_opdims, [4], types, [True], [3], [False], goal_size=250)

def lambda_benchmark(common_name, types, fun, name, framework_name, cast):
    goal_size = 2#00
    onek = 1000
    goal = onek * 1000 * goal_size
    for cont in [True, False]:
        for trans in [True, False]:
            for mag in [1, 2]:
                for dim in [4]:
                    for dtype in types:
                        size_ = int(onek * math.pow(10, mag))
                        count = goal / size_
                        tv = make_tensor(size_, dtype, cont, 3, trans)
                        status = start_stats(common_name, framework_name, name, mag, count, tv)
                        gc.collect()
                        gc.collect()
                        tstart = time.time()
                        for _ in range(count):
                            fun(tv)
                        elapsed = time.time() - tstart
                        print(status + finish_stats(dtype, dim, elapsed))
                        gc.collect()
                        gc.collect()

def numpy_comparison():
    all_fns = [
        ("acos", "arccos"),
        ("asin", "arcsin"),
        ("atan", "arctan"),
        "cos",
        "cosh",
        "sin",
        "tan",
        "sinh",
        "tanh",
        "abs",
        "ceil",
        "floor",
        "round",
        "sqrt",
        "trunc",
        "erf",
        "exp",
        "expm1",
        "log",
        "log10",
        "log1p",
        "log2",
        "rsqrt",
    ]
    for fn in all_fns:
        if isinstance(fn, tuple):
            torch_fn = fn[0]
            numpy_fn = fn[1]
        else:
            torch_fn = fn
            numpy_fn = fn
        try:
            lambda_benchmark(torch_fn, float_types, lambda x: getattr(np, numpy_fn)(x), numpy_fn, "numpy", lambda x: x.numpy())
        except AttributeError:
            eprint(numpy_fn + " not supported by numpy.")

	try:
            lambda_benchmark(torch_fn, float_types, lambda x: getattr(x, torch_fn)(), torch_fn, "torch", lambda x: x)
        except AttributeError:
            eprint(torch_fn + " not supported by torch.")

# TODO: Output csv for excel
# TODO: Output time per op
# TODO: Output amount of data in machine readable form
# TODO: Shuffle operations to remove more bias
# TODO: Create separate PR for benchmark repo

if __name__ == "__main__":
    numpy_comparison()
    # sleef_benchmark()
    # softmax_nn_benchmark()
    # lambda_benchmark(float_types + int_types, lambda x: x.fill_(1), "fill_")
    # lambda_benchmark(lambda x: x.clamp_(0, 1), "clamp_")
    # lambda_benchmark(lambda x: x.clamp(0, 1), "clamp")
    # lambda_benchmark(float_types, lambda x: x.sigmoid(), "sigmoid")
    # lambda_benchmark(float_types, lambda x: x.floor(), "floor", "torch", lambda x: x)
    # lambda_benchmark(float_types, lambda x: np.floor(x), "floor", "numpy", lambda x: x.numpy())
