import torch
import time
import gc
import argparse
import math

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
        tv = tv.transpose(dim -2, dim -1)
    return tv

def run(tv, count, fname, mag, dtype, dim):
    status = ""
    status += "{:<10}".format(fname)
    status += " memory: {:<10}".format("O(10^" + str(mag) + ")KB")
    status += " count: {:<6}".format(count)
    status += " size: {:<20}".format(list(tv.size()))
    status += " stride: {:<60}".format(list(map(lambda x: "{:>7}".format(x), list(tv.stride()))))
    status += " numel: {:<9}".format(tv.numel())
    f = getattr(tv, fname)
    gc.collect()
    gc.collect()
    tstart = time.time()
    for i in range(count):
        c = f()
    elapsed = time.time() - tstart
    status += " type: {:<18}".format(dtype)
    status += " dim: {:<5}".format(dim)
    status += " elapsed: {:8.4f}".format(elapsed)
    print(status)
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

    # Compare contiguous only
    float_fns = float_fns
    funcs = list(map(lambda x: x + "_", float_fns)) + float_fns
    run_all(funcs, [4], types, [True], [3], [False], goal_size=250)

    # # Check for regression
    # float_fns = float_fns_nonvec_touched
    # funcs = list(map(lambda x: x + "_", float_fns)) + float_fns
    # run_all(funcs, [4, 2, 1], types, [True, False], [5, 3], [True, False], goal_size=250)


if __name__ == "__main__":
    sleef_benchmark()
