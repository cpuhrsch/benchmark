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
    status += "{:<5}".format(fname)
    status += " size: 10^{:<3}".format(mag)
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
    status += " elapsed: {:8.4f}".format(elapsed)
    status += " type: {:<20}".format(dtype)
    status += " dim: " + str(dim)
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


if __name__ == "__main__":
    onek = 1000
    # process 1GB * sizeof(type) worth of data
    goal = onek * 1000 * 1000 * 1
    # fns = ["acos","asin","atan","cos","cosh","erf","exp","expm1","lgamma","log1p","log","sin","sinh","tan","tanh"]
    float_fns = ["cos", "sin", "exp", "log"]
    float_fns += ["ceil", "floor", "round", "trunc", "sqrt"]
    # run_all(float_fns, range(2, 6), [torch.double, torch.float])
    float_types = ['torch.FloatTensor', 'torch.DoubleTensor']
    int_types = ['torch.IntTensor', 'torch.LongTensor', 'torch.ShortTensor']
    # run_all(["abs"], range(2, 6), float_types + int_types, False)
    # run_all(["abs"], range(2, 4), float_types + int_types, False)
    # run_all(["sin"], range(2, 4), float_types, False)
    # run_all(["sin"], range(2, 4), float_types, False, [3])

    float_fns = ["sin"]
    types = float_types
    funcs = list(map(lambda x: x + "_", float_fns)) + float_fns
    run_all(funcs, [4, 2, 1], types, [True, False], [3, 5], [True, False], goal_size=100)

    # # LATEST BENCHMARK
    # float_fns = ["sqrt", "sin"]
    # types = float_types
    # funcs = list(map(lambda x: x + "_", float_fns)) + float_fns
    # run_all(funcs, [4, 2, 1], types, [True, False], [3, 5], [True, False], goal_size=100)

    # run_all(["sin"], [2, 2, 2], float_types, False, [3], goal_size=100)
    # run_all(["sin"], [2], float_types, False, [3], goal_size=1000)
