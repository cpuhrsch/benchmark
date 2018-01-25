#!/usr/bin/env python

import torch
import numpy as np
import random
import inspect
import time
import math
import gc

def find_common(array_pairs, no_arg=True):
    t = set(dir(torch.Tensor))
    nobj = np.array(1)
    n = set(dir(nobj))
    inter = t.intersection(n)
    match_ops = []
    match_shapes = []
    for opi in inter:
        if not opi.startswith("__"):
            for (a, b) in array_pairs:
                fun = getattr(a, opi)
                try:
                    if no_arg:
                        r1 = fun()
                    else:
                        r1 = fun(b)
                    (at, bt) = (torch.from_numpy(a), torch.from_numpy(b))
                    fun = getattr(at, opi)
                    if no_arg:
                        r2 = fun()
                    else:
                        r2 = fun(bt)
                    assert(np.isclose(r1, r2))
                    match_ops.append(opi)
                    match_shapes.append(a.shape)
                except:
                    # if no_arg:
                    #     print(opi + " failed for no_arg")
                    # else:
                    #     print(opi + " failed")
                    continue
    return match_ops, match_shapes

def gen_array_pairs():
    pairs = []
    dim_max = 4
    dim_min = 1
    for dim in range(dim_min, dim_max):
        arg = tuple([random.randint(2, 4) for _ in range(dim)])
        pairs.append((np.random.rand(*arg), np.random.rand(*arg)))
    return pairs

def gen_datum(shape, min_size=30, max_size=50):
    dim = len(shape)
    arg = tuple([random.randint(min_size, max_size) for _ in range(dim)])
    return np.random.rand(*arg)

def bench(ops, shapes, no_arg=True, smin_=100, smax_=1000, count_=1000):
    def norm(x):
        if dim > 1:
            return int(math.pow(float(x), 1/dim))
        else:
            return x
    op_times = []
    for (shape, op) in zip(shapes, ops):
        dim = float(len(shape))
        smin = norm(smin_)
        smax = norm(smax_)
        count = count_
        np_time = 0.0
        tr_time = 0.0
        if no_arg:
            datum = gen_datum(shape, min_size=smin, max_size=smax)
            tdatum = torch.from_numpy(datum)
            gc.collect()
            start = time.time()
            for _ in range(count):
                r1 = getattr(datum, op)()
            np_time += time.time() - start
            start = time.time()
            for _ in range(count):
                r2 = getattr(tdatum, op)()
            tr_time += time.time() - start
        else:
            raise(ValueError("No implement"))
        op_times.append((dim, op, np_time, tr_time))
        print("dim: {}, op: {}, numpy: {} torch: {}".format(*op_times[-1]))
    return op_times

if __name__ == "__main__":
    random.seed(42)
    ops, shapes = find_common(gen_array_pairs())
    bench(ops, shapes, smin_=1000000, smax_=1000000, count_=1000)
