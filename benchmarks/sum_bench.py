import numpy as np
import torch
import gc
import random
import time
import math
from torch.autograd import Variable


#TODO: Compare to CUDA

def generate_sizes(size, dims):
    sizes = []
    for d in range(dims):
        if d < dims - 1:
            rs = random.randint(int(math.pow(size / dims, 1.0 / dims)), int(math.pow(size, 1.2 / dims)))
            if rs >0:
                size = size / rs
        else:
            rs = size
        if rs <= 0:
            rs = 1
        sizes.append(rs)

    return sizes

def run_instance(sizes, dims, runs, test, numpy):

    # sizes = [5, 100]

    v = np.random.rand(*sizes).astype(np.float32)
    print(v.dtype)
    tv = Variable(torch.from_numpy(v).contiguous())
    gc.collect()
    start = time.time()
    for d in range(dims - 1):
        for _ in range(runs):
            if numpy or test:
                vr = v.sum(d)
            if not numpy or test:
                tvr = tv.sum(d)
            if test:
                vr = Variable(torch.from_numpy(vr))
                if not torch.equal(tvr, vr):
                    print("ERROR!")
                    print(((tvr - vr).abs() / vr).mean())
                    print("(tvr - vr).abs() / vr")
                    print((tvr - vr).abs() / vr)
                    print("tvr")
                    print(tvr)
                    print("vr")
                    print(vr)
                    print("ERROR!")
                    import sys
                    sys.exit(1)
    t_time = time.time() - start
    return t_time



if __name__ == "__main__":
    sample_size = 1
#    sample_size = 1
    to_process = 1000 * 1000 * 1000
    enable_numpy = True
    enable_test = False
    # enable_numpy = True
    # enable_torch = False
    ttime = 0.0
    all_settings = []
    random.seed(1234)
    for runs_ in range(2, 6):
        runs = math.pow(10, runs_)
        size = int(to_process / runs)
        runs = int(runs)
        for dims in range(1, 5):
            all_settings.append((runs, generate_sizes(size, dims), dims))
    for setting in all_settings:
        runs, sizes, dims = setting
        print("runs: {} sizes: {} dims: {}".format(str(runs), str(sizes), str(dims)))
        ttime += run_instance(sizes, dims, runs, enable_test, enable_numpy)
    print("ttime: " + str(ttime) + "s")
