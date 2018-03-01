import numpy as np
import torch
import gc
import random
import time
import math
from torch.autograd import Variable
import argparse


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

def run_instance(sizes, dims, runs, test, numpy, dattype, d):
    v = np.random.rand(*sizes).astype(dattype)
    tv = Variable(torch.from_numpy(v).contiguous())
    gc.collect()
    start = time.time()
    error = None
    for _ in range(runs):
        if numpy or test:
            if d is None:
                vr = v.sum()
            else:
                vr = v.sum(d)
        if not numpy or test:
            if d is None:
                tvr = tv.sum()
            else:
                tvr = tv.sum(d)
        if test:
            if d is not None:
                vr = Variable(torch.from_numpy(vr))
                if vr.nonzero().sum() != 0:
                    error = (tvr - vr).abs()
                    error = error.mean()
                    error = error.item()
                else:
                    error = 0
            else:
                error = abs(tvr.item() - vr.item())
            if error > 0:
                error = error
                assert(runs == 1)
            else:
                error = None
    t_time = time.time() - start
    return t_time, error

def run_settings(enable_numpy, all_settings, dattype):
    use_numpy = [False]
    if enable_numpy:
        use_numpy += [True]
    header = ["dtype"]
    first_setting = True
    all_speedup = 0
    for setting in all_settings:
        runs, sizes, dims, d = setting
        status = ""
        status += str(dattype) + ": "
        if first_setting:
            header[-1] += " " * (len(str(dattype)) - len(header[-1]) + 1)
        for en in use_numpy:
            ltime, error = run_instance(sizes, dims, runs, enable_test, en, dattype, d)
            if en:
                if first_setting:
                    header += ["library"]
                status += "numpy:  "
            else:
                if first_setting:
                    header += ["library"]
                status += "torch:  "
            header += [" "*6 + "runs", " sizes" + " " * 15, " "*4 + "time/run", " " * 4 + "dim"]
            d_str = str(d)
            if d is None:
                d_str = "all"
            status += "{:10}  {:20} {:10.2f}us {:8}".format(runs, str(sizes), ltime * 1e6 / runs, d_str)
            if en:
                ttime[1] += ltime
                speedup = ttime[1] / ttime[0]
                header += ["speedup"]
                status += str(speedup)
                all_speedup += speedup
            else:
                ttime[0] += ltime
            if error is not None:
                status += " error!: " + str(error)
        if first_setting:
            print(" ".join(header))
        first_setting = False
        print(status)
    print("Average speedup: " + str(all_speedup / len(all_settings)))
    print("-" * len(status))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--enable_numpy', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('mb', type=int, help='MB to process')
    
    args = parser.parse_args()

    sample_size = 1
    to_process = 1000 * 1000 * args.mb
    print("Processing " + str(to_process / 1e6) + "MB")
    enable_numpy = args.enable_numpy
    enable_test = args.test
    if enable_test:
        print("Testing!")
    ttime = [0.0, 0.0]
    all_settings = []
    random.seed(1234)
    for runs_ in range(2, 6):
        runs = math.pow(10, runs_)
        size = int(to_process / runs)
        runs = int(runs)
        if enable_test:
            runs = 1
        for dims in range(1, 5):
            for d in ([None] + range(dims - 1)):
                all_settings.append((runs, generate_sizes(size, dims), dims, d))
    for dattype in [np.float64, np.float32, np.int64, np.int32]:
        run_settings(enable_numpy, all_settings, dattype)
