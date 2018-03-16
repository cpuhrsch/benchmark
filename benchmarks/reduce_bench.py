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

def run_instance(op, sizes, dims, runs, test, numpy, dattype, d):
    if dattype != np.float64 and dattype != np.float32:
        v = np.ceil(np.random.binomial(n=1, p=(1.0 / max(1.0, (sum(sizes) / 1.5))), size=(sum(sizes))) + 1)
        v[np.random.permutation(range(len(v)))[:(len(v) / 2)]]*=-1
        v = np.resize(v, sizes)
    else:
        if op == "prod":
            v = np.random.randn(*sizes)
            v = (v / sum(sizes)) + 1
        else:
            v = np.random.randn(*sizes)

    v = v.astype(dattype)

    tv = Variable(torch.from_numpy(v).contiguous())
    # tv = tv.cuda()
    gc.collect()
    start = time.time()
    error = None
    for _ in range(runs):
        if numpy or test:
            if d is None:
                vr = getattr(v, op)()
            else:
                vr = getattr(v, op)(d)
        if not numpy or test:
            if d is None:
                tvr = getattr(tv, op)()
            else:
                tvr = getattr(tv, op)(d)
        if test:
            if d is not None:
                vr = Variable(torch.from_numpy(vr))
                if vr.nonzero().sum() != 0:
                    tvr = tvr.type('torch.DoubleTensor')
                    vr = vr.type('torch.DoubleTensor')
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
    all_speedup = {}
    ttime = [0.0, 0.0]
    for setting in all_settings:
        runs, sizes, dims, d, op = setting
        status = ""
        status += str(dattype) + ": "
        if first_setting:
            header[-1] += " " * (len(str(dattype)) - len(header[-1]) + 1)
        for en in use_numpy:
            ltime, error = run_instance(op, sizes, dims, runs, enable_test, en, dattype, d)
            if en:
                if first_setting:
                    header += ["library"]
                status += "numpy:  "
            else:
                if first_setting:
                    header += ["library"]
                status += "torch:  "
            header += [" "*6 + "runs", " sizes" + " " * 15, " "*4 + "time/run", " " * 4 + "dim", " " * 6 + "op"]
            d_str = str(d)
            if d is None:
                d_str = "all"
            status += "{:10}  {:20} {:10.2f}us {:8} {:8}".format(runs, str(sizes), ltime * 1e6 / runs, d_str, op)
            if en:
                ttime[1] += ltime
                speedup = ttime[1] / ttime[0]
                header += ["speedup"]
                status += str(speedup)
                if op not in all_speedup:
                    all_speedup[op] = (0.0, 0.0)
                all_speedup[op] = (all_speedup[op][0] + speedup, all_speedup[op][1] + 1)
            else:
                ttime[0] += ltime
            if error is not None:
                status += " error!: " + str(error)
        if first_setting:
            print(" ".join(header))
        first_setting = False
        print(status)
    for (k, v) in all_speedup.items():
        print(k + " Average speedup: " + str(v[0] / v[1]))
    print("-" * len(status))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--enable_numpy', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--threads', type=int)
    parser.add_argument('mb', type=int, help='MB to process')
    
    args = parser.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    sample_size = 1
    to_process = 1000 * 1000 * args.mb
    print("Processing " + str(to_process / 1e6) + "MB")
    enable_numpy = args.enable_numpy
    enable_test = args.test
    if enable_test:
        print("Testing!")
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
                for op in ("sum", "prod"):
                    all_settings.append((runs, generate_sizes(size, dims), dims, d, op))
    for dattype in [np.float32, np.float64, np.int64, np.int32, np.int16]:
        run_settings(enable_numpy, all_settings, dattype)
