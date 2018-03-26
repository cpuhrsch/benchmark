import torch
import time
import gc
import argparse
import math

def run(size, count, fname, mag):
    status = ""
    tv = torch.randn(size)
    status += fname + ":\t"
    status += "\tsize: 10^" + str(mag)
    status += "\tcount: " + str(count)
    f = getattr(tv, fname)
    gc.collect()
    tstart = time.time()
    for _ in range(count):
        c = f()
    elapsed = time.time() - tstart
    status += "\telapsed: " + str(elapsed)
    print(status)
    gc.collect()

if __name__ == "__main__":
    onek = 1000
    # process 1GB * sizeof(type) worth of data
    goal = onek * 1000 * 1000
    for fn in ["ceil", "floor", "round", "trunc", "sqrt", "clone"]:
        for mag in range(0, 9):
            size_ = int(onek * math.pow(10, mag))
            counts_ = goal / size_
            run(size_, counts_, fn, mag)

