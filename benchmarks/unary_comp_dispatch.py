import torch
import time
import gc
import argparse
import math

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--slow', action='store_true')
    parser.add_argument('--inline', action='store_true')
    parser.add_argument('--counts', type=int)
    args = parser.parse_args()

    counts_ = 100
    # counts_ = 10
    dtype = 'torch.FloatTensor'
    tv = torch.randn((128*128*128)).type(dtype)
    # c = torch.randn((128*128*128)).type(dtype)
    #gc.collect()
    #gc.collect()
    c = tv.sin()
    if not args.slow:
        c = tv.sin()
    #gc.collect()
    #gc.collect()
    tstart = time.time()
    for i in range(counts_):
        #gc.collect()
        #gc.collect()
        # torch.sin(tv, out=c)
        c = tv.sin()
        #gc.collect()
        #gc.collect()
    elapsed = time.time() - tstart
    #gc.collect()
    #gc.collect()
    print(elapsed)
