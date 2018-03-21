import torch
import time
import gc
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('threads', type=int, default=-1)
    parser.add_argument('--decrease', action='store_true')
    args = parser.parse_args()

    tv = torch.randn(1000 * 1000 * 10)
    if args.threads > 0:
        torch.set_num_threads(args.threads)
        num_thread = torch.get_num_threads()
    else:
        num_thread = 80
    gc.collect()
    tstart = time.time()
    for i in range(1, 100000):
        tv.sum()
        if i % 10000 == 0:
            elapsed = time.time() - tstart
            gc.collect()
            tstart = time.time()
            if args.decrease:
                num_thread = num_thread / 2
                print("decreasing to: " + str(num_thread) + " prev elapsed: " + str(elapsed))
                torch.set_num_threads(num_thread)
            else:
                num_thread = torch.get_num_threads()
                print("running with: " + str(num_thread) + " prev elapsed: " + str(elapsed))

        # print(tv.sum())
