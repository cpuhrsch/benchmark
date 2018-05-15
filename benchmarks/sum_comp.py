import torch
import time
import gc

if __name__ == "__main__":
    output = torch.randn(100, 100000).type('torch.FloatTensor')
    gc.collect()
    tstart = time.time()
    for _ in range(2000):
        number = output.sum(dim=1)
    print("elapsed: " + str(time.time() - tstart))
