import torch
import time
import gc

if __name__ == "__main__":
    softmax = torch.nn.LogSoftmax(dim=1)
    output = torch.randn(256, 20000).type('torch.FloatTensor')
    gc.collect()
    tstart = time.time()
    for _ in range(250):
        numbers = softmax(output)
    print("elapsed: " + str(time.time() - tstart))
