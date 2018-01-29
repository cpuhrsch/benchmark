import numpy as np
import torch
import gc
import time

if __name__ == "__main__":
    sample_size = 10000;
    size = 1000000;
    enable_numpy = False
    enable_torch = True
    # enable_numpy = True
    # enable_torch = False
    v = np.random.rand(size)
    tv = torch.from_numpy(v).contiguous()
    status = ""
    nps = 0
    ts = 0
    if enable_numpy:
        gc.collect()
        start = time.time()
        for _ in range(sample_size):
            s = v.sum()
            # s /= size
            # v *= s
        nps = s
        status += "numpy: {} nps: {}".format(time.time() - start, nps)
    if enable_torch:
        gc.collect()
        start = time.time()
        for _ in range(sample_size):
            s = tv.sum()
            # s /= size
            # tv *= s
        ts = s
        t_time = time.time() - start
        status += "torch: {} ts: {}".format(time.time() - start, ts)
    print(status)
