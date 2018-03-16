import numpy as np
import torch
import gc
import random
import time
import math
from torch.autograd import Variable
import argparse


if __name__ == "__main__":
    dattypes = [np.float64, np.float32, np.int64, np.int32, np.int16]
    dattypes = [np.int32, np.int16]
    for arg in [None]:
        for dattype in dattypes:
            v = np.random.randn(1, 2)
            print(str(dattype) + " tensor shape: " + str(v.shape))
            for exp in range(1, 20):
                num = math.pow(2, exp)
                v.fill(num)
                v = v.astype(dattype)
                # tv = Variable(torch.from_numpy(v).contiguous())
                tv = torch.from_numpy(v).contiguous()
                # print(tv)
                s = "2^" + str(exp) + ": "
                if arg:
                    s += " numpy: " + str(v.prod(arg)[0])
                else:
                    s += " numpy: " + str(v.prod())
                if arg:
                    tvp = tv.prod(arg)[0]
                else:
                    tvp = tv.prod()
                    # print(tvp)
                try:
                    s += "\ttorch: " + str(tvp.item())
                except AttributeError:
                    s += "\ttorch: " + str(tvp)
                print(s)
