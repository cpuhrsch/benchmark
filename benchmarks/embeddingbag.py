#!/usr/bin/env python

import torch
import random
import time
from torch import nn
from torch.autograd.variable import Variable

def perf_run(nume, size, num_input, num_offsets, max_offsets_size, runs, mode='mean', sparse=False):
    # an Embedding module containing 10 tensors of size 3
    embedding_sum = nn.EmbeddingBag(nume, size, mode=mode, sparse=sparse)
    # a batch of 2 samples of 4 indices each
     
    input = [random.randint(0, nume-1) for _ in range(num_input)]
    offsets = [0]
    for _ in range(num_offsets):
        offset = offsets[-1] + random.randint(1, max_offsets_size)
        if offset >= len(input):
            break
        offsets.append(offset)
    input = Variable(torch.LongTensor(input))
    offsets = Variable(torch.LongTensor(offsets))
    grad_output = torch.arange(1, len(offsets) * size + 1).view(len(offsets), size).type(torch.Tensor)
    start = time.time()
    for _ in range(runs):
        output = embedding_sum(input, offsets)
        # print(output)
        output.backward(grad_output)
        # print(grad_output)
        # print(embedding_sum.weight.data)
        # print(embedding_sum.weight.grad.data)
        # break
    return time.time() - start

if __name__ == "__main__":
    random.seed(42)
    # Main benchmakr! 
    # print(perf_run(10000, 100, 2000, 200, 20, 1000, sparse=True))
    # print(perf_run(10000, 100, 2000, 200, 20, 10000))
    print(perf_run(10000, 100, 2000, 200, 20, 10000))
    print(perf_run(10000, 100, 2000, 200, 20, 10000, sparse=True))

    # print(perf_run(10000, 128, 2000, 200, 20, 10000))
    # expect
    # Variable containing:
    # -0.7296 -4.6926  0.3295
    # -0.5186 -0.5631 -0.2792
    # [torch.FloatTensor of size 2x3]
