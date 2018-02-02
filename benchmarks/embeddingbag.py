#!/usr/bin/env python

import torch
import random
import time
from torch import nn
from torch.autograd.variable import Variable

def perf_run(nume, size, data, mode='mean', sparse=False, cuda=False):
    if sparse:
        try:
            embedding_sum = nn.EmbeddingBag(nume, size, mode=mode, sparse=sparse)
        except TypeError:
            return 0.0
    else:
        embedding_sum = nn.EmbeddingBag(nume, size, mode=mode)
    if cuda:
        embedding_sum = embedding_sum.cuda()
     
    total_time = 0.0
    run_num = 0
    for (input, offsets, grad_output) in data:
        if cuda:
            input = input.cuda()
            offsets = offsets.cuda()
            grad_output = grad_output.cuda()
        input = Variable(input)
        offsets = Variable(offsets)
        grad_ouput = Variable(grad_output)
        start = time.time()
        output = embedding_sum(input, offsets)
        output.backward(grad_output)
        if cuda:
            torch.cuda.synchronize()
        embedding_sum.weight.grad.data.zero_()
        if (run_num > 1000):
            total_time += time.time() - start
        run_num += 1
    return total_time

def make_data(nume, size, num_input, num_offsets, max_offsets_size, runs):
    data = []
    for _ in range(runs):
        input = [random.randint(0, nume-1) for _ in range(num_input)]
        offsets = [0]
        for _ in range(num_offsets):
            offset = offsets[-1] + random.randint(1, max_offsets_size)
            if offset >= len(input):
                break
            offsets.append(offset)
        input = torch.LongTensor(input)
        offsets = torch.LongTensor(offsets)
        grad_output = torch.arange(1, len(offsets) * size + 1).view(len(offsets), size).type(torch.Tensor)
        data.append((input, offsets, grad_output))
    return data

if __name__ == "__main__":
    random.seed(42)
    grid = [(10000, 100), (10000, 1000), (100000, 100), (100000, 1000)]
    runs = 10000
    num_input = 2000
    num_offsets = 200
    max_offsets_size = 30
    print("\n" + "\""*100)
    print("runs: {}\tnumber of indices: {}\tmaximum number of bags: {}\tmaximum bag size: {}\n"
            .format(runs, num_input, num_offsets, max_offsets_size))
    for (nume, size) in grid:
        print("="*100)
        print("dimension:\t{}\tx\t{}\t".format(nume, size, runs))
        print("-"*100)
        data = make_data(nume, size, num_input, num_offsets, max_offsets_size, runs)
        print("cpu dense\tcpu sparse\tcuda dense\tcuda sparse")
        print("{: 8.3f}s".format(perf_run(nume, size, data)) +
        "\t{: 8.3f}s".format(perf_run(nume, size, data, sparse=True)) +
        "\t{: 8.3f}s".format(perf_run(nume, size, data, cuda=True)) +
        "\t{: 8.3f}s".format(perf_run(nume, size, data, cuda=True, sparse=True)))
        print("")
