#!/usr/bin/env python

import argparse
import itertools

GRID_sizes = [
    (64, 64),
    (128, 128),(256, 256),(128, 512),(512, 128),(1024, 1024),(2048, 2048),(4096, 4096),(512, 4096),(4096, 512)
]

GRID_thresholds = [4096, 16384, 65536, 1048576, 4194304]

GRID_functions = ["-run_sum_tbb sum_impl_tbb_2", "-run_sum_tbb sum_impl_tbb", "-run_sum sum_impl21"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('cmd')
    parser.add_argument('repeat', type=int) # Repeat to increase benchmark precision
    
    args = parser.parse_args()
    grid = []
    for ((size1, size2), threshold, function) in list(itertools.product(GRID_sizes, GRID_thresholds, GRID_functions)):
        grid.append(args.cmd + " -size1 " + str(size1) + " -size2 " + str(size2) + " -epoch -1 " + function + " -threshold " + str(threshold))
    for g in grid:
        for _ in range(args.repeat):
            print(g)

# -run_sum -run_reducesum -epoch -1
# -run_sum -run_reducesum -epoch -1
# -run_sum -run_reducesum -epoch -1
# -run_sum -run_reducesum -epoch -1
# -run_sum -run_reducesum -epoch -1
# -run_sum -run_reducesum -epoch -1
# -run_sum -run_reducesum -epoch -1
# -run_sum -run_reducesum -epoch -1
# -run_sum -run_reducesum -epoch -1
# -run_sum -run_reducesum -epoch -1
