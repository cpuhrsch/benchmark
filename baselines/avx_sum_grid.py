#!/usr/bin/env python

import argparse

CMD = "./avx_sum.exe"

GRID = [
    (64, 64),
    (128, 128),(256, 256),(128, 512),(512, 128),(1024, 1024),(2048, 2048),(4096, 4096),(512, 4096),(4096, 512)
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('run')
    
    args = parser.parse_args()
    for (size1, size2) in GRID:
        print(CMD + " -size1 " + str(size1) + " -size2 " + str(size2) + " -epoch -1 " + args.run)

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
