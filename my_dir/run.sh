#!/bin/bash
numactl --cpubind 0 --membind 0 taskset -c 0 /private/home/cpuhrsch/miniconda3/envs/binary/bin/python one_off.py 1
