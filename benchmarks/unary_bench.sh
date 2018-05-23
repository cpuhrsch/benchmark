#!/bin/bash
set -c
set -x
echo "MASTER"
taskset -c 0 perf stat ~/miniconda2master/bin/python unary_comp.py
echo "BRANCH"
taskset -c 0 perf stat ~/miniconda2/bin/python unary_comp.py
