#!/usr/bin/env python

from tabulate import tabulate

if __name__ == "__main__":
    values = {}

    with open("avx_sum_result.csv") as f:
        for line in f:
            line = line.split(",")
            for cell in line:
                k, v =map(lambda x: x.strip(), cell.split(":"))
                k = k[1:]
                v = v[:-1]
                if k not in values:
                    values[k] = []
                values[k].append(v)
    print(",".join(values.keys()))
    keys = values.keys()
    for i in range(len(values[keys[0]])):
        line = []
        for k in keys:
            line.append(values[k][i])
        print(",".join(line))




