#pragma once

#include "benchmark_cpu.h"
#include "immintrin.h"
#include "xmmintrin.h"
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <vector>

constexpr size_t _ALIGNMENT = 32;
constexpr size_t _WIDTH = 16;
constexpr size_t _VSIZE = 8; // 8 floats - 256bits - one fetch has 1024 bits
constexpr size_t _ROW = 8;   // 4; // chunk of columns per tile
constexpr size_t _COL = 8;   // 4; // chunk of vector row per tile
constexpr size_t _SEED = 1;
constexpr size_t _HASHES_SIZE = 1024;

void reducesum_impl(const float *, float *, size_t, size_t, size_t, size_t,
                    size_t);
void reducesum_impl2(const float *, float *, size_t, size_t, size_t, size_t,
                     size_t);
void reducesum_impl11(const float *, float *, size_t, size_t, size_t, size_t,
                      size_t);
void reducesum_impl3_tile(const float *, float *, size_t, size_t, size_t,
                          size_t, size_t);
void reducesum_impl3(const float *, float *, size_t, size_t, size_t, size_t,
                     size_t);
void reducesum_impl33(const float *, float *, size_t, size_t, size_t, size_t,
                      size_t);
void reducesum_impl_naive(const float *, float *, size_t, size_t, size_t,
                          size_t, size_t);

std::map<std::string, void (*)(const float *, float *, size_t, size_t, size_t,
                               size_t, size_t)>
register_reducesum_impls();

void sum_impl4(float &, const float *, size_t, size_t);

void sum_impl4(float &, const float *, size_t, size_t);
void sum_impl31(float &, const float *, size_t, size_t);
void sum_impl3(float &, const float *, size_t, size_t);
void sum_impl2(float &, const float *, size_t, size_t);
void sum_impl_naive(float &, const float *, size_t, size_t);
void sum_impl_std(float &, const float *, size_t, size_t);
void sum_impl21(float &, const float *, size_t, size_t);
void sum_impl21_fma(float &, const float *, size_t, size_t);

std::map<std::string, void (*)(float &, const float *, size_t, size_t)>
register_sum_impls();
