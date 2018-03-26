#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <cmath>
#include <x86intrin.h>
#include <cassert>
#include "avx_sum.h"
#include "tbb_sum.h"
#include <gflags/gflags.h>


int ceil(float * a, const float*b, size_t numel) {
    size_t i = 0;
    for (; i + 8 < numel; i+=8) {
        a[i] = std::ceil(b[i]);
        a[i+1] = std::ceil(b[i+1]);
        a[i+2] = std::ceil(b[i+2]);
        a[i+3] = std::ceil(b[i+3]);
        a[i+4] = std::ceil(b[i+4]);
        a[i+5] = std::ceil(b[i+5]);
        a[i+6] = std::ceil(b[i+6]);
        a[i+7] = std::ceil(b[i+7]);
    }
    for (; i < numel; i++) {
        a[i] = std::ceil(b[i]);
    }
    return 0;
}

int ceil_v(float * a, const float*b, size_t numel) {
    size_t i = 0;
    __m256 bv;
    for (; i + 8 < numel; i += 8) {
        bv = _mm256_loadu_ps(b + i);
        bv = _mm256_ceil_ps(bv);
        _mm256_storeu_ps(a + i, bv);
    }
    for (; i < numel; i++) {
        a[i] = std::ceil(b[i]);
    }
    return 0;
}

void time_stats(uint64_t s, double floats) {
  std::cout << "(ns:\033[36m" << std::fixed << std::setw(15) << s << "\033[0m)"
            << ",(ops/ns: "
            << "\033[31m";
  std::cout << std::setw(12) << (double)floats / (double)s << "\033[0m)";
  std::cout << ",(s: " << s / (double)NSEC << ")";
}

void make_float_data(float **data_, size_t size) {
  if (posix_memalign((void **)data_, _ALIGNMENT, size * sizeof(float)))
    throw std::invalid_argument("received negative value");
  memset(*data_, 0, size * sizeof(float));
}

int main() {
    // size_t size = 1024 * 1024 * 1024;
    size_t size = 1024 * 1024 * 128;
    assert(size >= _ALIGNMENT);
    assert(size % _ALIGNMENT == 0);
    // size_t size = 20;
    // size_t counts = 1000 * 5;
    size_t counts = 1;
    float * a;
    float * a1;
    float * b;
    make_float_data(&a, size);
    make_float_data(&a1, size);
    make_float_data(&b, size);
    for (size_t i = 0; i < size; i++) {
      b[i] = (float)(i % 1024);// * 0.25;
    }
  auto start = get_time();
    for (size_t i = 0; i < counts; i++) {
        ceil(a, b, size);
        // ceil_v(a1, b, size);
    //  b[i] = a[i];
    }
  auto end = get_time();
  time_stats(timespec_subtract_to_ns(&start, &end), counts * size);
    // for (size_t i = 0; i < size; i++) {
    //         if(a[i] != a1[i]) {
    //             std::cerr << "a[" << i << "]: " << a[i] << std::endl;
    //             std::cerr << "a1[" << i << "]: " << a1[i] << std::endl;
    //             break;
    //         }
    // }
    // for (size_t i = 0; i < size; i++) {
    //     std::cout << "a[" << i << "]:\t" << a[i];
    //     std::cout << "\ta1[" << i << "]:\t" << a1[i];
    //     std::cout << "\tb[" << i << "]:\t" << b[i];
    //     std::cout << std::endl;
    // }
    return 0;
}
