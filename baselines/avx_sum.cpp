#include "avx_sum.h"

// TODO: Detailed understanding of highest throughput
// TODO: cycles per nanosecond

// TODO: look at avx switching cost and power settings

// TODO: L1, L2 has same throughput

// TODO: Optimize for applying the same operation to the same memory many many
// times

// TODO: Run benchmark multiple times and average (mean + stdev) and run it longer
// TODO: Use taskset


void reducesum_impl2(const float *arr, float *outarr, size_t size1b,
                     size_t size1e, size_t size2b, size_t size2e,
                     size_t size2) {
  for (size_t j = size2b; j < size2e; j += 1) {
    for (size_t i = size1b; i < size1e; i += 1) {
      outarr[j] += arr[i * size2 + j];
    }
  }
}

void reducesum_impl_naive(const float *arr, float *outarr, size_t size1b,
                          size_t size1e, size_t size2b, size_t size2e,
                          size_t size2) {
  for (size_t i = size1b; i < size1e; i += 1) {
    for (size_t j = size2b; j < size2e; j += 1) {
      outarr[j] += arr[i * size2 + j];
    }
  }
}

void reducesum_impl11(const float *arr, float *outarr, size_t size1b,
                      size_t size1e, size_t size2b, size_t size2e,
                      size_t size2) {
  for (size_t i = size1b; i < size1e; i += 1) {
    for (size_t j = size2b; j < size2e; j += 1) {
      outarr[j] += arr[i * size2 + j];
    }
  }
}


void reducesum_impl3_tile(const float *arr, float *outarr, size_t size1b, size_t size1e, size_t size2b, size_t size2e, size_t size2) {

  for (size_t i = size1b; i < size1e; i += _ROW) {
    for (size_t j = size2b / _VSIZE; j < size2e / _VSIZE; j += _COL) {
      __m256 tmp1[_COL];
      for (size_t j1 = 0; j1 < _COL; j1++) {
        __m256 tmp2[_ROW];
        tmp1[j1] = _mm256_load_ps(outarr + (j + j1) * _VSIZE);
        for (size_t i1 = 0; i1 < _ROW; i1++) {
          tmp2[i1] = _mm256_load_ps(arr + (i + i1) * size2 + (j + j1) * _VSIZE);
        }
        for (size_t i1 = 0; i1 < _ROW; i1++) {
          tmp1[j1] = _mm256_add_ps(tmp1[j1], tmp2[i1]);
        }
      }
      for (size_t j1 = 0; j1 < _COL; j1++) {
        _mm256_store_ps(outarr + (j + j1) * _VSIZE, tmp1[j1]);
      }
    }
  }
}

void reducesum_impl33(const float *arr, float *outarr, size_t size1b,
                      size_t size1e, size_t size2b, size_t size2e,
                      size_t size2) {
  assert(size2e > 7);
  size_t end = size2e - 7;
  for (size_t k = size2b; k < end; k += 8) {
    __m256 b = _mm256_loadu_ps(outarr + k);
    for (size_t i = size1b; i < size1e; i++) {
      __m256 a = _mm256_loadu_ps(arr + i * size2 + k);
      b = _mm256_add_ps(a, b);
    }
    _mm256_storeu_ps(outarr + k, b);
  }
}

void reducesum_impl3(const float *arr, float *outarr, size_t size1b,
                     size_t size1e, size_t size2b, size_t size2e,
                     size_t size2) {
  assert(size2e > 63);
  size_t end = _divup(size2e, 64) * 64;
  for (size_t k = size2b; k < end; k += 64) {
    __m256 a[8];
    __m256 b[8];
    for (int ib = 0; ib < 8; ib++) {
      b[ib] = _mm256_loadu_ps(outarr + k + ib * 8);
    }
    for (size_t i = size1b; i < size1e; i += 1) {
      for (int ib = 0; ib < 8; ib++) {
        a[ib] = _mm256_loadu_ps(arr + i * size2 + k + ib * 8);
        b[ib] = _mm256_add_ps(a[ib], b[ib]);
      }
    }
    for (int ib = 0; ib < 8; ib++) {
      _mm256_storeu_ps(outarr + k + ib * 8, b[ib]);
    }
  }
}

void reducesum_impl(const float *arr, float *outarr, size_t size1b,
                    size_t size1e, size_t size2b, size_t size2e, size_t size2) {
  register size_t k;
  for (size_t i = size1b; i < size1e; i += 1) {
    k = size2b;
    size_t end = size2e > 7 ? size2e - 8 : 0;
    for (; k < end; k += 8) {
      register __m256 a, b;
      a = _mm256_loadu_ps(arr + i * size2 + k);
      b = _mm256_loadu_ps(outarr + k);
      a = _mm256_add_ps(a, b);
      _mm256_storeu_ps(outarr + k, a);
    }
    for (; k < size2e; k++) {
      outarr[k] += arr[i * size2 + k];
    }
  }
}

void sum_impl4(float &sum, const float *arr, size_t start, size_t end) {
  assert((end - start) % _WIDTH == 0);
  float sums[_WIDTH];
  for (size_t i = 0; i < _WIDTH; i++) {
    sums[i] = 0;
  }
  size_t is[_WIDTH];
  for (size_t i = 0; i < _WIDTH; i++) {
    is[i] = ((end - start) / _WIDTH) * i;
  }
  for (size_t i = start; i < end; i += _WIDTH) {
    for (size_t j = 0; j < _WIDTH; j++) {
      sums[j] += arr[is[j]];
      is[j] += 1;
    }
  }
  sum = 0;
  for (size_t i = 0; i < _WIDTH; i++) {
    sum += sums[i];
  }
}

void sum_impl31(float &sum, const float *arr, size_t start, size_t end) {
  assert((end - start) % _WIDTH == 0);
  float sums[_WIDTH];
  for (size_t i = 0; i < _WIDTH; i++) {
    sums[i] = 0;
  }
  size_t is[_WIDTH];
  for (size_t i = 0; i < _WIDTH; i++) {
    is[i] = ((end - start) / _WIDTH) * i;
  }
  for (size_t i = start; i < end; i += _WIDTH * _WIDTH) {
    for (size_t j = 0; j < _WIDTH; j++) {
      for (size_t k = 0; k < _WIDTH; k++) {
        sums[j] += arr[is[j] + k];
      }
      is[j] += _WIDTH;
    }
  }
  sum = 0;
  for (size_t i = 0; i < _WIDTH; i++) {
    sum += sums[i];
  }
}

void sum_impl3(float &sum, const float *arr, size_t start, size_t end) {
  assert((end - start) % _WIDTH == 0);
  float sums[_WIDTH];
  for (size_t i = 0; i < _WIDTH; i++) {
    sums[i] = 0;
  }
  size_t is[_WIDTH];
  for (size_t i = 0; i < _WIDTH; i++) {
    is[i] = ((end - start) / _WIDTH) * i;
  }
  for (size_t i = start; i < end; i += _WIDTH) {
    for (size_t j = 0; j < _WIDTH; j++) {
      sums[j] += arr[is[j]];
      is[j] += 1;
    }
  }
  sum = 0;
  for (size_t i = 0; i < _WIDTH; i++) {
    sum += sums[i];
  }
}

void sum_impl21(float &sum, const float *arr, size_t start, size_t end) {
  register size_t k;
  float sarr[8];
  __m256 part_sum, tmp_sum[4], tmp_sum1, tmp_sum2, tmp_sum3, a[8];
  part_sum = _mm256_set1_ps(0);
  k = start;
  if (end % 64 != 0) {
    std::cout << "end: " << end << std::endl;
  }
  for (; k < end; k += 64) {
    for (size_t i = 0; i < 8; i++) {
      a[i] = _mm256_loadu_ps(arr + k + i * 8);
    }
    for (size_t i = 0; i < 8; i += 2) {
      tmp_sum[i / 2] = _mm256_add_ps(a[i], a[i + 1]);
    }
    tmp_sum1 = _mm256_add_ps(tmp_sum[0], tmp_sum[1]);
    tmp_sum2 = _mm256_add_ps(tmp_sum[2], tmp_sum[3]);
    tmp_sum3 = _mm256_add_ps(tmp_sum1, tmp_sum2);
    part_sum = _mm256_add_ps(part_sum, tmp_sum3);
  }
  _mm256_store_ps(sarr, part_sum);
  sum = sarr[0];
  for (int i = 1; i < 8; i++) {
    sum += sarr[i];
  }
}

void sum_impl21_fma(float &sum, const float *arr, size_t start, size_t end) {
  register size_t k;
  float sarr[8];
  __m256 part_sum, tmp_sum[4], tmp_sum1, tmp_sum2, tmp_sum3, a[8], useless;
  part_sum = _mm256_set1_ps(0);
  useless = _mm256_set1_ps(0);
  k = start;
  assert(end % 64 == 0);
  for (; k < end; k += 64) {
    for (size_t i = 0; i < 8; i++) {
      a[i] = _mm256_loadu_ps(arr + k + i * 8);
    }
    for (size_t i = 0; i < 8; i += 2) {
      tmp_sum[i / 2] = _mm256_fmadd_ps(useless, a[i], a[i + 1]);
    }
    tmp_sum1 = _mm256_fmadd_ps(useless, tmp_sum[0], tmp_sum[1]);
    tmp_sum2 = _mm256_fmadd_ps(useless, tmp_sum[2], tmp_sum[3]);
    tmp_sum3 = _mm256_fmadd_ps(useless, tmp_sum1, tmp_sum2);
    part_sum = _mm256_fmadd_ps(useless, part_sum, tmp_sum3);
  }
  _mm256_store_ps(sarr, part_sum);
  sum = sarr[0];
  for (int i = 1; i < 8; i++) {
    sum += sarr[i];
  }
}

void sum_impl2(float &sum, const float *arr, size_t start, size_t end) {
  assert(end % _WIDTH == 0);
  assert(start % _WIDTH == 0);
  sum = 0;
  for (size_t i = start; i < end; i += _WIDTH) {
    float slocal = 0;
    for (size_t j = 0; j < _WIDTH; j++) {
      slocal += arr[i + j];
    }
    sum += slocal;
  }
}

void sum_impl_naive(float &sum, const float *arr, size_t start, size_t end) {
  sum = 0;
  for (size_t i = start; i < end; i += 1) {
    sum += arr[i];
  }
}

void sum_impl_std(float &sum, const float *arr, size_t start, size_t end) {
  sum = std::accumulate(arr + start, arr + end, 0);
}

std::map<std::string, void (*)(const float *, float *, size_t, size_t,
                                      size_t, size_t, size_t)>
register_reducesum_impls() {
  std::map<std::string, void (*)(const float *, float *, size_t, size_t, size_t,
                                 size_t, size_t)>
      impls;
  impls["reducesum_impl"] = &reducesum_impl;
  impls["reducesum_impl2"] = &reducesum_impl2;
  impls["reducesum_impl11"] = &reducesum_impl11;
  impls["reducesum_impl3_tile"] = &reducesum_impl3_tile;
  impls["reducesum_impl3"] = &reducesum_impl3;
  impls["reducesum_impl33"] = &reducesum_impl33;
  impls["reducesum_impl_naive"] = &reducesum_impl_naive;
  return impls;
}

std::map<std::string, void (*)(float &, const float *, size_t, size_t)>
register_sum_impls() {
  std::map<std::string, void (*)(float &, const float *, size_t, size_t)> impls;
  impls["sum_impl4"] = &sum_impl4;
  impls["sum_impl31"] = &sum_impl31;
  impls["sum_impl3"] = &sum_impl3;
  impls["sum_impl2"] = &sum_impl2;
  impls["sum_impl_naive"] = &sum_impl_naive;
  // impls["sum_impl_std"] = &sum_impl_std; - simply too slow
  impls["sum_impl21"] = &sum_impl21;
  impls["sum_impl21_fma"] = &sum_impl21_fma;
  return impls;
}




