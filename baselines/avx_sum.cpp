#include "benchmark_cpu.h"
#include "immintrin.h"
#include "xmmintrin.h"
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <gflags/gflags.h>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <vector>

DEFINE_int64(size1, 1024, "size1 of matrix");
DEFINE_int64(size2, 1024, "size2 of matrix");
DEFINE_int64(epoch, 1, "number of runs with cache");
DEFINE_bool(test, false, "Show Debug matrix");
DEFINE_string(run_sum, "", "run given sum algorithm");
DEFINE_string(run_reducesum, "", "run given sumreduce algorithm");
DEFINE_bool(flush_cache, false, "flush cache before each run");
DEFINE_bool(list_run_sums, false, "list avail sum algorithms");
DEFINE_bool(list_run_reducesums, false, "list avail reducesum algorithms");

// TODO: Detailed understanding of highest throughput
// TODO: cycles per nanosecond

// TODO: look at avx switching cost

// TODO: L1, L2 has same throughput

// TODO: Optimize for applying the same operation to the same memory many many
// times

constexpr size_t _WIDTH = 16;

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

constexpr size_t _VSIZE = 8; // 8 floats - 256bits - one fetch has 1024 bits
constexpr size_t _ROW = 8;   // 4; // chunk of columns per tile
constexpr size_t _COL = 8;   // 4; // chunk of vector row per tile

void reducesum_impl3_tile(const float *arr, float *outarr, size_t size1b,
                          size_t size1e, size_t size2b, size_t size2e,
                          size_t size2) {

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
  size_t end = size2e - 63;
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
  assert(end % 64 == 0);
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

static std::map<std::string, void (*)(const float *, float *, size_t, size_t,
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

static std::map<std::string, void (*)(float &, const float *, size_t, size_t)>
register_sum_impls() {
  std::map<std::string, void (*)(float &, const float *, size_t, size_t)> impls;
  impls["sum_impl4"] = &sum_impl4;
  impls["sum_impl31"] = &sum_impl31;
  impls["sum_impl3"] = &sum_impl3;
  impls["sum_impl2"] = &sum_impl2;
  impls["sum_impl_naive"] = &sum_impl_naive;
  impls["sum_impl_std"] = &sum_impl_std;
  impls["sum_impl21"] = &sum_impl21;
  impls["sum_impl21_fma"] = &sum_impl21_fma;
  return impls;
}

constexpr size_t _SEED = 1;

constexpr size_t _HASHES_SIZE = 1024;

void make_vector(float *data_, size_t size) {
  // std::ifstream ifs(FLAGS_inpath, std::ifstream::binary);
  for (size_t i = 0; i < size; i++) {
    // float num;
    // ifs.read((char *)&num, sizeof(float));
    // data_[i] = num;
    data_[i] = (float)(i % 1024);
  }
  // ifs.close();
  // srand(1);
  // assert(size % _HASHES_SIZE == 0);
  // for (size_t c = 0; c < size; c += _HASHES_SIZE) {
  //   size_t start = rand();
  //   for (size_t ci = 0; ci < _HASHES_SIZE; ci++) {
  //     data_[c + ci] = (float)((start * ci) % (RAND_MAX)) / (float)RAND_MAX
  //     -0.5f;
  //   }
  // }
}

constexpr size_t _ALIGNMENT = 32;

void make_float_data(float **data_, size_t size) {
  if (posix_memalign((void **)data_, _ALIGNMENT, size * sizeof(float)))
    throw std::invalid_argument("received negative value");
  memset(*data_, 0, size * sizeof(float));
}

void time_stats(uint64_t s, double floats) {
  std::cout << "(ns:\033[36m" << std::fixed << std::setw(15) << s << "\033[0m)"
            << ",(ops/ns: "
            << "\033[31m";
  std::cout << std::setw(12) << (double)floats / (double)s << "\033[0m)";
  std::cout << ",(s: " << s / (double)NSEC << ")";
}

uint64_t sum_run(void (*fun)(float &, const float *, size_t, size_t),

                 const float *data, size_t size, size_t counts, size_t epoch) {
  float all_sum = 0;
  auto perm = rand_perm(counts);
  auto start = get_time();
  std::random_shuffle(perm.begin(), perm.end());
  for (size_t i_ = 0; i_ < counts; i_++) {
    for (size_t e = 0; e < epoch; e++) {
      size_t i = perm[i_];
      const float *datum = data + i * size;
      float sum;
      if (FLAGS_flush_cache)
        cacheflush(data);
      fun(sum, datum, 0, size);
      all_sum += sum;
    }
  }
  auto end = get_time();
  if (FLAGS_test) {
    assert(epoch == 1);
    assert(counts == 1);
    float sum1;
    fun(sum1, data, 0, size);
    assert(sum1 == all_sum);
  }
  return timespec_subtract_to_ns(&start, &end);
}

uint64_t reducesum_run(void (*fun)(const float *, float *, size_t, size_t,
                                   size_t, size_t, size_t),
                       const float *data, size_t size1, size_t size2,
                       size_t counts, size_t epoch) {
  float *outarr = NULL;
  make_float_data(&outarr, size2);
  auto perm = rand_perm(counts);
  auto start = get_time();
  for (size_t e = 0; e < epoch; e++) {
    std::random_shuffle(perm.begin(), perm.end());
    for (size_t i_ = 0; i_ < counts; i_++) {
      size_t i = perm[i_];
      const float *datum = data + i * size1 * size2;
      if (FLAGS_flush_cache)
        cacheflush(data);
      fun(datum, outarr, 0, size1, 0, size2, size2);
    }
  }
  auto end = get_time();
  if (FLAGS_test) {
    assert(epoch == 1);
    assert(counts == 1);
    float *outarr1 = NULL;
    make_float_data(&outarr1, size2);
    reducesum_impl_naive(data, outarr1, 0, size1, 0, size2, size2);
    bool error = false;
    for (size_t i = 0; i < size2; i++) {
      if (outarr1[i] != outarr[i])
        error = true;
    }
    if (error) {
      for (size_t i = 0; i < size2; i++) {
        std::cout << "a: " << outarr1[i] << "b: " << outarr[i] << std::endl;
      }
      exit(1);
    }
    std::free(outarr1);
  }
  if (outarr)
    std::free(outarr);
  return timespec_subtract_to_ns(&start, &end);
}

int main(int argc, char *argv[]) {
  gflags::SetUsageMessage("some usage message");
  gflags::SetVersionString("1.0.0");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (FLAGS_list_run_sums) {
    for (auto const &fun : register_sum_impls()) {
      std::cout << fun.first << std::endl;
    }
    return 0;
  }

  if (FLAGS_list_run_reducesums) {
    for (auto const &fun : register_reducesum_impls()) {
      std::cout << fun.first << std::endl;
    }
    return 0;
  }

  size_t size1 = (size_t)FLAGS_size1;
  size_t size2 = (size_t)FLAGS_size2;
  size_t counts =
      ((size_t)16 * (size_t)268435456) / (size1 * size2); // 16GiB of data
  assert(FLAGS_epoch > 0 || FLAGS_epoch == -1);
  size_t epoch = (size_t)FLAGS_epoch;
  if (epoch > counts) {
    epoch = counts;
  }
  assert(counts % epoch == 0);
  counts = counts / epoch;

  if (FLAGS_test) {
    counts = 1;
    epoch = 1;
  }

  assert(size1 >= _ALIGNMENT);
  assert(size2 >= _ALIGNMENT);
  assert(size1 % _ALIGNMENT == 0);
  assert(size2 % _ALIGNMENT == 0);

  float *data_ = NULL;
  make_float_data(&data_, counts * size1 * size2);

  make_vector(data_, size1 * size2 * counts);

  std::cout << "(size1:" << std::setw(8) << size1 << "),"
            << "(size2:" << std::setw(8) << size2 << "),"
            << "(counts:" << std::setw(8) << counts << "),"
            << "(epoch:" << std::setw(8) << epoch << ")";
  if (FLAGS_run_reducesum != "") {
    std::string funname = FLAGS_run_reducesum;
    auto funs = register_reducesum_impls();
    auto fun = funs[funname];
    std::cout << ",(fun: " << std::setw(25) << funname << "),";
    time_stats(reducesum_run(fun, data_, size1, size2, counts, epoch),
               size1 * size2 * counts * epoch);
  }
  if (FLAGS_run_sum != "") {
    std::string funname = FLAGS_run_sum;
    auto funs = register_sum_impls();
    auto fun = funs[funname];
    std::cout << ",(fun: " << std::setw(25) << funname << "),";
    time_stats(sum_run(fun, data_, size1 * size2, counts, epoch),
               size1 * size2 * counts * epoch);
  }
  std::cout << std::endl;
  std::free(data_);
  gflags::ShutDownCommandLineFlags();
}
