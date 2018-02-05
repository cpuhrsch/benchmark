#include "immintrin.h"
#include "tbb/blocked_range.h"
#include "tbb/parallel_reduce.h"
#include "tbb/task_scheduler_init.h"
#include "tbb/tbb.h"
#include "tbb/tick_count.h"
#include "xmmintrin.h"
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <gflags/gflags.h>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <time.h>
#include <vector>
#include <iomanip>

DEFINE_int64(size1, 1024, "size1 of matrix");
DEFINE_int64(size2, 1024, "size2 of matrix");
DEFINE_int64(t, 1, "number of threads");
DEFINE_int64(grain, 1, "granularity");
DEFINE_bool(show_baseline, false, "Include baseline benchmark");
DEFINE_bool(debug, false, "Show Debug matrix");
DEFINE_bool(run_sum, false, "run_sum");
DEFINE_bool(run_reducesum, false, "run_reducesum");

using namespace tbb;

#define USEC    1000000
#define NSEC    1000000000

/**
 * us_to_timespec - converts microseconds to a timespec
 * @us: number of microseconds
 * @t: the storage timespec
 */
static inline void us_to_timespec(uint64_t us, struct timespec *t)
{
    t->tv_sec = us / USEC;
    t->tv_nsec = (us - t->tv_sec * USEC) * (NSEC / USEC);
}

/**
 * timespec_to_us - converts a timespec to microseconds
 * @t: the timespec
 *
 * Returns microseconds.
 */
static inline uint64_t timespec_to_us(struct timespec *t)
{
    return t->tv_sec * USEC + t->tv_nsec / (NSEC / USEC);
}

/**
 * timespec_to_ns - converts a timespec to nanoseconds
 * @t: the timespec
 *
 * Returns nanoseconds.
 */
static inline uint64_t timespec_to_ns(struct timespec *t)
{
    return t->tv_sec * NSEC + t->tv_nsec;
}

/**
 * timespec_subtract - subtracts timespec y from timespec x
 * @x, @y: the timespecs to subtract
 * @result: a pointer to store the answer
 *
 * WARNING: It's not safe for @result to be @x or @y.
 *
 * Returns 1 if the difference is negative, otherwise 0.
 */
int timespec_subtract(struct timespec *x, struct timespec *y,
                 struct timespec *result)
{
    if (x->tv_nsec < y->tv_nsec) {
        int secs = (y->tv_nsec - x->tv_nsec) / NSEC + 1;
        y->tv_nsec -= NSEC * secs;
        y->tv_sec += secs;
    }

        if (x->tv_nsec - y->tv_nsec > NSEC) {
                int secs = (x->tv_nsec - y->tv_nsec) / NSEC;
                y->tv_nsec += NSEC * secs;
                y->tv_sec -= secs;
    }

    result->tv_sec = x->tv_sec - y->tv_sec;
    result->tv_nsec = x->tv_nsec - y->tv_nsec;

    return x->tv_sec < y->tv_sec;
}

constexpr size_t _ALIGNMENT = 32;

constexpr size_t WIDTH = 16;

void reducesum_impl2(const float *arr, float *outarr, size_t size1,
                     size_t size2) {
  for (size_t j = 0; j < size2; j += 1) {
    outarr[j] = 0;
  }
  for (size_t j = 0; j < size2; j += 1) {
    for (size_t i = 0; i < size1; i += 1) {
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

void reducesum_impl11(const float *arr, float *outarr, size_t size1,
                      size_t size2) {
  for (size_t j = 0; j < size2; j += 1) {
    outarr[j] = 0;
  }
  for (size_t i = 0; i < size1; i += 1) {
    for (size_t j = 0; j < size2; j += 1) {
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

void reducesum_impl3(const float *arr, float *outarr, size_t size1b,
                     size_t size1e, size_t size2b, size_t size2e,
                     size_t size2) {
  register size_t k;
  k = size2b;
  size_t end = size2e > 7 ? size2e - 8 : 0;
  for (; k < end; k += 8 * 8) {
    __m256 a[4];
    __m256 b[4];
    for (int ib = 0; ib < 4; ib++) {
      b[ib] = _mm256_loadu_ps(outarr + k + ib * 8);
    }
    for (size_t i = size1b; i < size1e; i += 1) {
      for (int ib = 0; ib < 4; ib++) {
        a[ib] = _mm256_loadu_ps(arr + i * size2 + k + ib * 8);
        b[ib] = _mm256_add_ps(a[ib], b[ib]);
      }
    }
    for (int ib = 0; ib < 4; ib++) {
      _mm256_storeu_ps(outarr + k + ib * 8, b[ib]);
    }
  }
}

void reducesum_impl(const float *arr, float *outarr, size_t size1b,
                    size_t size1e, size_t size2b, size_t size2e, size_t size2) {
  // for (size_t j = 0; j < size2; j += 1) {
  //  outarr[j] = 0;
  //}
  register size_t k;
  for (size_t i = size1b; i < size1e; i += 1) {
    k = size2b;
    size_t end = size2e > 7 ? size2e - 8 : 0;
    // std::cerr << "end k1: " << k << " end1: " << end << std::endl;
    for (; k < end; k += 8) {
      register __m256 a, b;
      // std::cerr << "avx!" << std::endl;
      // std::cerr << "k: " << k << std::endl;
      // std::cerr << "offset1: " << i * size2 + k << std::endl;
      a = _mm256_loadu_ps(arr + i * size2 + k);
      // std::cerr << "offset2: " << i * size2 + k << std::endl;
      b = _mm256_loadu_ps(outarr + k);
      // std::cerr << "add" << std::endl;
      a = _mm256_add_ps(a, b);
      // std::cerr << "store" << std::endl;
      _mm256_storeu_ps(outarr + k, a);
    }
    // std::cerr << "end k2 " << k << " end: " << end<< std::endl;

    for (; k < size2e; k++) {
      outarr[k] += arr[i * size2 + k];
      // if (k == 0) {
      // std::cerr << "outarr[" << k << "] " << outarr[k];
      // std::cerr << " - arr[" << i << " * " << size2 << " + " << k << "] " <<
      // arr[i * size2 + k] << std::endl;
      // }
    }
    // std::cerr << "end k3 " << k << " end: " << size2e<< std::endl;
  }
}

float sum_impl4(float *arr, size_t size) {
  assert(size % WIDTH == 0);
  float sums[WIDTH];
  for (size_t i = 0; i < WIDTH; i++) {
    sums[i] = 0;
  }
  size_t is[WIDTH];
  for (size_t i = 0; i < WIDTH; i++) {
    is[i] = (size / WIDTH) * i;
  }
  for (size_t i = 0; i < size; i += WIDTH) {
    for (size_t j = 0; j < WIDTH; j++) {
      sums[j] += arr[is[j]];
      is[j] += 1;
    }
  }
  float sum = 0;
  for (size_t i = 0; i < WIDTH; i++) {
    sum += sums[i];
  }
  return sum;
}

// TODO: Why won't instruction set show up in vtune? - compiler flag?

float sum_impl31(float *arr, size_t size) {
  assert(size % WIDTH == 0);
  float sums[WIDTH];
  for (size_t i = 0; i < WIDTH; i++) {
    sums[i] = 0;
  }
  size_t is[WIDTH];
  for (size_t i = 0; i < WIDTH; i++) {
    is[i] = (size / WIDTH) * i;
  }
  for (size_t i = 0; i < size; i += WIDTH * WIDTH) {
    for (size_t j = 0; j < WIDTH; j++) {
      for (size_t k = 0; k < WIDTH; k++) {
        sums[j] += arr[is[j] + k];
      }
      is[j] += WIDTH;
    }
  }
  float sum = 0;
  for (size_t i = 0; i < WIDTH; i++) {
    sum += sums[i];
  }
  return sum;
}

float sum_impl3(float *arr, size_t size) {
  assert(size % WIDTH == 0);
  float sums[WIDTH];
  for (size_t i = 0; i < WIDTH; i++) {
    sums[i] = 0;
  }
  size_t is[WIDTH];
  for (size_t i = 0; i < WIDTH; i++) {
    is[i] = (size / WIDTH) * i;
  }
  for (size_t i = 0; i < size; i += WIDTH) {
    for (size_t j = 0; j < WIDTH; j++) {
      sums[j] += arr[is[j]];
      is[j] += 1;
    }
  }
  float sum = 0;
  for (size_t i = 0; i < WIDTH; i++) {
    sum += sums[i];
  }
  return sum;
}

// TODO: Look at ns per flop
// TODO: Play with TBB to have repeated runs give same results - same order for
// merges no matter the num of threads

float sum_impl21(const float *arr, size_t start, size_t end) {
  register size_t k;
  float sarr[8];
  __m256 part_sum, tmp_sum[4], tmp_sum1, tmp_sum2, tmp_sum3, a[8];
  part_sum = _mm256_set1_ps(0);
  k = start;
  // std::cerr << "end k1: " << start << " end1: " << end << std::endl;
  // std::cerr << "end k: " << k << " end: " << end  - 16<< std::endl;
  assert(end % 64 == 0);
  for (; k < end; k += 64) {
    // std::cerr << "avx!" << std::endl;
    for (size_t i = 0; i < 8; i++) {
      a[i] = _mm256_loadu_ps(arr + k + i * 8);
    }
    // This tree reduction works better than adding each to part_sum
    for (size_t i = 0; i < 8; i += 2) {
      tmp_sum[i / 2] = _mm256_add_ps(a[i], a[i + 1]);
    }
    tmp_sum1 = _mm256_add_ps(tmp_sum[0], tmp_sum[1]);
    tmp_sum2 = _mm256_add_ps(tmp_sum[2], tmp_sum[3]);
    tmp_sum3 = _mm256_add_ps(tmp_sum1, tmp_sum2);
    part_sum = _mm256_add_ps(part_sum, tmp_sum3);
  }
  _mm256_store_ps(sarr, part_sum);
  float sum = sarr[0];
  for (int i = 1; i < 8; i++) {
    sum += sarr[i];
  }
  return sum;
}

float sum_impl21_fma(const float *arr, size_t start, size_t end) {
  register size_t k;
  float sarr[8];
  __m256 part_sum, tmp_sum[4], tmp_sum1, tmp_sum2, tmp_sum3, a[8], useless;
  part_sum = _mm256_set1_ps(0);
  useless = _mm256_set1_ps(0);
  k = start;
  // std::cerr << "end k1: " << start << " end1: " << end << std::endl;
  // std::cerr << "end k: " << k << " end: " << end  - 16<< std::endl;
  assert(end % 64 == 0);
  for (; k < end; k += 64) {
    // std::cerr << "avx!" << std::endl;
    for (size_t i = 0; i < 8; i++) {
      a[i] = _mm256_loadu_ps(arr + k + i * 8);
    }
    // This tree reduction works better than adding each to part_sum
    for (size_t i = 0; i < 8; i += 2) {
      tmp_sum[i / 2] = _mm256_add_ps(a[i], a[i + 1]);
    }
    tmp_sum1 = _mm256_fmadd_ps(useless, tmp_sum[0], tmp_sum[1]);
    tmp_sum2 = _mm256_fmadd_ps(useless, tmp_sum[2], tmp_sum[3]);
    tmp_sum3 = _mm256_fmadd_ps(useless, tmp_sum1, tmp_sum2);
    part_sum = _mm256_fmadd_ps(useless, part_sum, tmp_sum3);
  }
  _mm256_store_ps(sarr, part_sum);
  float sum = sarr[0];
  for (int i = 1; i < 8; i++) {
    sum += sarr[i];
  }
  return sum;
}

float sum_impl2(float *arr, size_t start, size_t end) {
  assert(end % WIDTH == 0);
  assert(start % WIDTH == 0);
  float sum = 0;
  for (size_t i = start; i < end; i += WIDTH) {
    float slocal = 0;
    for (size_t j = 0; j < WIDTH; j++) {
      slocal += arr[i + j];
    }
    sum += slocal;
  }
  return sum;
}

float sum_impl_naive(const float *arr, size_t start, size_t end) {
  float sum = 0;
  for (size_t i = start; i < end; i += 1) {
    sum += arr[i];
  }
  return sum;
}

class ReduceSumFoo {
  const float *my_a;
  size_t my_size2;

public:
  std::vector<float> my_sum;
  void operator()(const blocked_range2d<size_t> &r) {
    const float *a = my_a;
    float *sum = my_sum.data();
    size_t size2 = my_size2;
    reducesum_impl_naive(a, sum, r.rows().begin(), r.rows().end(),
                      r.cols().begin(), r.cols().end(), size2);
  }

  ReduceSumFoo(ReduceSumFoo &x, split)
      : my_a(x.my_a), my_size2(x.my_size2), my_sum(x.my_sum) {
  }

  void join(const ReduceSumFoo &y) {
    float *sum1 = my_sum.data();
    const float *sum2 = y.my_sum.data();
    register size_t k;
    size_t end = my_size2 > 7 ? my_size2 - 8 : 0;
    k = 0;
    for (; k < end; k += 8) {
      register __m256 a, b;
      a = _mm256_loadu_ps(sum1 + k);
      b = _mm256_loadu_ps(sum2 + k);
      a = _mm256_add_ps(a, b);
      _mm256_storeu_ps(sum1 + k, a);
    }
    for (; k < my_size2; k += 1) {
      sum1[k] += sum2[k];
    }
  }

  ReduceSumFoo(const float a[], size_t size2)
      : my_a(a), my_size2(size2), my_sum(size2, 0.0f) {}
};

void reducesum_impl_tbb(const float *arr, float *outarr, size_t size1,
                        size_t size2, size_t grain) {
  ReduceSumFoo sf(arr, size2);
  // static affinity_partitioner ap;
  // parallel_reduce(blocked_range2d<size_t>(0, size1, grain, 0, size2, grain),
  // sf);//, ap);
  parallel_reduce(blocked_range2d<size_t>(0, size1, grain, 0, size2, 64),
                  sf); //, ap);
  for (size_t i = 0; i < size2; i++) {
    outarr[i] = sf.my_sum[i];
  }
}

class SumFoo {
  float *my_a;

public:
  float my_sum;
  void operator()(const blocked_range<size_t> &r) {
    float *a = my_a;
    // size_t end = r.end();
    float sum = my_sum;
    // if (r.begin() == 0) {
    // std::cerr << "r.begin(): " << r.end() - r.begin() << std::endl;
    // }
    sum += sum_impl21(a, r.begin(), r.end());
    // std::cerr << "sum: " << sum << std::endl;
    // for (size_t i = r.begin(); i != end; ++i)
    //   sum += a[i];
    my_sum = sum;
  }

  SumFoo(SumFoo &x, split) : my_a(x.my_a), my_sum(0) {}

  void join(const SumFoo &y) { my_sum += y.my_sum; }

  SumFoo(float a[]) : my_a(a), my_sum(0) {}
};

float sum_impl_tbb(float *a, size_t size, size_t grain) {
  SumFoo sf(a);
  // static affinity_partitioner ap;
  parallel_reduce(blocked_range<size_t>(0, size, grain), sf); //, ap);
  return sf.my_sum;
}

float sum_impl_std(float *arr, size_t size) {
  float sum = std::accumulate(arr, arr + size, 0);
  return sum;
}

constexpr size_t _SEED = 1;

static inline uint16_t __mm_crc32_u64(uint64_t crc, uint64_t val)
{
    asm("crc32q %1, %0" : "+r" (crc) : "rm" (val));
    return crc;
}

unsigned int sfrand(unsigned int x)
{
    unsigned int seed = x * 16807;
    return ( (seed)>>9 ) | 0x40000000;
}

static inline size_t hash(size_t x) {
  size_t h = 2166136261;
  h = h ^ x;
  h = h * 16777619;
  return h;
}

std::vector<size_t> rand_perm(size_t max) {
  std::vector<size_t> perm;
  perm.reserve(max);
  for (size_t i = 0; i < max; i++) {
    size_t randi = rand() % max;
    while(std::find(perm.begin(), perm.end(), randi) != perm.end()) {
      randi = rand() % max;
    }
    perm.push_back(randi);
  }
  return perm;
}

//TODO: Random indexing into data
//TODO: use clock monotonic - make sure it's mapping to a good function

constexpr size_t _HASHES_SIZE = 1024;

void make_vector(float *data_, size_t size) {
  srand(1);
  std::vector<size_t> hashes(_HASHES_SIZE);
  size_t start = rand();
  for (size_t c = 0; c < _HASHES_SIZE; c++) {
    hashes[c] = hash(start + c);
  }
  //  parallel_for(blocked_range<size_t>(0, size, 10000000),
  //               [&](const blocked_range<size_t> &r) {
  for (size_t c = 0; c < size; c++) {
    start = (start + hashes[c % _HASHES_SIZE]) % RAND_MAX;
    data_[c] = ((float)(start) / (float)(RAND_MAX)) - 0.5f;
    //                   data_[c] = c;
  }
  //               });
}

void time_stats(uint64_t s, double floats){
  std::cerr << "ns:\033[36m" << std::fixed << std::setw(11) << s << "\033[0m"
            << " ops/ns: "
            << "\033[31m";
  std::cerr << (double)floats / (double)s << "\033[0m";
  std::cerr << " s: " << s / (double)NSEC;
}

uint64_t sum_run(bool baseline, const float * &data, int64_t size,
        size_t counts, int64_t threads, size_t grain) {
  (void)threads;
  (void)grain;
  int ret;
  float all_sum = 0;
  struct timespec start, finish, delta;
  uint64_t delta_ns = 0;
  auto perm = rand_perm(counts);
  for (size_t i_ = 0; i_ < counts; i_++) {
    size_t i = perm[i_];
    const float *datum = data + i * size;
    ret = clock_gettime(CLOCK_MONOTONIC, &start);
    if (ret == -1) {
      perror("clock_gettime()");
      exit(1);
    }
    if (baseline) {
      all_sum += sum_impl_naive(datum, 0, size);
    } else {
      all_sum += sum_impl21(datum, 0, size);
    }
    ret = clock_gettime(CLOCK_MONOTONIC, &finish);
    if (ret == -1) {
      perror("clock_gettime()");
      exit(1);
    }
    if (timespec_subtract(&finish, &start, &delta)) {
        fprintf(stderr, "clock not monotonic???\n");
        // ezyang: Exit here?
    }
    delta_ns += timespec_to_ns(&delta);
  }

  if (FLAGS_debug) {
    if (baseline) {
      std::cerr << " baseline_";
    }
    std::cerr << "sum: " << all_sum;
  }
  return delta_ns;
}

int main_sum_run(const float *data, int64_t size, size_t counts,
                 int64_t threads, size_t grain) {
  std::cerr << " sum: ";
  time_stats(sum_run(false, data, size, counts, threads, grain), size*counts);
  if (FLAGS_show_baseline) {
    std::cerr << " - ";
    time_stats(sum_run(true, data, size, counts, threads, grain), size*counts);
  }
  return 0;
}

uint64_t reducesum_run(bool baseline, const float *data, size_t size1,
                     size_t size2, size_t counts, int64_t threads,
                     size_t grain) {
  (void)threads;
  (void)grain;
  int ret;
  void *dat_ptr = NULL;
  if (posix_memalign(&dat_ptr, _ALIGNMENT, size2 * sizeof(float)))
    throw std::invalid_argument("received negative value");
  float *outarr = (float *)dat_ptr;
  memset(outarr, 0, size2 * sizeof(float));
  auto perm = rand_perm(counts);

  struct timespec start, finish, delta;
  uint64_t delta_ns = 0;
  for (size_t i_ = 0; i_ < counts; i_++) {
    size_t i = perm[i_];
    const float *datum = data + i * size1 * size2;
    ret = clock_gettime(CLOCK_MONOTONIC, &start);
    if (ret == -1) {
      perror("clock_gettime()");
      exit(1);
    }
    if (baseline) {
      reducesum_impl_naive(datum, outarr, 0, size1, 0, size2, size2);
    } else {
      reducesum_impl3(datum, outarr, 0, size1, 0, size2, size2);
    }
    ret = clock_gettime(CLOCK_MONOTONIC, &finish);
    if (ret == -1) {
      perror("clock_gettime()");
      exit(1);
    }
    if (timespec_subtract(&finish, &start, &delta)) {
        fprintf(stderr, "clock not monotonic???\n");
        // ezyang: Exit here?
    }
    delta_ns += timespec_to_ns(&delta);
  }

  if (FLAGS_debug) {
    std::cerr << std::endl;
    for (size_t j = 0; j < size2; j += 1) {
      std::cerr << "j: " << j << " " << outarr[j] << std::endl;
    }
    std::cerr << std::endl;
  }
  free(dat_ptr);
  return delta_ns;
}

int main_reducesum_run(const float *data, size_t size1, size_t size2,
                       size_t counts, int64_t threads, size_t grain) {
  std::cerr << " reduce_time: ";
  time_stats(reducesum_run(false, data, size1, size2, counts, threads, grain), size1*size2*counts);
  if (FLAGS_show_baseline) {
    std::cerr << " - ";
    time_stats(reducesum_run(true, data, size1, size2, counts, threads, grain), size1*size2*counts);
  }
  return 0;
}


// size_t size = 1000000; -- 1d bench -- size_t counts = 20000;
int main(int argc, char *argv[]) {
  gflags::SetUsageMessage("some usage message");
  gflags::SetVersionString("1.0.0");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  size_t size1 = (size_t)FLAGS_size1;
  size_t size2 = (size_t)FLAGS_size2;
  size_t counts = ((size_t)16 * (size_t)268435456) / (size1 * size2); // 16GiB of data

  if (FLAGS_debug) {
    counts = 1;
    size1 = 32;
    size2 = 32;
  }
  assert(size1 >= _ALIGNMENT);
  assert(size2 >= _ALIGNMENT);
  assert(size1 % _ALIGNMENT == 0);
  assert(size2 % _ALIGNMENT == 0);

  void *dat_ptr = NULL;
  if (posix_memalign(&dat_ptr, _ALIGNMENT,
                     counts * size1 * size2 * sizeof(float)))
    throw std::invalid_argument("received negative value");
  float *data_ = (float *)dat_ptr;

  int max_threads = tbb::task_scheduler_init::default_num_threads();
  assert(FLAGS_t <= max_threads);
  //  float* data_;
  make_vector(data_, size1 * size2 * counts);

  if (FLAGS_debug) {
    for (size_t i = 0; i < size1; i++) {
      for (size_t j = 0; j < size2; j++) {
        std::cerr << " " << data_[i * size2 + j];
      }
      std::cerr << std::endl;
    }
  }

  std::cerr << "size1: " << std::setw(8) << size1  << " "
            << "size2: " << std::setw(8) << size2 << " "
            << "counts: " << std::setw(8) << counts << " ";
  std::cerr << "threads: " << std::setw(8) << FLAGS_t;
  std::cerr << " granularity: " << std::setw(8) << FLAGS_grain;
  std::cerr << "\t";
  tbb::task_scheduler_init automatic(FLAGS_t);
  if (FLAGS_run_reducesum) {
    main_reducesum_run(data_, size1, size2, counts, FLAGS_t, FLAGS_grain);
  }
  if (FLAGS_run_sum) {
    main_sum_run(data_, size1 * size2, counts, FLAGS_t, FLAGS_grain);
  }
  std::cerr << std::endl;
  free(dat_ptr);
  gflags::ShutDownCommandLineFlags();
}
