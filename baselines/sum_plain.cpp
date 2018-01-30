#include <time.h>
#include <vector>
#include <cstdint>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include "xmmintrin.h"
#include "immintrin.h"
#include <memory>
#include <numeric>
#include "tbb/tick_count.h"
#include "tbb/parallel_reduce.h"
#include "tbb/task_scheduler_init.h"
#include "tbb/blocked_range.h"
#include "tbb/tbb.h"

using namespace tbb;

constexpr size_t WIDTH = 16;

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


//TODO: Why won't instruction set show up in vtune? - compiler flag?

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

float sum_impl21 (float *arr, size_t start, size_t end)
{
  register int k;
  float sarr[8];
  register __m256 a, b, part_sum, tmp_sum;
  part_sum = _mm256_set1_ps(0);
  tmp_sum = _mm256_set1_ps(0);
  k = start;
  for (; k < end - 16; k += 16)
  {
    a = _mm256_loadu_ps(arr + k);
    b = _mm256_loadu_ps(arr + k + 8);
    tmp_sum = _mm256_add_ps (a, b);
    part_sum = _mm256_add_ps (part_sum, tmp_sum);
  }
  _mm256_store_ps(sarr, part_sum);
  float sum = sarr[0];
  for (int i = 1; i < 8; i++) {
    sum += sarr[i];
  }
  //  std::cerr << "end k: " << k << " end: " << end << std::endl;
  for (;k < end; k ++) {
  //  std::cerr << "k: " << k << std::endl;
    sum += arr[k];
  }
  return sum;
}

float sum_impl2(float* arr, size_t size) {
  assert(size % 2 == 0);
  float sum = 0;
  for (size_t i = 0; i < size; i += WIDTH) {
    float slocal = 0;
    for (size_t j = 0; j < WIDTH; j++) {
     slocal += arr[i + j];
    }
    sum += slocal;
  }
  return sum;
}
 
float sum_impl(float* arr, size_t size) {
  float sum = 0;
  for (size_t i = 0; i < size; i += 1) {
     sum += arr[i];
  }
  return sum;
}

class SumFoo {
  float *my_a;

public:
  float my_sum;
  void operator()(const blocked_range<size_t> &r) {
    float *a = my_a;
    size_t end = r.end();
    float sum = my_sum;
    sum += sum_impl21(a, r.begin(), r.end());
    // std::cerr << "r.begin(): " << r.begin() << std::endl;
    // std::cerr << "r.end(): " << r.end() << std::endl;
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
    parallel_reduce(blocked_range<size_t>(0, size, grain), sf);//, ap);
    return sf.my_sum;
}

float sum_impl_std(float* arr, size_t size) {
  float sum = std::accumulate(arr, arr + size, 0);
  return sum;
}

std::vector<float> make_vector(size_t size) {
  srand (1);
  std::vector<float> v;
  v.reserve(size);
  for (size_t i = 0; i < size; i++) {
    v[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    v[i] = v[i] - 0.5;
    // std::cerr << "v " << vector[i] << std::endl;
  }
  return v;
}

int main_run(float* data, int64_t size, int64_t counts, int64_t threads, size_t grain) {
  float all_sum = 0;
  std::cerr << "threads: " << threads;
  std::cerr << "\tgranularity: " << grain;
  tbb::tick_count mainBeginMark = tbb::tick_count::now();
  all_sum = 0;
  for (int64_t i = 0; i < counts; i++) {
    // std::cerr << "i: " << i << std::endl;
    all_sum += sum_impl_tbb(data, size, grain);//0);
    all_sum += sum_impl21(data, 0, size);
  }
  std::cout <<  "\ttime: "
            << (tbb::tick_count::now() - mainBeginMark).seconds()
            << "s";
  std::cerr << "\tsum: " << all_sum;
  std::cerr << std::endl;
}


int main() {
  int64_t size = 1000000;
  int64_t counts = 20000;
  // size *= 100;
  // counts /= 100;
  // counts = 1;

  int max_threads = tbb::task_scheduler_init::default_num_threads();
  auto v = make_vector(size);
  float* data = v.data();

  for (int64_t threads = 1; threads < max_threads; threads *= 2) {
    tbb::task_scheduler_init init(threads);
    for (size_t grain = 2; grain < 1000000; grain = grain * grain) {
      main_run(data, size, counts, threads, grain);
    }
  }
}
