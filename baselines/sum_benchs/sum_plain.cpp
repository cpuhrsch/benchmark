#include <time.h>
#include <stdexcept>
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
#include <gflags/gflags.h>

DEFINE_int64(size1, 1024, "size1 of matrix");
DEFINE_int64(size2, 1024, "size2 of matrix");
DEFINE_int64(counts, 50000, "number of executions");
DEFINE_bool(show_baseline, false, "Include baseline benchmark");
DEFINE_bool(debug, false, "Show Debug matrix");

using namespace tbb;

constexpr size_t WIDTH = 16;

void reducesum_impl2(const float* arr, float* outarr, size_t size1, size_t size2) {
    for (size_t j = 0; j < size2; j += 1) {
      outarr[j] = 0;
    }
    for (size_t j = 0; j < size2; j += 1) {
      for (size_t i = 0; i < size1; i += 1) {
        outarr[j] += arr[i * size2 + j];
      }
    }
}

void reducesum_impl111(const float *arr, float *outarr, size_t size1b, size_t size1e,
    size_t size2b, size_t size2e,
                      size_t size2) {
  // for (size_t j = 0; j < size2; j += 1) {
  //   outarr[j] = 0;
  // }
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

constexpr size_t _VSIZE = 8; //8 floats - 256bits - one fetch has 1024 bits
constexpr size_t _ROW = 8; //4; // chunk of columns per tile
constexpr size_t _COL = 8; //4; // chunk of vector row per tile

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
          tmp2[i1] =
              _mm256_load_ps(arr + (i + i1) * size2 + (j + j1) * _VSIZE);
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
  // for (size_t j = 0; j < size2; j += 1) {
  //  outarr[j] = 0;
  //}
  register size_t k;
  register __m256 a, b;
  k = size2b;
  size_t end = size2e > 7 ? size2e - 8 : 0;
  // std::cerr << "end k1: " << k << " end1: " << end << std::endl;
  for (; k < end; k += 8) {
    b = _mm256_loadu_ps(outarr + k);
    for (size_t i = size1b; i < size1e; i += 1) {
      // std::cerr << "avx!" << std::endl;
      // std::cerr << "k: " << k << std::endl;
      // std::cerr << "offset1: " << i * size2 + k << std::endl;
      a = _mm256_loadu_ps(arr + i * size2 + k);
      //_mm256_stream_ps(arr + i * size2 + k, a);
      // std::cerr << "offset2: " << i * size2 + k << std::endl;
      // std::cerr << "add" << std::endl;
      b = _mm256_add_ps(a, b);
      // std::cerr << "store" << std::endl;
    }
    _mm256_storeu_ps(outarr + k, b);
  }
  // std::cerr << "end k2 " << k << " end: " << end<< std::endl;

  for (; k < size2e; k++) {
    for (size_t i = size1b; i < size1e; i += 1) {
      outarr[k] += arr[i * size2 + k];
      // if (k == 0) {
      // std::cerr << "outarr[" << k << "] " << outarr[k];
      // std::cerr << " - arr[" << i << " * " << size2 << " + " << k << "] " <<
      // arr[i * size2 + k] << std::endl;
      // }
    }
  }
  // std::cerr << "end k3 " << k << " end: " << size2e<< std::endl;
}

void reducesum_impl(const float *arr, float *outarr, size_t size1b, size_t size1e,
                    size_t size2b, size_t size2e, size_t size2) {
  //for (size_t j = 0; j < size2; j += 1) {
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

    for (; k < size2e; k++){
      outarr[k] += arr[i * size2 + k];
      // if (k == 0) {
      // std::cerr << "outarr[" << k << "] " << outarr[k];
      // std::cerr << " - arr[" << i << " * " << size2 << " + " << k << "] " << arr[i * size2 + k] << std::endl;
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

float sum_impl21 (const float *arr, size_t start, size_t end)
{
  register size_t k;
  float sarr[8];
  register __m256 a, b, part_sum, tmp_sum;
  part_sum = _mm256_set1_ps(0);
  tmp_sum = _mm256_set1_ps(0);
  k = start;
  // std::cerr << "end k1: " << start << " end1: " << end << std::endl;
  // std::cerr << "end k: " << k << " end: " << end  - 16<< std::endl;
  size_t end_ = end > 15 ? end - 16 : 0;
  for (; k < end_; k += 16)
  {
    // std::cerr << "avx!" << std::endl;
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
 
float sum_impl(const float* arr, size_t size) {
  float sum = 0;
  for (size_t i = 0; i < size; i += 1) {
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
    // size_t end = r.end();
    float *sum = my_sum.data();
    size_t size2 = my_size2;
    // if (r.begin() == 0) {
    // std::cerr << "r.begin(): " << r.end() - r.begin() << std::endl;
    // }
    // std::cerr << "r.rows().begin(): " << r.rows().end() - r.rows().begin() << std::endl;
    // std::cerr << "r.cols().begin(): " << r.cols().end() - r.cols().begin() << std::endl;
    // reducesum_impl(a, sum, r.rows().begin(), r.rows().end(), r.cols().begin(), r.cols().end(), size2);
   //  reducesum_impl3(a, sum, r.rows().begin(), r.rows().end(), r.cols().begin(), r.cols().end(), size2);
    reducesum_impl111(a, sum, r.rows().begin(), r.rows().end(), r.cols().begin(), r.cols().end(), size2);
    // std::cerr << "sum: " << sum << std::endl;
    // for (size_t i = r.begin(); i != end; ++i)
    //   sum += a[i];
  }

  ReduceSumFoo(ReduceSumFoo &x, split) : my_a(x.my_a), my_size2(x.my_size2), my_sum(x.my_sum) {} //std::cerr << "split!" << std::endl;}

  void join(const ReduceSumFoo &y) {
    // std::cerr << "join!" << std::endl;
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
      // std::cerr << "my_sum[" << i << "] " << my_sum[i] << std::endl;
    }
    for (; k < my_size2; k += 1) {
      sum1[k] += sum2[k];
    }
  }

  ReduceSumFoo(const float a[], size_t size2) : my_a(a), my_size2(size2), my_sum(size2, 0.0f) {}
};

void reducesum_impl_tbb(const float *arr, float *outarr, size_t size1,
                      size_t size2, size_t grain) {
    ReduceSumFoo sf(arr, size2);
    // static affinity_partitioner ap;
    // parallel_reduce(blocked_range2d<size_t>(0, size1, grain, 0, size2, grain), sf);//, ap);
    parallel_reduce(blocked_range2d<size_t>(0, size1, grain, 0, size2, 64), sf);//, ap);
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
    parallel_reduce(blocked_range<size_t>(0, size, grain), sf);//, ap);
    return sf.my_sum;
}

float sum_impl_std(float* arr, size_t size) {
  float sum = std::accumulate(arr, arr + size, 0);
  return sum;
}

void make_vector(float* data, size_t size) {
  srand (1);
  for (size_t i = 0; i < size; i++) {
    data[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    data[i] = data[i] - 0.5;
    data[i] = i;
  }
}

int main_sum_run(const float* data, int64_t size, int64_t counts, int64_t threads, size_t grain) {
  float all_sum = 0;
  std::cerr << "threads: " << threads;
  std::cerr << "\tgranularity: " << grain;
  tbb::tick_count mainBeginMark = tbb::tick_count::now();
  all_sum = 0;
  for (int64_t i = 0; i < counts; i++) {
    // std::cerr << "i: " << i << std::endl;
    // all_sum += sum_impl_tbb(data, size, grain);//0);
    // all_sum += sum_impl(data, size);//, grain);//0);
    all_sum += sum_impl21(data, 0, size);
  }
  std::cout <<  "\ttime: "
            << (tbb::tick_count::now() - mainBeginMark).seconds()
            << "s";
  std::cerr << "\tsum: " << all_sum;
  std::cerr << std::endl;
  return 0;
}

constexpr size_t _ALIGNMENT = 32;

int main_reducesum_run(const float *data, size_t size1, size_t size2,
                       int64_t counts, int64_t threads, size_t grain) {
  tbb::tick_count mainBeginMark;
  std::cerr << "threads: " << threads;
  std::cerr << "\tgranularity: " << grain;

  void *dat_ptr = NULL;
  if (posix_memalign(&dat_ptr, _ALIGNMENT, size2 * sizeof(float)))
    throw std::invalid_argument( "received negative value" );
  // float* data_ = (float*) dat_ptr;
  // std::vector<float> result(size2);
  float *outarr = (float*) dat_ptr;
  mainBeginMark = tbb::tick_count::now();
  for (int64_t i = 0; i < counts; i++) {
    reducesum_impl3_tile(data, outarr, 0, size1, 0, size2, size2);
  }
  std::cout << "\ttime: " << (tbb::tick_count::now() - mainBeginMark).seconds()
            << "s";

 if (false) {
//  if (true) {
    std::cerr << std::endl;
    for (size_t j = 0; j < size2; j += 1) {
      std::cerr << "j: " << j << " " << outarr[j] << std::endl;
    }
    std::cerr << std::endl;
  }
  free(dat_ptr);

  if (FLAGS_show_baseline) {

    void *dat_ptr2 = NULL;
    if (posix_memalign(&dat_ptr2, _ALIGNMENT, size2 * sizeof(float)))
      throw std::invalid_argument("received negative value");
    float *outarr2 = (float *)dat_ptr2;
    mainBeginMark = tbb::tick_count::now();
    for (int64_t i = 0; i < counts; i++) {
          reducesum_impl11(data, outarr2, size1, size2); // baseline
    }
    std::cout << "\tbaseline time: "
              << (tbb::tick_count::now() - mainBeginMark).seconds() << "s";
    free(dat_ptr2);
  }

  std::cout << std::endl;
  return 0;
}

  int main(int argc, char *argv[]) {
    gflags::SetUsageMessage("some usage message");
      gflags::SetVersionString("1.0.0");
        gflags::ParseCommandLineFlags(&argc, &argv, true);
  // size_t size = 1000000; -- 1d bench -- size_t counts = 20000;
  size_t size1 = (size_t)FLAGS_size1;
  size_t size2 = (size_t)FLAGS_size2;
  size_t counts = (size_t)FLAGS_counts;
  // size *= 100;
  // counts /= 100;
//    counts = 1;
//    size1 = 32;
//    size2 = 32;
    assert(size1 >= _ALIGNMENT);
    assert(size2 >= _ALIGNMENT);
    assert(size1 % _ALIGNMENT == 0);
    assert(size2 % _ALIGNMENT == 0);
//  counts = 20000;
 //  size1 = 8192;
 //  size2 = 8192;

  int max_threads = tbb::task_scheduler_init::default_num_threads();
//  float* data_;
  void *dat_ptr = NULL;
  if (posix_memalign(&dat_ptr, _ALIGNMENT, size1 * size2 * sizeof(float)))
    throw std::invalid_argument( "received negative value" );
  float* data_ = (float*) dat_ptr;
  make_vector(data_, size1 * size2);
  const float* data = (const float*) dat_ptr;

 if (false) {
//  if (true) {
 for (size_t i = 0; i < size1; i++) {
   for (size_t j = 0; j < size2; j++) {
     std::cerr << " " << data[i * size2 + j];
   }
   std::cerr << std::endl;
 }
  }

  std::cerr << "_VSIZE: " << _VSIZE << " _COL: " << _COL << " _ROW: " << _ROW
    << " size1: " << size1 << " size2: " << size2 << " counts: " << counts << " ";
//            std::cerr << std::endl;
  for (int64_t threads = 1; threads < max_threads; threads *= 2) {
    // tbb::task_scheduler_init automatic(threads); // - segfaults?
    for (size_t grain = 1; grain < size1; grain = grain * 2) {
    //for (size_t grain = 512; grain < size1 * size2; grain = grain * 2) {
      main_reducesum_run(data, size1, size2, counts, threads, grain);
      break;
       main_sum_run(data, size1 * size2, counts, threads, grain);
    }
    break;
  }
    free(dat_ptr);
        gflags::ShutDownCommandLineFlags();

}
