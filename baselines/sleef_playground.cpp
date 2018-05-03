#include <gflags/gflags.h>

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
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <sleef.h>

DEFINE_int64(size1, 1024, "size1 of matrix");
DEFINE_int64(size2, 1024, "size2 of matrix");
DEFINE_int64(epoch, 1, "number of runs with cache");
DEFINE_bool(test, false, "Show Debug matrix");
DEFINE_bool(run_sleef, false, "run using sleef");

constexpr size_t _ALIGNMENT = 32;

void time_stats(uint64_t s, double floats) {
  std::cout << "(ns:\033[36m" << std::fixed << std::setw(15) << s << "\033[0m)"
            << ",(ops/ns: "
            << "\033[31m";
  std::cout << std::setw(12) << (double)floats / (double)s << "\033[0m)";
  std::cout << ",(s: " << s / (double)NSEC << ")";
}

void make_vector(float *data_, uint64_t size) {
  for (size_t i = 0; i < size; i++) {
    data_[i] = (float)(i % 1024);
  }
}


void make_float_data(float **data_, size_t size) {
  if (posix_memalign((void **)data_, _ALIGNMENT, size * sizeof(float)))
    throw std::invalid_argument("received negative value");
  memset(*data_, 0, size * sizeof(float));
}

void make_test_float_data(float **data_) {
  uint64_t size = std::numeric_limits<uint64_t>::max();
  if (posix_memalign((void **)data_, _ALIGNMENT, size * sizeof(float)))
    throw std::invalid_argument("received negative value");
}

// Based on: http://en.cppreference.com/w/cpp/types/numeric_limits/epsilon
template<class T>
typename std::enable_if<!std::numeric_limits<T>::is_integer, bool>::type
    almost_equal(T x, T y, int ulp)
{
    // the machine epsilon has to be scaled to the magnitude of the values used
    // and multiplied by the desired precision in ULPs (units in the last place)
    return std::abs(x-y) <= std::numeric_limits<T>::epsilon() * std::abs(x+y) * ulp
    // unless the result is subnormal
           || std::abs(x-y) < std::numeric_limits<T>::min();
}

void _test() {
#ifdef __AVX2__
  uint64_t disagreements = 0;
  float wrong = 0;
  float wronga = 0;
  float wrongb = 0;
  float wrongmax = std::numeric_limits<float>::min();
  float wrongmin = std::numeric_limits<float>::max();
  float wrongminabs = std::numeric_limits<float>::max();
  bool go_on = true;
#pragma omp parallel for reduction(+ : disagreements) reduction(min : wrongmin)  reduction(max : wrongmax) reduction(min : wrongminabs)
  for (uint32_t i = 0; i < std::numeric_limits<uint32_t>::max(); i += 1) {
    if (go_on) {
      float input = *reinterpret_cast<float *>(&i);
      __m256 data = _mm256_set1_ps(input);
      data = Sleef_sinf8_u05(data);
      float ref = std::sin(input);
      float ref_d[8];
      _mm256_storeu_ps(ref_d, data);
      //    if (std::abs(ref - ref_d[0]) >
      //    std::numeric_limits<float>::epsilon())
      //    {
      if (!almost_equal(ref, ref_d[0], 1)) {
        disagreements++;
//#pragma omp critical
        {
          //      std::cout << "i: " << i << " - ref: " << ref << " input: " <<
          //      input
          //                << " ref_d[0]: " << ref_d[0];
          //      std::cout << std::endl;
//          wrong = input;
//          wronga = ref_d[0];
//          wrongb = ref;
          wrongmax = std::max(input, wrongmax);
          wrongmin = std::min(input, wrongmax);
          wrongminabs = std::min(wrongminabs, std::abs(input));
          // go_on = false;
        }
      }
    }
  }
  std::cout << "disagreements: " << disagreements << "/ "
            << std::numeric_limits<uint32_t>::max() << std::endl;
  std::cout << "wrong: " << wrong << " wronga: " << wronga
            << " wrongb: " << wrongb << std::endl;
  std::cout << "wrongmax: " << wrongmax
            << " wrongminabs: " << wrongminabs
            << " wrongmin: " << wrongmin << std::endl;
#else
  throw std::invalid_argument("need avx");
#endif
}

void _bench(size_t numel) {

  float *data_ = NULL;
  float *data_out_ = NULL;
  make_float_data(&data_, numel);
  make_float_data(&data_out_, numel);
  make_vector(data_, numel);

  // A ULP of 1.0 is enough based on:
  // https://www.gnu.org/software/libc/manual/html_node/Errors-in-Math-Functions.html
  auto start = get_time();
  size_t i = 0;
#ifdef __AVX2__
  if (FLAGS_run_sleef) {
    __m256 *data__ = (__m256 *)(data_);
    __m256 *data__out_ = (__m256 *)(data_out_);
    for (; i < numel / 8; i++) {
      std::cout << "1data_[" << i << "]: " << data_[i] << "\t-\t";
      data__out_[i] = Sleef_sinf8_u10(data__[i]);
      std::cout << "2data_[" << i << "]: " << data_out_[i] << std::endl;
    }
    i = i * 8;
  }
#endif
  for (; i < numel; i += 1) {
    data_out_[i] = std::sin(data_[i]);
  }
  auto end = get_time();
  time_stats(timespec_subtract_to_ns(&start, &end), numel);
  std::cout << std::endl;
  if (FLAGS_test) {
    for (i = 0; i < numel; i += 1) {
      if (!almost_equal(data_out_[i], std::sin(data_[i]), 1.0)) {
        std::cout << "ERROR!" << std::endl;
        std::cout << "data_[" << i << "]: " << data_[i] << std::endl;
        std::cout << "data_out_[" << i << "]: " << data_out_[i] << std::endl;
        std::cout << "std::sin(data_[" << i << "]): " << std::sin(data_[i]);
        std::cout << "(data_out_[" << i << "] - std::sin(data_[" << i
                  << "])): " << std::abs(data_out_[i] - std::sin(data_[i]));
        std::cout << std::endl;
      }
    }
  }
}


int main(int argc, char *argv[]) {
  gflags::SetUsageMessage("some usage message");
  gflags::SetVersionString("1.0.0");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  size_t size1 = (size_t)FLAGS_size1;
  size_t size2 = (size_t)FLAGS_size2;
  size_t counts =
      ((size_t)1 * (size_t)268435456) / (size1 * size2); // 16GiB of data
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

  // size_t numel = counts * size1 * size2;
  _test();
}
