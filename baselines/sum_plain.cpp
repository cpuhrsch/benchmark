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


constexpr size_t WIDTH = 16;

// float sum_impl4(float *arr, size_t size) {
//   assert(size % WIDTH == 0);
//   float sums[WIDTH];
//   for (size_t i = 0; i < WIDTH; i++) {
//     sums[i] = 0;
//   }
//   size_t is[WIDTH];
//   for (size_t i = 0; i < WIDTH; i++) {
//     is[i] = (size / WIDTH) * i;
//   }
//   for (size_t i = 0; i < size; i += WIDTH) {
//     for (size_t j = 0; j < WIDTH; j++) {
//       sums[j] += arr[is[j]];
//       is[j] += 1;
//     }
//   }
//   float sum = 0;
//   for (size_t i = 0; i < WIDTH; i++) {
//     sum += sums[i];
//   }
//   return sum;
// }
// 
// 
// //TODO: Why won't instruction set show up in vtune? - compiler flag?
// 
// float sum_impl31(float *arr, size_t size) {
//   assert(size % WIDTH == 0);
//   float sums[WIDTH];
//   for (size_t i = 0; i < WIDTH; i++) {
//     sums[i] = 0;
//   }
//   size_t is[WIDTH];
//   for (size_t i = 0; i < WIDTH; i++) {
//     is[i] = (size / WIDTH) * i;
//   }
//   for (size_t i = 0; i < size; i += WIDTH * WIDTH) {
//     for (size_t j = 0; j < WIDTH; j++) {
//       for (size_t k = 0; k < WIDTH; k++) {
//         sums[j] += arr[is[j] + k];
//       }
//       is[j] += WIDTH;
//     }
//   }
//   float sum = 0;
//   for (size_t i = 0; i < WIDTH; i++) {
//     sum += sums[i];
//   }
//   return sum;
// }

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

float sum_impl21 (float *arr, size_t size)
{
  register int limit = (size / 32) * 32;
  register int k;
  float sarr[8];
  register __m256 a, b, part_sum, tmp_sum;
  register __m256 a1, b1, part_sum1, tmp_sum1;
  // register __m256 a, b, part_sum, tmp_sum;
  part_sum = _mm256_set1_ps(0);
  tmp_sum = _mm256_set1_ps(0);
  part_sum1 = _mm256_set1_ps(0);
  tmp_sum1 = _mm256_set1_ps(0);
  // std::cerr << "size: " << size << std::endl;
  k = 0;
  for (; k < limit; k += 32)
  {
    // std::cerr << "k: " << k << std::endl;
    // std::cerr << "k1: " << k1 << std::endl;
    a = _mm256_loadu_ps(arr + k);
    b = _mm256_loadu_ps(arr + k + 8);
    tmp_sum = _mm256_add_ps (a, b);
    part_sum = _mm256_add_ps (part_sum, tmp_sum);

    a1 = _mm256_loadu_ps(arr + k + 16);
    b1 = _mm256_loadu_ps(arr + k + 24);
    tmp_sum1 = _mm256_add_ps (a1, b1);
    part_sum1 = _mm256_add_ps (part_sum1, tmp_sum1);
  }
  _mm256_store_ps(sarr, part_sum);
  float sum = sarr[0] + sarr[1] + sarr[2] + sarr[3];
  sum += sarr[4] + sarr[5] + sarr[6] + sarr[7];
  _mm256_store_ps(sarr, part_sum1);
  sum += sarr[0] + sarr[1] + sarr[2] + sarr[3];
  sum += sarr[4] + sarr[5] + sarr[6] + sarr[7];
  k = k * 2;
  for (;k < size; k ++) {
    sum += arr[k];
  }
  return sum;
}

// float sum_impl2(float* arr, size_t size) {
//   assert(size % 2 == 0);
//   float sum = 0;
//   for (size_t i = 0; i < size; i += WIDTH) {
//     float slocal = 0;
//     for (size_t j = 0; j < WIDTH; j++) {
//      slocal += arr[i + j];
//     }
//     sum += slocal;
//   }
//   return sum;
// }
// 
float sum_impl(float* arr, size_t size) {
  float sum = 0;
  for (size_t i = 0; i < size; i += 1) {
     sum += arr[i];
  }
  return sum;
}

void make_vector(float* arr, size_t size, float count) {
  srand (1);
  for (size_t i = 0; i < size; i++) {
    arr[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    arr[i] = arr[i] - 0.5;
    // std::cerr << "v " << vector[i] << std::endl;
  }
}

int main() {
  int64_t size = 1000000;
  // int64_t size = 10000;
  int64_t counts = 20000;
  // int64_t counts = 200000;
  // int64_t size = 1000;
  // int64_t counts = 1;
  size = (size / 32 + 1) * 32;
  size_t dsize = sizeof(float) * size;
  std::cerr << "dsize: " << size << std::endl;
  float* data = (float*)aligned_alloc(32, dsize);
  make_vector(data, size, (float)counts);
  std::cout << "allocated an int at " << (void*)data << '\n';
  std::cerr << "aligned: " << ((unsigned long)data & 31) << std::endl;
  float all_sum = 0;

  all_sum = 0;
  for (int64_t i = 0; i < counts; i++) {
    // std::cerr << "i: " << i << std::endl;
    all_sum += sum_impl21(data, size);
  }
  std::cerr << "sum_impl21: " << all_sum << std::endl;
  free(data);

  // all_sum = 0;
  // for (int64_t i = 0; i < counts; i++) {
  //   all_sum += sum_impl2(data, size);
  // }
  // std::cerr << "sum_impl2: " << all_sum << std::endl;
}
