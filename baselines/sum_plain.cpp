#include <time.h>
#include <vector>
#include <cstdint>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>


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
// float sum_impl(float* arr, size_t size) {
//   float sum = 0;
//   for (size_t i = 0; i < size; i += 1) {
//      sum += arr[i];
//   }
//   return sum;
// }

std::vector<float> make_vector(size_t size, float count) {
  srand (1);
  std::vector<float> vector;
  vector.reserve(size);
  for (size_t i = 0; i < size; i++) {
    float salt = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    vector.push_back(salt - 0.5);
    // std::cerr << "v " << vector[i] << std::endl;
  }
  return vector;
}

int main() {
  int64_t size = 1000000;
  int64_t counts = 20000;
  auto vector = make_vector(size, (float)counts);
  float* data = vector.data();
  float all_sum = 0;

  all_sum = 0;
  for (int64_t i = 0; i < counts; i++) {
    all_sum += sum_impl3(data, size);
  }
  std::cerr << "sum_impl3: " << all_sum << std::endl;

  // all_sum = 0;
  // for (int64_t i = 0; i < counts; i++) {
  //   all_sum += sum_impl2(data, size);
  // }
  // std::cerr << "sum_impl2: " << all_sum << std::endl;
}
