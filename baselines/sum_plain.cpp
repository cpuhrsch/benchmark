#include <time.h>
#include <vector>
#include <cstdint>
#include <iostream>
#include <cstdlib>

float sum_impl2(float* arr, size_t size) {
  float sum = 0;
  for (size_t i = 0; i < size; i += 2) {
     float slocal = arr[i] + arr[i + 1];
     sum += slocal;
  }
  return sum;
}

std::vector<float> make_vector(size_t size) {
  srand (1);
  std::vector<float> vector;
  vector.reserve(size);
  vector[0] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
  float salt = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
  for (size_t i = 1; i < size; i++) {
    vector.push_back(vector[i - 1] * salt);
  }
  return vector;
}

int main() {
  int64_t size = 1000000000;
  int64_t counts = 100;
  auto vector = make_vector(size);
  float* data = vector.data();
  float all_sum = 0;
  for (int64_t i = 0; i < counts; i++) {
    all_sum += sum_impl2(data, size);
  }
  std::cerr << "sum_impl2: " << all_sum << std::endl;
}
