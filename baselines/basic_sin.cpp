#include <chrono>
#include <iostream>
#include <cmath>
#include <vector>
#include <x86intrin.h>


int main() {
  std::vector<float> v(128 * 128 * 128, 1);
  std::vector<float> a(128 * 128 * 128, 0);
  for (size_t counts = 0; counts < 10; counts++) {
    auto t11 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < v.size(); i++) {

#ifdef __AVX__
      _mm256_zeroupper();
#endif
      a[i] = sinf(v[i]);// + a[i];
    }
    auto t21 = std::chrono::high_resolution_clock::now();
    std::cout << "asdf: " << std::chrono::duration_cast<std::chrono::microseconds>(t21 - t11).count() << " a[0]: " << a[0] << std::endl;
  }
}

