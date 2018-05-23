#include <cmath>
#include <vector>
#include <x86intrin.h>
#include <iostream>

#if defined(__AVX__) && defined(__GLIBC__) && __GLIBC_MINOR__ == 23
// #define ZEROUPPER
#define ZEROUPPER _mm256_zeroupper();
#else
#define ZEROUPPER
#endif

int main() {
  std::vector<float> v(128 * 128 * 128, 1);
  std::vector<float> a(128 * 128 * 128, 0);
  for (size_t counts = 0; counts < 10; counts++) {
    ZEROUPPER
    volatile float x = 1;
    x = sinf(x);
    for (size_t i = 0; i < v.size(); i++) {
      a[i] = sinf(v[i]); // + a[i];
    }
  }
}
