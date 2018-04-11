#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <time.h>
#include <iostream>

double diff(timespec start, timespec end)
{
  timespec temp;
  if ((end.tv_nsec-start.tv_nsec)<0) {
    temp.tv_sec = end.tv_sec-start.tv_sec-1;
    temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
  } else {
    temp.tv_sec = end.tv_sec-start.tv_sec;
    temp.tv_nsec = end.tv_nsec-start.tv_nsec;
  }
  return (double)temp.tv_sec + temp.tv_nsec * 1e-9;
}

template <typename Op>
double benchmark(Op op) {
  const int N = 10;
  timespec time1, time2;
  op();
  clock_gettime(CLOCK_MONOTONIC, &time1);
  for (int i = 0; i < N; i++)
    op();
  clock_gettime(CLOCK_MONOTONIC, &time2);
  double time = diff(time1, time2);
  std::cout << (time/N)*1e3 << " ms \n";
  return time;
}

void sin_loop(float* v, float* a, size_t size) {
  for (size_t counts = 0; counts < 1; counts++) {
    for (size_t i = 0; i < size; i++) {
      a[i] = std::sin(v[i]);
    }   
  }
}

float* output;

int main(int arg, char* argv[]) {
  size_t size = 128*128*128;
  float* input = (float*)malloc(size * sizeof(float));
  output = (float*)malloc(size * sizeof(float));
  //memset(input, 0, size * sizeof(float));
  benchmark([&]() {
    sin_loop(input, output, size); 
  });
  benchmark([&]() {
    sin_loop(input, output, size); 
  });
  return 0;
}

