#include "benchmark_common.h"

#include <time.h>
#include <vector>

int main() {
  int64_t size = 1000000;
  std::vector<float> vector(size);
  for (size_t i = 0; i < vector.size(); i++) {
    vector[i] = (float)i / (float)size;
  }

  int64_t sample_size = 10000;

  auto start_cpu_ns = getTime();
  float all_sum = 0;
  int64_t count = 0;
  for (int64_t sample = 0; sample < sample_size; sample++) {
    float sum = 0;
    for (size_t i = 0; i < vector.size(); i += 2) {
    // //   float slocal = vector[i] + vector[i + 1] + vector[i + 2] + vector[i + 3];
       float slocal = vector[i] + vector[i + 1];
       sum += slocal;
       count += 2;
    // // //   // sum += vector[i];
    // // //   // sum += vector[i];
     }
    // for (size_t i = 0; i < vector.size(); i += 1) {
    //   sum += vector[i];
    //   count++;
    // }
    // std::cerr << "sum: " << sum << std::endl;
    sum /= size;
    for (size_t i = 0; i < vector.size(); i++) {
      vector[i] += sum / (vector[i] + 1);
    }
    all_sum += sum;
  }
  auto end_cpu_ns = getTime();
  std::cerr << "all_sum: " << all_sum << std::endl;
  std::cerr << "count: " << count << std::endl;
  print_result_cpu_usecs("sum", 1, (end_cpu_ns - start_cpu_ns) / 1000.0,
                     sample_size);
}
