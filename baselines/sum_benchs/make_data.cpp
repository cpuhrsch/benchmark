#include "immintrin.h"
#include "tbb/blocked_range.h"
#include "tbb/parallel_reduce.h"
#include "tbb/task_scheduler_init.h"
#include "tbb/tbb.h"
#include "tbb/tick_count.h"
#include "xmmintrin.h"
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <gflags/gflags.h>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <time.h>
#include <vector>
#include <iomanip>
#include <fstream>

DEFINE_int64(size1, 1024, "size1 of matrix");
DEFINE_int64(size2, 1024, "size2 of matrix");
DEFINE_string(outpath, "/tmp/out.data", "Path of output data");

constexpr size_t _HASHES_SIZE = 1024;

void make_vector(size_t size) {
  std::ofstream ofs(FLAGS_outpath, std::ofstream::binary);
  srand(1);
  assert(size % _HASHES_SIZE == 0);
  for (size_t c = 0; c < size; c ++){
      float num = ((float)rand()/(float)(RAND_MAX)) - 0.5f;
      ofs.write((char *)&num, sizeof(float));
  }
  ofs.close();
}

int main(int argc, char *argv[]) {
  gflags::SetUsageMessage("some usage message");
  gflags::SetVersionString("1.0.0");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  size_t size1 = (size_t)FLAGS_size1;
  size_t size2 = (size_t)FLAGS_size2;
  size_t counts =
      ((size_t)1 * (size_t)268435456) / (size1 * size2); // 1GiB of data
  make_vector(size1 * size2 * counts);
  gflags::ShutDownCommandLineFlags();
  return 0;
}
