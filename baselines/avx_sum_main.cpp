#include "avx_sum.h"
#include "tbb_sum.h"
#include <gflags/gflags.h>

DEFINE_int64(size1, 1024, "size1 of matrix");
DEFINE_int64(size2, 1024, "size2 of matrix");
DEFINE_int64(num_thread, 0, "number of threads (if applicable)");
DEFINE_int64(threshold, 0, "parallelization threshold (if applicable)");
DEFINE_int64(epoch, 1, "number of runs with cache");
DEFINE_bool(test, false, "Show Debug matrix");
DEFINE_string(run_sum, "", "run given sum algorithm");
DEFINE_string(run_reducesum, "", "run given sumreduce algorithm");
DEFINE_bool(run_sum_tbb, false, "run given sumreduce algorithm");
DEFINE_bool(run_reducesum_tbb, false, "run given sumreduce algorithm");
DEFINE_bool(flush_cache, false, "flush cache before each run");
DEFINE_bool(list_run_sums, false, "list avail sum algorithms");
DEFINE_bool(list_run_reducesums, false, "list avail reducesum algorithms");

void make_vector(float *data_, size_t size) {
  // std::ifstream ifs(FLAGS_inpath, std::ifstream::binary);
  for (size_t i = 0; i < size; i++) {
    // float num;
    // ifs.read((char *)&num, sizeof(float));
    // data_[i] = num;
    data_[i] = (float)(i % 1024);
  }
  // ifs.close();
  // srand(1);
  // assert(size % _HASHES_SIZE == 0);
  // for (size_t c = 0; c < size; c += _HASHES_SIZE) {
  //   size_t start = rand();
  //   for (size_t ci = 0; ci < _HASHES_SIZE; ci++) {
  //     data_[c + ci] = (float)((start * ci) % (RAND_MAX)) / (float)RAND_MAX
  //     -0.5f;
  //   }
  // }
}

void make_float_data(float **data_, size_t size) {
  if (posix_memalign((void **)data_, _ALIGNMENT, size * sizeof(float)))
    throw std::invalid_argument("received negative value");
  memset(*data_, 0, size * sizeof(float));
}

void time_stats(uint64_t s, double floats) {
  std::cout << "(ns:\033[36m" << std::fixed << std::setw(15) << s << "\033[0m)"
            << ",(ops/ns: "
            << "\033[31m";
  std::cout << std::setw(12) << (double)floats / (double)s << "\033[0m)";
  std::cout << ",(s: " << s / (double)NSEC << ")";
}

void sum_test(float all_sum, const float *data, size_t size, size_t counts,
              size_t epoch) {
  if (FLAGS_test) {
    assert(epoch == 1);
    assert(counts == 1);
    float sum1;
    sum_impl_naive(sum1, data, 0, size);
    if (sum1 != all_sum) {
      std::cout << "\033[31msum test failed!\033[0m + ";
      std::cout << "ref sum: \033[31m" << sum1;
      std::cout << "\033[0m impl sum: \033[31m" << all_sum << "\033[0m + ";
      // for (size_t i = 0; i < size2; i++) {
      //   std::cout << "a: " << outarr1[i] << "b: " << outarr[i] << std::endl;
      // }
    } else {
      std::cout << "\033[37msum test succeeded!\033[0m + ";
    }
  }
}

uint64_t sum_tbb_run(const float *data, size_t size, size_t counts,
                     size_t epoch) {
  float all_sum = 0;
  auto perm = rand_perm(counts);
  auto start = get_time();
  for (size_t e = 0; e < epoch; e++) {
    std::random_shuffle(perm.begin(), perm.end());
    for (size_t i_ = 0; i_ < counts; i_++) {
      size_t i = perm[i_];
      const float *datum = data + i * size;
      float sum;
      if (FLAGS_flush_cache)
        mycacheflush(data);
      sum_impl_tbb(sum, datum, 0, size, FLAGS_threshold);
      //      sum_impl_tbb_2(sum, datum, 0, size, FLAGS_threshold);
      all_sum += sum;
    }
  }
  auto end = get_time();
  sum_test(all_sum, data, size, counts, epoch);
  return timespec_subtract_to_ns(&start, &end);
}

uint64_t sum_run(void (*fun)(float &, const float *, size_t, size_t),
                 const float *data, size_t size, size_t counts, size_t epoch) {
  float all_sum = 0;
  auto perm = rand_perm(counts);
  auto start = get_time();
  std::random_shuffle(perm.begin(), perm.end());
  for (size_t i_ = 0; i_ < counts; i_++) {
    for (size_t e = 0; e < epoch; e++) {
      size_t i = perm[i_];
      const float *datum = data + i * size;
      float sum;
      if (FLAGS_flush_cache)
        mycacheflush(data);
      fun(sum, datum, 0, size);
      all_sum += sum;
    }
  }
  auto end = get_time();
  sum_test(all_sum, data, size, counts, epoch);
  return timespec_subtract_to_ns(&start, &end);
}

void reducesum_test(const float *data, size_t size1, size_t size2,
                    size_t counts, size_t epoch, float *outarr) {
  if (FLAGS_test) {
    std::cout << " + Running test!";
    assert(epoch == 1);
    assert(counts == 1);
    float *outarr1 = NULL;
    make_float_data(&outarr1, size2);
    reducesum_impl_naive(data, outarr1, 0, size1, 0, size2, size2);
    bool error = false;
    for (size_t i = 0; i < size2; i++) {
      if (outarr1[i] != outarr[i])
        error = true;
    }
    if (error) {
      std::cout << "\033[31mreducesum test failed!\033[0m + ";
      // for (size_t i = 0; i < size2; i++) {
      //   std::cout << "a: " << outarr1[i] << "b: " << outarr[i] << std::endl;
      // }
    } else {
      std::cout << "\033[37mreducesum test succeeded!\033[0m + ";
    }
    std::free(outarr1);
  }
}

uint64_t reducesum_tbb_run(

    const float *data, size_t size1, size_t size2, size_t counts,
    size_t epoch) {
  float *outarr = NULL;
  make_float_data(&outarr, size2);
  auto perm = rand_perm(counts);
  auto start = get_time();
  for (size_t e = 0; e < epoch; e++) {
    std::random_shuffle(perm.begin(), perm.end());
    for (size_t i_ = 0; i_ < counts; i_++) {
      size_t i = perm[i_];
      const float *datum = data + i * size1 * size2;
      if (FLAGS_flush_cache)
        mycacheflush(data);
      //  fun(datum, outarr, 0, size1, 0, size2, size2);
      reducesum_impl_tbb(datum, outarr, 0, size1, 0, size2, size2,
                         FLAGS_num_thread);
    }
  }
  auto end = get_time();
  reducesum_test(data, size1, size2, counts, epoch, outarr);
  if (outarr)
    std::free(outarr);
  return timespec_subtract_to_ns(&start, &end);
}

uint64_t reducesum_run(void (*fun)(const float *, float *, size_t, size_t,
                                   size_t, size_t, size_t),
                       const float *data, size_t size1, size_t size2,
                       size_t counts, size_t epoch) {
  float *outarr = NULL;
  make_float_data(&outarr, size2);
  auto perm = rand_perm(counts);
  // std::cerr << "default num threads: " << n << std::endl;
  auto start = get_time();
  for (size_t e = 0; e < epoch; e++) {
    std::random_shuffle(perm.begin(), perm.end());
    for (size_t i_ = 0; i_ < counts; i_++) {
      size_t i = perm[i_];
      const float *datum = data + i * size1 * size2;
      if (FLAGS_flush_cache)
        mycacheflush(data);
      fun(datum, outarr, 0, size1, 0, size2, size2);
    }
  }
  auto end = get_time();
  reducesum_test(data, size1, size2, counts, epoch, outarr);
  if (outarr)
    std::free(outarr);
  return timespec_subtract_to_ns(&start, &end);
}

void print_settings(size_t size1, size_t size2, size_t counts, size_t epoch) {
  std::cout << "(size1:" << std::setw(8) << size1 << "),"
            << "(size2:" << std::setw(8) << size2 << "),"
            << "(size2*size1:" << std::setw(8) << size1 * size2 << "),"
            << "(counts:" << std::setw(8) << counts << "),"
            << "(epoch:" << std::setw(8) << epoch << "),";
  if (FLAGS_num_thread > 0) {
    std::cout << "(num_thread:" << std::setw(8) << FLAGS_num_thread << "),";
  }
  if (FLAGS_threshold > 0) {
    std::cout << "(threshold:" << std::setw(8) << FLAGS_threshold << "),";
  }
}

int main(int argc, char *argv[]) {
  gflags::SetUsageMessage("some usage message");
  gflags::SetVersionString("1.0.0");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (FLAGS_list_run_sums) {
    for (auto const &fun : register_sum_impls()) {
      std::cout << fun.first << std::endl;
    }
    return 0;
  }

  if (FLAGS_list_run_reducesums) {
    for (auto const &fun : register_reducesum_impls()) {
      std::cout << fun.first << std::endl;
    }
    return 0;
  }

  size_t size1 = (size_t)FLAGS_size1;
  size_t size2 = (size_t)FLAGS_size2;
  size_t counts =
      ((size_t)64 * (size_t)268435456) / (size1 * size2); // 16GiB of data
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

  float *data_ = NULL;
  make_float_data(&data_, counts * size1 * size2);

  make_vector(data_, size1 * size2 * counts);

  if (FLAGS_run_reducesum_tbb) {
    assert(FLAGS_num_thread > 0);
    print_settings(size1, size2, counts, epoch);
    time_stats(reducesum_tbb_run(data_, size1, size2, counts, epoch),
               size1 * size2 * counts * epoch);
    std::cout << std::endl;
  }
  if (FLAGS_run_sum_tbb) {
    assert(FLAGS_num_thread > 0);
    assert(FLAGS_threshold > 0);
    print_settings(size1, size2, counts, epoch);
    tbb::task_scheduler_init tbb_init(FLAGS_num_thread);
    time_stats(sum_tbb_run(data_, size1 * size2, counts, epoch),
               size1 * size2 * counts * epoch);
    std::cout << std::endl;
  }
  if (FLAGS_run_reducesum != "") {
    std::string funname = FLAGS_run_reducesum;
    auto funs = register_reducesum_impls();
    auto fun = funs[funname];
    print_settings(size1, size2, counts, epoch);
    std::cout << "(fun: " << std::setw(25) << funname << "),";
    time_stats(reducesum_run(fun, data_, size1, size2, counts, epoch),
               size1 * size2 * counts * epoch);
    std::cout << std::endl;
  }
  if (FLAGS_run_sum != "") {
    std::string funname = FLAGS_run_sum;
    auto funs = register_sum_impls();
    auto fun = funs[funname];
    print_settings(size1, size2, counts, epoch);
    std::cout << "(fun: " << std::setw(25) << funname << "),";
    time_stats(sum_run(fun, data_, size1 * size2, counts, epoch),
               size1 * size2 * counts * epoch);
    std::cout << std::endl;
  }
  std::free(data_);
  gflags::ShutDownCommandLineFlags();
}
