#include "tbb_sum.h"

using namespace tbb;

class SumFoo {
  const float *my_a;

public:
  float my_sum;
  void operator()(const blocked_range<size_t> &r) {
    const float *a = my_a;
    // size_t end = r.end();
    float sum;
    // if (r.begin() == 0) {
    // std::cerr << "r.begin(): " << r.end() - r.begin() << std::endl;
    // }
    sum_impl21(sum, a, r.begin(), r.end());
    // std::cerr << "sum: " << sum << std::endl;
    // for (size_t i = r.begin(); i != end; ++i)
    //   sum += a[i];
    my_sum += sum;
  }

  SumFoo(SumFoo &x, split) : my_a(x.my_a), my_sum(0) {}

  void join(const SumFoo &y) { my_sum += y.my_sum; }

  SumFoo(const float * a) : my_a(a), my_sum(0) {}
};


float RepeatableReduce( const float* a, size_t start, size_t end, size_t threshold) {
   if( end-start<=threshold ) {
      float sum;
      sum_impl21(sum, a, start, end);
      return sum;
   } else {
       // Do parallel divide-and-conquer reduction
       size_t mid = start + (end - start) / 2;
       float left, right;
       tbb::parallel_invoke(
           [&]{left=RepeatableReduce(a, start, mid, threshold);},
           [&]{right=RepeatableReduce(a, mid, end, threshold);}
       );
       return left+right;
   }
}

void sum_impl_tbb_2(float &sum, const float *a, size_t start, size_t end, size_t threshold, size_t max_num_thread) {
  (void) max_num_thread;
  sum = RepeatableReduce(a, start, end, threshold);
}

void sum_impl_tbb(float &sum, const float *a, size_t start, size_t end,
                  size_t threshold, size_t max_num_thread) {
  (void) max_num_thread;
  if (end - start < threshold) {
    sum_impl21(sum, a, start, end);
  } else {
    SumFoo sf(a);
    // static affinity_partitioner ap;
    parallel_reduce(blocked_range<size_t>(start, end, threshold), sf); //, ap);
    sum = sf.my_sum;
  }
}

class ReduceSumFoo {
  const float *my_a;
  size_t my_size2;

public:
  std::vector<float> my_sum;
  void operator()(const blocked_range2d<size_t> &r) {
    const float *a = my_a;
    float *sum = my_sum.data();
    // if (r.rows().begin() == 0) {
    //   std::cerr << "rb: " << r.rows().begin() << std::endl;
    //   std::cerr << "re: " << r.rows().end() << std::endl;
    //   std::cerr << "cb: " << r.cols().begin() << std::endl;
    //   std::cerr << "ce: " << r.cols().end() << std::endl;
    // }
    size_t size2 = my_size2;
    reducesum_impl3(a, sum, r.rows().begin(), r.rows().end(),
                         r.cols().begin(), r.cols().end(), size2);
    // if (r.rows().begin() == 0) {
    //   exit(1);
    // }
  }

  ReduceSumFoo(ReduceSumFoo &x, split)
      : my_a(x.my_a), my_size2(x.my_size2), my_sum(x.my_sum) {}

  void join(const ReduceSumFoo &y) {
    float *sum1 = my_sum.data();
    const float *sum2 = y.my_sum.data();
    register size_t k;
    size_t end = my_size2 > 7 ? my_size2 - 8 : 0;
    k = 0;
    for (; k < end; k += 8) {
      register __m256 a, b;
      a = _mm256_loadu_ps(sum1 + k);
      b = _mm256_loadu_ps(sum2 + k);
      a = _mm256_add_ps(a, b);
      _mm256_storeu_ps(sum1 + k, a);
    }
    for (; k < my_size2; k += 1) {
      sum1[k] += sum2[k];
    }
  }

  ReduceSumFoo(const float a[], size_t size2)
      : my_a(a), my_size2(size2), my_sum(size2, 0.0f) {}
};

void reducesum_impl_tbb(const float *arr, float *outarr, size_t size1b,
                        size_t size1e, size_t size2b, size_t size2e,
                        size_t size2, size_t num_thread) {
  (void)num_thread;
  ReduceSumFoo sf(arr, size2);
  // static affinity_partitioner ap;
  // parallel_reduce(blocked_range2d<size_t>(0, size1, grain, 0, size2,
  // grain),
  // sf);//, ap);
  parallel_reduce(
      blocked_range2d<size_t>(size1b, size1e, 64, size2b, size2e, 64),
      sf); //, ap);
  for (size_t i = 0; i < size2; i++) {
    outarr[i] = sf.my_sum[i];
  }
}

std::map<std::string,
         void (*)(float &, const float *, size_t, size_t, size_t, size_t)>
register_sum_impls_tbb() {
  std::map<std::string,
           void (*)(float &, const float *, size_t, size_t, size_t, size_t)>
      impls;
  impls["sum_impl_tbb"] = &sum_impl_tbb;
  impls["sum_impl_tbb_2"] = &sum_impl_tbb_2;
  return impls;
}
