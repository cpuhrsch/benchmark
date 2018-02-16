#pragma once

#include <map>
#include "avx_sum.h"
#include "tbb/task_arena.h"
#include "tbb/partitioner.h"
#include "tbb/blocked_range.h"
#include "tbb/parallel_reduce.h"
#include "tbb/task_scheduler_init.h"
#include "tbb/tbb.h"
#include "tbb/tick_count.h"
#include "tbb/task_scheduler_init.h"
#include <omp.h>

void reducesum_impl_tbb(const float *, float *, size_t, size_t, size_t, size_t, size_t, size_t);
void sum_impl_tbb(float &, const float *, size_t, size_t, size_t, size_t);
void sum_impl_tbb_2(float &, const float *, size_t, size_t, size_t, size_t);

std::map<std::string,
         void (*)(float &, const float *, size_t, size_t, size_t, size_t)>
register_sum_impls_tbb();
