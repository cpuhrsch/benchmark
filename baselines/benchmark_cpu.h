#pragma once

#include <time.h>
#include <numeric>
#include <algorithm>
#include "emmintrin.h"

#define USEC 1000000
#define NSEC 1000000000

static inline size_t _divup(size_t x, size_t y) {
  return ((x + y - 1) / y);
}

/**
 * us_to_timespec - converts microseconds to a timespec
 * @us: number of microseconds
 * @t: the storage timespec
 */
static inline void us_to_timespec(uint64_t us, struct timespec *t) {
  t->tv_sec = us / USEC;
  t->tv_nsec = (us - t->tv_sec * USEC) * (NSEC / USEC);
}

/**
 * timespec_to_us - converts a timespec to microseconds
 * @t: the timespec
 *
 * Returns microseconds.
 */
static inline uint64_t timespec_to_us(struct timespec *t) {
  return t->tv_sec * USEC + t->tv_nsec / (NSEC / USEC);
}

/**
 * timespec_to_ns - converts a timespec to nanoseconds
 * @t: the timespec
 *
 * Returns nanoseconds.
 */
static inline uint64_t timespec_to_ns(struct timespec *t) {
  return t->tv_sec * NSEC + t->tv_nsec;
}

/**
 * timespec_subtract - subtracts timespec y from timespec x
 * @x, @y: the timespecs to subtract
 * @result: a pointer to store the answer
 *
 * WARNING: It's not safe for @result to be @x or @y.
 *
 * Returns 1 if the difference is negative, otherwise 0.
 */
static inline void timespec_subtract(struct timespec *x, struct timespec *y,
                      struct timespec *result) {
  if (x->tv_nsec < y->tv_nsec) {
    int secs = (y->tv_nsec - x->tv_nsec) / NSEC + 1;
    y->tv_nsec -= NSEC * secs;
    y->tv_sec += secs;
  }

  if (x->tv_nsec - y->tv_nsec > NSEC) {
    int secs = (x->tv_nsec - y->tv_nsec) / NSEC;
    y->tv_nsec += NSEC * secs;
    y->tv_sec -= secs;
  }

  result->tv_sec = x->tv_sec - y->tv_sec;
  result->tv_nsec = x->tv_nsec - y->tv_nsec;

  if (x->tv_sec < y->tv_sec) {
    fprintf(stderr, "clock not monotonic???\n");
    exit(0);
  }
}

static inline void mycacheflush(void const* ptr) {
  _mm_clflush(ptr);
}

static inline uint64_t timespec_subtract_to_ns(struct timespec *start, struct timespec *finish) {
  struct timespec delta;
  timespec_subtract(finish, start, &delta);
  return timespec_to_ns(&delta);
}

static inline struct timespec get_time() {
  struct timespec start;
  int ret = clock_gettime(CLOCK_MONOTONIC, &start);
  if (ret == -1) {
    perror("clock_gettime()");
    exit(1);
  }
  return start;
}

static inline uint16_t __mm_crc32_u64(uint64_t crc, uint64_t val) {
  asm("crc32q %1, %0" : "+r"(crc) : "rm"(val));
  return crc;
}

static inline unsigned int sfrand(unsigned int x) {
  unsigned int seed = x * 16807;
  return ((seed) >> 9) | 0x40000000;
}

static inline size_t hash(size_t x) {
  size_t h = 2166136261;
  h = h ^ x;
  h = h * 16777619;
  return h;
}

static inline std::vector<size_t> rand_perm(size_t max) {
  std::vector<size_t> perm(max);
  std::iota(perm.begin(), perm.end(), 0);
  std::random_shuffle(perm.begin(), perm.end());
  return perm;
}
