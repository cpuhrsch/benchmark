TBB_USE_THREADING_TOOLS=1 g++ -std=c++11 -O3 -Wall -Wextra -pedantic -mavx -mavx2 -march=native -g sum_plain.cpp -ltbb -lrt -lgflags && perf stat ./a.out
