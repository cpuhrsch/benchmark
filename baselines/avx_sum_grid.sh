TBB_USE_THREADING_TOOLS=1 g++ -std=c++11 -funroll-loops -ftree-vectorize -ftree-vectorizer-verbose=1  -O3 -Wall -Wextra -pedantic -mavx -mavx2 -march=native -g avx_sum.cpp -ltbb -lrt -lgflags

./a.out -size1 32 -size2 32     -run_reducesum -show_baseline -epoch 1
./a.out -size1 64 -size2 64     -run_reducesum -show_baseline -epoch 1
./a.out -size1 128 -size2 128   -run_reducesum -show_baseline -epoch 1
./a.out -size1 256 -size2 256   -run_reducesum -show_baseline -epoch 1
./a.out -size1 128 -size2 512   -run_reducesum -show_baseline -epoch 1
./a.out -size1 512 -size2 128   -run_reducesum -show_baseline -epoch 1
./a.out -size1 1024 -size2 1024 -run_reducesum -show_baseline -epoch 1
./a.out -size1 2048 -size2 2048 -run_reducesum -show_baseline -epoch 1
./a.out -size1 4096 -size2 4096 -run_reducesum -show_baseline -epoch 1
./a.out -size1 512 -size2 4096  -run_reducesum -show_baseline -epoch 1
./a.out -size1 4096 -size2 512  -run_reducesum -show_baseline -epoch 1
