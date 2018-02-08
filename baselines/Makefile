%.exe: %.cpp benchmark_common.h
	LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${CONDA_PREFIX}/lib:${PYTORCH_HOME}/torch/lib:${CUDNN_LIB_DIR}:${LD_LIBRARY_PATH} \
	g++ -I${CUDA_HOME}/include \
	    -I${CONDA_PREFIX}/include \
	    -L${CUDA_HOME}/lib64 \
	    -L${CUDA_HOME}/lib64/stubs \
	    -L${CONDA_PREFIX}/lib \
	    -L${PYTORCH_HOME}/torch/lib \
	    -L${PYTORCH_HOME}/torch/_C.so \
	    -L${PYTORCH_HOME}/torch/lib/libATen.so.1 \
	    -L${CUDNN_LIB_DIR} \
	    -I${CUDNN_INCLUDE_DIR} \
	    -I${PYTORCH_HOME}/ \
	    -I${PYTORCH_HOME}/torch/csrc \
	    -I${PYTORCH_HOME}/torch/lib/pybind11/include \
	    -I${PYTORCH_HOME}/torch/lib/tmp_install/include \
	    -I${PYTORCH_HOME}/torch/lib/tmp_install/include/TH \
	    -I${PYTORCH_HOME}/torch/lib/tmp_install/include/THPP \
	    -I${PYTORCH_HOME}/torch/lib/tmp_install/include/THNN \
	    -I${PYTORCH_HOME}/torch/lib/tmp_install/include/ATen \
	    -lnvidia-ml -lcudart -lATen -lcuda -lnvrtc \
	    -fopenmp \
	    -std=c++11 -O3 -Wall -Wextra -pedantic -g -march=native\
	    $< -o $@

cudnn_%.exe: cudnn_%.cpp benchmark_common.h
	LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${CUDNN_LIB_DIR}:${LD_LIBRARY_PATH} \
	g++ -I${CUDA_HOME}/include \
	    -I${CUDNN_INCLUDE_DIR} \
	    -L${CUDA_HOME}/lib64 -L${CUDA_HOME}/lib64/stubs \
	    -L${CUDNN_LIB_DIR} \
	    -lnvidia-ml -lcudart -lcudnn \
	    -fopenmp \
	    -std=c++11 -O3 -g \
	    $< -o $@

# TODO: I'm not sure why I had to explicitly link against ATen. Seems like
# I'm picking up the wrong one sometimes...
lstm_variable.exe: lstm_variable.cpp benchmark_common.h
	LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${CONDA_PREFIX}/lib:${PYTORCH_HOME}/torch/lib:${CUDNN_LIB_DIR}:${LD_LIBRARY_PATH} \
	g++ -I${CUDA_HOME}/include \
	    -I${CUDNN_INCLUDE_DIR} \
	    -I${PYTORCH_HOME}/ \
	    -I${PYTORCH_HOME}/torch/csrc \
	    -I${PYTORCH_HOME}/torch/lib/pybind11/include \
	    -I${PYTORCH_HOME}/torch/lib/tmp_install/include \
	    -I${PYTORCH_HOME}/torch/lib/tmp_install/include/TH \
	    -I${PYTORCH_HOME}/torch/lib/tmp_install/include/THPP \
	    -I${PYTORCH_HOME}/torch/lib/tmp_install/include/THNN \
	    -I${PYTORCH_HOME}/torch/lib/tmp_install/include/ATen \
	    -L${CUDA_HOME}/lib64 -L${CUDA_HOME}/lib64/stubs -L${CONDA_PREFIX}/lib \
	    -lnvidia-ml -lcudart -lcuda -lnvrtc \
	    -fopenmp \
	    -Wl,${PYTORCH_HOME}/torch/lib/libATen.so.1 \
	    -Wl,${PYTORCH_HOME}/torch/_C.so \
	    -std=c++11 -O3 -g \
	    $< -o $@

run_%: %.exe
	MKL_NUM_THREADS=1 LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${CONDA_PREFIX}/lib:${PYTORCH_HOME}/torch/lib:${CUDNN_LIB_DIR}:${LD_LIBRARY_PATH} numactl --cpunodebind=0 --membind=0 taskset 0x1 perf stat ./$<

avx_sum.exec: avx_sum_main.cpp avx_sum.cpp tbb_sum.cpp
	TBB_USE_THREADING_TOOLS=1 \
	g++ -std=c++11 \
	    -funroll-loops \
	    -ftree-vectorize \
	    -ftree-vectorizer-verbose=1 \
	    -O3 \
	    -Wall \
	    -Wextra \
	    -pedantic \
	    -mavx \
	    -mavx2 \
	    -march=native \
	    -g \
	    -ltbb \
	    -lrt \
	    -lgflags \
	    $? -o $@

