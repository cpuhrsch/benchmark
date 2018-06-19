cmake_minimum_required (VERSION 3.0)
project (myproject C CXX)
find_package (Threads)
include("${CMAKE_CURRENT_SOURCE_DIR}/../../../../third_party/tbb/cmake/TBBGet.cmake")
tbb_get(TBB_ROOT tbb_root CONFIG_DIR TBB_DIR)
find_package(TBB REQUIRED)
SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -O3 -mavx2 -fopenmp")
set(CMAKE_MODULE_PATH "${CMAKE_HOME_DIRECTORY}/cmake/Modules")
set(CMAKE_BINARY_DIR "build/bin")
set (CMAKE_CXX_STANDARD 11)

set(GBENCHMARK_INCLUDE "${CMAKE_HOME_DIRECTORY}/../../build/benchmark_install/include")
set(GBENCHMARK_LIB
    "${CMAKE_HOME_DIRECTORY}/../../build/benchmark_install/lib/libbenchmark_main.a"
    "${CMAKE_HOME_DIRECTORY}/../../build/benchmark_install/lib/libbenchmark.a")


set(CONDA_INCLUDE "$ENV{CONDA_PREFIX}/include")
set(CONDA_LIBS "$ENV{CONDA_PREFIX}/lib/libiomp5.so")
message("CONDA_INCLUDE: ${CONDA_INCLUDE}")
message("CONDA_LIBS: ${CONDA_LIBS}")
include_directories (SYSTEM "${GBENCHMARK_INCLUDE}" "${CONDA_INCLUDE}")
add_executable (example conflict.cpp)
target_link_libraries(example TBB::tbb "${GBENCHMARK_LIB}")
target_link_libraries(example ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(example ${CONDA_LIBS})
