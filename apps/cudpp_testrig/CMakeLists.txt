###############################################################################
#
# Build script for project
#
###############################################################################
cmake_minimum_required(VERSION 2.8)

find_package(CUDA REQUIRED)

# Add include/lib paths for local cmake builds
include_directories(${CMAKE_SOURCE_DIR}/../../include)
link_directories(${CMAKE_SOURCE_DIR}/../../lib/)

set(CCFILES
  cudpp_testrig.cpp
  cudpp_testrig_options.cpp
  rand_gold.cpp
  spmvmult_gold.cpp
  test_compact.cpp
  test_mergesort.cpp
  test_radixsort.cpp
  test_rand.cpp
  test_reduce.cpp
  test_scan.cpp
  test_spmvmult.cpp
  test_stringsort.cpp
  test_tridiagonal.cpp
  test_compress.cpp
  test_listrank.cpp
  )

set(HFILES
  cudpp_testrig_options.h
  cudpp_testrig_utils.h
  compact_gold.h
  scan_gold.h
  tridiagonal_gold.h
  sparse.h
  listrank_gold.h
  )

include_directories(../common/include)

if (WIN32)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif (WIN32)

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  exec_program(uname ARGS -v  OUTPUT_VARIABLE DARWIN_VERSION)
  string(REGEX MATCH "[0-9]+" DARWIN_VERSION ${DARWIN_VERSION})
  # message(STATUS "Detected Darwin kernel version: ${DARWIN_VERSION}")
  if ((DARWIN_VERSION GREATER 13) OR (DARWIN_VERSION EQUAL 13))
    set(APPLE_MAVERICKS_OR_LATER 1 INTERNAL)
    set(CMAKE_CXX_FLAGS -stdlib=libstdc++)
    set(CMAKE_C_FLAGS -stdlib=libstdc++)
  endif ((DARWIN_VERSION GREATER 13) OR (DARWIN_VERSION EQUAL 13))
  # current CUDA on Mavericks uses libstdc++, not libc++
  # https://github.com/cudpp/cudpp/wiki/BuildingCUDPPWithMavericks
endif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

cuda_add_executable(cudpp_testrig ${CCFILES} ${HFILES})

target_link_libraries(cudpp_testrig
  cudpp
  )
