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
  hash_testrig.cpp
  random_numbers.cpp
  mt19937ar.cpp
  )

set(HFILES
  mt19937ar.h
  random_numbers.h
  )

include_directories(../common/include ./)

cuda_add_executable(cudpp_hash_testrig ${CCFILES} ${HFILES})

target_link_libraries(cudpp_hash_testrig
  cudpp_hash
  cudpp
  )
