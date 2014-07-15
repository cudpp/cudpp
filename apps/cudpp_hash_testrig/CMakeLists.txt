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

cuda_add_executable(cudpp_hash_testrig ${CCFILES} ${HFILES})

target_link_libraries(cudpp_hash_testrig
  cudpp_hash
  cudpp
  )
