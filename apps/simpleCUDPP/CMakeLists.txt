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
  scan_gold.cpp
  )

set(CUFILES
  simpleCUDPP.cu
  )

source_group("CUDA Source Files" FILES ${CUFILES})

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

cuda_add_executable(simpleCUDPP ${CCFILES} ${CUFILES})

target_link_libraries(simpleCUDPP cudpp)
