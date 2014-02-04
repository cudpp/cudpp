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

cuda_add_executable(simpleCUDPP ${CCFILES} ${CUFILES})

target_link_libraries(simpleCUDPP cudpp)
