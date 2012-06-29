###############################################################################
#
# Build script for project
#
###############################################################################

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
