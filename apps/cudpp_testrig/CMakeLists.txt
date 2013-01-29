###############################################################################
#
# Build script for project
#
###############################################################################

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

cuda_add_executable(cudpp_testrig ${CCFILES} ${HFILES})

target_link_libraries(cudpp_testrig
  cudpp
  )
