##############################################################################
#
# CUDA Data Parallel Primitives (CUDPP) CMake file
#
###############################################################################

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/../../include
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/app
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel
  ${CMAKE_CURRENT_SOURCE_DIR}/cta
  )

set(CCFILES
  cudpp.cpp
  cudpp_plan.cpp
  cudpp_manager.cpp
  )

set (HFILES
  cudpp_manager.h
  cudpp_plan.h
  cuda_util.h
  cudpp_util.h
  cudpp_globals.h
  cudpp_compact.h
  cudpp_compress.h
  cudpp_radixsort.h
  cudpp_rand.h
  cudpp_reduce.h
  cudpp_scan.h
  cudpp_segscan.h
  cudpp_spmvmult.h
  sharedmem.h
  )

set (CUHFILES
  cta/compress_cta.cuh
  cta/radixsort_cta.cuh
  cta/rand_cta.cuh
  cta/scan_cta.cuh
  cta/segmented_scan_cta.cuh
  kernel/compact_kernel.cuh
  kernel/compress_kernel.cuh
  kernel/radixsort_kernel.cuh
  kernel/rand_kernel.cuh
  kernel/reduce_kernel.cuh
  kernel/segmented_scan_kernel.cuh
  kernel/spmvmult_kernel.cuh
  kernel/vector_kernel.cuh
  kernel/tridiagonal_kernel.cuh
  )

set(CUFILES
  app/reduce_app.cu
  app/compact_app.cu
  app/compress_app.cu
  app/scan_app.cu
  app/segmented_scan_app.cu
  app/spmvmult_app.cu
  app/radixsort_app.cu
  app/rand_app.cu 
  app/tridiagonal_app.cu
  )

set(HFILES_PUBLIC
  ../../include/cudpp.h
  )

source_group("CUDA Source Files" FILES ${CUFILES})
source_group("CUDA Header Files" FILES ${CUHFILES})

set(GENCODE_SM10 -gencode=arch=compute_10,code=sm_10 -gencode=arch=compute_10,code=compute_10)
set(GENCODE_SM13 -gencode=arch=compute_13,code=sm_13 -gencode=arch=compute_13,code=compute_13)
set(GENCODE_SM20 -gencode=arch=compute_20,code=sm_20 -gencode=arch=compute_20,code=compute_20)

if (CUDA_VERBOSE_PTXAS)
  set(VERBOSE_PTXAS --ptxas-options=-v)
endif (CUDA_VERBOSE_PTXAS)

cuda_add_library(cudpp ${LIB_TYPE} 
  ${CCFILES} 
  ${HFILES}
  ${CUHFILES}
  ${HFILES_PUBLIC}
  ${CUFILES} 
  OPTIONS ${GENCODE_SM20} ${VERBOSE_PTXAS}
  #OPTIONS ${GENCODE_SM10} ${GENCODE_SM13} ${GENCODE_SM20} ${VERBOSE_PTXAS}
  )
  
install(FILES ${HFILES_PUBLIC}
  DESTINATION include
  )

install(TARGETS cudpp
  DESTINATION lib
  EXPORT cudpp-targets
  )

install(EXPORT cudpp-targets
  DESTINATION lib
  )
