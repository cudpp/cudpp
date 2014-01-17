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
  cudpp_listrank.h
  cudpp_mergesort.h
  cudpp_radixsort.h
  cudpp_rand.h
  cudpp_reduce.h
  cudpp_stringsort.h
  cudpp_scan.h
  cudpp_segscan.h
  cudpp_spmvmult.h
  sharedmem.h
  )

set (CUHFILES
  cta/compress_cta.cuh
  cta/mergesort_cta.cuh
  cta/radixsort_cta.cuh  
  cta/rand_cta.cuh
  cta/scan_cta.cuh
  cta/segmented_scan_cta.cuh
  cta/stringsort_cta.cuh  
  kernel/compact_kernel.cuh
  kernel/compress_kernel.cuh
  kernel/listrank_kernel.cuh
  kernel/mergesort_kernel.cuh
  kernel/radixsort_kernel.cuh
  kernel/rand_kernel.cuh
  kernel/reduce_kernel.cuh
  kernel/segmented_scan_kernel.cuh
  kernel/spmvmult_kernel.cuh
  kernel/stringsort_kernel.cuh
  kernel/vector_kernel.cuh
  kernel/tridiagonal_kernel.cuh
  )

set(CUFILES
  app/reduce_app.cu
  app/compact_app.cu
  app/compress_app.cu
  app/listrank_app.cu
  app/mergesort_app.cu
  app/scan_app.cu
  app/segmented_scan_app.cu
  app/spmvmult_app.cu
  app/stringsort_app.cu
  app/radixsort_app.cu
  app/rand_app.cu 
  app/tridiagonal_app.cu
  )

set(HFILES_PUBLIC
  ../../include/cudpp.h
  )

source_group("CUDA Source Files" FILES ${CUFILES})
source_group("CUDA Header Files" FILES ${CUHFILES})

#set(GENCODE_SM10 -gencode=arch=compute_10,code=sm_10 -gencode=arch=compute_10,code=compute_10)
#set(GENCODE_SM13 -gencode=arch=compute_13,code=sm_13 -gencode=arch=compute_13,code=compute_13)
#set(GENCODE_SM20 -gencode=arch=compute_20,code=sm_20 -gencode=arch=compute_20,code=compute_20)
#set(GENCODE_SM30 -gencode=arch=compute_30,code=sm_30 -gencode=arch=compute_30,code=compute_30)
set(GENCODE_SM35 -gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_35,code=compute_35)

set(GENCODE -gencode=arch=compute_35,code=compute_35) # at least generate PTX

#option(CUDPP_GENCODE_SM10
#       "ON to generate code for Compute Capability 1.0 devices (e.g. Tesla C870)"
#       ON)

#option(CUDPP_GENCODE_SM13
#       "ON to generate code for Compute Capability 1.3 devices (e.g. Tesla C1060)"
#       OFF)

option(CUDPP_GENCODE_SM20
       "ON to generate code for Compute Capability 2.0 devices (e.g. Tesla C2050)"
       ON)

option(CUDPP_GENCODE_SM30
       "ON to generate code for Compute Capability 3.0 devices (e.g. Tesla K10)"
       OFF)

option(CUDPP_GENCODE_SM35
       "ON to generate code for Compute Capability 3.5 devices (e.g. Tesla K20)"
       OFF)

#if (CUDPP_GENCODE_SM10)
#  set(GENCODE ${GENCODE} ${GENCODE_SM10})
#endif(CUDPP_GENCODE_SM10)

#if (CUDPP_GENCODE_SM13)
#  set(GENCODE ${GENCODE} ${GENCODE_SM13})
#endif(CUDPP_GENCODE_SM13)

if (CUDPP_GENCODE_SM20)
  set(GENCODE ${GENCODE} ${GENCODE_SM20})
endif(CUDPP_GENCODE_SM20)

if (CUDPP_GENCODE_SM30)
  set(GENCODE ${GENCODE} ${GENCODE_SM30})
endif(CUDPP_GENCODE_SM30)

if (CUDPP_GENCODE_SM35)
  set(GENCODE ${GENCODE} ${GENCODE_SM35})
endif(CUDPP_GENCODE_SM35)

if (CUDA_VERBOSE_PTXAS)
  set(VERBOSE_PTXAS --ptxas-options=-v)
endif (CUDA_VERBOSE_PTXAS)

cuda_add_library(cudpp ${LIB_TYPE} 
  ${CCFILES} 
  ${HFILES}
  ${CUHFILES}
  ${HFILES_PUBLIC}
  ${CUFILES} 
  OPTIONS ${GENCODE} ${VERBOSE_PTXAS}
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
