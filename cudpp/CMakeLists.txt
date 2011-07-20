##############################################################################
#
# CUDA Data Parallel Primitives (CUDPP) CMake file
#
###############################################################################

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR}/src/app
  ${CMAKE_CURRENT_SOURCE_DIR}/src/kernel
  ${CMAKE_CURRENT_SOURCE_DIR}/src/cta
  )

set(CCFILES
  src/cudpp.cpp
  src/cudpp_plan.cpp
  src/cudpp_manager.cpp
  src/cudpp_maximal_launch.cpp
  )

set (HFILES
  src/cudpp_manager.h
  src/cudpp_plan.h
  src/cuda_util.h
  src/cudpp_util.h
  src/cudpp_globals.h
  src/cudpp_maximal_launch.h
  src/cudpp_compact.h
  src/cudpp_radixsort.h
  src/cudpp_rand.h
  src/cudpp_reduce.h
  src/cudpp_scan.h
  src/cudpp_segscan.h
  src/cudpp_spmvmult.h
  src/sharedmem.h
  )

set (CUHFILES
  src/cta/radixsort_cta.cuh
  src/cta/rand_cta.cuh
  src/cta/scan_cta.cuh
  src/cta/segmented_scan_cta.cuh
  src/kernel/compact_kernel.cuh
  src/kernel/radixsort_kernel.cuh
  src/kernel/rand_kernel.cuh
  src/kernel/reduce_kernel.cuh
  src/kernel/segmented_scan_kernel.cuh
  src/kernel/spmvmult_kernel.cuh
  src/kernel/vector_kernel.cuh
  src/app/cr_app.cuh
  src/app/crpcr_app.cuh
  src/app/pcr_app.cuh
  src/kernel/cr_kernel.cuh
  src/kernel/crpcr_kernel.cuh
  src/kernel/pcr_kernel.cuh
  )

set(CUFILES
  src/app/reduce_app.cu
  src/app/compact_app.cu
  src/app/scan_app.cu
  src/app/segmented_scan_app.cu
  src/app/spmvmult_app.cu
  src/app/radixsort_app.cu
  src/app/rand_app.cu 
  src/app/tridiagonal_app.cu
  )

set(HFILES_PUBLIC
  include/cudpp.h
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
  OPTIONS ${DUMMY} ${GENCODE_SM10} ${GENCODE_SM13} ${GENCODE_SM20} ${VERBOSE_PTXAS}
  )
  
target_link_libraries(cudpp)

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
