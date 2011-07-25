##############################################################################
#
# CUDA Data Parallel Primitives (CUDPP) CMake file
#
###############################################################################

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/../../include
  ${CMAKE_CURRENT_SOURCE_DIR}/../../src/cudpp
  ${CMAKE_CURRENT_SOURCE_DIR}
  )

set(CCFILES
  cudpp_hash.cpp
  mt19937ar.cpp
  )

set (HFILES
  definitions.h
  hash_compacting.h
  hash_functions.h
  hash_multivalue.h
  hash_table.h
  mt19937ar.h
  )

set (CUHFILES
  debugging.cuh
  hash_table.cuh
  )

set(CUFILES
  debugging.cu
  hash_compacting.cu
  hash_multivalue.cu
  hash_table.cu
  hash_functions.cu
  )

set(HFILES_PUBLIC
  ../../include/cudpp_hash.h
  )

source_group("CUDA Source Files" FILES ${CUFILES})
source_group("CUDA Header Files" FILES ${CUHFILES})

set(GENCODE_SM20 -gencode=arch=compute_20,code=sm_20 -gencode=arch=compute_20,code=compute_20)

if (CUDA_VERBOSE_PTXAS)
  set(VERBOSE_PTXAS --ptxas-options=-v)
endif (CUDA_VERBOSE_PTXAS)

cuda_add_library(cudpp_hash ${LIB_TYPE} 
  ${CCFILES} 
  ${HFILES}
  ${CUHFILES}
  ${HFILES_PUBLIC}
  ${CUFILES} 
  OPTIONS ${GENCODE_SM20} ${VERBOSE_PTXAS}
  )
  
target_link_libraries(cudpp_hash cudpp)

install(FILES ${HFILES_PUBLIC}
  DESTINATION include
  )

install(TARGETS cudpp_hash
  DESTINATION lib
  EXPORT cudpp-targets
  )

install(EXPORT cudpp-targets
  DESTINATION lib
  )
