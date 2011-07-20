###############################################################################
#
# Build script for project
#
###############################################################################

set(CCFILES
  scan_gold.cpp
  )

set(CUFILES
  simpleCUDPP.cu
  )

source_group("CUDA Source Files" FILES ${CUFILES})

cuda_add_executable(simpleCUDPP ${CCFILES} ${CUFILES})

target_link_libraries(simpleCUDPP cudpp)
