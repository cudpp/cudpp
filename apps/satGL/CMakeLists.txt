###############################################################################
#
# Build script for project
#
###############################################################################

set(CCFILES
  main.cpp
  )

set(CUFILES
  satGL.cu
  )

source_group("CUDA Source Files" FILES ${CUFILES})
  
set(CMAKE_PREFIX_PATH ../common)
  
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(GLEW REQUIRED)

include_directories(  
  ../common/include
  ${OPENGL_INCLUDE_DIR}
  ${GLUT_INCLUDE_DIR}
  ${GLEW_INCLUDE_DIR}
  )

cuda_add_executable(satGL ${CCFILES} ${CUFILES})

target_link_libraries(satGL
  cudpp
  ${OPENGL_LIBRARIES}
  ${GLUT_LIBRARIES}
  ${GLEW_LIBRARY}
  )
