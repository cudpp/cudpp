project(cudpp)

set(cudpp_VERSION_MAJOR 2)
set(cudpp_VERSION_MINOR 0)
set(cudpp_VERSION_PATCH 0)

cmake_minimum_required(VERSION 2.8)

find_package(CUDA REQUIRED)

option(CUDPP_BUILD_SHARED_LIBS
  "On to build shared libraries, off for static libraries."
  OFF
  )

option(CUDPP_BUILD_APPLICATIONS
  "If on, builds the sample applications."
  OFF
  )

option(CUDA_VERBOSE_PTXAS
  "On to enable verbose output from the PTXAS assembler."
  OFF
  )

## Set the directory where the binaries will be stored
set(EXECUTABLE_OUTPUT_PATH
  ${PROJECT_BINARY_DIR}/bin
  CACHE PATH
  "Directory where all executables will be stored"
  )

## Set the directory where the libraries will be stored
set(LIBRARY_OUTPUT_PATH
  ${PROJECT_BINARY_DIR}/lib
  CACHE PATH
  "Directory where all the libraries will be stored"
  )

set(CMAKE_MODULE_PATH
  ${CMAKE_SOURCE_DIR}/cmake
  )

set(cudpp_INCLUDE_DIRS
  ${CMAKE_SOURCE_DIR}/include
  )
include_directories(${cudpp_INCLUDE_DIRS})

if (CMAKE_SIZEOF_VOID_P EQUAL 8)
  set(bits 64)
else ()
  set(bits 32)
endif (CMAKE_SIZEOF_VOID_P EQUAL 8)

string(TOUPPER "${CMAKE_CONFIGURATION_TYPES}" configs)
foreach (config ${configs})
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${config} "${EXECUTABLE_OUTPUT_PATH}")
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${config} "${LIBRARY_OUTPUT_PATH}")
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${config} "${LIBRARY_OUTPUT_PATH}")
  set(CMAKE_${config}_POSTFIX ${bits})
endforeach(config {configs})
set(CMAKE_DEBUG_POSTFIX "${bits}d")

if(CUDPP_BUILD_SHARED_LIBS)
  # User wants to build Dynamic Libraries, so change the LIB_TYPE variable to CMake keyword 'SHARED'
  set(LIB_TYPE SHARED)
else()
  set(LIB_TYPE STATIC)
  set(CUDPP_STATIC_LIB 1)
endif(CUDPP_BUILD_SHARED_LIBS)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/include/cudpp_config.h.in
  ${CMAKE_CURRENT_SOURCE_DIR}/include/cudpp_config.h
  )

set (CUDPP_APP_DATA_DIR
  "${CMAKE_SOURCE_DIR}/apps/data/"
  )
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/apps/common/include/common_config.h.in
  ${CMAKE_CURRENT_SOURCE_DIR}/apps/common/include/common_config.h
  )

# copy WIN32 DLLs
if (WIN32)
if (bits EQUAL 64)
file(COPY bin/win32/glut32.dll bin/win64/glew64.dll DESTINATION bin/)
elseif (bits EQUAL 32)
file(COPY bin/win32/glut32.dll bin/win32/glew32.dll DESTINATION bin/)
endif (bits EQUAL 64)
endif (WIN32)

# copy data files
file(COPY apps/data DESTINATION apps)

add_subdirectory(src/cudpp)
add_subdirectory(src/cudpp_hash)

if (CUDPP_BUILD_APPLICATIONS)
  add_subdirectory(apps/cudpp_testrig)
  add_subdirectory(apps/cudpp_hash_testrig)
  add_subdirectory(apps/simpleCUDPP)
  #add_subdirectory(apps/satGL)
endif (CUDPP_BUILD_APPLICATIONS)

# Make files that store configuration to help dependent CMake files use
# these libraries.
export(
  TARGETS cudpp cudpp_hash
  FILE ${LIBRARY_OUTPUT_PATH}/cudpp-targets.cmake
  )

set(cudpp_INCLUDE_DIRS_CONFIG ${cudpp_INCLUDE_DIRS})
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cudpp-config.cmake.in
  ${LIBRARY_OUTPUT_PATH}/cudpp-config.cmake
  @ONLY
  )

set(cudpp_INCLUDE_DIRS_CONFIG "\${_install_dir}/include")
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cudpp-config.cmake.in
  ${LIBRARY_OUTPUT_PATH}/cudpp-config.cmake.install
  @ONLY
  )
install(FILES ${LIBRARY_OUTPUT_PATH}/cudpp-config.cmake.install
  DESTINATION lib
  RENAME cudpp-config.cmake
  )

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cudpp-config-version.cmake.in
  ${LIBRARY_OUTPUT_PATH}/cudpp-config-version.cmake
  @ONLY
  )
install(FILES ${LIBRARY_OUTPUT_PATH}/cudpp-config-version.cmake
  DESTINATION lib
  )

set(CPACK_PACKAGE_VERSION_MAJOR ${cudpp_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${cudpp_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${cudpp_VERSION_PATCH})
include(CPack)
