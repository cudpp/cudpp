################################################################################
#
# Build script for project
#
################################################################################

# Add source files here
EXECUTABLE	:= satGL
# Cuda source files (compiled with cudacc)
CUFILES		:= satGL.cu
# C/C++ source files (compiled with gcc / c++)
CCFILES		:= main.cpp

INCLUDES	:= -I../../cudpp/include -I../common/

USEGLLIB	:= 1
USEGLUT		:= 1
USECUDPP	:= 1

################################################################################
# Rules and targets

include ../../linux_build/common.mk
