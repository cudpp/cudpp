##############################################################################
#
# CUDA Data Parallel Primitives (CUDPP) Makefile
#
###############################################################################

###############################################################################
# SOURCE VARS
CUFILES :=	src/hash_functions.cu \
		src/hash_table.cu \
		src/hash_compacting.cu \
		src/hash_multivalue.cu \
		src/debugging.cu
CCFILES :=      src/cudpp_hash.cpp

# disable SM10 since hash requires atomics that SM10 doesn't have
GENCODE_SM10 := 

SRCDIR := ./src/
CUSRCDIR := ./src/

# NVCCFLAGS := -D_DEBUG -g

INCLUDES := -I./include/ -I./src/ -I../cudpp/include -I../cudpp/src -I../apps/common

ROOTDIR := .

STATIC_LIB := libcudpp_hash.a

include ../linux_build/common.mk

