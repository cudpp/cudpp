##############################################################################
#
# CUDA Data Parallel Primitives (CUDPP) Makefile
#
###############################################################################

###############################################################################
# SOURCE VARS
CCFILES := 	src/cudpp.cpp \
		src/cudpp_plan.cpp \
		src/cudpp_plan_manager.cpp \
		src/cudpp_maximal_launch.cpp
CUFILES :=	src/app/compact_app.cu \
		src/app/scan_app.cu \
		src/app/segmented_scan_app.cu \
		src/app/spmvmult_app.cu \
		src/app/radixsort_app.cu \
		src/app/reduce_app.cu \
		src/app/rand_app.cu 

SRCDIR := ./src/
CUSRCDIR := ./src/app/

INCLUDES := -I./ -I./include/ -Isrc/ -Isrc/app/ -Isrc/kernel/ -Isrc/cta/

ROOTDIR := .

CUDPP_STATIC_LIB := libcudpp.a

include ../linux_build/common.mk

