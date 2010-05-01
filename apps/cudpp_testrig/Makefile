###############################################################################
#
# Build script for project
#
###############################################################################

###############################################################################
# SOURCE VARS
EXECUTABLE	:= cudpp_testrig

CCFILES :=	scan_gold.cpp compact_gold.cpp spmvmult_gold.cpp reduce_gold.cpp \
		cudpp_testrig_options.cpp
CUFILES :=	cudpp_testrig.cu

INCLUDES := -I../../cudpp/include -I../../common/inc

USECUDPP := 1

include ../../linux_build/common.mk

