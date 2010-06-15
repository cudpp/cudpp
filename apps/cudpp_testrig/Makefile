###############################################################################
#
# Build script for project
#
###############################################################################

###############################################################################
# SOURCE VARS
EXECUTABLE	:= cudpp_testrig

CCFILES :=	cudpp_testrig.cpp cudpp_testrig_options.cpp cudpp_testrig_utils.cpp \
		scan_gold.cpp compact_gold.cpp spmvmult_gold.cpp rand_gold.cpp \
		test_radixsort.cpp test_rand.cpp test_reduce.cpp \
		test_scan.cpp test_spmvmult.cpp test_compact.cpp

INCLUDES := -I../../cudpp/include -I../../common/inc

USECUDPP := 1

include ../../linux_build/common.mk

