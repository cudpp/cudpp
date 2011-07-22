###############################################################################
#
# Build script for project
#
###############################################################################

###############################################################################
# SOURCE VARS
EXECUTABLE	:= hash_testrig

CCFILES :=	hash_testrig.cpp random_numbers.cpp mt19937ar.cpp

INCLUDES := -I../../cudpp/include -I../../cudpp_hash/include -I../common/ 

USECUDPP := 1
USECUDPP_HASH := 1

include ../../linux_build/common.mk

