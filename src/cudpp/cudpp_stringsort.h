// -------------------------------------------------------------
// cuDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision$
// $Date$
// ------------------------------------------------------------- 
// This source code is distributed under the terms of license.txt
// in the root directory of this source distribution.
// ------------------------------------------------------------- 
#ifndef   __STRINGSORT_H__
#define   __STRINGSORT_H__

#include "cudpp_globals.h"
#include "cudpp.h"
#include "cudpp_plan.h"

extern "C"
void allocStringSortStorage(CUDPPStringSortPlan* plan);

extern "C"
void freeStringSortStorage(CUDPPStringSortPlan* plan);

extern "C"
void cudppStringSortDispatch(unsigned int       *keys,
                            unsigned int        *values,
                            unsigned int        *stringVals,
                            size_t      numElements,
							size_t      stringArrayLength,
                            const       CUDPPStringSortPlan *plan);


#endif // __STRINGSORT_H__
