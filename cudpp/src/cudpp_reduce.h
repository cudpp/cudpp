// -------------------------------------------------------------
// cuDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision$
// $Date$
// ------------------------------------------------------------- 
// This source code is distributed under the terms of license.txt
// in the root directory of this source distribution.
// ------------------------------------------------------------- 
#ifndef _CUDPP_REDUCE_H_
#define _CUDPP_REDUCE_H_

class CUDPPReducePlan;

extern "C"
void allocReduceStorage(CUDPPReducePlan* plan);

extern "C"
void freeReduceStorage(CUDPPReducePlan* plan);

extern "C"
void cudppReduceDispatch(void               *d_odata, 
                         const void         *d_idata, 
                         size_t             numElements, 
                         const CUDPPReducePlan *plan);

extern "C"
void tuneReduce(CUDPPReducePlan *plan, CUDPPTuneConfig *tuneConfig);

#endif


 // __RADIXSORT_H__
