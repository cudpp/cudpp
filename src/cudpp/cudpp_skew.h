// -------------------------------------------------------------
// cuDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision$
// $Date$
// ------------------------------------------------------------- 
// This source code is distributed under the terms of license.txt in
// the root directory of this source distribution.
// ------------------------------------------------------------- 

/**
* @file
* cudpp_skew.h
*
* @brief Suffix Array functionality header file - contains CUDPP interface (not public)
*/

#ifndef _CUDPP_SKEW_H_
#define _CUDPP_SKEW_H_

class CUDPPSkewPlan;

extern "C" 
void allocSkewStorage(CUDPPSkewPlan* plan);

extern "C" 
void freeSkewStorage(CUDPPSkewPlan* plan);

extern "C"
void cudppSuffixArrayDispatch(unsigned int* d_str, 
                              unsigned int* d_keys_sa, 
                              size_t d_str_length,
                              const CUDPPSkewPlan *plan);

#endif // _CUDPP_SKEW_H_
