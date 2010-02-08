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
* tridiagonal.h
*
* @brief tridiagonal functionality header file - contains CUDPP interface (not public)
*/

#ifndef __CUDPP_TRIDIAGONAL_H__
#define __CUDPP_TRIDIAGONAL_H__

#include "cudpp_globals.h"
#include "cudpp.h"
#include "cudpp_plan.h"

extern "C"
void cudppTridiagonalDispatch(void *a, void *b, void *c, void *d, void *x, int system_size, int num_systems, const CUDPPTridiagonalPlan * plan);

#endif //__CUDPP_TRIDIAGONAL_H__


