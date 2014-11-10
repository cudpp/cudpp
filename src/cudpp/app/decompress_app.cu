// -------------------------------------------------------------
// CUDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision$
// $Date$
// ------------------------------------------------------------- 
// This source code is distributed under the terms of license.txt 
// in the root directory of this source distribution.
// ------------------------------------------------------------- 

#include <stdio.h>
#include <stdlib.h>
#include <fstream>

#include "cuda_util.h"
#include "cudpp_globals.h"
#include "cudpp.h"
#include "cudpp_util.h"
#include "cudpp_plan.h"
#include "cudpp_sa.h"

#include "kernel/decompress_kernel.cuh"

using namespace std;

/**
 * @file
 * decompress_app.cu
 * 
 * @brief CUDPP application-level decompress routines
 */

/** \addtogroup cudpp_app 
 * @{
 */

/** @name Deompress Functions
 * @{
 */

void cudppDecompressDispatch(void ) { }

/* ========================= NOTES ==========================
 *
 *    - Functions in this file call kernel functions. 
 *    - They are not __global__ or __device__ functions as they are not CUDA code
 *
 */




#ifdef __cplusplus
}
#endif

/** @} */ // end decompress functions
/** @} */ // end cudpp_app
