// -------------------------------------------------------------
// CUDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision$
// $Date$
// ------------------------------------------------------------- 
// This source code is distributed under the terms of license.txt 
// in the root directory of this source distribution.
// -------------------------------------------------------------

/**
 * @file
 * decompress_cta.cu
 *
 * @brief CUDPP CTA-level decompress routines
 */

/** \addtogroup cudpp_cta 
 * @{
 */

/** @name Deompress Functions
 * @{
 */

#include <stdio.h>
#include <cudpp_globals.h>



/* ========================== NOTES ==========================
 *
 *    - This file contains CTA (CUDA block level) functions
 *    - These functions are defined by __device__ before the function definitions.
 *    - They can only be called from within kernel (__global__) functions.
 *    - They operate on code within a block/warp (ie - for searching, etc...)
 *
 */





/** @} */ // end compress functions
/** @} */ // end cudpp_cta
