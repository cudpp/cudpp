// -------------------------------------------------------------
// CUDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision$
// $Date$
// ------------------------------------------------------------- 
// This source code is distributed under the terms of license.txt 
// in the root directory of this source distribution.
// ------------------------------------------------------------- 

#include <cudpp_globals.h>
#include "sharedmem.h"
#include <stdio.h>
#include "cta/decompress_cta.cuh"

/**
 * @file
 * decompress_kernel.cu
 * 
 * @brief CUDPP kernel-level decompress routines
 */

/** \addtogroup cudpp_kernel
 * @{
 */

/** @name Decompress Functions
 * @{
 */



/* ========================= NOTES ============================
 *
 *    - This file contains kernel functions.
 *    - They are defined as __global__ before the function definition.
 *    - Contains CUDA code that runs on the device and is called from the host.
 *
 */







typedef unsigned int uint;
typedef unsigned char uchar;
typedef unsigned short ushort;

/** @} */ // end decompress functions
/** @} */ // end cudpp_kernel
