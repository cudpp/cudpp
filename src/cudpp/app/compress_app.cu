// -------------------------------------------------------------
// cuDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
//  $Revision$
//  $Date$
// ------------------------------------------------------------- 
// This source code is distributed under the terms of license.txt in
// the root directory of this source distribution.
// ------------------------------------------------------------- 

/**
  * @file
  * compress_app.cu
  * 
  * @brief CUDPP application-level compress routines
  */

#include <stdio.h>
#include <stdlib.h>

#include "cuda_util.h"
#include "cudpp.h"
#include "cudpp_util.h"
#include "cudpp_plan.h"

/** \addtogroup cudpp_app 
  * @{
  */

/** @name Compress Functions
 * @{
 */




#ifdef __cplusplus
extern "C" 
{
#endif

/** @brief Allocate intermediate arrays used by compression.
  *
  * @todo
  *
  * @param [in,out] plan Pointer to CUDPPCompressPlan object containing options and number 
  *                      of elements, which is used to compute storage requirements, and
  *                      within which intermediate storage is allocated.
  */
void allocCompressStorage(CUDPPReducePlan *plan)
{
    //todo
    CUDA_CHECK_ERROR("allocCompressStorage");
}

/** @brief Deallocate intermediate block arrays in a CUDPPCompressPlan object.
  *
  * @todo 
  *
  * @param[in,out] plan Pointer to CUDPPCompressPlan object initialized by allocCompressStorage().
  */
void freeCompressStorage(CUDPPReducePlan *plan)
{
    //todo
    CUDA_CHECK_ERROR("freeCompressStorage");
}

/** @brief Dispatch function to perform parallel compression on an
 * array with the specified configuration.
 *
 * @todo
 * 
 * @param[in]  d_a Uncompressed data
 * @param[out] d_x BWT Index
 * @param[out] d_y Histogram size
 * @param[out] d_z Histogram
 * @param[out] d_w Encoded offset table
 * @param[out] d_xx Size of compressed data
 * @param[out] d_yy Compressed data
 * @param[in]  numElements Number of elements to compress
 * @param[in]  plan     Pointer to CUDPPCompressPlan object containing
 *                      compress options and intermediate storage
 */
void cudppCompressDispatch(void *d_uncompressed,
                           void *d_bwtIndex,
                           void *d_histSize,
                           void *d_hist,
                           void *d_encodeOffset,
                           void *d_compressedSize,
                           void *d_compressed,
                           size_t numElements,
                           const CUDPPCompressPlan *plan)
{
}

#ifdef __cplusplus
}
#endif

/** @} */ // end compress functions
/** @} */ // end cudpp_app
