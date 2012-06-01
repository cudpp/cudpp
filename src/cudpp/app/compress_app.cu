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
    size_t numElts = plan->m_numElements;

    // BWT
    CUDA_SAFE_CALL(cudaMalloc((void**) &(plan->m_d_keys), numElts*sizeof(unsigned int) ));
    CUDA_SAFE_CALL(cudaMalloc((void**) &(plan->m_d_values), numElts*sizeof(unsigned int) ));
    CUDA_SAFE_CALL(cudaMalloc((void**) &(plan->m_d_bwtIndex), sizeof(int) ));
    CUDA_SAFE_CALL(cudaMalloc( (void**) &(plan->m_d_bwtOut), numElts*sizeof(unsigned char) ));

    CUDA_SAFE_CALL(cudaMalloc((void**) &(plan->m_d_bwtInRef), numElts*sizeof(unsigned int) ));
    CUDA_SAFE_CALL(cudaMalloc((void**) &(plan->m_d_bwtInRef2), numElts*sizeof(unsigned int) ));
    CUDA_SAFE_CALL(cudaMalloc((void**) &(plan->m_d_keys_dev), numElts*sizeof(unsigned int) ));
    CUDA_SAFE_CALL(cudaMalloc((void**) &(plan->m_d_values_dev), numElts*sizeof(unsigned int) ));

    CUDA_SAFE_CALL(cudaMalloc((void**)&(plan->m_d_partitionBeginA), 1024*sizeof(int)) );
    CUDA_SAFE_CALL(cudaMalloc((void**)&(plan->m_d_partitionSizeA), 1024*sizeof(int)) );
    CUDA_SAFE_CALL(cudaMalloc((void**)&(plan->m_d_partitionBeginB), 1024*sizeof(int)) );
    CUDA_SAFE_CALL(cudaMalloc((void**)&(plan->m_d_partitionSizeB), 1024*sizeof(int)) );

    // MTF
    CUDA_SAFE_CALL(cudaMalloc( (void**) &(plan->m_d_lists), (numElts/PER_THREAD)*256*sizeof(unsigned char)));
    CUDA_SAFE_CALL(cudaMalloc( (void**) &(plan->m_d_list_sizes), (numElts/PER_THREAD)*sizeof(unsigned short)));
    CUDA_SAFE_CALL(cudaMalloc( (void**) &(plan->m_d_mtfOut), numElts*sizeof(unsigned char) ));

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
    plan->m_d_mtfIn = plan->m_d_bwtOut;
}

#ifdef __cplusplus
}
#endif

/** @} */ // end compress functions
/** @} */ // end cudpp_app
