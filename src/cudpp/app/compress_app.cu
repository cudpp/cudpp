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

#include "cuda_util.h"
#include "cudpp_globals.h"
#include "cudpp.h"
#include "cudpp_util.h"
#include "cudpp_plan.h"

#include "kernel/compress_kernel.cuh"

/**
  * @file
  * compress_app.cu
  * 
  * @brief CUDPP application-level compress routines
  */

/** \addtogroup cudpp_app 
  * @{
  */

/** @name Compress Functions
 * @{
 */

/** @brief Perform the BWT
  * 
  * @todo
  *
  */
void burrowsWheelerTransform(unsigned char *d_uncompressed,
                             int *d_bwtIndex,
                             size_t numElements,
                             const CUDPPCompressPlan *plan)
{
    // set ptrs
    d_bwtIndex = plan->m_d_bwtIndex;

    size_t tThreads = (numElements%4 == 0) ? numElements/4 : numElements/4 + 1;
    size_t nThreads = BWT_CTA_BLOCK;
    bool fullBlocks = (tThreads%nThreads == 0);
    uint nBlocks = (fullBlocks) ? (tThreads/nThreads) : (tThreads/nThreads+1);
    dim3 grid_construct(nBlocks, 1, 1);
    dim3 threads_construct(nThreads, 1, 1);
    int numThreads = 64;
	int secondBlocks;
    size_t count;
    size_t mult;
    size_t numBlocks;
    int initSubPartitions;
    int subPartitions;
    int step;

    // Massage input to create sorting key-value pairs
    bwt_keys_construct_kernel<<< grid_construct, threads_construct >>>
        ((uchar4*)d_uncompressed, plan->m_d_bwtInRef,
        plan->m_d_keys, plan->m_d_values, plan->m_d_bwtInRef2, tThreads);
    CUDA_SAFE_CALL(cudaThreadSynchronize());

    // First satge -- block sort
    nBlocks = numElements/BWT_BLOCKSORT_SIZE;
    dim3 grid_blocksort(nBlocks, 1, 1);
    dim3 threads_blocksort(BWT_CTA_BLOCK, 1, 1);

    blockWiseStringSort<unsigned int, 8><<<grid_blocksort, threads_blocksort>>>
        (plan->m_d_keys, plan->m_d_values, (const unsigned int*)plan->m_d_bwtInRef, plan->m_d_bwtInRef2, BWT_BLOCKSORT_SIZE, numElements);
    CUDA_SAFE_CALL(cudaThreadSynchronize());

    // Start merging blocks
    // Second stage -- merge sorted blocks using simple merge
    count = 0;
    mult = 1;
    numBlocks = nBlocks;

    while(count < 6)
    {
        if(count%2 == 0)
        {
            simpleStringMerge<unsigned int, 2><<<numBlocks, BWT_CTASIZE_simple, sizeof(unsigned int)*(2*BWT_INTERSECT_B_BLOCK_SIZE_simple+2)>>>
                (plan->m_d_keys, plan->m_d_keys_dev, plan->m_d_values, plan->m_d_values_dev,
                plan->m_d_bwtInRef, BWT_BLOCKSORT_SIZE*mult, numBlocks*BWT_BLOCKSORT_SIZE, plan->m_d_bwtInRef2, numElements);
            CUDA_SAFE_CALL(cudaThreadSynchronize());

        }
        else
        {
            simpleStringMerge<unsigned int, 2><<<numBlocks, BWT_CTASIZE_simple, sizeof(unsigned int)*(2*BWT_INTERSECT_B_BLOCK_SIZE_simple+2)>>>
                (plan->m_d_keys_dev, plan->m_d_keys, plan->m_d_values_dev, plan->m_d_values,
                plan->m_d_bwtInRef, BWT_BLOCKSORT_SIZE*mult, numBlocks*BWT_BLOCKSORT_SIZE, plan->m_d_bwtInRef2, numElements);
            CUDA_SAFE_CALL(cudaThreadSynchronize());
        }

        mult*=2;
        count++;
        numBlocks /= 2;
    }

    // Third stage -- merge remaining blocks using multi-merge
    initSubPartitions = 2;
    subPartitions = initSubPartitions;
    secondBlocks = (2*numBlocks*initSubPartitions+numThreads-1)/numThreads;
    step = 1;

    while (numBlocks > 1)
    {
        if(count%2 == 1)
        {
            findMultiPartitions<unsigned int><<<secondBlocks, numThreads>>>
                (plan->m_d_keys_dev, subPartitions, numBlocks, BWT_BLOCKSORT_SIZE*mult,
                plan->m_d_partitionBeginA, plan->m_d_partitionSizeA, plan->m_d_partitionBeginB, plan->m_d_partitionSizeB, BWT_SIZE);
            CUDA_SAFE_CALL(cudaThreadSynchronize());

            stringMergeMulti<unsigned int, 2><<<numBlocks*subPartitions, BWT_CTASIZE_multi, (2*BWT_INTERSECT_B_BLOCK_SIZE_multi+5)*sizeof(unsigned int)>>>
                (plan->m_d_keys_dev, plan->m_d_keys, plan->m_d_values_dev, plan->m_d_values, plan->m_d_bwtInRef2, subPartitions, numBlocks,
                plan->m_d_partitionBeginA, plan->m_d_partitionSizeA, plan->m_d_partitionBeginB, plan->m_d_partitionSizeB, BWT_BLOCKSORT_SIZE*mult, step, numElements);
            CUDA_SAFE_CALL(cudaThreadSynchronize());
        }
        else
        {
            findMultiPartitions<unsigned int><<<secondBlocks, numThreads>>>
                (plan->m_d_keys, subPartitions, numBlocks, BWT_BLOCKSORT_SIZE*mult,
                plan->m_d_partitionBeginA, plan->m_d_partitionSizeA, plan->m_d_partitionBeginB, plan->m_d_partitionSizeB, BWT_SIZE);
            CUDA_SAFE_CALL(cudaThreadSynchronize());

            stringMergeMulti<unsigned int, 2><<<numBlocks*subPartitions, BWT_CTASIZE_multi, (2*BWT_INTERSECT_B_BLOCK_SIZE_multi+5)*sizeof(unsigned int)>>>
                (plan->m_d_keys, plan->m_d_keys_dev, plan->m_d_values, plan->m_d_values_dev, plan->m_d_bwtInRef2, subPartitions, numBlocks,
                plan->m_d_partitionBeginA, plan->m_d_partitionSizeA, plan->m_d_partitionBeginB, plan->m_d_partitionSizeB, BWT_BLOCKSORT_SIZE*mult, step, numElements);
            CUDA_SAFE_CALL(cudaThreadSynchronize());
        }
        numBlocks/=2;
        subPartitions*=2;
        count++;
        mult*=2;
        step++;
    }

    // Final stage -- compute BWT and BWT Index using sorted values
    if(count%2 == 0)
    {
        bwt_compute_final_kernel<<< grid_construct, threads_construct >>>
            (d_uncompressed, plan->m_d_values, d_bwtIndex, plan->m_d_bwtOut, numElements, tThreads);
        CUDA_SAFE_CALL(cudaThreadSynchronize());
    }
    else
    {
        bwt_compute_final_kernel<<< grid_construct, threads_construct >>>
            (d_uncompressed, plan->m_d_values_dev, d_bwtIndex, plan->m_d_bwtOut, numElements, tThreads);
        CUDA_SAFE_CALL(cudaThreadSynchronize());
    }

}


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
void allocCompressStorage(CUDPPCompressPlan *plan)
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
    CUDA_SAFE_CALL(cudaMalloc( (void**) &(plan->m_d_lists), (numElts/MTF_PER_THREAD)*256*sizeof(unsigned char)));
    CUDA_SAFE_CALL(cudaMalloc( (void**) &(plan->m_d_list_sizes), (numElts/MTF_PER_THREAD)*sizeof(unsigned short)));
    CUDA_SAFE_CALL(cudaMalloc( (void**) &(plan->m_d_mtfOut), numElts*sizeof(unsigned char) ));

    CUDA_CHECK_ERROR("allocCompressStorage");
}

/** @brief Deallocate intermediate block arrays in a CUDPPCompressPlan object.
  *
  * @todo 
  *
  * @param[in,out] plan Pointer to CUDPPCompressPlan object initialized by allocCompressStorage().
  */
void freeCompressStorage(CUDPPCompressPlan *plan)
{
    CUDA_SAFE_CALL( cudaFree(plan->m_d_keys));
    CUDA_SAFE_CALL( cudaFree(plan->m_d_values));
    CUDA_SAFE_CALL( cudaFree(plan->m_d_bwtIndex));
    CUDA_SAFE_CALL( cudaFree(plan->m_d_bwtOut));

    CUDA_SAFE_CALL( cudaFree(plan->m_d_bwtInRef));
    CUDA_SAFE_CALL( cudaFree(plan->m_d_bwtInRef2));
    CUDA_SAFE_CALL( cudaFree(plan->m_d_keys_dev));
    CUDA_SAFE_CALL( cudaFree(plan->m_d_values_dev));

    CUDA_SAFE_CALL( cudaFree(plan->m_d_partitionBeginA));
    CUDA_SAFE_CALL( cudaFree(plan->m_d_partitionSizeA));
    CUDA_SAFE_CALL( cudaFree(plan->m_d_partitionBeginB));
    CUDA_SAFE_CALL( cudaFree(plan->m_d_partitionSizeB));

    CUDA_SAFE_CALL( cudaFree(plan->m_d_lists));
    CUDA_SAFE_CALL( cudaFree(plan->m_d_list_sizes));
    CUDA_SAFE_CALL( cudaFree(plan->m_d_mtfOut));

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
    // Call to perform the Burrows-Wheeler transform
    burrowsWheelerTransform((unsigned char*)d_uncompressed, (int*)d_bwtIndex,
        numElements, plan);

    // Call to perform the move-to-front transform
    //moveToFrontTransform();
}

#ifdef __cplusplus
}
#endif

/** @} */ // end compress functions
/** @} */ // end cudpp_app
