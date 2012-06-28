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

/** @brief Perform Huffman encoding
 * 
 * @todo
 *
 */
void huffmanEncoding(unsigned int               *d_hist,
                     unsigned int               *d_encodeOffset,
                     unsigned int               *d_compressedSize,
                     unsigned int               *d_compressed,
                     size_t                     numElements,
                     const CUDPPCompressPlan    *plan)
{
    unsigned char* d_input  = plan->m_d_mtfOut;
    //d_hist                  = plan->m_d_histogram;
    //d_encodeOffset          = plan->m_d_encodeOffset;
    //d_compressedSize        = plan->m_d_totalEncodedSize;
    //d_compressed            = plan->m_d_encodedData;

    // Set work dimensions
    size_t nCodesPacked;
    size_t histBlocks = (numElements%(HUFF_WORK_PER_THREAD_HIST*HUFF_THREADS_PER_BLOCK_HIST)==0) ?
        numElements/(HUFF_WORK_PER_THREAD_HIST*HUFF_THREADS_PER_BLOCK_HIST) : numElements%(HUFF_WORK_PER_THREAD_HIST*HUFF_THREADS_PER_BLOCK_HIST)+1;
    size_t tThreads = ((numElements%HUFF_WORK_PER_THREAD) == 0) ? numElements/HUFF_WORK_PER_THREAD : numElements/HUFF_WORK_PER_THREAD+1;
    size_t nBlocks = ( (tThreads%HUFF_THREADS_PER_BLOCK) == 0) ? tThreads/HUFF_THREADS_PER_BLOCK : tThreads/HUFF_THREADS_PER_BLOCK+1;

    dim3 grid_hist(histBlocks, 1, 1);
    dim3 threads_hist(HUFF_THREADS_PER_BLOCK_HIST, 1, 1);

    dim3 grid_tree(1, 1, 1);
    dim3 threads_tree(128, 1, 1);

    dim3 grid_huff(nBlocks, 1, 1);
    dim3 threads_huff(HUFF_THREADS_PER_BLOCK, 1, 1);

    //---------------------------------------
    //  1) Build histogram from MTF output
    //---------------------------------------
    huffman_build_histogram_kernel<<< grid_hist, threads_hist>>>
        ((unsigned int*)d_input, plan->m_d_histograms, numElements);
    CUDA_SAFE_CALL(cudaThreadSynchronize());

    //----------------------------------------------------
    //  2) Compute final Histogram + Build Huffman codes
    //----------------------------------------------------
    huffman_build_tree_kernel<<< grid_tree, threads_tree>>>
        (d_input, plan->m_d_huffCodesPacked, plan->m_d_huffCodeLocations, plan->m_d_huffCodeLengths, plan->m_d_histograms,
	 d_hist, plan->m_d_nCodesPacked, d_compressedSize, histBlocks, numElements);
    CUDA_SAFE_CALL(cudaThreadSynchronize());

    //----------------------------------------------
    //  3) Main Huffman encoding step (encode data)
    //----------------------------------------------
    CUDA_SAFE_CALL(cudaMemcpy((void*)&nCodesPacked,  plan->m_d_nCodesPacked, sizeof(size_t), cudaMemcpyDeviceToHost));
    huffman_kernel_en<<< grid_huff, threads_huff, nCodesPacked*sizeof(unsigned char)>>>
        ((uchar4*)d_input, plan->m_d_huffCodesPacked, plan->m_d_huffCodeLocations, plan->m_d_huffCodeLengths,
	 plan->m_d_encoded, plan->m_d_nCodesPacked, tThreads);
    CUDA_SAFE_CALL(cudaThreadSynchronize());

    //--------------------------------------------------
    //  4) Pack together encoded data to determine how
    //     much encoded data needs to be transferred
    //--------------------------------------------------
    huffman_datapack_kernel<<<grid_huff, threads_huff>>>
        (plan->m_d_encoded, d_compressed, d_compressedSize, d_encodeOffset, nBlocks);
    CUDA_SAFE_CALL(cudaThreadSynchronize());
}


/** @brief Perform the MTF
 * 
 * @todo
 *
 */
template <class T>
void moveToFrontTransform(unsigned char             *d_mtfIn,
                          unsigned char             *d_mtfOut,
                          size_t                    numElements,
                          const T                   *plan)
{
    unsigned int nThreads = MTF_THREADS_BLOCK; 
    unsigned int nLists = numElements/MTF_PER_THREAD;
    unsigned int tThreads = numElements/MTF_PER_THREAD;
    unsigned int offset = 2;

    bool fullBlocks = (tThreads%nThreads == 0);
    unsigned int nBlocks = (fullBlocks) ? (tThreads/nThreads) : (tThreads/nThreads + 1);

    //-------------------------------------------
    //  Initial MTF lists + Initial Reduction
    //-------------------------------------------

    // Set work-item dimensions
    dim3 grid(nBlocks, 1, 1);
    dim3 threads(nThreads, 1, 1);

    // Kernel call
    mtf_reduction_kernel<<< grid, threads>>>
        (d_mtfIn, plan->m_d_lists, plan->m_d_list_sizes, nLists, offset);
    CUDA_SAFE_CALL(cudaThreadSynchronize());

    if(nBlocks > 1) 
    {
        //----------------------
        //  MTF Global Reduce
        //----------------------

        unsigned int init_offset = offset * nThreads;
        offset = init_offset;
        tThreads = nBlocks/2;
        fullBlocks = (tThreads%nThreads == 0);
        nBlocks = (fullBlocks) ? (tThreads/nThreads) : (tThreads/nThreads + 1);

        // Set work dimensions
        dim3 grid_GLred(nBlocks, 1, 1);
        dim3 threads_GLred(nThreads, 1, 1);

        while(offset <= nLists)
        {
            mtf_GLreduction_kernel<<< grid_GLred, threads_GLred>>>
                (plan->m_d_lists, plan->m_d_list_sizes, offset, tThreads, nLists);
            CUDA_SAFE_CALL(cudaThreadSynchronize());
            offset *= 2*nThreads;
        }

        //-----------------------------
        //  MTF Global Down-sweep
        //-----------------------------
        offset = nLists/4;
        unsigned int lastLevel = 0;

        // Work-dimensions
        dim3 grid_GLsweep(nBlocks, 1, 1);
        dim3 threads_GLsweep(nThreads, 1, 1);

        while(offset >= init_offset/2)
        {
            lastLevel = offset/nThreads;
            lastLevel = (lastLevel>=(init_offset/2)) ? lastLevel : init_offset/2;

            mtf_GLdownsweep_kernel<<< grid_GLsweep, threads_GLsweep>>>
                (plan->m_d_lists, plan->m_d_list_sizes, offset, lastLevel, nLists, tThreads);
            CUDA_SAFE_CALL(cudaThreadSynchronize());

            offset = lastLevel/2;
        }
    }

    //------------------------
    //      Local Scan
    //------------------------
    tThreads = numElements/MTF_PER_THREAD;
    offset = 2;
    fullBlocks = (tThreads%nThreads == 0);
    nBlocks = (fullBlocks) ? (tThreads/nThreads) : (tThreads/nThreads + 1);

    dim3 grid_loc(nBlocks, 1, 1);
    dim3 threads_loc(nThreads, 1, 1);

    mtf_localscan_lists_kernel<<< grid_loc, threads_loc>>>
        (d_mtfIn, d_mtfOut, plan->m_d_lists, plan->m_d_list_sizes, nLists, offset);
    CUDA_SAFE_CALL(cudaThreadSynchronize());

}

/** @brief Perform the BWT
 * 
 * @todo
 *
 */
template <class T>
void burrowsWheelerTransform(unsigned char              *d_uncompressed,
                             int                        *d_bwtIndex,
                             unsigned char              *d_bwtOut,
                             size_t                     numElements,
                             const T    *plan)
{
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
            (d_uncompressed, plan->m_d_values, d_bwtIndex, d_bwtOut, numElements, tThreads);
        CUDA_SAFE_CALL(cudaThreadSynchronize());
    }
    else
    {
        bwt_compute_final_kernel<<< grid_construct, threads_construct >>>
            (d_uncompressed, plan->m_d_values_dev, d_bwtIndex, d_bwtOut, numElements, tThreads);
        CUDA_SAFE_CALL(cudaThreadSynchronize());
    }

}

/** @brief Wrapper to call for calling BWT.
 *
 * @todo
 *
 */
void burrowsWheelerTransformWrapper(unsigned char *d_in,
                                    int *d_bwtIndex,
                                    size_t numElements,
                                    const CUDPPCompressPlan *plan)
{
    burrowsWheelerTransform<CUDPPCompressPlan>(d_in, d_bwtIndex, plan->m_d_bwtOut, numElements, plan);
}

/** @brief Wrapper to call for calling BWT.
 *
 * @todo
 *
 */
void burrowsWheelerTransformWrapper(unsigned char *d_in,
                                    int *d_bwtIndex,
                                    unsigned char *d_bwtOut,
                                    size_t numElements,
                                    const CUDPPBwtPlan *plan)
{
    burrowsWheelerTransform<CUDPPBwtPlan>(d_in, d_bwtIndex, d_bwtOut, numElements, plan);
}

/** @brief Wrapper to call for calling MTF.
 *
 * @todo
 *
 */
void moveToFrontTransformWrapper(size_t numElements,
                                 const CUDPPCompressPlan *plan)
{
    moveToFrontTransform<CUDPPCompressPlan>(plan->m_d_bwtOut, plan->m_d_mtfOut, numElements, plan);
}

/** @brief Wrapper to call for calling MTF.
 *
 * @todo
 *
 */
void moveToFrontTransformWrapper(unsigned char *d_in,
                                 unsigned char *d_mtfOut,
                                 size_t numElements,
                                 const CUDPPMtfPlan *plan)
{
    moveToFrontTransform<CUDPPMtfPlan>(d_in, d_mtfOut, numElements, plan);
}

#ifdef __cplusplus
extern "C" 
{
#endif

/** @brief Allocate intermediate arrays used by BWT.
 *
 * @todo
 *
 * @param [in,out] plan Pointer to CUDPPBwtPlan object containing options and number 
 *                      of elements, which is used to compute storage requirements, and
 *                      within which intermediate storage is allocated.
 */
void allocBwtStorage(CUDPPBwtPlan *plan)
{
    size_t numElts = plan->m_numElements;
    
    // BWT
    CUDA_SAFE_CALL(cudaMalloc((void**) &(plan->m_d_keys), numElts*sizeof(unsigned int) ));
    CUDA_SAFE_CALL(cudaMalloc((void**) &(plan->m_d_values), numElts*sizeof(unsigned int) ));
    
    CUDA_SAFE_CALL(cudaMalloc((void**) &(plan->m_d_bwtInRef), numElts*sizeof(unsigned int) ));
    CUDA_SAFE_CALL(cudaMalloc((void**) &(plan->m_d_bwtInRef2), numElts*sizeof(unsigned int) ));
    CUDA_SAFE_CALL(cudaMalloc((void**) &(plan->m_d_keys_dev), numElts*sizeof(unsigned int) ));
    CUDA_SAFE_CALL(cudaMalloc((void**) &(plan->m_d_values_dev), numElts*sizeof(unsigned int) ));
    
    CUDA_SAFE_CALL(cudaMalloc((void**)&(plan->m_d_partitionBeginA), 1024*sizeof(int)) );
    CUDA_SAFE_CALL(cudaMalloc((void**)&(plan->m_d_partitionSizeA), 1024*sizeof(int)) );
    CUDA_SAFE_CALL(cudaMalloc((void**)&(plan->m_d_partitionBeginB), 1024*sizeof(int)) );
    CUDA_SAFE_CALL(cudaMalloc((void**)&(plan->m_d_partitionSizeB), 1024*sizeof(int)) );
}
    
/** @brief Allocate intermediate arrays used by MTF.
 *
 * @todo
 *
 * @param [in,out] plan Pointer to CUDPPMtfPlan object containing
 *                      options and number of elements, which is used
 *                      to compute storage requirements, and within
 *                      which intermediate storage is allocated.
 */
void allocMtfStorage(CUDPPMtfPlan *plan)
{
    size_t numElts = plan->m_numElements;
    
    // MTF
    CUDA_SAFE_CALL(cudaMalloc( (void**) &(plan->m_d_lists), (numElts/MTF_PER_THREAD)*256*sizeof(unsigned char)));
    CUDA_SAFE_CALL(cudaMalloc( (void**) &(plan->m_d_list_sizes), (numElts/MTF_PER_THREAD)*sizeof(unsigned short)));
}
    
/** @brief Allocate intermediate arrays used by compression.
 *
 * @todo
 *
 * @param [in,out] plan Pointer to CUDPPCompressPlan object
 *                      containing options and number of elements,
 *                      which is used to compute storage
 *                      requirements, and within which intermediate
 *                      storage is allocated.
 */
void allocCompressStorage(CUDPPCompressPlan *plan)
{
    size_t numElts = plan->m_numElements;
    
    // BWT
    CUDA_SAFE_CALL(cudaMalloc((void**) &(plan->m_d_keys), numElts*sizeof(unsigned int) ));
    CUDA_SAFE_CALL(cudaMalloc((void**) &(plan->m_d_values), numElts*sizeof(unsigned int) ));
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
    
    // Huffman
    size_t numBitsAlloc = HUFF_NUM_CHARS*(HUFF_NUM_CHARS+1)/2;
    size_t numCharsAlloc = (numBitsAlloc%8 == 0) ? numBitsAlloc/8 : numBitsAlloc/8 + 1;
    size_t histBlocks = (numElts%(HUFF_WORK_PER_THREAD_HIST*HUFF_THREADS_PER_BLOCK_HIST)==0) ?
	numElts/(HUFF_WORK_PER_THREAD_HIST*HUFF_THREADS_PER_BLOCK_HIST) : numElts%(HUFF_WORK_PER_THREAD_HIST*HUFF_THREADS_PER_BLOCK_HIST)+1;
    size_t tThreads = ((numElts%HUFF_WORK_PER_THREAD) == 0) ? numElts/HUFF_WORK_PER_THREAD : numElts/HUFF_WORK_PER_THREAD+1;
    size_t nBlocks = ( (tThreads%HUFF_THREADS_PER_BLOCK) == 0) ? tThreads/HUFF_THREADS_PER_BLOCK : tThreads/HUFF_THREADS_PER_BLOCK+1;
    
    CUDA_SAFE_CALL(cudaMalloc( (void**) &(plan->m_d_huffCodesPacked), numCharsAlloc*sizeof(unsigned char) ));
    CUDA_SAFE_CALL(cudaMalloc( (void**) &(plan->m_d_huffCodeLocations), HUFF_NUM_CHARS*sizeof(size_t) ));
    CUDA_SAFE_CALL(cudaMalloc( (void**) &(plan->m_d_huffCodeLengths), HUFF_NUM_CHARS*sizeof(unsigned char) ));
    CUDA_SAFE_CALL(cudaMalloc( (void**) &(plan->m_d_histograms), histBlocks*256*sizeof(size_t) ));
    //CUDA_SAFE_CALL(cudaMalloc( (void**) &(plan->m_d_histogram), 256*sizeof(size_t) ));
    //CUDA_SAFE_CALL(cudaMalloc( (void**) &(plan->m_d_totalEncodedSize), sizeof(size_t)));
    //CUDA_SAFE_CALL(cudaMalloc( (void**) &(plan->m_d_encodedData), sizeof(size_t)*(HUFF_CODE_BYTES+1)*nBlocks));
    CUDA_SAFE_CALL(cudaMalloc( (void**) &(plan->m_d_nCodesPacked), sizeof(size_t)));
    CUDA_SAFE_CALL(cudaMalloc( (void**) &(plan->m_d_encoded), sizeof(encoded)*nBlocks));
    //CUDA_SAFE_CALL(cudaMalloc( (void**) &(plan->m_d_encodeOffset), sizeof(size_t)*nBlocks));
    
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
    // BWT
    CUDA_SAFE_CALL( cudaFree(plan->m_d_keys));
    CUDA_SAFE_CALL( cudaFree(plan->m_d_values));
    CUDA_SAFE_CALL( cudaFree(plan->m_d_bwtOut));
    
    CUDA_SAFE_CALL( cudaFree(plan->m_d_bwtInRef));
    CUDA_SAFE_CALL( cudaFree(plan->m_d_bwtInRef2));
    CUDA_SAFE_CALL( cudaFree(plan->m_d_keys_dev));
    CUDA_SAFE_CALL( cudaFree(plan->m_d_values_dev));
    
    CUDA_SAFE_CALL( cudaFree(plan->m_d_partitionBeginA));
    CUDA_SAFE_CALL( cudaFree(plan->m_d_partitionSizeA));
    CUDA_SAFE_CALL( cudaFree(plan->m_d_partitionBeginB));
    CUDA_SAFE_CALL( cudaFree(plan->m_d_partitionSizeB));

    // MTF
    CUDA_SAFE_CALL( cudaFree(plan->m_d_lists));
    CUDA_SAFE_CALL( cudaFree(plan->m_d_list_sizes));
    CUDA_SAFE_CALL( cudaFree(plan->m_d_mtfOut));

    // Huffman
    CUDA_SAFE_CALL(cudaFree(plan->m_d_histograms));
    //CUDA_SAFE_CALL(cudaFree(plan->m_d_histogram));
    CUDA_SAFE_CALL(cudaFree(plan->m_d_huffCodeLengths));
    CUDA_SAFE_CALL(cudaFree(plan->m_d_huffCodesPacked));
    CUDA_SAFE_CALL(cudaFree(plan->m_d_huffCodeLocations));
    //CUDA_SAFE_CALL(cudaFree(plan->m_d_totalEncodedSize));
    //CUDA_SAFE_CALL(cudaFree(plan->m_d_encodedData));
    CUDA_SAFE_CALL(cudaFree(plan->m_d_nCodesPacked));
    CUDA_SAFE_CALL(cudaFree(plan->m_d_encoded));
    //CUDA_SAFE_CALL(cudaFree(plan->m_d_encodeOffset));

    CUDA_CHECK_ERROR("freeCompressStorage");
}

/** @brief Deallocate intermediate block arrays in a CUDPPBwtPlan object.
 *
 * @todo 
 *
 * @param[in,out] plan Pointer to CUDPPBwtPlan object initialized by allocBwtStorage().
 */
void freeBwtStorage(CUDPPBwtPlan *plan)
{
    // BWT
    CUDA_SAFE_CALL( cudaFree(plan->m_d_keys));
    CUDA_SAFE_CALL( cudaFree(plan->m_d_values));

    CUDA_SAFE_CALL( cudaFree(plan->m_d_bwtInRef));
    CUDA_SAFE_CALL( cudaFree(plan->m_d_bwtInRef2));
    CUDA_SAFE_CALL( cudaFree(plan->m_d_keys_dev));
    CUDA_SAFE_CALL( cudaFree(plan->m_d_values_dev));

    CUDA_SAFE_CALL( cudaFree(plan->m_d_partitionBeginA));
    CUDA_SAFE_CALL( cudaFree(plan->m_d_partitionSizeA));
    CUDA_SAFE_CALL( cudaFree(plan->m_d_partitionBeginB));
    CUDA_SAFE_CALL( cudaFree(plan->m_d_partitionSizeB));
}

/** @brief Deallocate intermediate block arrays in a CUDPPMtfPlan object.
 *
 * @todo 
 *
 * @param[in,out] plan Pointer to CUDPPMtfPlan object initialized by allocMtfStorage().
 */
void freeMtfStorage(CUDPPMtfPlan *plan)
{
    // MTF
    CUDA_SAFE_CALL( cudaFree(plan->m_d_lists));
    CUDA_SAFE_CALL( cudaFree(plan->m_d_list_sizes));
}

/** @brief Dispatch function to perform parallel compression on an
 * array with the specified configuration.
 *
 * @todo
 * 
 * @param[in]  d_uncompressed Uncompressed data
 * @param[out] d_bwtIndex BWT Index
 * @param[out] d_histSize Histogram size
 * @param[out] d_hist Histogram
 * @param[out] d_encodeOffset Encoded offset table
 * @param[out] d_compressedSize Size of compressed data
 * @param[out] d_compressed Compressed data
 * @param[in]  numElements Number of elements to compress
 * @param[in]  plan     Pointer to CUDPPCompressPlan object containing
 *                      compress options and intermediate storage
 */
void cudppCompressDispatch(void *d_uncompressed,
			   void *d_bwtIndex,
			   void *d_histSize, // ignore
			   void *d_hist,
			   void *d_encodeOffset,
			   void *d_compressedSize,
			   void *d_compressed,
			   size_t numElements,
			   const CUDPPCompressPlan *plan)
{
    // Call to perform the Burrows-Wheeler transform
    burrowsWheelerTransformWrapper((unsigned char*)d_uncompressed, (int*)d_bwtIndex,
				   numElements, plan);

    // Call to perform the move-to-front transform
    moveToFrontTransformWrapper(numElements, plan);

    // Call to perform the Huffman encoding
    huffmanEncoding((unsigned int*)d_hist, (unsigned int*)d_encodeOffset,
		    (unsigned int*)d_compressedSize, (unsigned int*)d_compressed, numElements, plan);
}


/** @brief Dispatch function to perform the Burrows-Wheeler transform
 *
 * @todo
 * 
 * @param[in]  d_bwtIn     Input data
 * @param[out] d_bwtOut    Transformed data
 * @param[out] d_bwtIndex  BWT Index
 * @param[in]  numElements Number of elements to compress
 * @param[in]  plan        Pointer to CUDPPBwtPlan object containing
 *                         compress options and intermediate storage
 */
void cudppBwtDispatch(void *d_bwtIn,
		      void *d_bwtOut,
		      void *d_bwtIndex,
		      size_t numElements,
		      const CUDPPBwtPlan *plan)
{
    // Call to perform the Burrows-Wheeler transform
    burrowsWheelerTransformWrapper((unsigned char*)d_bwtIn, (int*)d_bwtIndex,
				   (unsigned char*) d_bwtOut, numElements, 
				   plan);
}


/** @brief Dispatch function to perform the Move-to-Front transform
 *
 * @todo
 * 
 * @param[in]  d_mtfIn     Input data
 * @param[out] d_mtfOut    Transformed data
 * @param[in]  numElements Number of elements to compress
 * @param[in]  plan        Pointer to CUDPPMtfPlan object containing
 *                         compress options and intermediate storage
 */
void cudppMtfDispatch(void *d_mtfIn,
		      void *d_mtfOut,
		      size_t numElements,
		      const CUDPPMtfPlan *plan)
{
    // Call to perform the Burrows-Wheeler transform
    moveToFrontTransformWrapper((unsigned char*) d_mtfIn, 
				(unsigned char*) d_mtfOut, numElements, plan);
}

#ifdef __cplusplus
}
#endif

/** @} */ // end compress functions
/** @} */ // end cudpp_app
