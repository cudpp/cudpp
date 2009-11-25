// ***************************************************************
//  cuDPP -- CUDA Data Parallel Primitives library
//  -------------------------------------------------------------
//  $Revision: 3505 $
//  $Date: 2007-07-06 09:26:06 -0700 (Fri, 06 Jul 2007) $
//  -------------------------------------------------------------
// This source code is distributed under the terms of license.txt in
// the root directory of this source distribution.
// ------------------------------------------------------------- 

/**
* @file
* segmented_scan_app.cu
*
* @brief CUDPP application-level scan routines
*/

/** \addtogroup cudpp_app
  *
  */

/** @name Segmented Scan Functions
* @{
*/

#include "cudpp.h"
#include "cudpp_util.h"
#include "cudpp_plan.h"
#include "kernel/segmented_scan_kernel.cu"
#include "kernel/vector_kernel.cu"


#include <cutil.h>
#include <cstdlib>
#include <cstdio>
#include <assert.h>

/** @brief Perform recursive scan on arbitrary size arrays
*
* This is the CPU-side workhorse function of the segmented scan
* engine. This function invokes the CUDA kernels which perform the
* segmented scan on individual blocks.
*
* Scans of large arrays must be split (possibly recursively) into a
* hierarchy of block scans, where each block is scanned by a single
* CUDA thread block. At each recursive level of the
* segmentedScanArrayRecursive first invokes a kernel to scan all blocks of
* that level, and if the level has more than one block, it calls
* itself recursively. On returning from each recursive level, the
* total sum of each block from the level below is added to all
* elements of the first segment of the corresponding block in this
* level.
*
* Template parameter T is the data type of the input data.
* Template parameter op is the binary operator of the segmented scan.
* Template parameter isBackward specifies whether the direction is backward
* (not implemented). It is forward if it is false.
* Template parameter isExclusive specifies whether the segmented scan
* is exclusive (true) or inclusive (false).
*
* @param[out] d_out The output array for the segmented scan results
* @param[in] d_idata The input array to be scanned
* @param[in] d_iflags The input flags vector which specifies the
* segments. The first element of a segment is marked by a 1 in the
* corresponding position in d_iflags vector. All other elements of
* d_iflags is 0.
* @param[out] d_blockSums Array of arrays of per-block sums (one
* array per recursive level, allocated
* by allocScanStorage())
* @param[out] d_blockFlags Array of arrays of per-block OR-reductions
* of flags (one array per recursive level, allocated by
* allocScanStorage())
* @param[out] d_blockIndices Array of arrays of per-block
* min-reductions of indices (one array per recursive level, allocated
* by allocSegmentedScanStorage()). An index for a particular position \c i in
* a block is calculated as - if \c d_iflags[i] is set then it is the
* 1-based index of that position (i.e if \c d_iflags[10] is set then
* index is \c 11) otherwise the index is \c INT_MAX (the identity
* element of a min operator)
* @param[in] numElements The number of elements in the array to scan
* @param[in] level The current recursive level of the scan
*/
template <class T, CUDPPOperator op, bool isBackward, bool isExclusive, bool doShiftFlagsLeft>
void segmentedScanArrayRecursive(T                  *d_out, 
                                 const T            *d_idata, 
                                 const unsigned int *d_iflags,
                                 T                  **d_blockSums,
                                 unsigned int       **d_blockFlags,
                                 unsigned int       **d_blockIndices,
                                 int                numElements,
                                 int                level)
{
    unsigned int numBlocks = 
        max(1, (int)ceil((double)numElements / 
        ((double)SEGSCAN_ELTS_PER_THREAD * SCAN_CTA_SIZE)));

    // This is the number of elements per block that the 
    // CTA level API is aware of
    unsigned int numEltsPerBlock = SCAN_CTA_SIZE * 2;

    // Space to store flags - we need two sets. One gets modified and the
    // other doesn't
    unsigned int flagSpace = numEltsPerBlock * sizeof(unsigned int);

    // Space to store indices
    unsigned int idxSpace = numEltsPerBlock * sizeof(unsigned int);

    // Total shared memory space
    unsigned int sharedMemSize = 
        sizeof(T) * (numEltsPerBlock) + idxSpace + flagSpace;

    // setup execution parameters
    dim3  grid(max(1, numBlocks), 1, 1);
    dim3  threads(SCAN_CTA_SIZE, 1, 1);

    // make sure there are no CUDA errors before we start
    CUT_CHECK_ERROR("segmentedScanArrayRecursive before kernels");

    bool fullBlock = (numElements == 
        (numBlocks * SEGSCAN_ELTS_PER_THREAD * SCAN_CTA_SIZE));    

    bool sm12OrBetterHw;
    cudaDeviceProp deviceProp;
    int dev;
    CUDA_SAFE_CALL(cudaGetDevice(&dev));
    CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, dev));
    if (deviceProp.minor >= 2)
        sm12OrBetterHw = true;
    else
        sm12OrBetterHw = false;

    unsigned int traitsCode = 0;
    if (numBlocks > 1)  traitsCode |= 1;
    if (fullBlock)      traitsCode |= 2;
    if (sm12OrBetterHw) traitsCode |= 4;

    switch(traitsCode)
    {
    case 0: // single block, single row, non-full last block
        segmentedScan4<T, SegmentedScanTraits<T, op, isBackward, isExclusive, doShiftFlagsLeft, false, false,
                       false> >
            <<< grid, threads, sharedMemSize >>>
            (d_out, d_idata, d_iflags, numElements, 0, 0, 0);
        break;
    case 1: // multi block, single row, non-full last block
        segmentedScan4<T, SegmentedScanTraits<T, op, isBackward, isExclusive, doShiftFlagsLeft, false, true,
                       false> >
            <<< grid, threads, sharedMemSize >>>
            (d_out, d_idata, d_iflags, numElements,
            d_blockSums[level], d_blockFlags[level],
            d_blockIndices[level]);
        break;
    case 2: // single block, single row, full last block
        segmentedScan4<T, SegmentedScanTraits<T, op, isBackward, isExclusive, doShiftFlagsLeft, true, false,
                       false> >
            <<< grid, threads, sharedMemSize >>>
            (d_out, d_idata, d_iflags, numElements, 0, 0, 0);
        break;
    case 3: // multi block, single row, full last block
        segmentedScan4<T, SegmentedScanTraits<T, op, isBackward, isExclusive, doShiftFlagsLeft, true, true,
                       false> >
            <<< grid, threads, sharedMemSize >>>
            (d_out, d_idata, d_iflags, numElements,
            d_blockSums[level], d_blockFlags[level],
            d_blockIndices[level]);
        break;
    case 4: // single block, single row, non-full last block
        segmentedScan4<T, SegmentedScanTraits<T, op, isBackward, isExclusive, doShiftFlagsLeft, false, false,
                       true> >
            <<< grid, threads, sharedMemSize >>>
            (d_out, d_idata, d_iflags, numElements, 0, 0, 0);
        break;
    case 5: // multi block, single row, non-full last block
        segmentedScan4<T, SegmentedScanTraits<T, op, isBackward, isExclusive, doShiftFlagsLeft, false, true,
                       true> >
            <<< grid, threads, sharedMemSize >>>
            (d_out, d_idata, d_iflags, numElements,
            d_blockSums[level], d_blockFlags[level],
            d_blockIndices[level]);
        break;
    case 6: // single block, single row, full last block
        segmentedScan4<T, SegmentedScanTraits<T, op, isBackward, isExclusive, doShiftFlagsLeft, true, false,
                       true> >
            <<< grid, threads, sharedMemSize >>>
            (d_out, d_idata, d_iflags, numElements, 0, 0, 0);
        break;
    case 7: // multi block, single row, full last block
        segmentedScan4<T, SegmentedScanTraits<T, op, isBackward, isExclusive, doShiftFlagsLeft, true, true,
                       true> >
            <<< grid, threads, sharedMemSize >>>
            (d_out, d_idata, d_iflags, numElements,
            d_blockSums[level], d_blockFlags[level],
            d_blockIndices[level]);
        break;
    }

    CUT_CHECK_ERROR("segmentedScanArrayRecursive after block level scans");

    if (numBlocks > 1)
    {
        // After scanning all the sub-blocks, we are mostly done. But
        // now we need to take all of the last values of the
        // sub-blocks and segment scan those. This will give us a new value
        // that must be sdded to the first segment of each block to get 
        // the final results.
        segmentedScanArrayRecursive<T, op, isBackward, false, false>
            ((T*)d_blockSums[level], (const T*)d_blockSums[level], 
            d_blockFlags[level], (T **)d_blockSums,
            d_blockFlags, d_blockIndices,
            numBlocks, level + 1);

        if (isBackward)
        {
            if (fullBlock)
                vectorSegmentedAddUniformToRight4<T, op, true><<<grid, threads>>>
                (d_out, d_blockSums[level], d_blockIndices[level], 
                numElements, 0, 0);
            else
                vectorSegmentedAddUniformToRight4<T, op, false><<<grid, threads>>>
                (d_out, d_blockSums[level], d_blockIndices[level], 
                numElements, 0, 0);
        }
        else
        {
            if (fullBlock)
                vectorSegmentedAddUniform4<T, op, true><<<grid, threads>>>
                (d_out, d_blockSums[level], d_blockIndices[level], 
                numElements, 0, 0);
            else
                vectorSegmentedAddUniform4<T, op, false><<<grid, threads>>>
                (d_out, d_blockSums[level], d_blockIndices[level], 
                numElements, 0, 0);
        }

        CUT_CHECK_ERROR("vectorSegmentedAddUniform4");
    }
}

#ifdef __cplusplus
extern "C" 
{
#endif

    // file scope
    /** @brief Allocate intermediate block sums, block flags and block
    *        indices arrays in a CUDPPSegmentedScanPlan class.
    *
    * Segmented scans of large arrays must be split (possibly
    * recursively) into a hierarchy of block segmented scans, where each
    * block is scanned by a single CUDA thread block. At each recursive
    * level of the scan, we need an array in which to store the total
    * sums of all blocks in that level. Also at this level we have two
    * more arrays - one which contains the OR-reductions of flags of all
    * blocks at that level and the second which contains the
    * min-reductions of indices of all blocks at that levels This
    * function computes the amount of storage needed and allocates it.
    *
    * @param[in] plan Pointer to CUDPPSegmentedScanPlan object containing segmented scan
    * options and number of elements, which is used to compute storage
    * requirements.
    */
    void allocSegmentedScanStorage(CUDPPSegmentedScanPlan *plan)
    {
        plan->m_numEltsAllocated = plan->m_numElements;

        size_t numElts = plan->m_numElements;

        size_t level = 0;

        do
        {       
            size_t numBlocks = 
                max(1, (unsigned int)ceil
                ((double)numElts / 
                ((double)SEGSCAN_ELTS_PER_THREAD * SCAN_CTA_SIZE)));
            if (numBlocks > 1)
            {
                level++;
            }
            numElts = numBlocks;
        } while (numElts > 1);

        size_t elementSize = 0;

        switch(plan->m_config.datatype)
        {
        case CUDPP_INT:
            plan->m_blockSums = (void**) malloc(level * sizeof(int*));
            elementSize = sizeof(int);
            break;
        case CUDPP_UINT:
            plan->m_blockSums = (void**) malloc(level * sizeof(unsigned int*));
            elementSize = sizeof(unsigned int);
            break;
        case CUDPP_FLOAT:
            plan->m_blockSums = (void**) malloc(level * sizeof(float*));
            elementSize = sizeof(float);
            break;
        default:
            break;
        }

        plan->m_blockFlags = 
            (unsigned int**) malloc(level * sizeof(unsigned int*));
        plan->m_blockIndices = 
            (unsigned int**) malloc(level * sizeof(unsigned int*));

        plan->m_numLevelsAllocated = level;
        numElts = plan->m_numElements;

        level = 0;

        do
        {       
            size_t numBlocks = 
                max(1, 
                (unsigned int)ceil((double)numElts / 
                ((double)SEGSCAN_ELTS_PER_THREAD * SCAN_CTA_SIZE)));
            if (numBlocks > 1) 
            {
                CUDA_SAFE_CALL(cudaMalloc((void**) &(plan->m_blockSums[level]),
                    numBlocks * elementSize));
                CUDA_SAFE_CALL(cudaMalloc((void**) &(plan->m_blockFlags[level]),
                    numBlocks * sizeof(unsigned int)));
                CUDA_SAFE_CALL(cudaMalloc((void**) &(plan->m_blockIndices[level]),  
                    numBlocks * sizeof(unsigned int)));
                level++;
            }
            numElts = numBlocks;
        } while (numElts > 1);

        CUT_CHECK_ERROR("allocSegmentedScanStorage");
    }

    /** @brief Deallocate intermediate block sums, block flags and block
    *        indices arrays in a CUDPPSegmentedScanPlan class.
    *
    * These arrays must have been allocated by allocSegmentedScanStorage(),
    * which is called by the constructor of CUDPPSegmentedScanPlan.
    *
    * @param[in] plan CUDPPSegmentedScanPlan class initialized by its constructor.
    */
    void freeSegmentedScanStorage(CUDPPSegmentedScanPlan *plan)
    {
        for (unsigned int i = 0; i < plan->m_numLevelsAllocated; i++)
        {
            cudaFree(plan->m_blockSums[i]);
            cudaFree(plan->m_blockFlags[i]);
            cudaFree(plan->m_blockIndices[i]);
        }

        CUT_CHECK_ERROR("freeSegmentedScanStorage");

        free((void**)plan->m_blockSums);
        free((void**)plan->m_blockFlags);
        free((void**)plan->m_blockIndices);

        plan->m_blockSums = 0;
        plan->m_blockFlags = 0;
        plan->m_blockIndices = 0;
        plan->m_numEltsAllocated = 0;
        plan->m_numLevelsAllocated = 0;
    }

    /** @brief Dispatch function to perform a scan (prefix sum) on an
    * array with the specified configuration.
    *
    * This is the dispatch routine which calls segmentedScanArrayRecursive() with 
    * appropriate template parameters and arguments to achieve the scan as 
    * specified in \a plan. 
    * 
    * @param[in]  numElements The number of elements to scan
    * @param[in]  plan        Segmented Scan configuration (plan), initialized 
    *                         by CUDPPSegmentedScanPlan constructor
    * @param[in]  d_idata     The input array
    * @param[in]  d_iflags    The input flags array

    * @param[out] d_out    The output array of segmented scan results
    */
    void cudppSegmentedScanDispatch (void                         *d_out, 
        const void                   *d_idata,
        const unsigned int           *d_iflags,
        int                          numElements,
        const CUDPPSegmentedScanPlan *plan
        )
    {    
        if (CUDPP_OPTION_EXCLUSIVE & plan->m_config.options)
        {
            if (CUDPP_OPTION_BACKWARD & plan->m_config.options)
            {
                switch(plan->m_config.datatype)
                {
                case CUDPP_INT:
                    switch(plan->m_config.op)
                    {
                    case CUDPP_MAX:
                        segmentedScanArrayRecursive<int, CUDPP_MAX, true, true, true>
                            ((int *)d_out, (int *)d_idata, d_iflags, 
                            (int **)plan->m_blockSums, plan->m_blockFlags,
                            plan->m_blockIndices, numElements, 0);
                        break;
                    case CUDPP_ADD:
                        segmentedScanArrayRecursive<int, CUDPP_ADD, true, true, true>
                            ((int *)d_out, (int *)d_idata, d_iflags, 
                            (int **)plan->m_blockSums, plan->m_blockFlags,
                            plan->m_blockIndices, numElements, 0);
                        break;
                    case CUDPP_MULTIPLY:
                        segmentedScanArrayRecursive<int, CUDPP_MULTIPLY, true, true, true>
                            ((int *)d_out, (int *)d_idata, d_iflags, 
                            (int **)plan->m_blockSums, plan->m_blockFlags,
                            plan->m_blockIndices, numElements, 0);
                        break;
                    case CUDPP_MIN:
                        segmentedScanArrayRecursive<int, CUDPP_MIN, true, true, true>
                            ((int *)d_out, (int *)d_idata, d_iflags, 
                            (int **)plan->m_blockSums, plan->m_blockFlags,
                            plan->m_blockIndices, numElements, 0);
                        break;
                    default:
                        break;
                    }
                    break;
                case CUDPP_UINT:
                    switch(plan->m_config.op)
                    {
                    case CUDPP_MAX:
                        segmentedScanArrayRecursive<unsigned int, CUDPP_MAX, true, true, true>
                            ((unsigned int *)d_out, (unsigned int *)d_idata, d_iflags, 
                            (unsigned int **)plan->m_blockSums, plan->m_blockFlags,
                            plan->m_blockIndices, numElements, 0);
                        break;
                    case CUDPP_ADD:
                        segmentedScanArrayRecursive<unsigned int, CUDPP_ADD, true, true, true>
                            ((unsigned int *)d_out, (unsigned int *)d_idata, d_iflags, 
                            (unsigned int **)plan->m_blockSums, plan->m_blockFlags,
                            plan->m_blockIndices, numElements, 0);
                        break;
                    case CUDPP_MULTIPLY:
                        segmentedScanArrayRecursive<unsigned int, CUDPP_MULTIPLY, true, true, true>
                            ((unsigned int *)d_out, (unsigned int *)d_idata, d_iflags, 
                            (unsigned int **)plan->m_blockSums, plan->m_blockFlags,
                            plan->m_blockIndices, numElements, 0);
                        break;
                    case CUDPP_MIN:
                        segmentedScanArrayRecursive<unsigned int, CUDPP_MIN, true, true, true>
                            ((unsigned int *)d_out, (unsigned int *)d_idata, d_iflags, 
                        (unsigned int **)plan->m_blockSums, plan->m_blockFlags,
                        plan->m_blockIndices, numElements, 0);
                        break;
                    default:
                        break;
                    }
                    break;
                case CUDPP_FLOAT:
                    switch(plan->m_config.op)
                    {
                    case CUDPP_MAX:
                        segmentedScanArrayRecursive<float, CUDPP_MAX, true, true, true>
                            ((float *)d_out, (float *)d_idata, d_iflags, 
                            (float **)plan->m_blockSums, plan->m_blockFlags,
                            plan->m_blockIndices, numElements, 0);
                        break;
                    case CUDPP_ADD:
                        segmentedScanArrayRecursive<float, CUDPP_ADD, true, true, true>
                            ((float *)d_out, (float *)d_idata, d_iflags, 
                            (float **)plan->m_blockSums, plan->m_blockFlags,
                            plan->m_blockIndices, numElements, 0);
                        break;
                    case CUDPP_MULTIPLY:
                        segmentedScanArrayRecursive<float, CUDPP_MULTIPLY, true, true, true>
                            ((float *)d_out, (float *)d_idata, d_iflags, 
                            (float **)plan->m_blockSums, plan->m_blockFlags,
                            plan->m_blockIndices, numElements, 0);
                        break;
                    case CUDPP_MIN:
                        segmentedScanArrayRecursive<float, CUDPP_MIN, true, true, true>
                            ((float *)d_out, (float *)d_idata, d_iflags, 
                            (float **)plan->m_blockSums, plan->m_blockFlags,
                            plan->m_blockIndices, numElements, 0);
                        break;
                    default:
                        break;
                    }
                    break;
                default:
                    break;
                }
            }
            else
            {
                switch(plan->m_config.datatype)
                {
                case CUDPP_INT:
                    switch(plan->m_config.op)
                    {
                    case CUDPP_MAX:
                        segmentedScanArrayRecursive<int, CUDPP_MAX, false, true, false>
                            ((int *)d_out, (int *)d_idata, d_iflags, 
                            (int **)plan->m_blockSums, plan->m_blockFlags,
                            plan->m_blockIndices, numElements, 0);
                        break;
                    case CUDPP_ADD:
                        segmentedScanArrayRecursive<int, CUDPP_ADD, false, true, false>
                            ((int *)d_out, (int *)d_idata, d_iflags, 
                            (int **)plan->m_blockSums, plan->m_blockFlags,
                            plan->m_blockIndices, numElements, 0);
                        break;
                    case CUDPP_MULTIPLY:
                        segmentedScanArrayRecursive<int, CUDPP_MULTIPLY, false, true, false>
                            ((int *)d_out, (int *)d_idata, d_iflags, 
                            (int **)plan->m_blockSums, plan->m_blockFlags,
                            plan->m_blockIndices, numElements, 0);
                        break;
                    case CUDPP_MIN:
                        segmentedScanArrayRecursive<int, CUDPP_MIN, false, true, false>
                            ((int *)d_out, (int *)d_idata, d_iflags, 
                            (int **)plan->m_blockSums, plan->m_blockFlags,
                            plan->m_blockIndices, numElements, 0);
                        break;
                    default:
                        break;
                    }
                    break;
                case CUDPP_UINT:
                    switch(plan->m_config.op)
                    {
                    case CUDPP_MAX:
                        segmentedScanArrayRecursive<unsigned int, CUDPP_MAX, false, true, false>
                            ((unsigned int *)d_out, (unsigned int *)d_idata, d_iflags, 
                            (unsigned int **)plan->m_blockSums, plan->m_blockFlags,
                            plan->m_blockIndices, numElements, 0);
                        break;
                    case CUDPP_ADD:
                        segmentedScanArrayRecursive<unsigned int, CUDPP_ADD, false, true, false>
                            ((unsigned int *)d_out, (unsigned int *)d_idata, d_iflags, 
                            (unsigned int **)plan->m_blockSums, plan->m_blockFlags,
                            plan->m_blockIndices, numElements, 0);
                        break;
                    case CUDPP_MULTIPLY:
                        segmentedScanArrayRecursive<unsigned int, CUDPP_MULTIPLY, false, true, false>
                            ((unsigned int *)d_out, (unsigned int *)d_idata, d_iflags, 
                            (unsigned int **)plan->m_blockSums, plan->m_blockFlags,
                            plan->m_blockIndices, numElements, 0);
                        break;
                    case CUDPP_MIN:
                        segmentedScanArrayRecursive<unsigned int, CUDPP_MIN, false, true, false>
                            ((unsigned int *)d_out, (unsigned int *)d_idata, d_iflags, 
                        (unsigned int **)plan->m_blockSums, plan->m_blockFlags,
                        plan->m_blockIndices, numElements, 0);
                        break;
                    default:
                        break;
                    }
                    break;
                case CUDPP_FLOAT:
                    switch(plan->m_config.op)
                    {
                    case CUDPP_MAX:
                        segmentedScanArrayRecursive<float, CUDPP_MAX, false, true, false>
                            ((float *)d_out, (float *)d_idata, d_iflags, 
                            (float **)plan->m_blockSums, plan->m_blockFlags,
                            plan->m_blockIndices, numElements, 0);
                        break;
                    case CUDPP_ADD:
                        segmentedScanArrayRecursive<float, CUDPP_ADD, false, true, false>
                            ((float *)d_out, (float *)d_idata, d_iflags, 
                            (float **)plan->m_blockSums, plan->m_blockFlags,
                            plan->m_blockIndices, numElements, 0);
                        break;
                    case CUDPP_MULTIPLY:
                        segmentedScanArrayRecursive<float, CUDPP_MULTIPLY, false, true, false>
                            ((float *)d_out, (float *)d_idata, d_iflags, 
                            (float **)plan->m_blockSums, plan->m_blockFlags,
                            plan->m_blockIndices, numElements, 0);
                        break;
                    case CUDPP_MIN:
                        segmentedScanArrayRecursive<float, CUDPP_MIN, false, true, false>
                            ((float *)d_out, (float *)d_idata, d_iflags, 
                            (float **)plan->m_blockSums, plan->m_blockFlags,
                            plan->m_blockIndices, numElements, 0);
                        break;
                    default:
                        break;
                    }
                    break;
                default:
                    break;
                }
            }
        }
        else
        {
            if (CUDPP_OPTION_BACKWARD & plan->m_config.options)
            {
                switch(plan->m_config.datatype)
                {
                case CUDPP_INT:
                    switch(plan->m_config.op)
                    {
                    case CUDPP_MAX:
                        segmentedScanArrayRecursive<int, CUDPP_MAX, true, false, true>
                            ((int *)d_out, (int *)d_idata, d_iflags,
                            (int **)plan->m_blockSums, plan->m_blockFlags,
                            plan->m_blockIndices, numElements, 0);
                        break;
                    case CUDPP_ADD:
                        segmentedScanArrayRecursive<int, CUDPP_ADD, true, false, true>
                            ((int *)d_out, (int *)d_idata, d_iflags,
                            (int **)plan->m_blockSums, plan->m_blockFlags,
                            plan->m_blockIndices, numElements, 0);
                        break;
                    case CUDPP_MULTIPLY:
                        segmentedScanArrayRecursive<int, CUDPP_MULTIPLY, true, false, true>
                            ((int *)d_out, (int *)d_idata, d_iflags,
                            (int **)plan->m_blockSums, plan->m_blockFlags,
                            plan->m_blockIndices, numElements, 0);
                        break;
                    case CUDPP_MIN:
                        segmentedScanArrayRecursive<int, CUDPP_MIN, true, false, true>
                            ((int *)d_out, (int *)d_idata, d_iflags,
                            (int **)plan->m_blockSums, plan->m_blockFlags,
                            plan->m_blockIndices, numElements, 0);
                        break;
                    default:
                        break;
                    }
                break;
                case CUDPP_UINT:
                    switch(plan->m_config.op)
                    {
                    case CUDPP_MAX:
                        segmentedScanArrayRecursive<unsigned int, CUDPP_MAX, true, false, true>
                            ((unsigned int *)d_out, (unsigned int *)d_idata, d_iflags,
                            (unsigned int **)plan->m_blockSums, plan->m_blockFlags,
                            plan->m_blockIndices, numElements, 0);
                        break;
                    case CUDPP_ADD:
                        segmentedScanArrayRecursive<unsigned int, CUDPP_ADD, true, false, true>
                            ((unsigned int *)d_out, (unsigned int *)d_idata, d_iflags,
                            (unsigned int **)plan->m_blockSums, plan->m_blockFlags,
                            plan->m_blockIndices, numElements, 0);
                        break;
                    case CUDPP_MULTIPLY:
                        segmentedScanArrayRecursive<unsigned int, CUDPP_MULTIPLY, true, false, true>
                            ((unsigned int *)d_out, (unsigned int *)d_idata, d_iflags,
                            (unsigned int **)plan->m_blockSums, plan->m_blockFlags,
                            plan->m_blockIndices, numElements, 0);
                        break;
                    case CUDPP_MIN:
                        segmentedScanArrayRecursive<unsigned int, CUDPP_MIN, true, false, true>
                            ((unsigned int *)d_out, (unsigned int *)d_idata, d_iflags,
                            (unsigned int **)plan->m_blockSums, plan->m_blockFlags,
                            plan->m_blockIndices, numElements, 0);
                        break;
                    default:
                        break;
                    }
                    break;
                case CUDPP_FLOAT:
                    switch(plan->m_config.op)
                    {
                    case CUDPP_MAX:
                        segmentedScanArrayRecursive<float, CUDPP_MAX, true, false, true>
                            ((float *)d_out, (float *)d_idata, d_iflags,
                            (float **)plan->m_blockSums, plan->m_blockFlags,
                            plan->m_blockIndices, numElements, 0);
                        break;
                    case CUDPP_ADD:
                        segmentedScanArrayRecursive<float, CUDPP_ADD, true, false, true>
                            ((float *)d_out, (float *)d_idata, d_iflags,
                            (float **)plan->m_blockSums, plan->m_blockFlags,
                            plan->m_blockIndices, numElements, 0);
                        break;
                    case CUDPP_MULTIPLY:
                        segmentedScanArrayRecursive<float, CUDPP_MULTIPLY, true, false, true>
                            ((float *)d_out, (float *)d_idata, d_iflags,
                            (float **)plan->m_blockSums, plan->m_blockFlags,
                            plan->m_blockIndices, numElements, 0);
                        break;
                    case CUDPP_MIN:
                        segmentedScanArrayRecursive<float, CUDPP_MIN, true, false, true>
                            ((float *)d_out, (float *)d_idata, d_iflags,
                            (float **)plan->m_blockSums, plan->m_blockFlags,
                            plan->m_blockIndices, numElements, 0);
                        break;
                    default:
                        break;
                    }
                    break;
                default:
                    break;
                }
            }
            else
            {
                switch(plan->m_config.datatype)
                {
                case CUDPP_INT:
                    switch(plan->m_config.op)
                    {
                    case CUDPP_MAX:
                        segmentedScanArrayRecursive<int, CUDPP_MAX, false, false, false>
                            ((int *)d_out, (int *)d_idata, d_iflags,
                            (int **)plan->m_blockSums, plan->m_blockFlags,
                            plan->m_blockIndices, numElements, 0);
                        break;
                    case CUDPP_ADD:
                        segmentedScanArrayRecursive<int, CUDPP_ADD, false, false, false>
                            ((int *)d_out, (int *)d_idata, d_iflags,
                            (int **)plan->m_blockSums, plan->m_blockFlags,
                            plan->m_blockIndices, numElements, 0);
                        break;
                    case CUDPP_MULTIPLY:
                        segmentedScanArrayRecursive<int, CUDPP_MULTIPLY, false, false, false>
                            ((int *)d_out, (int *)d_idata, d_iflags,
                            (int **)plan->m_blockSums, plan->m_blockFlags,
                            plan->m_blockIndices, numElements, 0);
                        break;
                    case CUDPP_MIN:
                        segmentedScanArrayRecursive<int, CUDPP_MIN, false, false, false>
                            ((int *)d_out, (int *)d_idata, d_iflags,
                            (int **)plan->m_blockSums, plan->m_blockFlags,
                            plan->m_blockIndices, numElements, 0);
                        break;
                    default:
                        break;
                    }
                break;
                case CUDPP_UINT:
                    switch(plan->m_config.op)
                    {
                    case CUDPP_MAX:
                        segmentedScanArrayRecursive<unsigned int, CUDPP_MAX, false, false, false>
                            ((unsigned int *)d_out, (unsigned int *)d_idata, d_iflags,
                            (unsigned int **)plan->m_blockSums, plan->m_blockFlags,
                            plan->m_blockIndices, numElements, 0);
                        break;
                    case CUDPP_ADD:
                        segmentedScanArrayRecursive<unsigned int, CUDPP_ADD, false, false, false>
                            ((unsigned int *)d_out, (unsigned int *)d_idata, d_iflags,
                            (unsigned int **)plan->m_blockSums, plan->m_blockFlags,
                            plan->m_blockIndices, numElements, 0);
                        break;
                    case CUDPP_MULTIPLY:
                        segmentedScanArrayRecursive<unsigned int, CUDPP_MULTIPLY, false, false, false>
                            ((unsigned int *)d_out, (unsigned int *)d_idata, d_iflags,
                            (unsigned int **)plan->m_blockSums, plan->m_blockFlags,
                            plan->m_blockIndices, numElements, 0);
                        break;
                    case CUDPP_MIN:
                        segmentedScanArrayRecursive<unsigned int, CUDPP_MIN, false, false, false>
                            ((unsigned int *)d_out, (unsigned int *)d_idata, d_iflags,
                            (unsigned int **)plan->m_blockSums, plan->m_blockFlags,
                            plan->m_blockIndices, numElements, 0);
                        break;
                    default:
                        break;
                    }
                    break;
                case CUDPP_FLOAT:
                    switch(plan->m_config.op)
                    {
                    case CUDPP_MAX:
                        segmentedScanArrayRecursive<float, CUDPP_MAX, false, false, false>
                            ((float *)d_out, (float *)d_idata, d_iflags,
                            (float **)plan->m_blockSums, plan->m_blockFlags,
                            plan->m_blockIndices, numElements, 0);
                        break;
                    case CUDPP_ADD:
                        segmentedScanArrayRecursive<float, CUDPP_ADD, false, false, false>
                            ((float *)d_out, (float *)d_idata, d_iflags,
                            (float **)plan->m_blockSums, plan->m_blockFlags,
                            plan->m_blockIndices, numElements, 0);
                        break;
                    case CUDPP_MULTIPLY:
                        segmentedScanArrayRecursive<float, CUDPP_MULTIPLY, false, false, false>
                            ((float *)d_out, (float *)d_idata, d_iflags,
                            (float **)plan->m_blockSums, plan->m_blockFlags,
                            plan->m_blockIndices, numElements, 0);
                        break;
                    case CUDPP_MIN:
                        segmentedScanArrayRecursive<float, CUDPP_MIN, false, false, false>
                            ((float *)d_out, (float *)d_idata, d_iflags,
                            (float **)plan->m_blockSums, plan->m_blockFlags,
                            plan->m_blockIndices, numElements, 0);
                        break;
                    default:
                        break;
                    }
                    break;
                default:
                    break;
                }
            }
        }
    }

#ifdef __cplusplus
}
#endif

/** @} */ // end segmented scan functions
/** @} */ // end cudpp_app
