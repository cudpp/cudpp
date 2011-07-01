// ***************************************************************
//  cuDPP -- CUDA Data Parallel Primitives library
//  -------------------------------------------------------------
//  $Revision: 3512 $
//  $Date: 2007-07-06 15:39:28 -0700 (Fri, 06 Jul 2007) $
//  -------------------------------------------------------------
// This source code is distributed under the terms of license.txt in
// the root directory of this source distribution.
// ------------------------------------------------------------- 

/**
* @file
* segmented_scan_cta.cu
*
* @brief CUDPP CTA-level scan routines
*/

/** \addtogroup cudpp_cta 
* @{
*/

/** @name Segmented scan Functions
* @{
*/

#include "cta/scan_cta.cu"
#include <cudpp_globals.h>
#include <math.h>
#include <cstdio>

/**
  * @brief Template class containing compile-time parameters to the segmented scan functions
  *
  * SegmentedScanTraits is passed as a template parameter to all segmented scan functions.  By 
  * using these compile-time functions we can enable generic code while 
  * maintaining the highest performance.  This is crucial for the performance 
  * of low-level workhorse algorithms like segmented scan.
  *
  * @param T The datatype of the segmented scan
  * @param oper The ::CUDPPOperator to use for the segmented scan (add, max, etc.)
  * @param unroll True if scan inner loops should be unrolled
  * @param sums True if each block should write it's sum to the d_blockSums array (false for single-block scans)
  * @param backward True if this is a backward scan, False if this is a forward scan
  * @param fullBlock True if all blocks in this scan are full (CTA_SIZE * SCAN_ELEMENTS_PER_THREAD elements)
  * @param exclusivity True for exclusive scans, false for inclusive scans
*/
template <typename T, class Oper, bool backward, bool exclusivity, 
          bool doShiftFlags, bool fullBlock, bool sums, bool sm12OrBetter>
class SegmentedScanTraits
{
public:
    //! Returns true if this is a backward scan
    static __device__ bool isBackward()   { return backward;   }
    //! Returns true if this is an exclusive scan
    static __device__ bool isExclusive()  { return exclusivity; }
    //! Returns true if this scan needs to shift flags to the left. This is only needed for the first level scan
    //! in a multi-block scan
    static __device__ bool shiftFlags() { return doShiftFlags; }
    //! Returns true if this is a full scan -- all blocks process CTA_SIZE * SCAN_ELEMENTS_PER_THREAD elements
    static __device__ bool isFullBlock() { return fullBlock;        }
    //! Returns true if this scan writes the sum of each block to the d_blockSums array (multi-block scans)
    static __device__ bool writeSums() { return sums; }
    //! Returns true if we are sm12 or better hardware
    static __device__ bool isSM12OrBetter() { return sm12OrBetter; }

    typedef Oper Op; //!< The operator functor used for segmented scan
};

/**
* @brief Handles loading input s_data from global memory to shared memory 
* (vec4 version)
*
* Load a chunk of 8*blockDim.x elements from global memory into a 
* shared memory array.  Each thread loads two T4 elements (where
* T4 is, e.g. int4 or float4), computes the segmented scan of those two vec4s in 
* thread local arrays (in registers), and writes the two total sums of the
* vec4s into shared memory, where they will be cooperatively scanned with 
* the other partial sums by all threads in the CTA.
*
* @param[out] s_odata The output (shared) memory array
* @param[out] threadScan0 Intermediate per-thread partial sums array 1
* @param[out] threadScan1 Intermediate per-thread partial sums array 2
* @param[out] threadFlag Intermediate array which holds 8 flags as follows
* Temporary register threadFlag0[4] - the flags for the first 4 elements read
* Temporary register threadFlag1[4] - the flags for the second 4 elements read
* Temporary register threadScanFlag0[4] - the inclusive OR-scan for the flags in threadFlag0[4]
* Temporary register threadScanFlag1[4] - the inclusive OR-scan for the flags in threadFlag1[4]
* We storing the 16 flags 32 bits of threadFlag
* Bits 0...3 contains threadFlag0[0]...threadFlag0[3]
* Bits 4...7 contains threadFlag1[0]...threadFlag1[3]
* Bits 8...11 contains threadScanFlag0[0]...threadScanFlag0[3]
* Bits 11...15 contains threadScanFlag1[0]...threadScanFlag1[3]
* @param[out] s_oflags Output (shared) memory array of segment head flags
* @param[out] s_oindices Output (shared) memory array of indices. If a flag for a position (1-based)
*                        is set then index for that position is the position, 0 otherwise.
* @param[in] d_idata The input (device) memory array
* @param[in] d_iflags The input (device) memory array of segment head flags
* @param[in] numElements The number of elements in the array being scanned
* @param[in] iDataOffset the offset of the input array in global memory for this 
* thread block
* @param[out] ai The shared memory address for the thread's first element 
* (returned for reuse)
* @param[out] bi The shared memory address for the thread's second element 
* (returned for reuse)
* @param[out] aiDev The device memory address for this thread's first element 
* (returned for reuse)
* @param[out] biDev The device memory address for this thread's second element 
* (returned for reuse)
*/
template <class T, typename traits>
inline __device__ 
void 
loadForSegmentedScanSharedChunkFromMem4(
                                        T *s_odata,
                                        T threadScan0[4],
                                        T threadScan1[4],
                                        unsigned int& threadFlag,
                                        unsigned int* s_oflags,
                                        unsigned int* s_oindices,
                                        const T *d_idata,
                                        const unsigned int *d_iflags,
                                        int numElements,
                                        int iDataOffset,
                                        int& ai, 
                                        int& bi, 
                                        int& aiDev, 
                                        int& biDev
                                        )
{
    int thid = threadIdx.x;

    aiDev = iDataOffset + threadIdx.x;
    biDev = aiDev + blockDim.x;

    ai = thid;
    bi = thid + blockDim.x;

    bool isLastBlock = (blockIdx.x == (gridDim.x-1));

    // convert to 4-vector
    typename typeToVector<T,4>::Result* iData = (typename typeToVector<T,4>::Result*)d_idata;
    typename typeToVector<unsigned int,4>::Result* iFlags = (typename typeToVector<unsigned,4>::Result*)d_iflags;
    typename typeToVector<T,4>::Result tempData;

    uint4 tempFlag;

    unsigned int gIndex = (aiDev) * 4;

    if (traits::shiftFlags() && traits::isSM12OrBetter())
    {
        if (traits::isFullBlock() || (gIndex+4) <  numElements)
        {
            tempFlag.x = d_iflags[gIndex+1];
            tempFlag.y = d_iflags[gIndex+2];
            tempFlag.z = d_iflags[gIndex+3];
            tempFlag.w = d_iflags[gIndex+4];
        }
        else
        {
            tempFlag.x = ((gIndex+1) < numElements) ? d_iflags[gIndex+1] : 0;
            tempFlag.y = ((gIndex+2) < numElements) ? d_iflags[gIndex+2] : 0;
            tempFlag.z = ((gIndex+3) < numElements) ? d_iflags[gIndex+3] : 0;
            tempFlag.w = 0;
        }   
    }
    else
    {
        tempFlag = iFlags[aiDev];

        if (isLastBlock && !traits::isFullBlock())
        {
            if (gIndex >= numElements) tempFlag.x = 0;
            if ((gIndex+1) >= numElements) tempFlag.y = 0;
            if ((gIndex+2) >= numElements) tempFlag.z = 0;
            if ((gIndex+3) >= numElements) tempFlag.w = 0;
        }
    }

    // Pad values beyond numElements with identity elements 
    if (traits::shiftFlags() && !traits::isSM12OrBetter())
    {
        if (ai == 0)
        {
            unsigned int t = (iDataOffset + blockDim.x)*(4);
            if (isLastBlock)
                s_oflags[blockDim.x-1] = (t < numElements) ? d_iflags[t] : 0;
            else
                s_oflags[blockDim.x-1] = d_iflags[t];
        }
        else
        {
            s_oflags[ai-1] = tempFlag.x;
        }
        
        // Inside an if but the if should be evaluated at compile time
        __syncthreads();

        tempFlag.x = tempFlag.y;
        tempFlag.y = tempFlag.z;
        tempFlag.z = tempFlag.w;
        tempFlag.w = s_oflags[ai];

        // Do I need a __syncthreads here - I don't think so
    }

    // Store the first 4 flags in threadFlag[0]...threadFlag[3]
    threadFlag = 0;
    threadFlag |= tempFlag.x;
    threadFlag |= (tempFlag.y << 1);
    threadFlag |= (tempFlag.z << 2);
    threadFlag |= (tempFlag.w << 3);

    // instantiate operator functor
    typename traits::Op op;

    // Read 4 data
    // Pad values beyond numElements with identity elements
    tempData = iData[aiDev];
    if (isLastBlock && !traits::isFullBlock())
    {
        if (gIndex     >= numElements) tempData.x = op.identity();
        if ((gIndex+1) >= numElements) tempData.y = op.identity();
        if ((gIndex+2) >= numElements) tempData.z = op.identity();
        if ((gIndex+3) >= numElements) tempData.w = op.identity();
    }

    // Computed inclusive segmented scan and store result in
    // threadScan0
    if (traits::isBackward())
    {
        threadScan0[3] = tempData.w;
        threadScan0[2] = 
            op(tempData.z, tempFlag.z ? op.identity() : threadScan0[3]);
        threadScan0[1] = 
            op(tempData.y, tempFlag.y ? op.identity() : threadScan0[2]);
        threadScan0[0] = s_odata[ai] = 
            op(tempData.x, tempFlag.x ? op.identity() : threadScan0[1]);
    }
    else
    {
        threadScan0[0] = tempData.x;
        threadScan0[1] = 
            op(tempData.y, tempFlag.y ? op.identity() : threadScan0[0]);
        threadScan0[2] = 
            op(tempData.z, tempFlag.z ? op.identity() : threadScan0[1]);
        threadScan0[3] = s_odata[ai] =
            op(tempData.w, tempFlag.w ? op.identity() : threadScan0[2]);
    }

    unsigned int indexVec[4];

    if (traits::isBackward())
    {
        // Compute 4 indices. The logic is if a flag in this position
        // is 1 then the index is set to the 1-based position (i.e if
        // gIndex is 10 then index is set to 11). If flag is 0 then 
        // index is set to the identity element for max which is 
        // 0
        indexVec[0] = (gIndex + 1 + 0) * tempFlag.x;
        indexVec[1] = (gIndex + 1 + 1) * tempFlag.y;
        indexVec[2] = (gIndex + 1 + 2) * tempFlag.z;
        indexVec[3] = (gIndex + 1 + 3) * tempFlag.w;
    }
    else
    {
        // Compute 4 indices. The logic is if a flag in this position
        // is 1 then the index is set to the 1-based position (i.e if
        // gIndex is 10 then index is set to 11). If flag is 0 then 
        // index is set to the identity element for min which is 
        // UINT_MAX
        indexVec[0] = 
            (gIndex + 1 + 0) * tempFlag.x + (1 - tempFlag.x) * UINT_MAX;
        indexVec[1] = 
            (gIndex + 1 + 1) * tempFlag.y + (1 - tempFlag.y) * UINT_MAX;
        indexVec[2] = 
            (gIndex + 1 + 2) * tempFlag.z + (1 - tempFlag.z) * UINT_MAX;
        indexVec[3] = 
            (gIndex + 1 + 3) * tempFlag.w + (1 - tempFlag.w) * UINT_MAX;
    }

    unsigned int m_index;

    if (traits::isBackward())
    {
        // Compute maximum of 4 indices
        m_index = 
            max(max(max(indexVec[0], indexVec[1]), indexVec[2]), indexVec[3]);
    }
    else
    {
        // Compute minimum of 4 indices
        m_index = 
            min(min(min(indexVec[0], indexVec[1]), indexVec[2]), indexVec[3]);
    }

    // Store the minimum/maximum index in shared memory
    s_oindices[ai] = m_index;

    // Store inclusive OR-scan of 4 flags read in threadFlag[8]...threadFlag[11]
    if (traits::isBackward())
    {
        threadFlag |= 
            ((tempFlag.w | tempFlag.z | tempFlag.y | tempFlag.x ) << 8);
        threadFlag |= ((tempFlag.w | tempFlag.z | tempFlag.y) << 9);
        threadFlag |= ((tempFlag.w | tempFlag.z) << 10);
        threadFlag |= (tempFlag.w << 11);
    }
    else
    {
        threadFlag |= (tempFlag.x << 8);
        threadFlag |= ((tempFlag.x | tempFlag.y) << 9);
        threadFlag |= ((tempFlag.x | tempFlag.y | tempFlag.z) << 10);
        threadFlag |= 
            ((tempFlag.x | tempFlag.y | tempFlag.z | tempFlag.w) << 11);
    }

    // Store the OR-reduce of 4 flags in shared memory
    if (traits::isBackward())
        s_oflags[ai] = ((threadFlag >> 8) & 1);
    else
        s_oflags[ai] = ((threadFlag >> 11) & 1);

    gIndex = biDev * 4;

    // Read 4 flags
    // Pad values beyond numElements with identity elements
    if (traits::shiftFlags() && traits::isSM12OrBetter())
    {
        if (traits::isFullBlock() || (gIndex+4) < numElements)
        {   
            tempFlag.x = d_iflags[gIndex+1];
            tempFlag.y = d_iflags[gIndex+2];
            tempFlag.z = d_iflags[gIndex+3];
            if (isLastBlock && (bi==((blockDim.x<<1)-1)))
                tempFlag.w = 0;
            else
                tempFlag.w = d_iflags[gIndex+4];
        }
        else
        {
            tempFlag.x = ((gIndex+1) < numElements) ? d_iflags[gIndex+1] : 0;
            tempFlag.y = ((gIndex+2) < numElements) ? d_iflags[gIndex+2] : 0;
            tempFlag.z = ((gIndex+3) < numElements) ? d_iflags[gIndex+3] : 0;
            tempFlag.w = 0;
        }   
    }
    else
    {
        tempFlag = iFlags[biDev];

        if (isLastBlock && !traits::isFullBlock())
        {
            if (gIndex >= numElements) tempFlag.x = 0;
            if ((gIndex+1) >= numElements) tempFlag.y = 0;
            if ((gIndex+2) >= numElements) tempFlag.z = 0;
            if ((gIndex+3) >= numElements) tempFlag.w = 0;
        }
    } 

    if (traits::shiftFlags() && !traits::isSM12OrBetter())
    {
        if (bi == blockDim.x)
        {
            if (isLastBlock)
                s_oflags[(blockDim.x<<1)-1] = 0;
            else
                s_oflags[(blockDim.x<<1)-1] =
                    d_iflags[(iDataOffset + (blockDim.x<<1))*4];
        }
        else
        {
            s_oflags[bi-1] = tempFlag.x;
        }
        
        // Inside an if but the if should be evaluated at compile time
        __syncthreads();

        tempFlag.x = tempFlag.y;
        tempFlag.y = tempFlag.z;
        tempFlag.z = tempFlag.w;
        tempFlag.w = s_oflags[bi];

        // Do I need a __syncthreads here - I don't think so
    }

    // Store the first 4 flags in threadFlag[4]...threadFlag[7]
    threadFlag |= (tempFlag.x << 4);
    threadFlag |= (tempFlag.y << 5);
    threadFlag |= (tempFlag.z << 6);
    threadFlag |= (tempFlag.w << 7);

    // Read 4 data
    // Pad values beyond numElements with identity elements
    tempData = iData[biDev];

    // Pad values beyond numElements with identity elements 
    if (isLastBlock && !traits::isFullBlock())
    {
        if (gIndex     >= numElements) tempData.x = op.identity();
        if ((gIndex+1) >= numElements) tempData.y = op.identity();
        if ((gIndex+2) >= numElements) tempData.z = op.identity();
        if ((gIndex+3) >= numElements) tempData.w = op.identity();
    }

    // Computed inclusive segmented scan and store result in
    // threadScan1
    if (traits::isBackward())
    {
        threadScan1[3] = tempData.w;
        threadScan1[2] = 
            op(tempData.z, tempFlag.z ? op.identity() : threadScan1[3]);
        threadScan1[1] = 
            op(tempData.y, tempFlag.y ? op.identity() : threadScan1[2]);
        threadScan1[0] = s_odata[bi] = 
            op(tempData.x, tempFlag.x ? op.identity() : threadScan1[1]);
    }
    else
    {
        threadScan1[0] = tempData.x;
        threadScan1[1] = 
            op(tempData.y, tempFlag.y ? op.identity() : threadScan1[0]);
        threadScan1[2] = 
            op(tempData.z, tempFlag.z ? op.identity() : threadScan1[1]);
        threadScan1[3] = s_odata[bi] = 
            op(tempData.w, tempFlag.w ? op.identity() : threadScan1[2]);
    }

    if (traits::isBackward())
    {
        // Compute 4 indices. Thelogic is if a flag in this position
        // is 1 then the index is set to the 1-based position (i.e if
        // gIndex is 10 then index is set to 11). If flag is 0 then 
        // index is set to the identity element for max which is 
        // 0
        indexVec[0] = 
            (gIndex + 1 + 0) * tempFlag.x;
        indexVec[1] = 
            (gIndex + 1 + 1) * tempFlag.y;
        indexVec[2] = 
            (gIndex + 1 + 2) * tempFlag.z;
        indexVec[3] = 
            (gIndex + 1 + 3) * tempFlag.w;
    }
    else
    {
        // Compute 4 indices. Thelogic is if a flag in this position
        // is 1 then the index is set to the 1-based position (i.e if
        // gIndex is 10 then index is set to 11). If flag is 0 then 
        // index is set to the identity element for min which is 
        // INT_MAX
        indexVec[0] = 
            (gIndex + 1 + 0) * tempFlag.x + (1 - tempFlag.x) * UINT_MAX;
        indexVec[1] = 
            (gIndex + 1 + 1) * tempFlag.y + (1 - tempFlag.y) * UINT_MAX;
        indexVec[2] = 
            (gIndex + 1 + 2) * tempFlag.z + (1 - tempFlag.z) * UINT_MAX;
        indexVec[3] = 
            (gIndex + 1 + 3) * tempFlag.w + (1 - tempFlag.w) * UINT_MAX;
    }

    if (traits::isBackward())
    {
        // Compute the minimum of 4 indices
        m_index = 
            max(max(max(indexVec[0], indexVec[1]), indexVec[2]), indexVec[3]);
    }
    else
    {
        // Compute the minimum of 4 indices
        m_index = 
            min(min(min(indexVec[0], indexVec[1]), indexVec[2]), indexVec[3]);
    }

    // Store the minimum index in shared memory
    s_oindices[bi] = m_index;

    // Store inclusive OR-scan of 4 flags read in threadFlag[12]...threadFlag[15]
    if (traits::isBackward())
    {
        threadFlag |= 
            ((tempFlag.w | tempFlag.z | tempFlag.y | tempFlag.x) << 12);
        threadFlag |= 
            ((tempFlag.w | tempFlag.z | tempFlag.y) << 13);
        threadFlag |= 
            ((tempFlag.w | tempFlag.z) << 14);
        threadFlag |= 
            (tempFlag.w << 15);
    }
    else
    {
        threadFlag |= (tempFlag.x << 12);
        threadFlag |= ((tempFlag.x | tempFlag.y) << 13);
        threadFlag |= ((tempFlag.x | tempFlag.y | tempFlag.z) << 14);
        threadFlag |= 
            ((tempFlag.x | tempFlag.y | tempFlag.z | tempFlag.w) << 15);
    }

    // Store the OR-reduce of 4 flags in shared memory
    if (traits::isBackward())
        s_oflags[bi] = ((threadFlag >> 12) & 1);
    else
        s_oflags[bi] = ((threadFlag >> 15) & 1);

    __syncthreads();
}



/**
* @brief Handles storing result s_data from shared memory to global memory 
* (vec4 version)
*
* Store a chunk of 8*blockDim.x elements from shared memory into a 
* device memory array.  Each thread stores reads two elements from shared
* memory, adds them while respecting segment bouldaries, to the intermediate
* sums computed in loadForSegmentedScanSharedChunkFromMem4(), and writes two T4 
* elements (where T4 is, e.g. int4 or float4) to global memory.
*
* @param[out] d_odata The output (device) memory array
* @param[out] threadScan0 Intermediate per-thread partial sums array 1
* (contents computed in loadForSegmentedScanSharedChunkFromMem4())
* @param[in] threadScan1 Intermediate per-thread partial sums array 2
* (contents computed in loadForSegmentedScanSharedChunkFromMem4())
* @param[in] threadFlag Various flags that loadForSegmentedScanSharedChunkFromMem4()
*            needs to pass
* @param[in] s_idata The input (shared) memory array
* @param[in] numElements The number of elements in the array being scanned
* @param[in] oDataOffset the offset of the output array in global memory 
* for this thread block
* @param[in] ai The shared memory address for the thread's first element 
* (computed in loadForSegmentedScanSharedChunkFromMem4())
* @param[in] bi The shared memory address for the thread's second element 
* (computed in loadForSegmentedScanSharedChunkFromMem4())
* @param[in] aiDev The device memory address for this thread's first element 
* (computed in loadForSegmentedScanSharedChunkFromMem4())
* @param[in] biDev The device memory address for this thread's second element 
* (computed in loadForSegmentedScanSharedChunkFromMem4())
*/
template <class T, class traits>
inline __device__ 
void storeForSegmentedScanSharedChunkToMem4(T *d_odata,
                                            T threadScan0[4],
                                            T threadScan1[4],
                                            unsigned int threadFlag,
                                            T *s_idata, 
                                            unsigned int numElements,
                                            int oDataOffset,
                                            int ai, 
                                            int bi, 
                                            int aiDev, 
                                            int biDev
                                            )
{
    // instantiate operator functor
    typename traits::Op op;

    bool isLastBlock = (blockIdx.x == (gridDim.x-1));

    // Convert to 4-vector
    typename typeToVector<T,4>::Result tempData;
    typename typeToVector<T,4>::Result* oData = (typename typeToVector<T,4>::Result*)d_odata;

    T temp;
    // To make it exclusive
    if (traits::isBackward())
    {
        temp = s_idata[ai+1];
    }
    else
    {
        if (ai == 0)
            temp = op.identity(); 
        else
            temp = s_idata[ai-1]; 
    }

    // perform a 4-tuple wide segmented scan (either exclusive
    // or inclusive)
    if (traits::isExclusive())
    {
        if (traits::isBackward())
        {
            tempData.x = 
                op(((threadFlag >> 8) & 1) ? op.identity() : temp,
                       ((threadFlag >> 0) & 1) ? op.identity() : threadScan0[1]);
            tempData.y = 
                op(((threadFlag >> 9) & 1) ? op.identity() : temp,
                       ((threadFlag >> 1) & 1) ? op.identity() : threadScan0[2]); 
            tempData.z = 
                op(((threadFlag >> 10) & 1) ? op.identity() : temp,
                       ((threadFlag >> 2) & 1) ? op.identity() : threadScan0[3]); 
            tempData.w = 
                ((threadFlag >> 11) & 1) ? op.identity() : temp;
        }
        else
        {
            tempData.x = 
                ((threadFlag >> 8) & 1) ? op.identity() : temp;
            tempData.y = 
                op(((threadFlag >> 9) & 1) ? op.identity() : temp,
                       ((threadFlag >> 1) & 1) ? op.identity() : threadScan0[0]); 
            tempData.z = 
                op(((threadFlag >> 10) & 1) ? op.identity() : temp,
                       ((threadFlag >> 2) & 1) ? op.identity() : threadScan0[1]); 
            tempData.w = 
                op(((threadFlag >> 11) & 1) ? op.identity() : temp,
                       ((threadFlag >> 3) & 1) ? op.identity() : threadScan0[2]);
        }
    }
    else
    {
            tempData.x =
                op(((threadFlag >> 8) & 1) ? op.identity() : temp, 
                       threadScan0[0]);
            tempData.y = 
                op(((threadFlag >> 9) & 1) ? op.identity() : temp, 
                       threadScan0[1]);
            tempData.z =
                op(((threadFlag >> 10) & 1) ? op.identity() : temp, 
                       threadScan0[2]);
            tempData.w =
                op(((threadFlag >> 11) & 1) ? op.identity() : temp, 
                       threadScan0[3]);
    }

    // write results to global memory
    if (isLastBlock && !traits::isFullBlock())
    {            
        unsigned int i = aiDev * 4;
        if (i < numElements) {d_odata[i] = tempData.x;}
        if ((i+1) < numElements) {d_odata[i+1] = tempData.y;}
        if ((i+2) < numElements) {d_odata[i+2] = tempData.z;}
        if ((i+3) < numElements) {d_odata[i+3] = tempData.w;}
    }
    else
    {   
        oData[aiDev] = tempData; 
    }

    // To make it inclusive
    if (traits::isBackward())
    {
        if (bi == ((blockDim.x<<1)-1))
            temp = op.identity(); 
        else
            temp = s_idata[bi+1]; 
    }
    else
    {
        temp = s_idata[bi-1]; 
    }

    // perform a 4-tuple wide segmented scan (either exclusive
    // or inclusive)
    if (traits::isExclusive())
    {
        if (traits::isBackward())
        {
            tempData.x = 
                op(((threadFlag >> 12) & 1) ? op.identity() : temp,
                       ((threadFlag >>  4) & 1) ? op.identity() : threadScan1[1]);
            tempData.y = 
                op(((threadFlag >> 13) & 1) ? op.identity() : temp,
                       ((threadFlag >>  5) & 1) ? op.identity() : threadScan1[2]);
            tempData.z = 
                op(((threadFlag >> 14) & 1) ? op.identity() : temp,
                       ((threadFlag >>  6) & 1) ? op.identity() : threadScan1[3]);
            tempData.w = ((threadFlag >> 15) & 1) ? op.identity() : temp;
        }
        else
        {
            tempData.x = 
                ((threadFlag >> 12) & 1) ? op.identity() : temp;
            tempData.y = 
                op(((threadFlag >> 13) & 1) ? op.identity() : temp,
                       ((threadFlag >>  5) & 1) ? op.identity() : threadScan1[0]);
            tempData.z = 
                op(((threadFlag >> 14) & 1) ? op.identity() : temp,
                       ((threadFlag >>  6) & 1) ? op.identity() : threadScan1[1]);
            tempData.w =
                op(((threadFlag >> 15) & 1) ? op.identity() : temp,
                       ((threadFlag >>  7) & 1) ? op.identity() : threadScan1[2]);
        }
    }
    else
    {
        tempData.x = 
            op(((threadFlag >> 12) & 1) ? op.identity() : temp, threadScan1[0]);    
        tempData.y =
            op(((threadFlag >> 13) & 1) ? op.identity() : temp, threadScan1[1]);    
        tempData.z =
            op(((threadFlag >> 14) & 1) ? op.identity() : temp, threadScan1[2]);    
        tempData.w =
            op(((threadFlag >> 15) & 1) ? op.identity() : temp, threadScan1[3]);    
    }

    // write results to global memory
    if (isLastBlock && !traits::isFullBlock())
    {
        unsigned int i = biDev * 4;
        if (i < numElements) {d_odata[i] = tempData.x;}
        if ((i+1) < numElements) {d_odata[i+1] = tempData.y;}
        if ((i+2) < numElements) {d_odata[i+2] = tempData.z;}
        if ((i+3) < numElements) {d_odata[i+3] = tempData.w;}

    }
    else
    {
        oData[biDev] = tempData;
    }
}

template <class T, class traits, unsigned int blockSize>
__device__ T
reduceCTA(volatile T *s_data)
{
    // instantiate operator functor
    typename traits::Op op;

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    T t = s_data[tid];

    // do reduction in shared mem
    if (blockSize >= 512) { if (tid < 256) { s_data[tid] = t = op(t, s_data[tid + 256]); } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { s_data[tid] = t = op(t, s_data[tid + 128]); } __syncthreads(); }
    if (blockSize >= 128) { if (tid <  64) { s_data[tid] = t = op(t, s_data[tid +  64]); } __syncthreads(); }
    
    if (tid < 32)
    {
        if (blockSize >=  64) { s_data[tid] = t = op(t, s_data[tid + 32]); }
        if (blockSize >=  32) { s_data[tid] = t = op(t, s_data[tid + 16]); }
        if (blockSize >=  16) { s_data[tid] = t = op(t, s_data[tid +  8]); }
        if (blockSize >=   8) { s_data[tid] = t = op(t, s_data[tid +  4]); }
        if (blockSize >=   4) { s_data[tid] = t = op(t, s_data[tid +  2]); }
        if (blockSize >=   2) { s_data[tid] = t = op(t, s_data[tid +  1]); }
    }
    
    // write result for this block to global mem 
    return s_data[0];
}

template<class T, class traits, bool isExclusive, unsigned int maxlevel>
__device__ void warpSegScan(T val,
                            unsigned int flag,
                            volatile T *s_data,
                            volatile unsigned int *s_flags,
                            T& oVal,
                            unsigned int& oFlag)
{
    // instantiate operator functor
    typename traits::Op op;

    int idx;
    if (traits::isBackward())
    {
        idx = 2 * threadIdx.x - (threadIdx.x & (WARP_SIZE-1)) + WARP_SIZE;
    }
    else
    {
        idx = 2 * threadIdx.x - (threadIdx.x & (WARP_SIZE-1));
    }

    s_data[idx] = op.identity(); s_flags[idx] = 0;

    if (traits::isBackward())
    {
        idx -= WARP_SIZE;
    }
    else
    {
        idx += WARP_SIZE;
    }

    T t = s_data[idx] = val; 
    unsigned int f = s_flags[idx] = flag;

    if (0 <= maxlevel)
    {
        if (traits::isBackward())
        {
            s_data[idx]  = t = f ? t : op(s_data[idx +  1] , t);
            s_flags[idx] = f = s_flags[idx +  1] | f;
        }
        else
        {
            s_data[idx]  = t = f ? t : op(s_data[idx -  1] , t);
            s_flags[idx] = f = s_flags[idx -  1] | f;
        }
    }
    if (1 <= maxlevel)
    {
        if (traits::isBackward())
        {
            s_data[idx]  = t = f ? t : op(s_data[idx +  2] , t);
            s_flags[idx] = f = s_flags[idx +  2] | f;
        }
        else
        {
            s_data[idx]  = t = f ? t : op(s_data[idx -  2] , t);
            s_flags[idx] = f = s_flags[idx -  2] | f;
        }
    }
    if (2 <= maxlevel)
    {
        if (traits::isBackward())
        {
            s_data[idx]  = t = f ? t : op(s_data[idx +  4] , t);
            s_flags[idx] = f = s_flags[idx +  4] | f;
        }
        else
        {
            s_data[idx]  = t = f ? t : op(s_data[idx -  4] , t);
            s_flags[idx] = f = s_flags[idx -  4] | f;
        }
    }
    if (3 <= maxlevel)
    {
        if (traits::isBackward())
        {
            s_data[idx]  = t = f ? t : op(s_data[idx +  8] , t);
            s_flags[idx] = f = s_flags[idx +  8] | f;
        }
        else
        {
            s_data[idx]  = t = f ? t : op(s_data[idx -  8] , t);
            s_flags[idx] = f = s_flags[idx -  8] | f;
        }
    }
    if (4 <= maxlevel)
    {
        if (traits::isBackward())
        {
            s_data[idx]  = t = f ? t : op(s_data[idx + 16] , t);
            s_flags[idx] = f = s_flags[idx + 16] | f;
        }
        else
        {
            s_data[idx]  = t = f ? t : op(s_data[idx - 16] , t);
            s_flags[idx] = f = s_flags[idx - 16] | f;
        }
    }

    if( isExclusive ) 
        if (traits::isBackward())
            oVal = (!flag) ? s_data[idx+1] : op.identity();
        else
            oVal = (!flag) ? s_data[idx-1] : op.identity();
    else
        oVal =  t;

    oFlag = f;
}


template<class T, class traits>
__device__ void segmentedScanWarps(T val1,
                                   unsigned int flag1,
                                   T val2,
                                   unsigned int flag2,
                                   T *s_data, 
                                   unsigned int *s_flags)
{
    // instantiate operator functor
    typename traits::Op op;

    const unsigned int idx = threadIdx.x;

    // Phase 1: Intra-warp prefix sums

    // Seg scan for (0 ... blockDim.x - 1)
    T oVal1; unsigned int oFlag1;
    warpSegScan<T, traits, false, 4>(val1, flag1, s_data, s_flags,
                                     oVal1, oFlag1);
    __syncthreads();

    // Seg scan for (blockDim.x ... 2*blockDim.x - 1)
    T oVal2; unsigned int oFlag2;
    warpSegScan<T, traits, false, 4>(val2, flag2, s_data, s_flags, 
                                     oVal2, oFlag2);
    __syncthreads(); // FIXME - this is needed why?
        
    // Phase 2: Sum across warps of the CTA

    const unsigned int lane   = idx&(WARP_SIZE-1);
    const unsigned int warpid = idx >> LOG_WARP_SIZE;
    const unsigned int warpid2 = (idx + blockDim.x) >> LOG_WARP_SIZE;

    //  - write per-warp partial sums
    if (traits::isBackward())
    {
        const unsigned int num_warps = ((blockDim.x << 1) >> LOG_WARP_SIZE);
        const unsigned int offset = blockDim.x - num_warps;

        // The effect of adding the offset is to shift the input values of the second
        // level seg-scan to the last warp. This is needed because the data movement
        // in backward segmented scan in right to left (as opposed to left to right in 
        // forward segmented scan). This implies that spurious values in other warps
        // will not affect the values that we need in the last warp when we do the 
        // second level segmented scan. This would not have been a concern if the
        // compiler bug noted below is fixed. In that case we could have done a 
        // segmented scan on a single warp instead of a whole CTA
        if( lane == 0 )  
        {
            s_data[warpid + offset] = oVal1; 
            s_data[warpid2 + offset] = oVal2;

            s_flags[warpid + offset] = oFlag1;
            s_flags[warpid2 + offset] = oFlag2;
        }
    }
    else
    {
        if( lane == (WARP_SIZE-1) )  
        {
            s_data[warpid] = oVal1; 
            s_data[warpid2] = oVal2;

            s_flags[warpid] = oFlag1;
            s_flags[warpid2] = oFlag2;
        }
    }
    __syncthreads();

    T oVal3; unsigned int oFlag3;

    T tdata = s_data[idx];
    T tflag = s_flags[idx];
    __syncthreads();

    //  - use 1 warp for prefix sum over them
    // MJH: This optimization saves very little time in practice, and it
    // breaks backward segscans for some reason, so commenting it out.
    //    if ( warpid==0 )   
    {
        warpSegScan<T, traits, false, (LOG_SCAN_CTA_SIZE-LOG_WARP_SIZE+1)>
            (tdata, tflag, s_data, s_flags, oVal3, oFlag3);
    }
    __syncthreads(); // This looks unnecessary but won't work without it

    s_data[idx] = oVal3;
    s_flags[idx] = oFlag3;
    __syncthreads();

     //  - add the results back into each thread
    if (traits::isBackward())
    {
        // FIXME - this shouldn't need to be defined twice
        const unsigned int num_warps = ((blockDim.x << 1) >> LOG_WARP_SIZE);
        const unsigned int offset = blockDim.x - num_warps;

        oVal1 = oFlag1 ? oVal1 : op(s_data[offset+warpid+1], oVal1);

        if (warpid2 < (num_warps-1)) oVal2 = oFlag2 ? oVal2 : op(s_data[offset+warpid2+1], oVal2);
    }
    else
    {
        if (warpid > 0) oVal1 = oFlag1 ? oVal1 : op(s_data[warpid-1], oVal1);

        oVal2 = oFlag2 ? oVal2 : op(s_data[warpid2-1], oVal2);
    }
    __syncthreads(); // This looks unnecessary

     //  - and we're done
     s_data[idx] = oVal1;
     s_data[idx + blockDim.x] = oVal2;
     
     __syncthreads(); // make sure the caller sees all our s_data[] writes
}


/**
* @brief CTA-level segmented scan routine; 
* 
* Performs segmented scan on \a s_data in shared memory in each thread block 
* with head flags in \a s_flags (\a s_tflags is a read-write copy of the head 
* flags which are modified).
*
* This function is the main CTA-level segmented scan function.  It may be called 
* by other CUDA __global__ or __device__ functions.
* \note This code is intended to be run on a CTA of 128 threads.  Other sizes are
* untested.
* 
* @param[in] s_data Array to be scanned in shared memory
* @param[in] s_flags Read-only version of flags in shared memory
* @param[in] s_indices Temporary read-write indices array
* @param[out] d_blockSums Array of per-block sums
* @param[out] d_blockFlags Array of per-block OR-reduction of flags
* @param[out] d_blockIndices Array of per-block min-reduction of indices
*/
template<class T, class traits>
__device__
void segmentedScanCTA(T            *s_data, 
                      unsigned int *s_flags,
                      unsigned int *s_indices,
                      T            *d_blockSums = 0, 
                      unsigned int *d_blockFlags = 0,
                      unsigned int *d_blockIndices = 0)
{
    T val = s_data[threadIdx.x];
    T val2 = s_data[threadIdx.x + blockDim.x];
    unsigned int flag = s_flags[threadIdx.x];
    unsigned int flag2 = s_flags[threadIdx.x + blockDim.x];

    unsigned int cta_is_closed = s_flags[0];

    __syncthreads();

    segmentedScanWarps<T, traits>(val, flag, val2, flag2, 
                                  s_data, s_flags);

    if (traits::isBackward())
    {
        if (traits::writeSums() && (threadIdx.x == 0))
        {
            d_blockSums[blockIdx.x] = s_data[0];

            const unsigned int num_warps = ((blockDim.x << 1) >> LOG_WARP_SIZE);
            const unsigned int offset = blockDim.x - num_warps;
            d_blockFlags[blockIdx.x] = (s_flags[offset+0] != 0);
        }
    }
    else
    {
        if (traits::writeSums() && (threadIdx.x == blockDim.x - 1))
        {
            d_blockSums[blockIdx.x] = s_data[threadIdx.x + blockDim.x];
            d_blockFlags[blockIdx.x] = cta_is_closed || (s_flags[(1 << (LOG_SCAN_CTA_SIZE-LOG_WARP_SIZE+1))-1] != 0);
        }
    }

    unsigned int mIndex;

    if (traits::writeSums())
    {
        if (traits::isBackward())
        {
            mIndex = 
                reduceCTA<unsigned int, ScanTraits<unsigned int, OperatorMax<unsigned int>, false, false, false, false, true>,
                      (2 * SCAN_CTA_SIZE)>(s_indices);
        }
        else
        {
            mIndex =
                reduceCTA<unsigned int, ScanTraits<unsigned int, OperatorMin<unsigned int>, false, false, false, false, true>,
                      (2 * SCAN_CTA_SIZE)>(s_indices);
        }
    }

    if (traits::writeSums() && (threadIdx.x == 0))
    {
        d_blockIndices[blockIdx.x] = mIndex;
    }
}

/** @} */ // end segmented scan functions
/** @} */ // end cudpp_cta
