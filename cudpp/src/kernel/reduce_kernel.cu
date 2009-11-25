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
 * reduce_kernel.cu
 *   
 * @brief CUDPP kernel-level reduction routines
 */

/** \addtogroup cudpp_kernel
  * @{
 */

/** @name Reduce Functions
 * @{
 */

#include <cudpp_globals.h>
#include "sharedmem.h"
#include "cta/radixsort_cta.cu"

#ifdef __DEVICE_EMULATION__
#define __EMUSYNC  __syncthreads()
#else
#define __EMUSYNC
#endif

/**
  * @brief Main reduction kernel
  *
  * This reduction kernel adds multiple elements per thread sequentially, and then the threads
  * work together to produce a block sum in shared memory.  The code is optimized using
  * warp-synchronous programming to eliminate unnecessary barrier synchronization. Performing
  * sequential work in each thread before performing the log(N) parallel summation reduces the
  * overall cost of the algorithm while keeping the work complexity O(n) and the step complexity 
  * O(log n). (Brent's Theorem optimization)
  *
  * @param[out] odata The output data pointer.  Each block writes a single output element.
  * @param[in]  idata The input data pointer.  
  * @param[in]  n     The number of elements to be reduced.
*/
template <typename T, class Oper, unsigned int blockSize, bool nIsPow2>
__global__ void reduce(T *odata, const T *idata, unsigned int n)
{
    SharedMemory<T> smem;
    T* sdata = smem.getPointer();

    Oper op;

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;
    T mySum = op.identity();

    // we reduce multiple elements per thread.  The number is determined by the 
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {         
        //sdata[tid] += idata[i];
        mySum = op(mySum, idata[i]);
        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (nIsPow2 || i + blockSize < n) 
            mySum = op(mySum, idata[i+blockSize]);  
        i += gridSize;
    } 
    sdata[tid] = mySum;
    __syncthreads();

    // do reduction in shared mem
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] = op(sdata[tid], sdata[tid + 256]); } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] = op(sdata[tid], sdata[tid + 128]); } __syncthreads(); }
    if (blockSize >= 128) { if (tid <  64) { sdata[tid] = op(sdata[tid], sdata[tid +  64]); } __syncthreads(); }
    
#ifndef __DEVICE_EMULATION__
    if (tid < 32)
#endif
    {
        if (blockSize >=  64) { sdata[tid] = op(sdata[tid], sdata[tid + 32]); __EMUSYNC; }
        if (blockSize >=  32) { sdata[tid] = op(sdata[tid], sdata[tid + 16]); __EMUSYNC; }
        if (blockSize >=  16) { sdata[tid] = op(sdata[tid], sdata[tid +  8]); __EMUSYNC; }
        if (blockSize >=   8) { sdata[tid] = op(sdata[tid], sdata[tid +  4]); __EMUSYNC; }
        if (blockSize >=   4) { sdata[tid] = op(sdata[tid], sdata[tid +  2]); __EMUSYNC; }
        if (blockSize >=   2) { sdata[tid] = op(sdata[tid], sdata[tid +  1]); __EMUSYNC; }
    }
    
    // write result for this block to global mem 
    if (tid == 0) odata[blockIdx.x] = sdata[0];
}

/** @} */ // end reduce functions
/** @} */ // end cudpp_kernel
