// -------------------------------------------------------------
// cuDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
//  $Revision:
//  $Date:
// ------------------------------------------------------------- 
// This source code is distributed under the terms of license.txt 
// in the root directory of this source distribution.
// ------------------------------------------------------------- 

/**
 * @file
 * crpcr_app.cu
 *
 * @brief CUDPP app-level CR-PCR tridiagonal solver
 */

#include "kernel/crpcr_kernel.cuh"

/**
 * @brief Hybrid CR-PCR solver (CRPCR)
 *
 * This is a wrapper function for the GPU CR-PCR kernel.
 *
 * @param[out] d_x Solution vector
 * @param[in] d_a Lower diagonal
 * @param[in] d_b Main diagonal
 * @param[in] d_c Upper diagonal
 * @param[in] d_d Right hand side
 * @param[in] systemSize The size of the linear system
 * @param[in] numSystems The number of systems to be solved
 */

unsigned int roundUpToNextPowerOfTwo(unsigned int x)
{
    x--;
    x |= x >> 1;  // handle  2 bit numbers
    x |= x >> 2;  // handle  4 bit numbers
    x |= x >> 4;  // handle  8 bit numbers
    x |= x >> 8;  // handle 16 bit numbers
    x |= x >> 16; // handle 32 bit numbers
    x++; 
    return x;
}

template <class T>
void crpcr(T *d_a, T *d_b, T *d_c, T *d_d, T *d_x, int systemSizeOriginal, int numSystems)
{
    int systemSize = roundUpToNextPowerOfTwo((unsigned int)systemSizeOriginal);
    
    const unsigned int num_threads_block = systemSize/2;
    int restSystemSize = systemSize/2;
  
    // setup execution parameters
    dim3  grid(numSystems, 1, 1);
    dim3  threads(num_threads_block, 1, 1);

    crpcrKernel<<< grid, threads,(systemSize+1)*5*sizeof(T)+restSystemSize*(5+0)*sizeof(T)>>>(d_a, d_b, d_c, d_d, d_x, systemSizeOriginal);

    CUDA_CHECK_ERROR("crpcr");
}

