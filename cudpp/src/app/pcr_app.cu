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
 * pcr_app.cu
 *
 * @brief CUDPP app-level PCR tridiagonal solver
 */

/** \addtogroup cudpp_app
  * @{
  */
/** @name Parallel cyclic reduction solver (PCR)
 * @{
 */

#include "kernel/pcr_kernel.cu"

/**
 * @brief Parallel cyclic reduction solver (PCR)
 *
 * This is a wrapper function for the GPU PCR kernel.
 *
 * @param[out] d_x Solution vector
 * @param[in] d_a Lower diagonal
 * @param[in] d_b Main diagonal
 * @param[in] d_c Upper diagonal
 * @param[in] d_d Right hand side
 * @param[in] systemSize The size of the linear system
 * @param[in] numSystems The number of systems to be solved
 */

template <class T>
void pcr(T *d_a, T *d_b, T *d_c, T *d_d, T *d_x, int systemSize, int numSystems)
{
    const unsigned int num_threads_block = systemSize;

    // setup execution parameters
    dim3  grid(numSystems, 1, 1);
    dim3  threads(num_threads_block, 1, 1);

    pcrKernel<<< grid, threads,(systemSize+1)*5*sizeof(T)>>>(d_a, d_b, d_c, d_d, d_x);  
}

/** @} */ // end Parallel cyclic reduction solver (PCR)
/** @} */ // end cudpp_app
