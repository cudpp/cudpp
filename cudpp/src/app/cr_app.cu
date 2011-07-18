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
 * cr_app.cu
 *
 * @brief CUDPP application-level CR tridiagonal solver
 */

/** \addtogroup cudpp_app
  * @{
  */
/** @name Cyclic reduction solver (CR)
 * @{
 */

// #include "stopwatch.h"
#include "kernel/cr_kernel.cu"

/**
 * @brief Cyclic reduction solver (CR)
 *
 * This is a wrapper function for the GPU CR kernel.
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
void cr(T *d_a, T *d_b, T *d_c, T *d_d, T *d_x, int systemSize, int numSystems)
{
    const unsigned int num_threads_block = systemSize/2;

    // setup execution parameters
    dim3  grid(numSystems, 1, 1);
    dim3  threads(num_threads_block, 1, 1);

    crKernel<<< grid, threads,systemSize*5*sizeof(T)>>>(d_a, d_b, d_c, d_d, d_x);
}
/** @} */ // end Cyclic reduction solver (CR)
/** @} */ // end cudpp_app

