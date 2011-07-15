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
void cr(T *a, T *b, T *c, T *d, T *x, int systemSize, int numSystems)
{
    const unsigned int num_threads_block = systemSize/2;
    const unsigned int memSize = sizeof(T)*numSystems*systemSize;

    // allocate device memory input and output arrays
    T* d_a;
    T* d_b;
    T* d_c;
    T* d_d;
    T* d_x;

    // cudpp_app::StopWatch timer;
    // timer.reset();
    // timer.start();

    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_a,memSize));
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_b,memSize));
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_c,memSize));
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_d,memSize));
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_x,memSize));

    // timer.stop();            
    // printf("GPU cudaMalloc time: %f ms\n", timer.getTime());

   // copy host memory to device input array
    CUDA_SAFE_CALL( cudaMemcpy( d_a, a,memSize, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL( cudaMemcpy( d_b, b,memSize, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL( cudaMemcpy( d_c, c,memSize, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL( cudaMemcpy( d_d, d,memSize, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL( cudaMemcpy( d_x, x,memSize, cudaMemcpyHostToDevice));

    // setup execution parameters
    dim3  grid(numSystems, 1, 1);
    dim3  threads(num_threads_block, 1, 1);

    crKernel<<< grid, threads,systemSize*5*sizeof(T)>>>(d_a, d_b, d_c, d_d, d_x);
    //cudaThreadSynchronize();

    // copy result from device to host
    CUDA_SAFE_CALL( cudaMemcpy(x, d_x,memSize, cudaMemcpyDeviceToHost));

    // cleanup memory
    CUDA_SAFE_CALL(cudaFree(d_a));
    CUDA_SAFE_CALL(cudaFree(d_b));
    CUDA_SAFE_CALL(cudaFree(d_c));
    CUDA_SAFE_CALL(cudaFree(d_d));
    CUDA_SAFE_CALL(cudaFree(d_x));
}
/** @} */ // end Cyclic reduction solver (CR)
/** @} */ // end cudpp_app

