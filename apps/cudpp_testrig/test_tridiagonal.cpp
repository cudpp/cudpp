// -------------------------------------------------------------
// cuDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision: $
// $Date: $
// -------------------------------------------------------------
// This source code is distributed under the terms of license.txt in
// the root directory of this source distribution.
// -------------------------------------------------------------

/**
 * @file
 * test_tridiagonal.cpp
 *
 * @brief Host testrig routines to exercise cudpp's tridiagonal solver functionality.
 */

#include <memory.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include <cstdlib>
#include <cstdio>

#include <cuda_runtime_api.h>
#include "cuda_util.h"

#include "cudpp_testrig_options.h"
#include "tridiagonal_gold.h"
#include "stopwatch.h"

template <typename T>
int testTridiagonalDataType(CUDPPConfiguration &config)
{
    int retval = 0;
    CUDPPHandle tridiagonalPlan = 0;
    CUDPPResult result;
    
    CUDPPHandle theCudpp;
    result = cudppCreate(&theCudpp);
    if(result != CUDPP_SUCCESS)
    {
        printf("Error initializing CUDPP Library.\n");
        retval = 1;
        return retval;
    }

    result = cudppPlan(theCudpp, &tridiagonalPlan, config, 0, 0, 0);
    if (CUDPP_SUCCESS != result)
    {
        printf("Error creating CUDPPPlan here\n");
        exit(-1);
    }

    int numSystems = 512;
    int systemSize = 512;
    const unsigned int memSize = sizeof(T)*numSystems*systemSize;

    T* a = (T*) malloc(memSize);
    T* b = (T*) malloc(memSize);
    T* c = (T*) malloc(memSize);
    T* d = (T*) malloc(memSize);
    T* x1 = (T*) malloc(memSize);
    T* x2 = (T*) malloc(memSize);

    for (int i = 0; i < numSystems; i++)
    {
        testGeneration(&a[i*systemSize], &b[i*systemSize], &c[i*systemSize], &d[i*systemSize], &x1[i*systemSize], systemSize);
    }

    // allocate device memory input and output arrays
    T* d_a;
    T* d_b;
    T* d_c;
    T* d_d;
    T* d_x;

    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_a,memSize));
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_b,memSize));
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_c,memSize));
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_d,memSize));
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_x,memSize));

   // copy host memory to device input array
    CUDA_SAFE_CALL( cudaMemcpy( d_a, a, memSize, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL( cudaMemcpy( d_b, b, memSize, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL( cudaMemcpy( d_c, c, memSize, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL( cudaMemcpy( d_d, d, memSize, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL( cudaMemcpy( d_x, x1, memSize, cudaMemcpyHostToDevice));

    // warm up the GPU to avoid the overhead time for the next timing
    cudppTridiagonal(tridiagonalPlan, d_a, d_b, d_c, d_d, d_x, systemSize, numSystems);
    if (config.datatype == CUDPP_FLOAT) printf("Runing a CR-PCR tridiagonal solver solving %d systems with each of %d equations with SINGLE precision\n", numSystems, systemSize);
    if (config.datatype == CUDPP_DOUBLE) printf("Runing a CR-PCR tridiagonal solver solving %d systems with each of %d equations with DOUBLE precision\n", numSystems, systemSize);            
    
    cudpp_app::StopWatch timer;
    timer.reset();
    timer.start();
    
    cudppTridiagonal(tridiagonalPlan, d_a, d_b, d_c, d_d, d_x, systemSize, numSystems);
    cudaThreadSynchronize();

    timer.stop();            
    printf("GPU execution time: %f ms\n", timer.getTime());
    
    // copy result from device to host
    CUDA_SAFE_CALL( cudaMemcpy(x2, d_x, memSize, cudaMemcpyDeviceToHost));

    // cleanup memory
    CUDA_SAFE_CALL(cudaFree(d_a));
    CUDA_SAFE_CALL(cudaFree(d_b));
    CUDA_SAFE_CALL(cudaFree(d_c));
    CUDA_SAFE_CALL(cudaFree(d_d));
    CUDA_SAFE_CALL(cudaFree(d_x));

    timer.reset();
    timer.start();
    
    serialManySystems<T>(a,b,c,d,x1,systemSize,numSystems);

    timer.stop();            
    printf("CPU execution time: %f ms\n", timer.getTime());
    
    retval = compareManySystems<T>(x1, x2, systemSize, numSystems, 0.001f);
    
    if (retval == 0)
        printf("test PASSED\n");
    else
        printf("test FAILED\n");
    
    printf("\n");
    
    free(a);
    free(b);
    free(c);
    free(d);
    free(x1);
    free(x2);

    return retval;
    
}

int testTridiagonal(int argc, const char** argv, const CUDPPConfiguration *configPtr)
{
    int retval = 0;

    CUDPPConfiguration config;
    config = *configPtr;
    
    if ((*configPtr).datatype == CUDPP_FLOAT)
        retval = testTridiagonalDataType<float>(config);
    
    if ((*configPtr).datatype == CUDPP_DOUBLE)
        retval = testTridiagonalDataType<double>(config);  
    
    return retval;
    
}
