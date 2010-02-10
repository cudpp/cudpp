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

#include <cutil.h>
#include <memory.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include <cutil.h>
#include <cstdlib>
#include <cstdio>

#include "cudpp_testrig_options.h"
#include "tridiagonal_gold.h"

//#define TRIDIAGONAL_DOUBLE

int testTridiagonal(int argc, const char** argv)
{
    int retval =0;
    CUDPPHandle tridiagonalPlan = 0;
    CUDPPResult result;
    CUDPPConfiguration config;
    #ifdef TRIDIAGONAL_DOUBLE
        config.datatype = CUDPP_DOUBLE;
        typedef double T;
    #else
        config.datatype = CUDPP_FLOAT;
        typedef float T;
    #endif
    config.algorithm = CUDPP_TRIDIAGONAL_CR;
    config.options = 0;

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
        printf("Error creating CUDPPPlan\n");
        exit(-1);
    }

    int numSystems = 256;
    int systemSize = 256;
    const unsigned int memSize = sizeof(T)*numSystems*systemSize;

    T* a = (T*) malloc(memSize);
    T* b = (T*) malloc(memSize);
    T* c = (T*) malloc(memSize);
    T* d = (T*) malloc(memSize);
    T* x1 = (T*) malloc(memSize);
    T* x2 = (T*) malloc(memSize);

    for (int i = 0; i < numSystems; i++)
    {
        testGeneration(&a[i*systemSize],&b[i*systemSize],&c[i*systemSize],&d[i*systemSize],&x1[i*systemSize],systemSize);
    }

    unsigned int timer1, timer2;

    CUT_SAFE_CALL(cutCreateTimer(&timer1));
    cutStartTimer(timer1);
    cudppTridiagonal(tridiagonalPlan, a, b, c, d, x2, systemSize, numSystems);
    cutStopTimer(timer1);
    printf("numSystems: %d, systemSize: %d, GPU execution time: %f ms\n", numSystems, systemSize, cutGetTimerValue(timer1));

    CUT_SAFE_CALL(cutCreateTimer(&timer2));
    cutStartTimer(timer2);
    serialManySystems<T>(a,b,c,d,x1,systemSize,numSystems);
    cutStopTimer(timer2);
    printf("numSystems: %d, systemSize: %d, CPU execution time: %f ms\n", numSystems, systemSize, cutGetTimerValue(timer2));

    writeResultToFile<T>(x1,numSystems,systemSize,"cpu_result.txt");
    writeResultToFile<T>(x2,numSystems,systemSize,"gpu_result.txt");

    retval = compareManySystems<T>(x1,x2,systemSize,numSystems,0.001f);

    free(a);
    free(b);
    free(c);
    free(d);
    free(x1);
    free(x2);

    return retval;
}
