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

int testTridiagonal(int argc, const char** argv)
{
    int retval =0;
    CUDPPHandle tridiagonalPlan = 0;
    CUDPPResult result;
    CUDPPConfiguration config;
    config.datatype = CUDPP_FLOAT;
    typedef float T;
    //config.datatype = CUDPP_DOUBLE;
    //typedef double T;
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

    int num_systems = 512;
    int system_size = 512;
    const unsigned int mem_size = sizeof(T)*num_systems*system_size;

    T* a = (T*) malloc(mem_size);
    T* b = (T*) malloc(mem_size);
    T* c = (T*) malloc(mem_size);
    T* d = (T*) malloc(mem_size);
    T* x1 = (T*) malloc(mem_size);
    T* x2 = (T*) malloc(mem_size);

    for (int i = 0; i < num_systems; i++)
    {
        test_gen(&a[i*system_size],&b[i*system_size],&c[i*system_size],&d[i*system_size],&x1[i*system_size],system_size);
    }

    unsigned int timer1, timer2;

    CUT_SAFE_CALL(cutCreateTimer(&timer2));
    cutStartTimer(timer2);
    cudppTridiagonal(tridiagonalPlan, a, b, c, d, x2, system_size, num_systems);
    cutStopTimer(timer2);
    printf("num_systems: %d, system_size: %d, GPU execution time: %f ms\n", num_systems, system_size, cutGetTimerValue(timer2));

    CUT_SAFE_CALL(cutCreateTimer(&timer1));
    cutStartTimer(timer1);
    serial_small_systems<T>(a,b,c,d,x1,system_size,num_systems);
    cutStopTimer(timer1);
    printf("num_systems: %d, system_size: %d, CPU execution time: %f ms\n", num_systems, system_size, cutGetTimerValue(timer1));

    //file_write_small_systems<T>(x1,num_systems,system_size,"cpu_result.txt");
    //file_write_small_systems<T>(x2,num_systems,system_size,"gpu_result.txt");

    retval = compare_small_systems<T>(x1,x2,system_size,num_systems,0.001f);

    free(a);
    free(b);
    free(c);
    free(d);
    free(x1);
    free(x2);

    return retval;
}
