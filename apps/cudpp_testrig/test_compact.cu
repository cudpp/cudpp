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
 * test_compact.cu
 *
 * @brief Host testrig routines to exercise cudpp's compact functionality.
 */

#include <stdio.h>
#include <cutil.h>
#include <time.h>
#include <limits.h>

#include "cudpp.h"

#include "cudpp_testrig_options.h"


extern "C"
unsigned int compactGold(float* reference, const float* idata,
                         const unsigned int *isValid, const unsigned int len,
                         const CUDPPConfiguration &config);

/**
 * testCompact exercises cudpp's compact functionality.
 * Possible command line arguments:
 * - --forward, --backward: sets direction of scan
 * - --exclusive, --inclusive: sets exclusivity of scan
 * - --n=#: number of elements in scan
 * - --prob=#: fraction (0.0-1.0) of elements that are valid (default: 0.3)
 * - Also "global" options (see setOptions)
 * @param argc Number of arguments on the command line, passed
 * directly from main
 * @param argv Array of arguments on the command line, passed directly
 * from main
 * @return Number of tests that failed regression (0 for all pass)
 * @see setOptions, cudppCompact
 */
int testCompact(int argc, const char **argv, const CUDPPConfiguration *configPtr)
{
    int retval = 0;

    testrigOptions testOptions;
    setOptions(argc, argv, testOptions);

    unsigned int timer;
    CUT_SAFE_CALL(cutCreateTimer(&timer));

    CUDPPConfiguration config;
    config.algorithm = CUDPP_COMPACT;
    config.datatype = CUDPP_FLOAT;

    bool quiet = (cutCheckCmdLineFlag(argc, (const char**)argv, "quiet") == CUTTrue);	

    if (configPtr != NULL)
    {
        config = *configPtr;
    }
    else
    {
        config.options = CUDPP_OPTION_FORWARD;

        if (CUTTrue == cutCheckCmdLineFlag(argc, argv, "backward"))
        {
            config.options = CUDPP_OPTION_BACKWARD;
        }  
    }
   
    int numElements = 8388608; // maximum test size
    float probValid = 0.3f;

    bool oneTest = false;

    if (CUTTrue == cutGetCmdLineArgumenti(argc, (const char**) argv, "n",
        &numElements))
    {
        oneTest = true;
    }

    unsigned int test[] = {39, 128, 256, 512, 1000, 1024, 1025, 32768, 45537, 65536, 131072,
        262144, 500001, 524288, 1048577, 1048576, 1048581, 2097152, 4194304, 8388608};

    int numTests = sizeof(test) / sizeof(test[0]);

    if (oneTest)
    {
        numTests = 1;
        test[0] = numElements;
    }   

    cutGetCmdLineArgumentf(argc, (const char**) argv, "prob", &probValid);

    CUDPPHandle plan;
    CUDPPResult result = CUDPP_SUCCESS;
    result = cudppPlan(&plan, config, numElements, 1, 0);

    if (result != CUDPP_SUCCESS)
    {
        if (!quiet)
            fprintf(stderr, "Error creating plan for Compact\n");
        retval = (oneTest) ? 1 : numTests;
        return retval;
    }

    unsigned int memSize = sizeof(float) * numElements;
    
    // allocate host memory to store the input data
    float* h_data = (float*) malloc( memSize);
    unsigned int *h_isValid = (unsigned int*) malloc(sizeof(unsigned int) * numElements);

    // allocate and compute reference solution
    float* reference = (float*) malloc( memSize);

    // allocate device memory input and output arrays
    float* d_idata     = NULL;
    float* d_odata     = NULL;
    unsigned int* d_isValid   = NULL;
    size_t* d_numValid  = NULL;

    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_idata, memSize));
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_odata, memSize));
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_isValid, sizeof(unsigned int) * numElements));
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_numValid, sizeof(size_t)));

    size_t *numValidElements = (size_t*)malloc(sizeof(size_t));

    // numTests = numTests;
    for (int k = 0; k < numTests; ++k)
    {
        if (!quiet)
        {
            printf("Running a %sstream-compact of %d elements\n", 
                   config.options & CUDPP_OPTION_BACKWARD ? "backward " : "", test[k]);
        }
        fflush(stdout);

        //srand((unsigned int)time(NULL));
        srand(222);

        for( unsigned int i = 0; i < test[k]; ++i)
        {
            if (rand() / (float)RAND_MAX > probValid)
                h_isValid[i] = 0;
            else
                h_isValid[i] = 1;
            h_data[i] = (float)(rand() + 1);
        }

        memset(reference, 0, sizeof(float) * test[k]);
        size_t c_numValidElts =
            compactGold( reference, h_data, h_isValid, test[k], config);
        CUDA_SAFE_CALL( cudaMemcpy(d_idata, h_data, sizeof(float) * test[k],
                                   cudaMemcpyHostToDevice) );

        CUDA_SAFE_CALL( cudaMemcpy(d_isValid, h_isValid, sizeof(unsigned int) * test[k],
                                   cudaMemcpyHostToDevice) );

        CUDA_SAFE_CALL( cudaMemset(d_odata, 0, sizeof(float) * test[k]));

        // run once to avoid timing startup overhead.
#ifndef __DEVICE_EMULATION__
        cudppCompact(plan, d_odata, d_numValid, d_idata, d_isValid, test[k]);
#endif

        cutStartTimer(timer);
        for (int i = 0; i < testOptions.numIterations; i++)
        {
            cudppCompact(plan, d_odata, d_numValid, d_idata, d_isValid, test[k]);
        }
        cudaThreadSynchronize();
        cutStopTimer(timer);

        // get number of valid elements back to host
        CUDA_SAFE_CALL( cudaMemcpy(numValidElements, d_numValid, sizeof(size_t), 
                                   cudaMemcpyDeviceToHost) );

        // allocate host memory to store the output data

        float* o_data = (float*) malloc( sizeof(float) * *numValidElements);

        // copy result from device to host
        CUDA_SAFE_CALL(cudaMemcpy(o_data, d_odata,
                                  sizeof(float) * *numValidElements,
                                  cudaMemcpyDeviceToHost));
        // check if the result is equivalent to the expected soluion
        if (!quiet)
            printf("numValidElements: %d\n", *numValidElements);
        CUTBoolean result = cutComparefe( reference, o_data, *numValidElements, 0.001f);

        free(o_data);

        if (c_numValidElts != *numValidElements)
        {
            retval += 1;
            if (!quiet)
            {
                printf("Number of valid elements does not match reference solution.\n");
                printf("Test FAILED\n");
            }
        }
        else
        {
            retval += (CUTTrue == result) ? 0 : 1;
            if (!quiet)
            {
                printf("%s test %s\n", testOptions.runMode,
                       (CUTTrue == result) ? "PASSED" : "FAILED");
            }
        }
        if (!quiet)
        {
            printf("Average execution time: %f ms\n",
                   cutGetTimerValue(timer) / testOptions.numIterations);
        }
        else
            printf("\t%10d\t%0.4f\n", test[k], cutGetTimerValue(timer) / testOptions.numIterations);

        cutResetTimer(timer);
    }
    if (!quiet)
        printf("\n");

    result = cudppDestroyPlan(plan);
    if (result != CUDPP_SUCCESS)
    {
        if (!quiet)
            printf("Error destroying CUDPPPlan for Scan\n");
    }

    // cleanup memory
    cutDeleteTimer(timer);
    free( h_data);
    free( h_isValid);
    free( reference);
    cudaFree( d_odata);
    cudaFree( d_idata);
    cudaFree( d_isValid);
    cudaFree( d_numValid);
    return retval;
}
