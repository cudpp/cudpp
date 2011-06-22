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
 * test_scan.cu
 *
 * @brief Host testrig routines to exercise cudpp's scan functionality.
 */

#include <stdio.h>
#include <cutil.h>
#include <time.h>
#include <limits.h>
#include <cstring>

#include "cudpp.h"
#include "cudpp_testrig_options.h"
#include "cudpp_testrig_utils.h"
#include "arraycompare.h"

#include "scan_gold.cpp" // this file is all templates now; must be included

/**
 * testScan exercises cudpp's unsegmented scan functionality.
 * Possible command line arguments:
 * - --op=OP: sets scan operation to OP (sum, max, min and multiply.)
 * - --forward, --backward: sets direction of scan
 * - --exclusive, --inclusive: sets exclusivity of scan
 * - --n=#: number of elements in scan
 * - Also "global" options (see setOptions)
 * @param argc Number of arguments on the command line, passed
 * directly from main
 * @param argv Array of arguments on the command line, passed directly
 * from main
 * @param configPtr Configuration for scan, set by caller
 * @return Number of tests that failed regression (0 for all pass)
 * @see CUDPPConfiguration, setOptions, cudppScan
 */
template <typename T>
int scanTest(int argc, const char **argv, const CUDPPConfiguration &config, 
             const testrigOptions &testOptions)
{
    int retval = 0;

    unsigned int timer;

    CUT_SAFE_CALL(cutCreateTimer(&timer));

    int numElements = 8388608; // maximum test size

    bool quiet = (CUTTrue == cutCheckCmdLineFlag(argc, (const char**) argv, "quiet"));

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
        test[0] = numElements;
        numTests = 1;
    }

    // Initialize CUDPP
    CUDPPResult result = CUDPP_SUCCESS;
    CUDPPHandle theCudpp;
    result = cudppCreate(&theCudpp);    
    if (result != CUDPP_SUCCESS)
    {
        fprintf(stderr, "Error initializing CUDPP Library\n");
        retval = (oneTest) ? 1 : numTests;
        return retval;
    }

    CUDPPHandle scanPlan;
    
    result = cudppPlan(theCudpp, &scanPlan, config, numElements, 1, 0);

    if (result != CUDPP_SUCCESS)
    {
        fprintf(stderr, "Error creating plan for Scan\n");
        retval = (oneTest) ? 1 : numTests;
        return retval;
    }
 
    unsigned int memSize = sizeof(T) * numElements;
 
    // allocate host memory to store the input data
    T* i_data = (T*) malloc( memSize);
 
    // allocate host memory to store the output data
    T* o_data = (T*) malloc( memSize);
 
    // host memory to store input flags
  
    // initialize the input data on the host
    for(int i = 0; i < numElements; ++i)
    {
        i_data[i] = (float)(rand() & 1);
    }

    // allocate and compute reference solution
    T* reference = (T*) malloc( memSize);
 
    // allocate device memory input and output arrays
    T* d_idata     = NULL;
    T* d_odata     = NULL;

    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_idata, memSize));
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_odata, memSize));
     
    // copy host memory to device input array
    CUDA_SAFE_CALL( cudaMemcpy(d_idata, i_data, memSize,
                               cudaMemcpyHostToDevice) );
    // initialize all the other device arrays to be safe
    CUDA_SAFE_CALL( cudaMemcpy(d_odata, o_data, memSize,
                               cudaMemcpyHostToDevice) );
 
    for (int k = 0; k < numTests; ++k)
    {
        char op[10];
        switch (config.op)
        {
        case CUDPP_ADD:
            strcpy(op, "sum");
            break;
        case CUDPP_MULTIPLY:
            strcpy(op, "multiply");
            break;
        case CUDPP_MAX:
            strcpy(op, "max");
            break;
        case CUDPP_MIN:
            strcpy(op, "min");
            break;
        case CUDPP_OPERATOR_INVALID:
            fprintf(stderr, "testScan called with invalid operator\n");
            break;
        }
        if (!quiet)
        {
            printf("Running a%s%s %s-scan of %d %s elements\n",
                   (config.options & CUDPP_OPTION_BACKWARD) ? " backward" : "",
                   (config.options & CUDPP_OPTION_INCLUSIVE) ? " inclusive" : "",
                   op,
                   test[k],
                   datatype_to_string[(int) config.datatype]);
            fflush(stdout);
        }

        cutResetTimer(timer);
        cutStartTimer(timer);
         
        if (config.op == CUDPP_ADD)
            computeSumScanGold( reference, i_data, test[k], config);
        else if (config.op == CUDPP_MULTIPLY)
            computeMultiplyScanGold( reference, i_data, test[k], config);
        else if (config.op == CUDPP_MAX)
            computeMaxScanGold( reference, i_data, test[k], config);     
        else if (config.op == CUDPP_MIN)
            computeMinScanGold( reference, i_data, test[k], config);    

        cutStopTimer(timer);
     
        if (!quiet)
            printf("CPU execution time = %f\n", cutGetTimerValue(timer));
        cutResetTimer(timer);
  
        // Run the scan
        // run once to avoid timing startup overhead.
#ifndef __DEVICE_EMULATION__
        cudppScan(scanPlan, d_odata, d_idata, test[k]);
#endif

        cutStartTimer(timer);
        for (int i = 0; i < testOptions.numIterations; i++)
        {
            cudppScan(scanPlan, d_odata, d_idata, test[k]);
        }
        cudaThreadSynchronize();
        cutStopTimer(timer);
     
        // copy result from device to host
        CUDA_SAFE_CALL(cudaMemcpy( o_data, d_odata, sizeof(T) * test[k],
                                   cudaMemcpyDeviceToHost));
          
        // check if the result is equivalent to the expected solution
        ArrayComparator<T> compare;
        CUTBoolean result = compare.compare_e( reference, o_data, 
                                               test[k], 0.001f);

        retval += (CUTTrue == result) ? 0 : 1;
        if (!quiet)
        {
            printf("%s test %s\n", testOptions.runMode,
                   (CUTTrue == result) ? "PASSED" : "FAILED");
            printf("Average execution time: %f ms\n",
                   cutGetTimerValue(timer) / testOptions.numIterations);
        }
        else
        {
            printf("\t%10d\t%0.4f\n", test[k], cutGetTimerValue(timer) / testOptions.numIterations);
        }
        if (testOptions.debug)
        {
            printArray(i_data, numElements);
            printArray(o_data, numElements);
        }
     
        cutResetTimer(timer); 
    }
    if (!quiet)
        printf("\n");

    result = cudppDestroyPlan(scanPlan);

    if (result != CUDPP_SUCCESS)
    {
        printf("Error destroying CUDPPPlan for Scan\n");
    }

    result = cudppDestroy(theCudpp);

    if (result != CUDPP_SUCCESS)
    {
        printf("Error shutting down CUDPP Library\n");
    }
 
    // cleanup memory
    cutDeleteTimer(timer);

    free(i_data);
    free(o_data);
    free(reference);
    cudaFree(d_odata);
    cudaFree(d_idata);
    return retval;
}

/**
 * testSegmentedScan exercises cudpp's unsegmented scan functionality.
 * Possible command line arguments:
 * - --op=OP: sets scan operation to OP (sum, max, min and multiply.)
 * - --forward: sets direction of scan
 * - --exclusive, --inclusive: sets exclusivity of scan
 * - --n=#: number of elements in scan
 * - Also "global" options (see setOptions)
 * @param argc Number of arguments on the command line, passed
 * directly from main
 * @param argv Array of arguments on the command line, passed directly
 * from main
 * @param configPtr Configuration for scan, set by caller
 * @return Number of tests that failed regression (0 for all pass)
 * @see CUDPPConfiguration, setOptions, cudppSegmentedScan
 */
int testSegmentedScan(int argc, const char **argv, const CUDPPConfiguration *configPtr)
{
    int retval = 0;

    testrigOptions testOptions;
    setOptions(argc, argv, testOptions);

    unsigned int timer;

    CUT_SAFE_CALL(cutCreateTimer(&timer));

    CUDPPConfiguration config;
    config.algorithm = CUDPP_SEGMENTED_SCAN;

    if (configPtr != NULL)
    {
        config = *configPtr;
    }
    else
    {
        CUDPPOption direction = CUDPP_OPTION_FORWARD;
        CUDPPOption inclusivity = CUDPP_OPTION_EXCLUSIVE;

        //default segmented sum scan
        config.op = CUDPP_ADD;
        config.datatype = CUDPP_FLOAT;

        if (testOptions.op && !strcmp(testOptions.op, "max"))
        {
            config.op = CUDPP_MAX;
        }

        if (testOptions.op && !strcmp(testOptions.op, "multiply"))
        {
            config.op = CUDPP_MULTIPLY;
        }

        if (testOptions.op && !strcmp(testOptions.op, "min"))
        {
            config.op = CUDPP_MIN;
        }

        if (CUTTrue == cutCheckCmdLineFlag(argc, argv, "backward"))
        {
            direction = CUDPP_OPTION_BACKWARD;
        }
     
        if (CUTTrue == cutCheckCmdLineFlag(argc, argv, "exclusive"))
        {
            inclusivity = CUDPP_OPTION_EXCLUSIVE;
        }

        if (CUTTrue == cutCheckCmdLineFlag(argc, argv, "inclusive"))
        {
            inclusivity = CUDPP_OPTION_INCLUSIVE;
        }
     
        config.options = direction | inclusivity;
    }
 
    int numElements = 8388608; // maximum test size
    int numFlags = 4;

    bool quiet = (CUTTrue == cutCheckCmdLineFlag(argc, (const char**) argv, "quiet"));

    bool oneTest = false;
    if (CUTTrue == cutGetCmdLineArgumenti(argc, (const char**) argv, "n",
                                          &numElements))
    {
        oneTest = true;
    }

    unsigned int test[] = {32, 128, 256, 512, 1024, 1025, 32768, 45537, 65536, 131072,
                           262144, 500001, 524288, 1048577, 1048576, 1048581, 2097152, 4194304, 8388608};

    int numTests = sizeof(test) / sizeof(test[0]);
    if (oneTest)
    {
        test[0] = numElements;
        numTests = 1;
    }

    // Initialize CUDPP
    CUDPPResult result = CUDPP_SUCCESS;
    CUDPPHandle theCudpp;
    result = cudppCreate(&theCudpp);
    if (result != CUDPP_SUCCESS)
    {
        fprintf(stderr, "Error initializing CUDPP Library.\n");
        retval = (oneTest) ? 1 : numTests;
        return retval;
    }

    CUDPPHandle segmentedScanPlan;
    result = cudppPlan(theCudpp, &segmentedScanPlan, config, numElements, 1, 0);

    if (result != CUDPP_SUCCESS)
    {
        fprintf(stderr, "Error creating plan for Segmented Scan\n");
        retval = (oneTest) ? 1 : numTests;
        return retval;
    }
 
    unsigned int memSize = sizeof(float) * numElements;
 
    // allocate host memory to store the input data
    float* i_data = (float*) malloc( memSize);

    // allocate host memory to store the input data
    unsigned int* i_flags = 
        (unsigned int*) malloc(sizeof(unsigned int) * numElements);

    // Set all flags to 0
    memset(i_flags, 0, sizeof(unsigned int) * numElements);
 
    // allocate host memory to store the output data
    float* o_data = (float*) malloc( memSize);
 
    // host memory to store input flags
  
    // initialize the input data on the host
    for(int i = 0; i < numElements; ++i)
    {
        i_data[i] = (float) 1; // (rand() & 1);
    }

    // allocate and compute reference solution
    float* reference = (float*) malloc( memSize);
 
    // allocate device memory input and output arrays
    float* d_idata         = NULL;
    unsigned int *d_iflags = NULL;
    float* d_odata         = NULL;


    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_idata, memSize));
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_iflags, 
                                sizeof(unsigned int) * numElements));
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_odata, memSize));
     
    // copy host memory to device input array
    CUDA_SAFE_CALL( cudaMemcpy(d_idata, i_data, memSize,
                               cudaMemcpyHostToDevice) );
    // initialize all the other device arrays to be safe
    CUDA_SAFE_CALL( cudaMemcpy(d_odata, o_data, memSize,
                               cudaMemcpyHostToDevice) );
 
    for (int k = 0; k < numTests; ++k)
    {
        // Generate flags
        for(int i = 0; i < numFlags; ++i) 
        {
            unsigned int idx;

            // The flag at the first position is implicitly set
            // so try to generate non-zero positions
            while((idx = (unsigned int)
                   ((test[k] - 1) * (rand() / (float)RAND_MAX))) 
                  == 0)
            {
            }
            
            // printf("Setting flag at pos %d\n", idx);
            i_flags[idx] = 1;
        }
        // i_flags[5]=1;
        // Copy flags to GPU
        CUDA_SAFE_CALL( cudaMemcpy(d_iflags, i_flags, 
                                   sizeof(unsigned int) * test[k],
                                   cudaMemcpyHostToDevice) );

        char op[10];
        switch (config.op)
        {
        case CUDPP_ADD:
            strcpy(op, "sum");
            break;
        case CUDPP_MULTIPLY:
            strcpy(op, "multiply");
            break;
        case CUDPP_MAX:
            strcpy(op, "max");
            break;
        case CUDPP_MIN:
            strcpy(op, "min");
            break;
        case CUDPP_OPERATOR_INVALID:
            fprintf(stderr, "testSegmentedScan called with invalid operator\n");
            break;
        }

        if (!quiet)
        {
            printf("Running a%s%s %s-segmented scan of %d elements\n",               
                   (config.options & CUDPP_OPTION_BACKWARD) ? " backward" : "",
                   (config.options & CUDPP_OPTION_INCLUSIVE) ? " inclusive" : "",
                   op,
                   test[k]);
            fflush(stdout);
        }

        fflush(stdout);

        cutResetTimer(timer);
        cutStartTimer(timer);
         
        if(config.op == CUDPP_ADD)
            computeSumSegmentedScanGold(reference, i_data, i_flags, test[k], config);
        else if (config.op == CUDPP_MAX)
            computeMaxSegmentedScanGold(reference, i_data, i_flags, test[k], config);
        else if (config.op == CUDPP_MULTIPLY)
            computeMultiplySegmentedScanGold(reference, i_data, i_flags, test[k], config);
        else if (config.op == CUDPP_MIN)
            computeMinSegmentedScanGold(reference, i_data, i_flags, test[k], config);
        
        cutStopTimer(timer);
     
        if (!quiet)
        {
            printf("CPU execution time = %f\n", cutGetTimerValue(timer));
        }
        cutResetTimer(timer);
  
        // Run the scan
        // run once to avoid timing startup overhead.
#ifndef __DEVICE_EMULATION__
        cudppSegmentedScan(segmentedScanPlan, d_odata, d_idata, d_iflags, test[k]);
#endif

        cutStartTimer(timer);
        for (int i = 0; i < testOptions.numIterations; i++)
        {
            cudppSegmentedScan(segmentedScanPlan, d_odata, d_idata, d_iflags, test[k]);       
        }
        cudaThreadSynchronize();
        cutStopTimer(timer);
     
        // copy result from device to host
        CUDA_SAFE_CALL(cudaMemcpy( o_data, d_odata, sizeof(float) * test[k],
                                   cudaMemcpyDeviceToHost));
          
        // check if the result is equivalent to the expected soluion
        CUTBoolean result = cutComparefe( reference, o_data, test[k], 0.001f);

        retval += (CUTTrue == result) ? 0 : 1;
        if (!quiet)
        {
            printf("%s test %s\n", testOptions.runMode,
                   (CUTTrue == result) ? "PASSED" : "FAILED");
            printf("Average execution time: %f ms\n",
                   cutGetTimerValue(timer) / testOptions.numIterations);
        }
        else
        {
            printf("\t%10d\t%0.4f\n", test[k], cutGetTimerValue(timer) / testOptions.numIterations);
        }

        if (testOptions.debug)
        {
            for (unsigned int i = 0; i < test[k]; ++i)
            {
                if (reference[i] != o_data[i]) printf("%d %f %f\n", i, o_data[i], reference[i]);
                // printf("%f %f\n", reference[i], o_data[i]);
            }
            // printf("\n");
            // for (unsigned int i = 0; i < test[k]; ++i)
            // {
            //    printf("%f ", reference[i]);
            //}
            // printf("\n");
        }
     
        cutResetTimer(timer); // needed after CUT alpha2
    }
    if (!quiet)
        printf("\n");

    result = cudppDestroyPlan(segmentedScanPlan);

    if (result != CUDPP_SUCCESS)
    {
        printf("Error destroying CUDPPPlan for Scan\n");
    }

    result = cudppDestroy(theCudpp);

    if (result != CUDPP_SUCCESS)
    {
        printf("Error shutting down CUDPP Library.\n");
    }
 
    // cleanup memory
    cutDeleteTimer(timer);

    free(i_data);
    free(i_flags);
    free(o_data);
    free(reference);
    cudaFree(d_odata);
    cudaFree(d_idata);
    cudaFree(d_iflags);
    return retval;
}

/**
 * testMultiSumScan exercises cudpp's multiple-unsegmented-scan functionality.
 * @param argc Number of arguments on the command line, passed
 * directly from main
 * @param argv Array of arguments on the command line, passed directly
 * from main
 * @return Number of tests that failed regression (0 for all pass)
 * @see cudppMultiScan
 */
// template<class T>
int testMultiSumScan(int argc, const char **argv)
{
    int retval = 0;
    testrigOptions testOptions;
    setOptions(argc, argv, testOptions);

    unsigned int timer;
    CUT_SAFE_CALL(cutCreateTimer(&timer));

    CUDPPConfiguration config;
    CUDPPOption direction = CUDPP_OPTION_FORWARD;
    CUDPPOption inclusivity = CUDPP_OPTION_EXCLUSIVE;
     
    if (CUTTrue == cutCheckCmdLineFlag(argc, argv, "backward"))
    {
        direction = CUDPP_OPTION_BACKWARD;
    }
         
    config.algorithm = CUDPP_SCAN;
    config.options = direction | inclusivity;
    config.op = CUDPP_ADD;
    config.datatype = CUDPP_FLOAT;
 
    int numElements = 1024; // maximum test size
    int numRows = 1024;

    //bool oneTest = false;
    if (CUTTrue == cutGetCmdLineArgumenti(argc, (const char**) argv, "n",
                                          &numElements))
    {
        //   oneTest = true;
    }
    if (CUTTrue == cutGetCmdLineArgumenti(argc, (const char**) argv, "r",
                                          &numRows))
    {
        //   oneTest = true;
    }

    size_t myPitch = numElements * sizeof(float);
    size_t hmemSize = numRows * myPitch;
 
    // allocate host memory to store the input data
    float* i_data = (float*) malloc( hmemSize);
 
    // allocate host memory to store the output data
    float* o_data = (float*) malloc( hmemSize);
    
    for( int i = 0; i < numElements * numRows; ++i)
    {
        i_data[i] = (float)(rand() & 31);
        o_data[i] = -1;
    }

    // allocate and compute reference solution
    float* reference = (float*) malloc(hmemSize);
    computeMultiRowSumScanGold( reference, i_data, numElements, numRows, config);
 
    // allocate device memory input and output arrays
    float* d_idata     = NULL;
    float* d_odata     = NULL;

    size_t d_ipitch = 0;
    size_t d_opitch = 0;
 
    CUDA_SAFE_CALL( cudaMallocPitch( (void**) &d_idata, &d_ipitch,
                                     myPitch, numRows));
    CUDA_SAFE_CALL( cudaMallocPitch( (void**) &d_odata, &d_opitch,
                                     myPitch, numRows));
    // copy host memory to device input array
    CUDA_SAFE_CALL( cudaMemcpy2D(d_idata, d_ipitch, i_data, myPitch, myPitch,
                                 numRows, cudaMemcpyHostToDevice) );
    // initialize all the other device arrays to be safe
    CUDA_SAFE_CALL( cudaMemcpy2D(d_odata, d_ipitch, o_data, myPitch, myPitch,
                                 numRows, cudaMemcpyHostToDevice) );

    size_t rowPitch = d_ipitch / sizeof(float);

       
    CUDPPResult ret;
    CUDPPHandle theCudpp;
    ret = cudppCreate(&theCudpp);

    if (ret != CUDPP_SUCCESS)
    {
        fprintf(stderr, "Error Initializing CUDPP Library.\n");
        retval = 1;
        return retval;
    }

    CUDPPHandle multiscanPlan = 0;
    ret = cudppPlan(theCudpp, &multiscanPlan, config, numElements, numRows, rowPitch);

    if (ret != CUDPP_SUCCESS)
    {
        fprintf(stderr, "Error creating CUDPP Plan for multi-row Scan.\n");
        retval = 1;
        return retval;
    }

    printf("Running a%s sum-scan of %d rows of %d elements\n",
           (config.options & CUDPP_OPTION_BACKWARD) ? " backward" : "",
           numRows,
           numElements);
    fflush(stdout);

    // run once to avoid timing startup overhead.
#ifndef __DEVICE_EMULATION__
    cudppMultiScan(multiscanPlan, d_odata, d_idata, numElements, numRows);
#endif
    cutStartTimer(timer);
    for (int i = 0; i < testOptions.numIterations; i++)
    {
        cudppMultiScan(multiscanPlan, d_odata, d_idata, numElements, numRows);

    }
    cudaThreadSynchronize();
    cutStopTimer(timer);

    // copy result from device to host
    CUDA_SAFE_CALL(cudaMemcpy2D( o_data, myPitch, d_odata, d_opitch,
                                 myPitch, numRows, cudaMemcpyDeviceToHost));
     
    // check if the result is equivalent to the expected solution
    // ArrayComparator<T> compare;
    // CUTBoolean result = compare.compare_e( reference, o_data, 
    // numElements*numRows, 0.001f);
    CUTBoolean result = cutComparefe( reference, o_data, 
                                      numElements*numRows, 0.001f);
    retval += (CUTTrue == result) ? 0 : 1;
    printf("%s test %s\n", testOptions.runMode,
           (CUTTrue == result) ? "PASSED" : "FAILED");
    printf("Average execution time: %f ms\n",
           cutGetTimerValue(timer) / testOptions.numIterations);
    printf("\n");

    ret = cudppDestroyPlan(multiscanPlan);

    if (ret != CUDPP_SUCCESS)
    {
        printf("Error destroying CUDPPPlan for Multiscan\n");
    }

    ret = cudppDestroy(theCudpp);

    if (ret != CUDPP_SUCCESS)
    {
        printf("Error shutting down CUDPP Library.\n");
    }

    // cleanup memory
    cutDeleteTimer(timer);
    free( i_data);
    free(o_data);
    free( reference);
    cudaFree( d_odata);
    cudaFree( d_idata);
    return retval;
}

int testScan(int argc, const char **argv, const CUDPPConfiguration *configPtr)
{
    testrigOptions testOptions;
    setOptions(argc, argv, testOptions);

    CUDPPConfiguration config;
    config.algorithm = CUDPP_SCAN;

    if (configPtr != NULL)
    {
        config = *configPtr;
    }
    else
    {
        CUDPPOption direction = CUDPP_OPTION_FORWARD;
        CUDPPOption inclusivity = CUDPP_OPTION_EXCLUSIVE;

        //default sum scan
        config.op = CUDPP_ADD;
        config.datatype = getDatatypeFromArgv(argc, argv);

        if (testOptions.op && !strcmp(testOptions.op, "max"))
        {
            config.op = CUDPP_MAX;
        }
        else if (testOptions.op && !strcmp(testOptions.op, "min"))
        {
            config.op = CUDPP_MIN;
        }
        else if (testOptions.op && !strcmp(testOptions.op, "multiply"))
        {
            config.op = CUDPP_MULTIPLY;
        }

        if (CUTTrue == cutCheckCmdLineFlag(argc, argv, "backward"))
        {
            direction = CUDPP_OPTION_BACKWARD;
        }
     
        if (CUTTrue == cutCheckCmdLineFlag(argc, argv, "exclusive"))
        {
            inclusivity = CUDPP_OPTION_EXCLUSIVE;
        }

        if (CUTTrue == cutCheckCmdLineFlag(argc, argv, "inclusive"))
        {
            inclusivity = CUDPP_OPTION_INCLUSIVE;
        }
     
        config.options = direction | inclusivity;
    }

    switch(config.datatype)
    {
    case CUDPP_INT:
        return scanTest<int>(argc, argv, config, testOptions);
        break;
    case CUDPP_UINT:
        return scanTest<unsigned int>(argc, argv, config, testOptions);
        break;
    case CUDPP_FLOAT:
        return scanTest<float>(argc, argv, config, testOptions);
        break;
    case CUDPP_DOUBLE:
        return scanTest<double>(argc, argv, config, testOptions);
        break;
    case CUDPP_LONGLONG:
        return scanTest<long long>(argc, argv, config, testOptions);
        break;
    case CUDPP_ULONGLONG:
        return scanTest<unsigned long long>(argc, argv, config, testOptions);
        break;
    default:
        return 0;
        break;
    }
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
