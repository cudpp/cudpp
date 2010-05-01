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
 * test_reduce.cu
 *
 * @brief Host testrig routines to exercise cudpp's reduction functionality.
 */

#include "cudpp.h"
#include "cudpp_testrig_options.h"
#include <iostream>

extern "C" 
void computeSumReduceGold( float &out, const float* idata, const unsigned int len);

extern "C" 
void computeMultiplyReduceGold( float &out, const float* idata, const unsigned int len);

extern "C" 
void computeMaxReduceGold( float &out, const float* idata, const unsigned int len);

extern "C" 
void computeMinReduceGold( float &out, const float* idata, const unsigned int len);


int reduceTest(int argc, const char **argv, const CUDPPConfiguration &config, testrigOptions &testOptions)
{
    int retval = 0;

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

    CUDPPHandle plan;
    CUDPPTuneConfig tuneConfig;

    result = CUDPP_SUCCESS;
    result = cudppPlan(&plan, config, numElements, 1, 0);
    if(result != CUDPP_SUCCESS)
    {
        printf("Error in plan creation\n");
        retval = numTests;
        cudppDestroyPlan(plan);       
        return retval;
    }

    unsigned int memSize = sizeof(float) * numElements;

    // allocate host memory to store the input data
    float* i_data = new float[numElements];

    // initialize the input data on the host
    float range = 100;
    if (config.op == CUDPP_MULTIPLY)
    {

        if (config.datatype == CUDPP_FLOAT) range = 1;
        else                                range = 2;
    }        

     for(int i = 0; i < numElements; ++i)
    {
        i_data[i] = (float)(rand() & 1);
    }
    
    float reference = 0;

    // allocate device memory input and output arrays
    float* d_idata     = NULL;
    float* d_odata     = NULL;

    CUDA_SAFE_CALL( cudaMalloc( &d_idata, memSize));
    CUDA_SAFE_CALL( cudaMalloc( &d_odata, sizeof(float)));

    // copy host memory to device input array
    CUDA_SAFE_CALL( cudaMemcpy(d_idata, i_data, memSize, cudaMemcpyHostToDevice) );

    CUDA_SAFE_CALL( cudaMemset(d_odata, 0, sizeof(float)) );

    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    tuneConfig.reTune = false;
    if(CUTTrue == cutCheckCmdLineFlag(argc, argv, "retune"))
        tuneConfig.reTune = true;
    for (int k = 0; k < numTests; ++k)
    {
        
        if(CUTTrue == cutCheckCmdLineFlag(argc, argv, "tune"))
        {
            tuneConfig.numElements = test[k];
            tuneConfig.tuneFilePath = "reductionTuned.res";
                
            cudppTune(plan, tuneConfig);
            tuneConfig.reTune = false;
            
        }
     
        char op[10];
        switch (config.op)
        {
        case CUDPP_ADD:
        default:
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
        }
        char dt[10];
        switch (config.datatype)
        {
        case CUDPP_UINT:
        default:
            strcpy(dt, "uint");
            break;
        case CUDPP_INT:
            strcpy(dt, "int");
            break;
        case CUDPP_FLOAT:
            strcpy(dt, "float");
            break;
        }

        if (!quiet)
        {
            printf("Running a %s-reduction of %d %s elements\n", 
                   op, test[k], dt);
            fflush(stdout);
        }

        if (config.op == CUDPP_ADD)
            computeSumReduceGold( reference, i_data, test[k]);
        else if (config.op == CUDPP_MULTIPLY)
            computeMultiplyReduceGold( reference, i_data, test[k]);
        else if (config.op == CUDPP_MAX)
            computeMaxReduceGold( reference, i_data, test[k]);
        else if (config.op == CUDPP_MIN)
            computeMinReduceGold( reference, i_data, test[k]);

        // Run the reduction
        // run once to avoid timing startup overhead.
#ifndef __DEVICE_EMULATION__
        cudppReduce(plan, d_odata, d_idata, test[k]);
#endif

        CUDA_SAFE_CALL( cudaEventRecord(startEvent, 0) );
        for (int i = 0; i < testOptions.numIterations; i++)
        {
            cudppReduce(plan, d_odata, d_idata, test[k]);
        }
        CUDA_SAFE_CALL( cudaEventRecord(stopEvent, 0) );
        CUDA_SAFE_CALL( cudaEventSynchronize(stopEvent) );

        float time = 0;
        CUDA_SAFE_CALL( cudaEventElapsedTime(&time, startEvent, stopEvent));
        time /= testOptions.numIterations;
        double bandwidth = 1.0e-6 * test[k] * sizeof(float) / time;

        float o_data = 0;
        // copy result from device to host
        CUDA_SAFE_CALL(cudaMemcpy( &o_data, d_odata, sizeof(float), cudaMemcpyDeviceToHost));

        double threshold = (config.op == CUDPP_MULTIPLY && config.datatype == CUDPP_FLOAT) ? 1 : test[k] * 5e-6;
        bool correct = (abs((double)reference - (double)o_data) < threshold);

        // correct result?
        retval += (correct) ? 0 : 1;

        if (!quiet)
        {
            printf("%s test %s\n", testOptions.runMode,
                   (correct) ? "PASSED" : "FAILED");
            if (!correct)
                std::cout << o_data << " != " << reference << std::endl;
            printf("Average execution time: %f ms, ", time);
            printf(": %0.4f GB/s\n", bandwidth);
        }
        else
        {
            printf("\t%10d\t%0.4f\t%0.4f\n", test[k], time, bandwidth);
        }
    }

    result = cudppDestroyPlan(plan);

    if (result != CUDPP_SUCCESS)
    {	
        printf("Error destroying CUDPPPlan for Scan\n");
        retval = numTests;
    }


    if (result != CUDPP_SUCCESS)
    {	
        printf("Error shutting down CUDPP Library.\n");
        retval = numTests;
    }

    delete [] i_data;
    cudaFree(d_odata);
    cudaFree(d_idata);
    return retval;
}

int testReduce(int argc, const char **argv, const CUDPPConfiguration *configPtr)
{
    testrigOptions testOptions;
    setOptions(argc, argv, testOptions);

    CUDPPConfiguration config;
    config.algorithm = CUDPP_REDUCE;
    config.options = 0;

    if (configPtr != NULL)
    {
        config = *configPtr;
    }
    else
    {
        //default sum scan
        config.op = CUDPP_ADD;

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

        // default float
        config.datatype = CUDPP_FLOAT;

        if( cutCheckCmdLineFlag(argc, (const char**)argv, "float") )
        {     
            config.datatype = CUDPP_FLOAT;
        }
        else if( cutCheckCmdLineFlag(argc, (const char**)argv, "uint") )
        {        
            config.datatype = CUDPP_UINT;
        }
        else if( cutCheckCmdLineFlag(argc, (const char**)argv, "int") )
        {        
            config.datatype = CUDPP_INT;
        }
    }

   
    return reduceTest(argc, argv, config, testOptions);
   
}
