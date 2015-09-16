// -------------------------------------------------------------
// cuDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision$
// $Date$
// -------------------------------------------------------------
// This source code is distributed under the terms of license.txt
// in the root directory of this source distribution.
// -------------------------------------------------------------

#include <stdio.h>
#include <math.h>
#include <cuda_runtime_api.h>

#include "cudpp.h"
#include "cudpp_testrig_options.h"
#include "cudpp_testrig_utils.h"
#include "cuda_util.h"
#include "commandline.h"

#ifdef WIN32
#undef min
#undef max
#endif

#include <limits>


typedef unsigned long long int uint64;
using namespace cudpp_app;
/*

int verifyStringSort(unsigned int *valuesSorted,
                     unsigned char* stringVals, size_t numElements,
                     int stringSize, unsigned char termC)
{
    int retval = 0;

    for(unsigned int i = 0; i < numElements-1; ++i)
    {
        unsigned int add1, add2;
        add1 = valuesSorted[i];
        add2 = valuesSorted[i+1];

        unsigned char c1, c2;

        do
        {
            c1 = (stringVals[add1]);
            c2 = (stringVals[add2]);


            add1++;
            add2++;

        }
        while(c1 == c2 && c1 != termC && c2 != termC &&
              add1 < stringSize && add2 < stringSize);

        if (c1 > c2)
        {
            printf("Error comparing index %d to %d (%d > %d) "
                   "(add1 %d add2 %d)\n",
                   i, i+1, c1, c2, valuesSorted[i], valuesSorted[i+1]);
            return 1;
        }

    }
    return retval;
}*/

void randomPermute(unsigned int* input, unsigned int numElements)
{
  //Uses knuth's method to randomly permute
  for(int i = 0; i < numElements; i++)
    input[i] = i; //rand() + rand() << 15;

  for(int i = 0; i < numElements; i++)
  {
    unsigned int rand1 = rand();
    unsigned int rand2 = (rand() << 15) + rand1;
    unsigned int swap = i + (rand2%(numElements-i));

    unsigned int temp = input[i];
    input[i] = input[swap];
    input[swap] = temp;
  }
}

int multiSplitTest(CUDPPHandle theCudpp, CUDPPConfiguration config,
                   size_t *tests, unsigned int numTests, size_t maxNumElements,
                   testrigOptions testOptions, bool quiet)
{
  int retval = 0;
    srand(44);
    unsigned int numBuckets = 32;

    unsigned int* elements = (unsigned int*) malloc(sizeof(unsigned int)*maxNumElements);
    // an arbitrary initialization
    //for(int i = 0; i<maxNumElements; i++)
    //  elements[i] = i;
    //randomPermute(elements, maxNumElements);

    unsigned int *d_elements = NULL;
    CUDA_SAFE_CALL(cudaMalloc((void**) &d_elements, maxNumElements*sizeof(unsigned int)));   // gpu input (keys)


    CUDPPHandle plan;
    CUDPPResult result = cudppPlan(theCudpp, &plan, config, maxNumElements, 1, 0);

    if(result != CUDPP_SUCCESS)
    {
        printf("Error in plan creation\n");
        retval = numTests;
        cudppDestroyPlan(plan);
        return retval;
    }

    cudaEvent_t start_event, stop_event;
    CUDA_SAFE_CALL( cudaEventCreate(&start_event) );
    CUDA_SAFE_CALL( cudaEventCreate(&stop_event) );

    for (unsigned int k = 0; k < numTests; ++k)
    {
        if(numTests == 1)
            tests[0] = maxNumElements;

        if (!quiet)
        {

            printf("Running a multi-split on %ld elements\n", tests[k]);
            fflush(stdout);
        }

        float totalTime = 0;

        // an arbitrary initialization
        for(int i = 0; i<maxNumElements; i++)
          elements[i] = i;
        randomPermute(elements, maxNumElements);

        CUDA_SAFE_CALL( cudaMemcpy(d_elements, elements,
              tests[k] * sizeof(unsigned int),
              cudaMemcpyHostToDevice) );
        CUDA_SAFE_CALL( cudaEventRecord(start_event, 0) );

        cudppMultiSplit(plan, d_elements, tests[k], numBuckets);

        CUDA_SAFE_CALL( cudaEventRecord(stop_event, 0) );
        CUDA_SAFE_CALL( cudaEventSynchronize(stop_event) );

        float time = 0;
        CUDA_SAFE_CALL( cudaEventElapsedTime(&time, start_event,
              stop_event));

        CUDA_CHECK_ERROR("testMultiSplit - cudppMultiSplit");

        // copy results
        CUDA_SAFE_CALL( cudaMemcpy(elements,
                                   d_elements,
                                   tests[k] * sizeof(unsigned int),
                                   cudaMemcpyDeviceToHost));

//        retval = verifyStringSort(h_valuesSorted,
//            stringVals, tests[k], stringSize, 0);
        if(!quiet)
        {
            printf("test %s\n", (retval == 0) ? "PASSED" : "FAILED");
            printf("Average execution time: %f ms\n", totalTime / testOptions.numIterations);
        }
        else
        {
            printf("\t%10ld\t%0.4f\n", tests[k], totalTime / testOptions.numIterations);
        }
    }
    printf("\n");

    CUDA_CHECK_ERROR("after multi-split");

    result = cudppDestroyPlan(plan);

    if (result != CUDPP_SUCCESS)
    {
        printf("Error destroying CUDPPPlan for multi-split\n");
        retval = numTests;
    }

    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);

    cudaFree(d_elements);
    free(elements);

    return retval;
}

/**
 * testStringSort tests cudpp's merge sort
 * Possible command line arguments:
 * - -n=#, number of elements in sort
 * @param argc Number of arguments on the command line, passed
 * directly from main
 * @param argv Array of arguments on the command line, passed directly
 * from main
 * @param configPtr Configuration for scan, set by caller
 * @return Number of tests that failed regression (0 for all pass)
 * @see cudppSort
 */
int testMultiSplit(int argc, const char **argv,
                   const CUDPPConfiguration *configPtr)
{

    int cmdVal;
    int retval = 0;

    bool quiet = checkCommandLineFlag(argc, argv, "quiet");
    testrigOptions testOptions;
    setOptions(argc, argv, testOptions);

    CUDPPConfiguration config;
    config.algorithm = CUDPP_MULTISPLIT;
    config.datatype = CUDPP_UINT;
    if(configPtr != NULL)
    {
        config = *configPtr;
    }

    size_t test_num_elements[] = {128, 256, 512, 513, 1000, 1024, 1025, 32768,
                     45537, 65536, 131072, 262144, 500001, 524288,
                     1048577, 1048576, 1048581, 2097152, 4194304};

    size_t test_num_buckets[] = {2, 8, 16, 32, 64};


    int numTests = sizeof(test_num_elements)/sizeof(test_num_elements[0]);

    // small GPUs are susceptible to running out of memory,
    // restrict the tests to only those where we have enough
    size_t freeMem, totalMem;
    CUDA_SAFE_CALL(cudaMemGetInfo(&freeMem, &totalMem));
    printf("freeMem: %d, totalMem: %d\n", int(freeMem), int(totalMem));
    while (freeMem < 90 * test_num_elements[numTests - 1]) // 90B/item appears to be enough
    {
        numTests--;
        if (numTests <= 0)
        {
            // something has gone very wrong
            printf("Not enough free memory to run any multisplit tests.\n");
            return -1;
        }
    }

    size_t numElements = test_num_elements[numTests - 1];

    if( commandLineArg( cmdVal, argc, (const char**)argv, "n" ) )
    {
        numElements = cmdVal;
        numTests = 1;
    }


    CUDPPResult result = CUDPP_SUCCESS;
    CUDPPHandle theCudpp;
    result = cudppCreate(&theCudpp);

    if(result != CUDPP_SUCCESS)
    {
        printf("Error initializing CUDPP Library.\n");
        retval = numTests;
        return retval;
    }

    retval = multiSplitTest(theCudpp, config, test_num_elements, numTests, numElements,
                            testOptions, quiet);
    result = cudppDestroy(theCudpp);
    
    return retval;
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
