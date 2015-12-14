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
//  for(unsigned int i = 0; i < numElements; i++)
//    input[i] = i; //rand() + rand() << 15;

  for(unsigned int i = 0; i < numElements; i++)
  {
    unsigned int rand1 = rand();
    unsigned int rand2 = (rand() << 15) + rand1;
    unsigned int swap = i + (rand2%(numElements-i));

    unsigned int temp = input[i];
    input[i] = input[swap];
    input[swap] = temp;
  }
}

//===============================================
void cpu_multisplit(uint* key_input, uint* key_output, uint n, uint buckets,
    int bucket_mode) {
  // Performs the mutlisplit with arbitrary bucket distribution on cpu:
  // bucket_mode == 0: equal number of elements per bucket
  // bucket_mode == 1: most significant bits of input represent bucket ID
  // n: number of elements

  uint log_buckets = ceil(log2(buckets));
  uint *bins = new uint[buckets]; // histogram results holder
  uint *scan_bins = new uint[buckets];
  uint *current_idx = new uint[buckets];
  // Computing histograms:
  uint bucketId;

  uint elsPerBucket = (n + buckets - 1)/buckets;
  uint msb_shift = 32 - log_buckets;

  for(unsigned int k = 0; k<buckets; k++)
    bins[k] = 0;

  for(unsigned int i = 0; i<n ; i++)
  {
    bucketId = ((bucket_mode == 0)?(key_input[i]/elsPerBucket):(key_input[i]>>msb_shift));
    if (bucketId >= buckets)
      printf(" %d %d %d\n", bucketId, elsPerBucket, key_input[i]);
    bins[bucketId]++;
  }

  // computing exclusive scan operation on the inputs:
  scan_bins[0] = 0;
  for(unsigned int j = 1; j<buckets; j++)
    scan_bins[j] = scan_bins[j-1] + bins[j-1];

  // Placing items in their new positions:
  for(unsigned int k = 0; k<buckets; k++)
    current_idx[k] = 0;

  for(unsigned int i = 0; i<n; i++)
  {
    bucketId = ((bucket_mode == 0)?(key_input[i]/elsPerBucket):(key_input[i]>>msb_shift));
    key_output[scan_bins[bucketId] + current_idx[bucketId]] = key_input[i];
    current_idx[bucketId]++;
  }
  // releasing memory:
  delete[] bins;
  delete[] scan_bins;
  delete[] current_idx;
}

int verifySplit(unsigned int *keysSorted, unsigned int *valuesSorted,
    unsigned int *keysUnsorted, size_t len)
{
    int retval = 0;

/*    for(unsigned int i=0; i<len-1; ++i)
    {
        bool unordered =
                                 : (keysSorted[i])>(keysSorted[i+1]);
        if (unordered)
        {
            cout << "Unordered key[" << i << "]:" << keysSorted[i]
                 << (reverse ? " < " : " > ") << "key["
                 << i+1 << "]:" << keysSorted[i+1] << endl;
            retval = 1;
            break;
        }
    }*/

  /*  if (valuesSorted)
    {
        for(unsigned int i=0; i<len; ++i)
        {
            if( keysUnsorted[valuesSorted[i]] != keysSorted[i] )
            {
                cout << "Incorrectly sorted value[" << i << "] ("
                     << valuesSorted[i] << ") " << keysUnsorted[valuesSorted[i]]
                     << " != " << keysSorted[i] << endl;
                retval = 1;
                break;
            }
        }
    }*/

    return retval;
}

int multiSplitTest(CUDPPHandle theCudpp, CUDPPConfiguration config,
    size_t *element_tests, size_t *bucket_tests, unsigned int num_element_tests,
    unsigned int num_bucket_tests, size_t maxNumElements,
    testrigOptions testOptions, bool quiet) {
  int retval = 0;
    srand(44);
    unsigned int numBuckets = 32;

    unsigned int* elements = (unsigned int*) malloc(sizeof(unsigned int)*maxNumElements);
    unsigned int* h_result = (unsigned int*) malloc(sizeof(unsigned int)*maxNumElements);

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
        retval = num_element_tests;
        cudppDestroyPlan(plan);
        return retval;
    }

    cudaEvent_t start_event, stop_event;
    CUDA_SAFE_CALL( cudaEventCreate(&start_event) );
    CUDA_SAFE_CALL( cudaEventCreate(&stop_event) );

    for (unsigned int k = 0; k < num_element_tests; ++k)
    {
      for (unsigned int b = 0; b < num_bucket_tests; ++b)
      {

        if(num_element_tests == 1)
            element_tests[0] = maxNumElements;

        if (!quiet)
        {

        printf("Running a multi-split on %ld elements and %ld buckets\n",
            element_tests[k], bucket_tests[b]);
            fflush(stdout);
        }

        float totalTime = 0;

        // an arbitrary initialization
        for(unsigned int i = 0; i<maxNumElements; i++)
          elements[i] = rand();
        randomPermute(elements, maxNumElements);

        CUDA_SAFE_CALL( cudaMemcpy(d_elements, elements,
              element_tests[k] * sizeof(unsigned int),
              cudaMemcpyHostToDevice) );
        CUDA_SAFE_CALL( cudaEventRecord(start_event, 0) );

        cudppMultiSplit(plan, d_elements, element_tests[k], bucket_tests[b]);

        CUDA_SAFE_CALL( cudaEventRecord(stop_event, 0) );
        CUDA_SAFE_CALL( cudaEventSynchronize(stop_event) );

        float time = 0;
        CUDA_SAFE_CALL( cudaEventElapsedTime(&time, start_event,
              stop_event));

        CUDA_CHECK_ERROR("testMultiSplit - cudppMultiSplit");

        // copy results
        CUDA_SAFE_CALL( cudaMemcpy(h_result,
                                   d_elements,
                                   element_tests[k] * sizeof(unsigned int),
                                   cudaMemcpyDeviceToHost));
        // === Sanity check:
        uint count = 0;
        uint *h_output = new uint[element_tests[k]];
        cpu_multisplit(elements, h_output, element_tests[k], bucket_tests[b], 1);
        for(unsigned int i = 0; i < 10; i++)
        {
          unsigned int elems_per_bucket = element_tests[k] / numBuckets;
/*
      printf("cpu val %d bucket %d gpu val %d bucket %d\n", h_output[i],
          h_output[i] / elems_per_bucket, h_result[i],
          h_result[i] / elems_per_bucket);
*/
          if(h_output[i] != h_result[i]){
            count++;
            printf(" ##### index %d, correct = %d, computed = %d\n", i, h_output[i], h_result[i]);
            if (i == 9)
              printf("...\n");
            retval = -1;
          }
        }
        delete h_output;

        if(!quiet)
        {
            printf("test %s\n", (retval == 0) ? "PASSED" : "FAILED");
            printf("Average execution time: %f ms\n", totalTime / testOptions.numIterations);
        }
        else
        {
            printf("\t%10ld\t%0.4f\n", element_tests[k], totalTime / testOptions.numIterations);
        }
      }
    }
    printf("\n");

    CUDA_CHECK_ERROR("after multi-split");

    result = cudppDestroyPlan(plan);

    if (result != CUDPP_SUCCESS)
    {
        printf("Error destroying CUDPPPlan for multi-split\n");
        retval = num_element_tests;
    }

    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);

    cudaFree(d_elements);
    free(elements);
    free(h_result);

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
    config.options = CUDPP_OPTION_KEYS_ONLY;
    if(configPtr != NULL)
    {
        config = *configPtr;
    }

/*
    size_t test_num_elements[] = {128, 256, 512, 513, 1000, 1024, 1025, 32768,
                     45537, 65536, 131072, 262144, 500001, 524288,
                     1048577, 1048576, 1048581, 2097152, 4194304};
*/

    // The last test size should be the largest
    size_t element_tests[] = {2097152, 4194304, 8388608, 16777216};
    size_t bucket_tests[] = {12, 13, 32, 97, 98, 99, 112, 128, 129, 230, 12, 13, 32, 8, 16, 32, 64, 96};

    int num_element_tests = sizeof(element_tests)/sizeof(element_tests[0]);
    int num_bucket_tests = sizeof(bucket_tests)/sizeof(bucket_tests[0]);

    // small GPUs are susceptible to running out of memory,
    // restrict the tests to only those where we have enough
    size_t freeMem, totalMem;
    CUDA_SAFE_CALL(cudaMemGetInfo(&freeMem, &totalMem));
    printf("freeMem: %d, totalMem: %d\n", int(freeMem), int(totalMem));
    while (freeMem < 40 * element_tests[num_element_tests - 1]) // 90B/item appears to be enough
    {
        num_element_tests--;
        if (num_element_tests <= 0)
        {
            // something has gone very wrong
            printf("Not enough free memory to run any multisplit tests.\n");
            return -1;
        }
    }

    size_t numElements = element_tests[num_element_tests - 1];

    if( commandLineArg( cmdVal, argc, (const char**)argv, "n" ) )
    {
        numElements = cmdVal;
        num_element_tests = 1;
    }


    CUDPPResult result = CUDPP_SUCCESS;
    CUDPPHandle theCudpp;
    result = cudppCreate(&theCudpp);

    if(result != CUDPP_SUCCESS)
    {
        printf("Error initializing CUDPP Library.\n");
        retval = num_element_tests;
        return retval;
    }

  retval = multiSplitTest(theCudpp, config, element_tests,
      bucket_tests, num_element_tests, num_bucket_tests, numElements, testOptions, quiet);
    result = cudppDestroy(theCudpp);
    
    return retval;
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
