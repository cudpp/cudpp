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


using namespace cudpp_app;

int stringSortTest(CUDPPHandle theCudpp, CUDPPConfiguration config, size_t *tests, 
                  unsigned int numTests, size_t numElements,
                  testrigOptions testOptions, bool quiet)
{
    int retval = 0;
    
    unsigned int *h_keys, *h_keysSorted, *d_keys;
    unsigned int *h_values, *h_valuesSorted, *d_values;
    unsigned int *string_length;
    unsigned int *stringVals;

    h_keys       = (unsigned int*)malloc(numElements*sizeof(unsigned int));
    h_keysSorted = (unsigned int*)malloc(numElements*sizeof(unsigned int));

    
    h_values       = (unsigned int*)malloc(numElements*sizeof(unsigned int));
    h_valuesSorted = (unsigned int*)malloc(numElements*sizeof(unsigned int));


    unsigned int stringSize = 0;
    string_length = (unsigned int*)malloc(numElements*sizeof(unsigned int));
    //We want to create numElements unique strings with varying size
    //We add its starting address to the end to make it unique 
    //For the sake of simplicity in this test we make all strings 
    //multiples of 4 characters
    for(unsigned int i=0; i < numElements; ++i)                     
    {
        h_address[i] = 2 + rand(maxStringLength);            
	stringSize += h_address[i];
    }
    stringVals = (unsigned int*) malloc(sizeof(unsigned int));
    unsigned int index = 0;
    for(unsigned int i = 0; i < numElements; ++i)
    {
	    for(unsigned int j = 0; j < h_address[i]-1; j++);
	    {
		    unsigned int val = rand(UINT_MAX);
		    stringValues[index++] = val;
		    if(j == 0)
			    h_keys[i] = val;

	    }
	    if( i > 0 )
		    h_values[i] += h_values[i-1];
	    stringValues[index++] = i << 8; // Make sure there is a character with a 0
	    //so the algorithm knows the string has terminated
    }
    

                                                                                                                                       
	
  
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_keys, numElements*sizeof(unsigned int)));
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_values, numElements*sizeof(unsigned int)));
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_stringVals, stringSize*sizeof(unsigned int)));
    CUDA_SAFE_CALL(cudaMemcpy(d_stringVals, stringVals, stringSize*sizeof(unsigned int), cudaMemcpyHostToDevice));

  
    
    CUDPPHandle plan;   
    CUDPPResult result = cudppPlan(theCudpp, &plan, config, numElements, 1, 0);     

	

    if(result != CUDPP_SUCCESS)
    {
        printf("Error in plan creation\n");
        retval = numTests;
        cudppDestroyPlan(plan);
        return retval;
    }

    // run multiple iterations to compute an average sort time
    cudaEvent_t start_event, stop_event;
    CUDA_SAFE_CALL( cudaEventCreate(&start_event) );
    CUDA_SAFE_CALL( cudaEventCreate(&stop_event) );

	
    for (unsigned int k = 0; k < numTests; ++k)
    {
        if(numTests == 1)
            tests[0] = numElements;    

        if (!quiet)
        {
			
            printf("Running a string sort of %ld keys\n", tests[k]);
            fflush(stdout);
        }                                        
            
        float totalTime = 0;

        for (int i = 0; i < testOptions.numIterations; i++)
        {		
			
			
            CUDA_SAFE_CALL(cudaMemcpy(d_keys, h_keys, tests[k] * sizeof(T), cudaMemcpyHostToDevice));
          			
            CUDA_SAFE_CALL( cudaMemcpy(d_values, h_values, tests[k] * sizeof(unsigned int), cudaMemcpyHostToDevice) );

            
            CUDA_SAFE_CALL( cudaEventRecord(start_event, 0) );

			
            cudppStringSort(plan, d_keys, d_values, stringVals, tests[k], stringSize);

            CUDA_SAFE_CALL( cudaEventRecord(stop_event, 0) );
            CUDA_SAFE_CALL( cudaEventSynchronize(stop_event) );

            float time = 0;
            CUDA_SAFE_CALL( cudaEventElapsedTime(&time, start_event, stop_event));
            totalTime += time;
        }
        
        CUDA_CHECK_ERROR("testmergeSort - cudppMergeSort");
		
        // copy results
        CUDA_SAFE_CALL(cudaMemcpy(h_keysSorted, d_keys, tests[k] * sizeof(T), cudaMemcpyDeviceToHost));
       
        CUDA_SAFE_CALL( cudaMemcpy((void*)h_valuesSorted, 
                                   (void*)d_values, 
                                   tests[k] * sizeof(unsigned int), 
                                   cudaMemcpyDeviceToHost) );
       
        retval += VectorSupport<T>::verifySort(h_keysSorted, h_valuesSorted, h_keys, tests[k], 
                                               (config.options & CUDPP_OPTION_BACKWARD) != 0);

	//Verify that the keys make sense
	//TODO: Verify that all strings are in correct order using addresses

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
	//printf("%f %f\n", FLT_MAX, -FLT_MAX);

	
    CUDA_CHECK_ERROR("after mergesort");
	
    result = cudppDestroyPlan(plan);
	

    if (result != CUDPP_SUCCESS)
    {   
        printf("Error destroying CUDPPPlan for Scan\n");
        retval = numTests;
    }

    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);

    cudaFree(d_keys);
    cudaFree(d_vals);
    cudaFree(d_stringVals);
	
    free(h_keys);
    free(h_keysSorted);
    free(h_vals);
    free(h_valsSorted);
    free(stringVals);
    

    return retval;
}

/**
 * testMergeSort tests cudpp's merge sort
 * Possible command line arguments:
 * - --keysonly, tests only a set of keys
 * - --keyval, tests a set of keys with associated values
 * - --n=#, number of elements in sort
 * @param argc Number of arguments on the command line, passed
 * directly from main
 * @param argv Array of arguments on the command line, passed directly
 * from main
 * @param configPtr Configuration for scan, set by caller
 * @return Number of tests that failed regression (0 for all pass)
 * @see cudppSort
*/
int testMergeSort(int argc, const char **argv, const CUDPPConfiguration *configPtr)
{

    int cmdVal;
    int retval = 0;
    
    bool quiet = checkCommandLineFlag(argc, argv, "quiet");        
    testrigOptions testOptions;
    setOptions(argc, argv, testOptions);        
    
    CUDPPConfiguration config;
    config.algorithm = CUDPP_SORT_MERGE;
    config.datatype = CUDPP_UINT;
    config.options = CUDPP_OPTION_KEY_VALUE_PAIRS;
             
    size_t test[] = {39, 128, 256, 512, 513, 1000, 1024, 1025, 32768, 
                     45537, 65536, 131072, 262144, 500001, 524288, 
                     1048577, 1048576, 1048581, 2097152, 4194304, 
                     8388608};
    
    int numTests = sizeof(test)/sizeof(test[0]);
    
    size_t numElements = test[numTests - 1];

    if(configPtr != NULL)
    {
        config = *configPtr;
    }
    else
    {
        config.datatype = getDatatypeFromArgv(argc, argv);

        bool keysOnly = checkCommandLineFlag(argc, argv, "keysonly");     
        bool backward = checkCommandLineFlag(argc, argv, "backward");
        
        config.options = CUDPP_OPTION_KEY_VALUE_PAIRS;
        
	}

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
        
    switch(config.datatype)
    {        
    //case CUDPP_CHAR:
      //  retval = mergeSortTest<char>(theCudpp, config, test, numTests, numElements, testOptions, quiet);    
        //break;
    //case CUDPP_UCHAR:
      //  retval = mergeSortTest<unsigned char>(theCudpp, config, test, numTests, numElements, testOptions, quiet);    
        //break;
    case CUDPP_INT:
        retval = mergeSortTest<int>(theCudpp, config, test, numTests, numElements, testOptions, quiet);
        break;
    case CUDPP_UINT:
        retval = mergeSortTest<unsigned int>(theCudpp, config, test, numTests, numElements, testOptions, quiet);
        break;
    case CUDPP_FLOAT:   
        retval = mergeSortTest<float>(theCudpp, config, test, numTests, numElements, testOptions, quiet);
        break;
    case CUDPP_DOUBLE:   
        retval = mergeSortTest<double>(theCudpp, config, test, numTests, numElements, testOptions, quiet);
        break;
    //case CUDPP_LONGLONG:   
     //   retval = mergeSortTest<long long>(theCudpp, config, test, numTests, numElements, testOptions, quiet);
     //   break;
   // case CUDPP_ULONGLONG:   
     //   retval = mergeSortTest<unsigned long long>(theCudpp, config, test, numTests, numElements, testOptions, quiet);
     //   break;
    default:
        break;
    }

    result = cudppDestroy(theCudpp);

    if (result != CUDPP_SUCCESS)
    {   
        printf("Error shutting down CUDPP Library.\n");
        retval = numTests;
    }
                          
    return retval;
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
