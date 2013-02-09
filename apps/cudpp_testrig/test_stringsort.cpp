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

//TODO: right now just checking the keys, need to check in case of ties (and values)
int verifyStringSort(unsigned int* keysSorted, unsigned int *valuesSorted, unsigned int* keysUnsorted,
					  unsigned int* stringVals, size_t numElements, int stringSize)
{
	int retval = 0;
	
	for(unsigned int i = 0; i < numElements-1; ++i)
	{
		bool unordered = (keysSorted[i])>(keysSorted[i+1]);
        if (unordered)
        {
            cout << "Unordered key[" << i << "]:" << keysSorted[i] 
                 << (" > ") << "key["
                 << i+1 << "]:" << keysSorted[i+1] << endl;
            retval = 1;
            break;
        } 
	}	
	return retval;
}

int stringSortTest(CUDPPHandle theCudpp, CUDPPConfiguration config, size_t *tests, 
                  unsigned int numTests, size_t numElements,
                  testrigOptions testOptions, bool quiet)
{
    int retval = 0;
    srand(1);
    unsigned int *h_keys, *h_keysSorted, *d_keys;
    unsigned int *h_values, *h_valuesSorted, *h_valSend, *d_values;
    unsigned int *string_length;
	unsigned int *d_stringVals;
    unsigned int *stringVals;
	config.algorithm = CUDPP_SORT_STRING;
	config.datatype = CUDPP_UINT;
	config.options = CUDPP_OPTION_FORWARD;

	unsigned int maxStringLength = 4;
    h_keys       = (unsigned int*)malloc(numElements*sizeof(unsigned int));
    h_keysSorted = (unsigned int*)malloc(numElements*sizeof(unsigned int));

    
    h_values       = (unsigned int*)malloc(numElements*sizeof(unsigned int));
	h_valSend = (unsigned int*)malloc(numElements*sizeof(unsigned int));
    h_valuesSorted = (unsigned int*)malloc(numElements*sizeof(unsigned int));


    unsigned int stringSize = 0;
    string_length = (unsigned int*)malloc(numElements*sizeof(unsigned int));
    //We want to create numElements unique strings with varying size
    //We add its starting address to the end to make it unique 
    //For the sake of simplicity in this test we make all strings 
    //multiples of 4 characters
	//printf("making strings\n");
    for(unsigned int i=0; i < numElements; ++i)                     
    {				
        h_values[i] = 2 + rand()%maxStringLength;    
		h_valSend[i] = i == 0 ? 0 : h_values[i-1];
	    stringSize += h_values[i];		
    }
    stringVals = (unsigned int*) malloc(sizeof(unsigned int)*stringSize);
    unsigned int index = 0;
	printf("%u elements and %d characters\n", numElements, 4*stringSize);
	unsigned int temp = 0;
	char c1, c2, c3, c4;
    for(unsigned int i = 0; i < numElements; ++i)
    {
	    for(unsigned int j = 0; j < h_values[i]-1; j++)
	    {
		    c1 = (rand()%255)+1;
			c2 = (rand()%255)+1;
			c3 = (rand()%255)+1;
			c4 = (rand()%255)+1;
			unsigned int val = (c1 << 24) + (c2 << 16) + (c3 << 8) + c4;
		    stringVals[index++] = val;
		    if(j == 0)			
				h_keys[i] = val;				
			

	    }
		

		c1 = rand()%255+1;
		c2 = rand()%255+1;
		c3 = rand()%255+1;		
		unsigned int val = (c1 << 24) + (c2 << 16) + (c3 << 8);
		
	    if( i > 0 )					
		    h_valSend[i] += h_valSend[i-1];					
		

	    stringVals[index++] = val; // Make sure there is a character with a 0
	    //so the algorithm knows the string has terminated
    }
    

    //printf("made strings %d long %d stringSize\n", index, stringSize);
	
	
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
			
			
            CUDA_SAFE_CALL(cudaMemcpy(d_keys, h_keys, tests[k] * sizeof(unsigned int), cudaMemcpyHostToDevice));
          			
            CUDA_SAFE_CALL( cudaMemcpy(d_values, h_valSend, tests[k] * sizeof(unsigned int), cudaMemcpyHostToDevice) );

            
            CUDA_SAFE_CALL( cudaEventRecord(start_event, 0) );

			
            cudppStringSort(plan, d_keys, d_values, d_stringVals, tests[k], stringSize);

            CUDA_SAFE_CALL( cudaEventRecord(stop_event, 0) );
            CUDA_SAFE_CALL( cudaEventSynchronize(stop_event) );

            float time = 0;
            CUDA_SAFE_CALL( cudaEventElapsedTime(&time, start_event, stop_event));
            totalTime += time;
        }
        
        CUDA_CHECK_ERROR("teststringSort - cudppStringSort");
		
        // copy results
        CUDA_SAFE_CALL(cudaMemcpy(h_keysSorted, d_keys, tests[k] * sizeof(unsigned int), cudaMemcpyDeviceToHost));
       
        CUDA_SAFE_CALL( cudaMemcpy((void*)h_valuesSorted, 
                                   (void*)d_values, 
                                   tests[k] * sizeof(unsigned int), 
                                   cudaMemcpyDeviceToHost) );
       
		retval += verifyStringSort(h_keysSorted, h_valuesSorted, h_keys,
					  stringVals, tests[k], stringSize);
        
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

	
    CUDA_CHECK_ERROR("after stringsort");
	
    result = cudppDestroyPlan(plan);
	

    if (result != CUDPP_SUCCESS)
    {   
        printf("Error destroying CUDPPPlan for StringSort\n");
        retval = numTests;
    }

    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);

    cudaFree(d_keys);
    cudaFree(d_values);
    cudaFree(d_stringVals);
	
    free(h_keys);
    free(h_keysSorted);
    free(h_values);
    free(h_valuesSorted);
    free(stringVals);
    

    return retval;
}


/**
 * testStringSort tests cudpp's merge sort
 * Possible command line arguments: 
 * - --n=#, number of elements in sort
 * @param argc Number of arguments on the command line, passed
 * directly from main
 * @param argv Array of arguments on the command line, passed directly
 * from main
 * @param configPtr Configuration for scan, set by caller
 * @return Number of tests that failed regression (0 for all pass)
 * @see cudppSort
*/
int testStringSort(int argc, const char **argv, const CUDPPConfiguration *configPtr)
{
	
	int cmdVal;
    int retval = 0;
    
    bool quiet = checkCommandLineFlag(argc, argv, "quiet");        
    testrigOptions testOptions;
    setOptions(argc, argv, testOptions);        
    
    CUDPPConfiguration config;
    config.algorithm = CUDPP_SORT_STRING;
    config.datatype = CUDPP_UINT;

             
    size_t test[] = {39, 128, 256, 512, 513, 1000, 1024, 1025, 32768, 
                     45537, 65536, 131072, 262144, 500001, 524288, 
                     1048577, 1048576, 1048581, 2097152, 4194304};
    
    int numTests = sizeof(test)/sizeof(test[0]);
    
    size_t numElements = test[numTests - 1];
	
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

	retval = stringSortTest(theCudpp, config, test, numTests, numElements, testOptions, quiet);
	result = cudppDestroy(theCudpp);
    return retval;

}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
