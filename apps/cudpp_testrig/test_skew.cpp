// -------------------------------------------------------------
// CUDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision: $
// $Date: $
// -------------------------------------------------------------
// This source code is distributed under the terms of license.txt in
// the root directory of this source distribution.
// -------------------------------------------------------------

/**
 * @file
 * test_skew.cpp
 *
 * @brief Host testrig routines to exercise cudpp's suffix array functionality.
 */

#include <cstring>
#include <iostream>
#include <cuda_runtime_api.h>
#include <time.h>

#include "cudpp.h"
#include "cudpp_testrig_options.h"
#include "cudpp_testrig_utils.h"
#include "cuda_util.h"
#include "stopwatch.h"
#include "commandline.h"
#include "comparearrays.h"

using namespace cudpp_app;

int suffixArrayTest(int argc, const char **argv, const CUDPPConfiguration &config,
                    const testrigOptions &testOptions)
{
    int retval = 0;

    cudpp_app::StopWatch timer;

    bool quiet = checkCommandLineFlag(argc, (const char**) argv, "quiet");

    bool oneTest = false;
    int numElements;
    if (commandLineArg(numElements, argc, (const char**) argv, "n"))
    {   
        oneTest = true;
    }   

    unsigned int test[] = {'m','m','i','i','s','s','i','i','s','s','i','i','p','p','i','i'};
 
    int numTests = sizeof(test) / sizeof(test[0]);
    numElements = test[numTests-1]; // maximum test size

    if (oneTest)
    {   
        test[0] = numElements;
        numTests = 1;
    }   

    // Initialize CUDPP
    CUDPPHandle plan;
    CUDPPResult result = CUDPP_SUCCESS;
    CUDPPHandle theCudpp;
    result = cudppCreate(&theCudpp);
     if (result != CUDPP_SUCCESS)
    {    
        fprintf(stderr, "Error initializing CUDPP Library\n");
        retval = 1; 
        return retval;
    }    

    result = cudppPlan(theCudpp, &plan, config, numElements, 1, 0);

    if(result != CUDPP_SUCCESS)
    {    
        printf("Error in plan creation\n");
        retval = numTests;
        cudppDestroyPlan(plan);
        cudppDestroy(theCudpp);
        return retval;
    }    


}



int testSuffixArray(int argc, const char **argv,
                    const CUDPPConfiguration *configPtr)
{
    testrigOptions testOptions;
    setOptions(argc, argv, testOptions);

    CUDPPConfiguration config;
    config.algorithm = CUDPP_SA;
    config.options = 0; 

    if (configPtr != NULL)
    {    
        config = *configPtr;
    }    
    else 
    {    
        config.datatype = CUDPP_UINT;
    }    

    return suffixArrayTest(argc, argv, config, testOptions);
}


// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
     

