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
 * test_compress.cpp
 *
 * @brief Host testrig routines to exercise cudpp's compress functionality.
 */

#include <cstring>
#include <iostream>
#include <cuda_runtime_api.h>

#include "cudpp.h"
#include "cudpp_testrig_options.h"
#include "cudpp_testrig_utils.h"
#include "cuda_util.h"
#include "commandline.h"

using namespace cudpp_app;

void computeMtfGold( unsigned char* out, const unsigned char* idata, const unsigned int len)
{
    unsigned char* list = new unsigned char[256];
    unsigned int j = 0;
    
    // init mtf list
    for(unsigned int i=0; i<256; i++)
        list[i] = i;

    for (unsigned int i = 0; i < len; i++)
    {
        // Find the character in the list of characters
        for (j = 0; j < 256; j++)
        {
            if (list[j] == idata[i])
            {
                // Found the character
                out[i] = j;
                break;
            }
        }

        // Move the current character to the front of the list
        for (; j > 0; j--)
        {
            list[j] = list[j - 1];
        }
        list[0] = idata[i];
    }

    delete [] list;
}

int mtfTest(int argc, const char **argv, const CUDPPConfiguration &config,
               const testrigOptions &testOptions)
{
    int retval = 0;
    int numElements = 1048576; // test size

    bool quiet = checkCommandLineFlag(argc, argv, "quiet");
    int numTests = 1;
    bool oneTest = true;

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

    unsigned int memSize = sizeof(unsigned char) * numElements;
    
    // allocate host memory to store the input data
    unsigned char* i_data = new unsigned char[numElements];

    // initialize the input data on the host
    float range = (float)(sizeof(unsigned char)*8);
        
    VectorSupport<unsigned char>::fillVector(i_data, numElements, range);
    
    unsigned char* reference = new unsigned char[numElements];

    // allocate device memory input and output arrays
    unsigned char* d_idata     = (unsigned char *) NULL;
    unsigned char* d_odata     = (unsigned char *) NULL;

    CUDA_SAFE_CALL( cudaMalloc( (void **) &d_idata, memSize));
    CUDA_SAFE_CALL( cudaMalloc( (void **) &d_odata, memSize));

    CUDA_SAFE_CALL( cudaMemcpy(d_idata, i_data, memSize, cudaMemcpyHostToDevice) );

    char dt[10];
    strcpy(dt, "uchar");

    if (!quiet)
    {
        printf("Running a mtf of %d %s elements\n", 
            numElements, dt);
        fflush(stdout);
    }

    computeMtfGold( reference, i_data, numElements);

    // Run the reduction
    // run once to avoid timing startup overhead.
    cudppMoveToFrontTransform(plan, d_idata, d_odata, (unsigned int)numElements);

    // copy result from device to host
    unsigned char* o_data = new unsigned char[numElements];
    CUDA_SAFE_CALL(cudaMemcpy( o_data, d_odata, memSize, cudaMemcpyDeviceToHost));


    // check results
    bool error = false;
    for(int i=0; i<numElements; i++)
    {
        if(o_data[i] != reference[i])
        {
            error = true;
            break;
        }
    }

    printf("test %s\n", (error) ? "FAILED" : "PASSED");

    result = cudppDestroyPlan(plan);

    if (result != CUDPP_SUCCESS)
    {   
        printf("Error destroying CUDPPPlan for MTF\n");
        retval = numTests;
    }

    result = cudppDestroy(theCudpp);

    if (result != CUDPP_SUCCESS)
    {   
        printf("Error shutting down CUDPP Library.\n");
        retval = numTests;
    }

    delete [] reference;
    delete [] o_data;

    return retval;
}

int testMtf(int argc, const char **argv, const CUDPPConfiguration *configPtr)
{
    testrigOptions testOptions;
    setOptions(argc, argv, testOptions);

    CUDPPConfiguration config;
    config.algorithm = CUDPP_MTF;
    config.options = 0;

    if (configPtr != NULL)
    {
        config = *configPtr;
    }
    else
    {
        config.datatype = CUDPP_UCHAR;
    }

    return mtfTest(argc, argv, config, testOptions);
}
