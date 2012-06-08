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
unsigned int blockSize;
unsigned char *block;

#define Wrap(value, limit)      (((value) < (limit)) ? (value) : ((value) - (limit)))

int ComparePresorted(const void *s1, const void *s2)
{
    int offset1, offset2;
    int i;
    int result;

    offset1 = *((int *)s1);
    offset2 = *((int *)s2);

    /***********************************************************************
    * Compare 1 character at a time until there's difference or the end of
    * the block is reached.  Since we're only sorting strings that already
    * match at the first two characters, start with the third character.
    ***********************************************************************/
    for(i = 2; i < (int)blockSize; i++)
    {
        result = (int)block[Wrap((offset1 + i), (int)blockSize)] -
            (int)block[Wrap((offset2 + i), (int)blockSize)];

        if (result != 0)
        {
            return result;
        }
    }

    /* strings are identical */
    return 0;
}

void computeBwtGold(unsigned char *block_out, int &s0Idx, unsigned int numElements)
{
    unsigned int i, k;
    int j;
    unsigned int *rotationIdx = new unsigned int[numElements];
    unsigned int *v = new unsigned int[numElements];
    unsigned int *counters = new unsigned int[256];
    unsigned int *offsetTable = new unsigned int[256];
    blockSize = numElements;

    /*******************************************************************
    * Sort the rotated strings in the block.  A radix sort is performed
    * on the first two characters of all the rotated strings (2nd
    * character then 1st).  All rotated strings with matching initial
    * characters are then quicksorted. - Q4..Q7
    *******************************************************************/

    /*** radix sort on second character in rotation ***/

    /* count number of characters for radix sort */
    memset(counters, 0, 256 * sizeof(int));
    for (i = 0; i < blockSize; i++)
    {
        counters[block[i]]++;
    }

    offsetTable[0] = 0;

    for(i = 1; i < 256; i++)
    {
        /* determine number of values before those sorted under i */
        offsetTable[i] = offsetTable[i - 1] + counters[i - 1];
    }

    /* sort on 2nd character */
    for (i = 0; i < blockSize - 1; i++)
    {
        j = block[i + 1];
        v[offsetTable[j]] = i;
        offsetTable[j] = offsetTable[j] + 1;
    }

    /* handle wrap around for string starting at end of block */
    j = block[0];
    v[offsetTable[j]] = i;
    offsetTable[0] = 0;

    /*** radix sort on first character in rotation ***/

    for(i = 1; i < 256; i++)
    {
        /* determine number of values before those sorted under i */
        offsetTable[i] = offsetTable[i - 1] + counters[i - 1];
    }

    for (i = 0; i < blockSize; i++)
    {
        j = v[i];
        j = block[j];
        rotationIdx[offsetTable[j]] = v[i];
        offsetTable[j] = offsetTable[j] + 1;
    }

    /*******************************************************************
    * now rotationIdx contains the sort order of all strings sorted
    * by their first 2 characters.  Use qsort to sort the strings
    * that have their first two characters matching.
    *******************************************************************/
    for (i = 0, k = 0; (i <= UCHAR_MAX) && (k < (blockSize - 1)); i++)
    {
        for (j = 0; (j <= UCHAR_MAX) && (k < (blockSize - 1)); j++)
        {
            unsigned int first = k;

            /* count strings starting with ij */
            while ((i == block[rotationIdx[k]]) &&
                (j == block[Wrap(rotationIdx[k] + 1,  blockSize)]))
            {
                k++;

                if (k == blockSize)
                {
                    /* we've searched the whole block */
                    break;
                }
            }

            if (k - first > 1)
            {
                /* there are at least 2 strings staring with ij, sort them */
                qsort(&rotationIdx[first], k - first, sizeof(int),
                    ComparePresorted);
            }
        }
    }

    /* find last characters of rotations (L) - C2 */
    s0Idx = 0;
    for (i = 0; i < blockSize; i++)
    {
        if (rotationIdx[i] != 0)
        {
            block_out[i] = block[rotationIdx[i] - 1];
        }
        else
        {
            /* unrotated string 1st character is end of string */
            s0Idx = i;
            block_out[i] = block[blockSize - 1];
        }
    }

    delete [] rotationIdx;
    delete [] v;
    delete [] counters;
    delete [] offsetTable;
}

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
    CUDA_SAFE_CALL( cudaMemset(d_odata, 0, memSize) );

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
            retval = 1;
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
    delete [] i_data;
    cudaFree(d_odata);
    cudaFree(d_idata);
    return retval;
}

int bwtTest(int argc, const char **argv, const CUDPPConfiguration &config,
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
        
    //VectorSupport<unsigned char>::fillVector(i_data, numElements, range);
    srand(95123);
    for(int j = 0; j < numElements; j++)
    {
        i_data[j] = (unsigned char)(rand()%245+1);
    }
    
    unsigned char* reference = new unsigned char[numElements];
    int ref_index;

    // allocate device memory input and output arrays
    unsigned char* d_idata      = (unsigned char *) NULL;
    unsigned char* d_odata      = (unsigned char *) NULL;
    int* d_oindex               = (int *) NULL;

    CUDA_SAFE_CALL( cudaMalloc( (void **) &d_idata, memSize));
    CUDA_SAFE_CALL( cudaMalloc( (void **) &d_odata, memSize));
    CUDA_SAFE_CALL( cudaMalloc( (void **) &d_oindex, sizeof(int)));

    CUDA_SAFE_CALL( cudaMemcpy(d_idata, i_data, memSize, cudaMemcpyHostToDevice) );
    CUDA_SAFE_CALL( cudaMemset(d_odata, 0, memSize) );

    char dt[10];
    strcpy(dt, "uchar");

    if (!quiet)
    {
        printf("Running a bwt of %d %s elements\n", 
            numElements, dt);
        fflush(stdout);
    }

    block = i_data;
    computeBwtGold( reference, ref_index, numElements);

    // Run the reduction
    // run once to avoid timing startup overhead.
    cudppBurrowsWheelerTransform(plan, d_idata, d_odata, d_oindex, (unsigned int)numElements);

    // copy result from device to host
    unsigned char* o_data = new unsigned char[numElements];
    int o_index;
    CUDA_SAFE_CALL(cudaMemcpy( o_data, d_odata, memSize, cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy( &o_index, d_oindex, sizeof(int), cudaMemcpyDeviceToHost));


    // check results
    bool error = false;
    for(int i=0; i<numElements; i++)
    {
        if(o_data[i] != reference[i])
        {
            error = true;
            retval = 1;
            break;
        }
    }
    if(o_index != ref_index) {
        error = true;
        retval = 1;
    }

    printf("test %s\n", (error) ? "FAILED" : "PASSED");

    result = cudppDestroyPlan(plan);

    if (result != CUDPP_SUCCESS)
    {   
        printf("Error destroying CUDPPPlan for BWt\n");
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
    delete [] i_data;
    cudaFree(d_odata);
    cudaFree(d_idata);
    cudaFree(d_oindex);
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


int testBwt(int argc, const char **argv, const CUDPPConfiguration *configPtr)
{
    testrigOptions testOptions;
    setOptions(argc, argv, testOptions);

    CUDPPConfiguration config;
    config.algorithm = CUDPP_BWT;
    config.options = 0;

    if (configPtr != NULL)
    {
        config = *configPtr;
    }
    else
    {
        config.datatype = CUDPP_UCHAR;
    }

    return bwtTest(argc, argv, config, testOptions);
}
