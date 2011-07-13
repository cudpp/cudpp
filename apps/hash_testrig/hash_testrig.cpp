/*! @file hash_testrig.cu
 *  @brief This file demonstrates how to use the basic hash table.
 */

#include <cudpp_hash.h>
#include <cuda_util.h>
#include <mt19937ar.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdio>
#include <map>

#include "random_numbers.h"
#define CUDPP_APP_COMMON_IMPL
#include "stopwatch.h"
#include "commandline.h"

using namespace cudpp_app;

//! Run a single test on a basic hash table.
int main(int argc, const char **argv) {
    bool quiet = checkCommandLineFlag(argc, argv, "quiet"); 

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        fprintf(stderr, "error: no devices supporting CUDA.\n");
        exit(EXIT_FAILURE);
    }
    int dev = 0;
    commandLineArg(dev, argc, argv, "device");
    if (dev < 0) dev = 0;
    if (dev > deviceCount-1) dev = deviceCount - 1;
    cudaSetDevice(dev);

    cudaDeviceProp prop;
    if (!quiet && cudaGetDeviceProperties(&prop, dev) == 0)
    {
        printf("Using device %d:\n", dev);
        printf("%s; global mem: %dB; compute v%d.%d; clock: %d kHz\n",
               prop.name, (int)prop.totalGlobalMem, (int)prop.major, 
               (int)prop.minor, (int)prop.clockRate);
    }

    int computeVersion = prop.major * 10 + prop.minor;

    int retval = 0;

    init_genrand(time(NULL));

    /// Parse the command line.
    /// @TODO fix to correspond with 
    if (argc != 2) {
        printf("Usage: %s test_size\n", argv[0]);
        return 1;
    }
    const unsigned kInputSize = atoi(argv[1]);

    /// Allocate memory.
    /* We will need a pool of random numbers to create test input and queries from:
     * we need N input keys and N query keys to produce failed queries.
     */
    const unsigned pool_size = kInputSize * 2;
    unsigned *number_pool = new unsigned[pool_size];
  
    // Set aside memory for input keys and values.
    unsigned *input_keys = new unsigned[kInputSize];
    unsigned *input_vals = new unsigned[kInputSize];
    unsigned *query_keys = new unsigned[kInputSize];
    unsigned *query_vals = new unsigned[kInputSize];

    // Allocate the GPU memory.
    unsigned *d_test_keys = NULL, *d_test_vals = NULL;
    CUDA_SAFE_CALL(cudaMalloc((void**) &d_test_keys, sizeof(unsigned) * kInputSize));
    CUDA_SAFE_CALL(cudaMalloc((void**) &d_test_vals, sizeof(unsigned) * kInputSize));

    CUDPPHandle theCudpp;
    CUDPPResult result = cudppCreate(&theCudpp);
    if (result != CUDPP_SUCCESS)
    {
        if (!quiet)
            fprintf(stderr, "Error initializing CUDPP Library.\n");
        retval = 1;
        return retval;
    }

    const unsigned kMaxIterations = 100;
    for (unsigned iteration = 0; iteration < kMaxIterations; ++iteration) {
        printf("Iteration %u\n", iteration);

        // Generate random data.
        GenerateUniqueRandomNumbers(number_pool, pool_size);
        for (unsigned i = 0; i < kInputSize; ++i) {
            input_vals[i] = genrand_int32();
        }

        // The unique numbers are pre-shuffled by the generator.  Take the first half as the input keys.
        memcpy(input_keys, number_pool, sizeof(unsigned) * kInputSize);

        // Save the original input for checking the results.
        std::map<unsigned, unsigned> pairs;
        for (unsigned i = 0; i < kInputSize; ++i)
        {
            pairs[input_keys[i]] = input_vals[i];
        }

        const float kSpaceUsagesToTest[] = {1.05, 1.15, 1.25, 1.5, 2};
        const unsigned kNumSpaceUsagesToTest = 5;

        for (unsigned i = 0; i < kNumSpaceUsagesToTest; ++i)
        {
            float space_usage = kSpaceUsagesToTest[i];
            printf("\tSpace usage: %f\n", space_usage);

            CUDA_SAFE_CALL(cudaMemcpy(d_test_keys, input_keys, sizeof(unsigned) * kInputSize, cudaMemcpyHostToDevice));
            CUDA_SAFE_CALL(cudaMemcpy(d_test_vals, input_vals, sizeof(unsigned) * kInputSize, cudaMemcpyHostToDevice));

            /// ----------------------------------- Create and build the basic hash table.
            CUDPPHashTableConfig config;
            config.type = CUDPP_BASIC_HASH_TABLE;
            config.kInputSize = kInputSize;
            config.space_usage = space_usage;
            CUDPPHandle basic_table_handle;
            cudppHashTable(&basic_table_handle, &config);

            /// CudaHT::CuckooHashing::HashTable basic_table;
            /// basic_table.Initialize(kInputSize, space_usage);
  
            cudpp_app::StopWatch timer;
            timer.reset();
            timer.start();
            /// basic_table.Build(kInputSize, d_test_keys, d_test_vals);
            cudppHashInsert(basic_table_handle, d_test_keys, d_test_vals,
                            kInputSize);
            timer.stop();
            printf("\t\tBasic table build: %f ms\n", timer.getTime());
            /// --------------------------------------------------------------------------
  
            for (unsigned failure = 0; failure <= 10; ++failure)
            {
                // Generate a set of queries comprised of keys both from and not from the input.
                float failure_rate = failure / 10.0f;
                GenerateQueries(kInputSize, failure_rate, number_pool, query_keys);
                CUDA_SAFE_CALL(cudaMemcpy(d_test_keys, query_keys, sizeof(unsigned) * kInputSize, cudaMemcpyHostToDevice));
                CUDA_SAFE_CALL(cudaMemset(d_test_vals, 0, sizeof(unsigned) * kInputSize));
  
                /// --------------------------------------------------------- Query the table.
                timer.reset();
                timer.start();
                /// basic_table.Retrieve(kInputSize, d_test_keys, d_test_vals);
                cudppHashRetrieve(basic_table_handle, d_test_keys, d_test_vals,
                                  kInputSize);
                timer.stop();
                printf("\t\tBasic table retrieve with %3u%% chance of failed queries: %f ms\n", failure * 10, timer.getTime());
                /// --------------------------------------------------------------------------
  
#ifdef _DEBUG
                // Check the results.
                CUDA_SAFE_CALL(cudaMemcpy(query_vals, d_test_vals, sizeof(unsigned) * kInputSize, cudaMemcpyDeviceToHost));
                for (unsigned i = 0; i < kInputSize; ++i) {
                    unsigned actual_value = 0xffffffffu;
                    if (pairs.find(query_keys[i]) != pairs.end()) {
                        actual_value = pairs[query_keys[i]];
                    }
          
                    if (actual_value != query_vals[i]) {
                        printf("\t\t\tError for key %10u: Actual value is %10u, but hash returned %10u.\n", query_keys[i], actual_value, query_vals[i]);
                    }
                }
#endif
            }
  
            /// ---------------------------------------------------------- Free the table.
            cudppDestroyHashTable(basic_table_handle);
            /// basic_table.Release();
            /// --------------------------------------------------------------------------
        }
    }

    CUDA_SAFE_CALL(cudaFree(d_test_keys));
    CUDA_SAFE_CALL(cudaFree(d_test_vals));
    delete [] number_pool;
    delete [] input_keys;
    delete [] input_vals;
    delete [] query_keys;
    delete [] query_vals;

    result = cudppDestroy(theCudpp);
    if (result != CUDPP_SUCCESS)
    {
        if (!quiet)
            printf("Error shutting down CUDPP Library.\n");
    }

    return retval;
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
