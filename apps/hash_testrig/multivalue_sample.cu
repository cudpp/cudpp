/*! @file multivalue_sample.cu
 *  @brief This file demonstrates how to use the multi-value hash table.
 */
#include "random_numbers.h"
#include "timers.h"

#include <cutil.h>
#include <hash_table.h>
#include <hash_multivalue.h>
#include <mt19937ar.h>

#include <algorithm>
#include <cstdio>
#include <map>
#include <vector>

void CheckResults(const unsigned  kInputSize,
                  const std::map<unsigned, std::vector<unsigned> > &pairs,
                  const unsigned *sorted_values,
                  const unsigned *query_keys,
                  const uint2    *query_vals) {
  // Check to see if the correct counts and values were placed in the right locations
  // for each query.
  for (unsigned i = 0; i < kInputSize; ++i) {
    std::map<unsigned, std::vector<unsigned> >::const_iterator itr = pairs.find(query_keys[i]);
    if (itr != pairs.end()) {
      // The query key was part of the input.  Confirm that the number of values is right.
      if (query_vals[i].y != itr->second.size()) {
        fprintf(stderr,
                "\t\t\t\t!!! Input query key %10u returned %u values instead of %u.\n",
                query_keys[i],
                query_vals[i].y,
                itr->second.size());
      }

      // Confirm that all of the values can be found.
      std::vector<unsigned> hash_table_values;
      for (unsigned j = 0; j < query_vals[i].y; ++j) {
        hash_table_values.push_back(sorted_values[query_vals[i].x + j]);
      }
      std::sort(hash_table_values.begin(), hash_table_values.end());

      for (unsigned j = 0; j < query_vals[i].y; ++j) {
        if (hash_table_values[j] != itr->second[j]) {
          fprintf(stderr,
                  "\t\t\t\t!!! Values didn't match: %10u != %10u\n",
                  hash_table_values[j],
                  itr->second[j]);
        }
      }
    } else {
      // The query key was not part of the input.  Confirm that there are 0 values.
      if (query_vals[i].y != 0) {
        fprintf(stderr,
                "\t\t\t\t!!! Invalid query key %10u has %u values instead of 0.\n",
                query_keys[i],
                query_vals[i].y);
      }
    }
  }
}                  

//! Run a single test on a multi-value hash table.
int main(int argc, char **argv) {
  init_genrand(time(NULL));

 /// Parse the command line.
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
  uint2    *query_vals = new uint2[kInputSize];

  // Allocate the GPU memory.
  unsigned *d_test_keys = NULL, *d_test_vals = NULL;
  CUDA_SAFE_CALL(cudaMalloc((void**) &d_test_keys, sizeof(unsigned) * kInputSize));
  CUDA_SAFE_CALL(cudaMalloc((void**) &d_test_vals, sizeof(unsigned) * kInputSize));

  uint2 *d_query_vals = NULL;
  CUDA_SAFE_CALL(cudaMalloc((void**) &d_query_vals, sizeof(uint2) * kInputSize));

  const unsigned kMaxIterations = 100;
  for (unsigned iteration = 0; iteration < kMaxIterations; ++iteration) {
    printf("Iteration %u\n", iteration);

    for (unsigned multiplicity = 1; multiplicity <= 2048; multiplicity *= 2) {
      float chance_of_repeating = 1.0 - 1.0/multiplicity;
      printf("\tAverage multiplicity of keys: %u\n", multiplicity);

      // Generate random input keys and query keys.
      GenerateUniqueRandomNumbers(number_pool, pool_size);
      for (unsigned i = 0; i < kInputSize; ++i) {
        input_vals[i] = genrand_int32();
      }
      GenerateMultiples(kInputSize, chance_of_repeating, number_pool);
      GenerateMultiples(kInputSize, chance_of_repeating, number_pool + kInputSize);
      Shuffle(kInputSize, number_pool);
      Shuffle(kInputSize, number_pool + kInputSize);

      // The unique numbers are pre-shuffled by the generator.  Take the first half as the input keys.
      memcpy(input_keys, number_pool, sizeof(unsigned) * kInputSize);

      // Save the original input for checking the results.
      std::map<unsigned, std::vector<unsigned> > pairs;
      for (unsigned i = 0; i < kInputSize; ++i) {
        pairs[input_keys[i]].push_back(input_vals[i]);
      }
      for (std::map<unsigned, std::vector<unsigned> >::iterator itr = pairs.begin();
           itr != pairs.end();
           itr++) {
        std::sort(itr->second.begin(), itr->second.end());
      }

      const float kSpaceUsagesToTest[] = {1.05, 1.15, 1.25, 1.5, 2};
      const unsigned kNumSpaceUsagesToTest = 5;

      for (unsigned i = 0; i < kNumSpaceUsagesToTest; ++i) {
        float space_usage = kSpaceUsagesToTest[i];
        printf("\t\tSpace usage: %f\n", space_usage);

        CUDA_SAFE_CALL(cudaMemcpy(d_test_keys, input_keys, sizeof(unsigned) * kInputSize, cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(d_test_vals, input_vals, sizeof(unsigned) * kInputSize, cudaMemcpyHostToDevice));

       /// ----------------------------------- Create and build the multivalue hash table.
        CudaHT::CuckooHashing::MultivalueHashTable multivalue_table;
        if (multivalue_table.Initialize(kInputSize, space_usage) == false) {
          fprintf(stderr, "!!! Failed to initialize table; aborting.\n");
          exit(1);
        }
  
        TimerType timer = StartTimer();
        multivalue_table.Build(kInputSize, d_test_keys, d_test_vals);
        printf("\t\t\tMultivalue table build: %f ms\n", StopTimer(timer));
       /// --------------------------------------------------------------------------

#ifdef _DEBUG
        unsigned *sorted_values = new unsigned[multivalue_table.get_values_size()];
        CUDA_SAFE_CALL(cudaMemcpy(sorted_values,
                                  multivalue_table.get_all_values(),
                                  sizeof(unsigned) * multivalue_table.get_values_size(),
                                  cudaMemcpyDeviceToHost));
#endif
  
        for (unsigned failure = 0; failure <= 10; ++failure) {
          // Generate a set of queries comprised of keys both from and not from the input.
          float failure_rate = failure / 10.0f;
          GenerateQueries(kInputSize, failure_rate, number_pool, query_keys);
          CUDA_SAFE_CALL(cudaMemcpy(d_test_keys, query_keys, sizeof(unsigned) * kInputSize, cudaMemcpyHostToDevice));
          CUDA_SAFE_CALL(cudaMemset(d_test_vals, 0, sizeof(unsigned) * kInputSize));
  
         /// --------------------------------------------------------- Query the table.
          timer = StartTimer();
          multivalue_table.Retrieve(kInputSize, d_test_keys, d_query_vals);
          printf("\t\t\tMultivalue table retrieve with %3u%% chance of failed queries: %f ms\n", failure * 10, StopTimer(timer));
         /// -------------------------------------------------------------------------- 

         /// Check results.
#ifdef _DEBUG         
          CUDA_SAFE_CALL(cudaMemcpy(query_vals,
                                    d_query_vals,
                                    sizeof(uint2) * kInputSize,
                                    cudaMemcpyDeviceToHost));
          CheckResults(kInputSize, pairs, sorted_values, query_keys, query_vals);
#endif          
        }

#ifdef _DEBUG
        delete [] sorted_values;
#endif        

     /// ---------------------------------------------------------- Free the table.
        multivalue_table.Release();
     /// --------------------------------------------------------------------------
      }
    }
  }

  CUDA_SAFE_CALL(cudaFree(d_test_keys));
  CUDA_SAFE_CALL(cudaFree(d_test_vals));
  CUDA_SAFE_CALL(cudaFree(d_query_vals));
  delete [] number_pool;
  delete [] input_keys;
  delete [] input_vals;
  delete [] query_keys;
  delete [] query_vals;

  return 0;
}
