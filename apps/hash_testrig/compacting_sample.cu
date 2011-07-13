/*! @file compacting_sample.cu
 *  @brief This file demonstrates how to use the compacting hash table.
 */
#include "random_numbers.h"
#include "timers.h"

#include <cutil.h>
#include <hash_table.h>
#include <hash_compacting.h>
#include <mt19937ar.h>

#include <algorithm>
#include <cstdio>
#include <map>
#include <set>
#include <vector>

void CheckResults(const unsigned            kInputSize,
                  const std::set<unsigned> &pairs,
                  const unsigned           *query_keys,
                  const unsigned           *query_vals) {
  std::map<unsigned, unsigned> id_to_key_map;
  std::map<unsigned, unsigned> key_to_id_map;
  std::vector<unsigned> ids;
  ids.reserve(kInputSize);

  for (unsigned j = 0; j < kInputSize; ++j) {
    // Confirm that all valid queries get a valid ID back, and that bad queries
    // get an invalid one.
    if (pairs.find(query_keys[j]) != pairs.end()) {
      if (query_vals[j] >= kInputSize) {
        fprintf(stderr, "\t\t\t\t!!! Valid query returned bad ID: %10u %10u\n", query_keys[j], query_vals[j]);
      }
    } else {
      if (query_vals[j] < kInputSize) {
        fprintf(stderr, "\t\t\t\t!!! Invalid query returned good ID: %10u %10u\n", query_keys[j], query_vals[j]);
      }
    }

    // Track which unique IDs were returned.
    if (query_vals[j] != CudaHT::CuckooHashing::kNotFound) {
      ids.push_back(query_vals[j]);
    }

    // Track which keys mapped to which unique IDs.
    if (pairs.find(query_keys[j]) != pairs.end()) {
      // Make sure all copies of a key get the same ID back.
      if (key_to_id_map.find(query_keys[j]) != key_to_id_map.end()) {
        if (key_to_id_map[query_keys[j]] != query_vals[j]) {
          fprintf(stderr,
                  "\t\t\t\t!!! Key %10u had two IDs: %10u %10u\n",
                  query_keys[j],
                  key_to_id_map[query_keys[j]],
                  query_vals[j]);
        }
      }

      // Make sure all copies of the same ID have the same key.
      if (id_to_key_map.find(query_vals[j]) != id_to_key_map.end()) {
        if (id_to_key_map[query_vals[j]] != query_keys[j]) {
          fprintf(stderr,
                  "\t\t\t\t!!! ID %10u had two keys: %10u %10u\n",
                  query_vals[j],
                  id_to_key_map[query_vals[j]],
                  query_keys[j]);
        }
      }

      key_to_id_map[query_keys[j]] = query_vals[j];
      id_to_key_map[query_vals[j]] = query_keys[j];
    }
  }

  std::sort(ids.begin(), ids.end());
  if (ids.back() >= pairs.size()) {
    fprintf(stderr, "\t\t\t\t!!! Biggest ID >= number of input items\n");
  }

  if (key_to_id_map.size() != id_to_key_map.size()) {
    fprintf(stderr, "\t\t\t\t!!! Number of unique IDs doesn't match the number of input items in the query set\n");
  }

  for (std::map<unsigned, unsigned>::iterator itr = key_to_id_map.begin();
       itr != key_to_id_map.end(); ++itr) {
    unsigned current_key    = itr->first;
    unsigned expected_value = itr->second;
    if (id_to_key_map[expected_value] != current_key) {
      fprintf(stderr,
              "\t\t\t\t!!! Translation mismatch: %u has ID %u, but ID is mapped to %u\n",
              current_key, expected_value, id_to_key_map[expected_value]);
    }
  }
}

//! Run a single test on a basic hash table.
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
  unsigned *query_keys = new unsigned[kInputSize];
  unsigned *query_vals = new unsigned[kInputSize];

  // Allocate the GPU memory.
  unsigned *d_test_keys = NULL, *d_test_vals = NULL;
  CUDA_SAFE_CALL(cudaMalloc((void**) &d_test_keys, sizeof(unsigned) * kInputSize));
  CUDA_SAFE_CALL(cudaMalloc((void**) &d_test_vals, sizeof(unsigned) * kInputSize));

  const unsigned kMaxIterations = 100;
  for (unsigned iteration = 0; iteration < kMaxIterations; ++iteration) {
    printf("Iteration %u\n", iteration);

    for (unsigned multiplicity = 1; multiplicity <= 2048; multiplicity *= 2) {
      float chance_of_repeating = 1.0 - 1.0/multiplicity;
      printf("\tAverage multiplicity of keys: %u\n", multiplicity);

      // Generate random input keys and query keys.
      GenerateUniqueRandomNumbers(number_pool, pool_size);
      GenerateMultiples(kInputSize, chance_of_repeating, number_pool);
      GenerateMultiples(kInputSize, chance_of_repeating, number_pool + kInputSize);
      Shuffle(kInputSize, number_pool);
      Shuffle(kInputSize, number_pool + kInputSize);

      // The unique numbers are pre-shuffled by the generator.  Take the first half as the input keys.
      memcpy(input_keys, number_pool, sizeof(unsigned) * kInputSize);

      // Save the original input for checking the results.
#ifdef _DEBUG      
      std::set<unsigned> pairs;
      for (unsigned i = 0; i < kInputSize; ++i) {
        pairs.insert(input_keys[i]);
      }
#endif      

      const float kSpaceUsagesToTest[] = {1.05, 1.15, 1.25, 1.5, 2};
      const unsigned kNumSpaceUsagesToTest = 5;

      for (unsigned i = 0; i < kNumSpaceUsagesToTest; ++i) {
        float space_usage = kSpaceUsagesToTest[i];
        printf("\t\tSpace usage: %f\n", space_usage);

        CUDA_SAFE_CALL(cudaMemcpy(d_test_keys, input_keys, sizeof(unsigned) * kInputSize, cudaMemcpyHostToDevice));

       /// ----------------------------------- Create and build the basic hash table.
        CudaHT::CuckooHashing::CompactingHashTable table;
        if (table.Initialize(kInputSize, space_usage) == false) {
          fprintf(stderr, "!!! Failed to initialize table; aborting.\n");
          exit(1);
        }
  
        TimerType timer = StartTimer();
        table.Build(kInputSize, d_test_keys, NULL);
        printf("\t\t\tCompacting table build: %f ms\n", StopTimer(timer));
       /// --------------------------------------------------------------------------

        for (unsigned failure = 0; failure <= 10; ++failure) {
          // Generate a set of queries comprised of keys both from and not from the input.
          float failure_rate = failure / 10.0f;
          GenerateQueries(kInputSize, failure_rate, number_pool, query_keys);
          CUDA_SAFE_CALL(cudaMemcpy(d_test_keys, query_keys, sizeof(unsigned) * kInputSize, cudaMemcpyHostToDevice));
          CUDA_SAFE_CALL(cudaMemset(d_test_vals, 0, sizeof(unsigned) * kInputSize));
  
         /// --------------------------------------------------------- Query the table.
          timer = StartTimer();
          table.Retrieve(kInputSize, d_test_keys, d_test_vals);
          printf("\t\t\tCompacting table retrieve with %3u%% chance of failed queries: %f ms\n", failure * 10, StopTimer(timer));
         /// --------------------------------------------------------------------------

#ifdef _DEBUG         
         /// Check the results.
          CUDA_SAFE_CALL(cudaMemcpy(query_vals, d_test_vals, sizeof(unsigned) * kInputSize, cudaMemcpyDeviceToHost));
          CheckResults(kInputSize, pairs, query_keys, query_vals);
#endif
        }

     /// ---------------------------------------------------------- Free the table.
        table.Release();
     /// --------------------------------------------------------------------------
      }
    }
  }

  CUDA_SAFE_CALL(cudaFree(d_test_keys));
  CUDA_SAFE_CALL(cudaFree(d_test_vals));
  delete [] number_pool;
  delete [] input_keys;
  delete [] query_keys;
  delete [] query_vals;

  return 0;
}
