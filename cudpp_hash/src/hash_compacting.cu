#ifndef CUDAHT__CUCKOO__SRC__LIBRARY__HASH_COMPACTING__CUH
#define CUDAHT__CUCKOO__SRC__LIBRARY__HASH_COMPACTING__CUH

#include "debugging.cuh"
#include "hash_compacting.h"
#include "hash_functions.h"
#include "hash_table.cuh"

#include <cudpp.h>
#include <cutil.h>

#include <set>

namespace CudaHT {
namespace CuckooHashing {

/* --------------------------------------------------------------------------
   Retrieval functions.
   -------------------------------------------------------------------------- */
//! Answers a single query from a compacting hash table.
/*! @ingroup PublicInterface
 *  @param[in]  key                   Query key
 *  @param[in]  table_size            Size of the hash table
 *  @param[in]  table                 The contents of the hash table
 *  @param[in]  constants             The hash functions used to build the table
 *  @param[in]  stash_constants       Constants used by the stash hash function
 *  @param[in]  stash_count           Number of items contained in the stash
 *  @param[out] num_probes_required   Debug only: The number of probes required to resolve the query.
 *
 *  @returns The ID of the query key is returned if the key exists in the table.  Otherwise, \ref kNotFound will be returned.
 */
template <unsigned kNumHashFunctions> __device__
unsigned retrieve_compacting(const unsigned                      query_key,
                             const unsigned                      table_size,
                             const Entry                        *table,
                             const Functions<kNumHashFunctions>  constants,
                             const uint2                         stash_constants,
                             const unsigned                      stash_count,
                                   unsigned                     *num_probes_required = NULL)
{
  // Identify all of the locations that the key can be located in.
  unsigned locations[kNumHashFunctions];
  KeyLocations(constants, table_size, query_key, locations);

  // Check each location until the key is found.
  // Short-circuiting is disabled because the duplicate removal step breaks it.
  unsigned num_probes = 1;
  Entry    entry      = table[locations[0]];

  #pragma unroll
  for (unsigned i = 1; i < kNumHashFunctions; ++i) {
    if (get_key(entry) != query_key) {
      num_probes++;
      entry = table[locations[i]];
    }
  }

  // Check the stash.
  if (stash_count && get_key(entry) != query_key) {
    num_probes++;
    const Entry *stash = table + table_size;
    unsigned slot = stash_hash_function(stash_constants, query_key);
    entry = stash[slot];
  }

#ifdef TRACK_ITERATIONS
  if (num_probes_required) {
    *num_probes_required = num_probes;
  }
#endif

  if (get_key(entry) == query_key) {
    return get_value(entry);
  } else {
    return kNotFound;
  }
}


//! Returns the unique identifier for every query key.  Each thread manages a single query.
/*! @param[in]  n_queries             Number of query keys
 *  @param[in]  keys_in               Query keys
 *  @param[in]  table_size            Size of the hash table
 *  @param[in]  table                 The contents of the hash table
 *  @param[in]  constants             The hash functions used to build the table
 *  @param[in]  stash_constants       Constants used by the stash hash function
 *  @param[in]  stash_count           Number of items contained in the stash
 *  @param[out] values_out            The unique identifiers for each query key
 *  @param[out] num_probes_required   Debug only: The number of probes required to resolve the query.
 *
 *  The ID of the query key is written out if the key exists in the table.
 *  Otherwise, \ref kNotFound will be.
 */
template <unsigned kNumHashFunctions> __global__
void hash_retrieve_compacting(const unsigned                      n_queries,
                              const unsigned                     *keys_in,
                              const unsigned                      table_size,
                              const Entry                        *table,
                              const Functions<kNumHashFunctions>  constants,
                              const uint2                         stash_constants,
                              const unsigned                      stash_count,
                                    unsigned                     *values_out,
                                    unsigned                     *num_probes_required = NULL)
{
  // Get the key.
  unsigned thread_index = threadIdx.x +
                          blockIdx.x * blockDim.x +
                          blockIdx.y * blockDim.x * gridDim.x;
  if (thread_index >= n_queries)
    return;
  unsigned key = keys_in[thread_index];

  values_out[thread_index] = retrieve_compacting<kNumHashFunctions>
                                                (key,
                                                 table_size,
                                                 table,
                                                 constants,
                                                 stash_constants,
                                                 stash_count,
                                                 (num_probes_required ? num_probes_required + thread_index : NULL));
}	


/*! @name Internal
 *  @{
 */
//! Builds a compacting hash table.
template <unsigned kNumHashFunctions>
__global__
void hash_build_compacting(const int                           n,
                           const unsigned                     *keys,
                           const unsigned                      table_size,
                           const Functions<kNumHashFunctions>  constants,
                           const uint2                         stash_constants,
                           const unsigned                      max_iteration_attempts,
                                 unsigned                     *table,
                                 unsigned                     *stash_count,
                                 unsigned                     *failures)
{	
  // Check if this thread has an item and if any previous threads failed.
  unsigned int thread_index = threadIdx.x +
                              blockIdx.x * blockDim.x +
                              blockIdx.y * blockDim.x * gridDim.x;
  if (thread_index >= n || *failures)
    return;

  // Read the key that this thread should insert.  It always uses its first slot.
  unsigned key      = keys[thread_index];
  unsigned location = hash_function(constants, 0, key) % table_size;

  // Keep inserting until an empty slot is found, a copy was found,
  // or the eviction chain grows too large.
  unsigned old_key = kKeyEmpty;
  for (int its = 1; its < max_iteration_attempts; its++) {
    old_key = key;

    // Insert the new entry.
    key = atomicExch(&table[location], key);

    // If no unique key was evicted, we're done.
    if (key == kKeyEmpty || key == old_key)
      return;

    location = determine_next_location(constants, table_size, key, location);
  };

  // Shove it into the stash.
  if (key != kKeyEmpty) {
    unsigned slot = stash_hash_function(stash_constants, key);
    unsigned *stash = table + table_size;
    unsigned  replaced_key = atomicExch(stash + slot, key);
    if (replaced_key == kKeyEmpty || replaced_key == key) {
      atomicAdd(stash_count, 1);
      return;
    }
  }

  // The eviction chain grew too large.  Report failure.
#ifdef COUNT_UNINSERTED
  atomicAdd(failures, 1);
#else
  *failures = 1;
#endif
}	


//! Removes all key duplicates from a compacting hash table.
/*! The unspecialized version is significantly slower than the explicitly specialized ones.
 */
template <unsigned kNumHashFunctions> __global__
void hash_remove_duplicates(const unsigned                      table_size,
                            const unsigned                      total_table_size,
                            const Functions<kNumHashFunctions>  constants,
                            const uint2                         stash_constants,
                                  unsigned                     *keys,
                                  unsigned                     *is_unique) {
  // Read out the key that may be duplicated.
  unsigned int thread_index = threadIdx.x +
                              blockIdx.x * blockDim.x +
                              blockIdx.y * blockDim.x * gridDim.x;
  if (thread_index >= total_table_size)
    return;
  unsigned key = keys[thread_index];

  // Determine all the locations that the key could be in.
  unsigned first_location = table_size + stash_hash_function(stash_constants, key);
  #pragma unroll
  for (int i = kNumHashFunctions-1; i >= 0; --i) {
    unsigned location = hash_function(constants, i, key) % table_size;
    first_location = (keys[location] == key ? location : first_location);
  }

  // If this thread got a later copy of the key, remove this thread's copy from the table.
  if (first_location != thread_index || key == kKeyEmpty) {
    keys[thread_index] = kKeyEmpty;
    is_unique[thread_index] = 0;
  } else {
    is_unique[thread_index] = 1;
  }
}                                  
/// @}

//! @name Explicit template specializations
/// @{
#if 1
template <> __global__
void hash_remove_duplicates<2>(const unsigned      table_size,
                               const unsigned      total_table_size,
                               const Functions<2>  constants,
                               const uint2         stash_constants,
                                     unsigned     *keys,
                                     unsigned     *is_unique) {	
  // Read out the key that may be duplicated.
  unsigned int thread_index = threadIdx.x +
                              blockIdx.x * blockDim.x +
                              blockIdx.y * blockDim.x * gridDim.x;
  if (thread_index >= total_table_size)
    return;
  unsigned key = keys[thread_index];

  // Determine all the locations that the key could be in.
  unsigned location_0 = hash_function(constants, 0, key) % table_size;
  unsigned location_1 = hash_function(constants, 1, key) % table_size;
  unsigned stash_loc  = table_size + stash_hash_function(stash_constants, key);

  // Figure out where the key is first located.
  unsigned first_index;
       if (keys[location_0] == key) first_index = location_0;
  else if (keys[location_1] == key) first_index = location_1;
  else                              first_index = stash_loc;

  // If this thread got a later copy of the key, remove this thread's copy from the table.
  if (first_index != thread_index || key == kKeyEmpty) {
    keys[thread_index] = kKeyEmpty;
    is_unique[thread_index] = 0;
  } else {
    is_unique[thread_index] = 1;
  }
}	

template <> __global__
void hash_remove_duplicates<3>(const unsigned      table_size,
                               const unsigned      total_table_size,
                               const Functions<3>  constants,
                               const uint2         stash_constants,
                                     unsigned     *keys,
                                     unsigned     *is_unique) {	
  // Read out the key that may be duplicated.
  unsigned int thread_index = threadIdx.x +
                              blockIdx.x * blockDim.x +
                              blockIdx.y * blockDim.x * gridDim.x;
  if (thread_index >= total_table_size)
    return;
  unsigned key = keys[thread_index];

  // Determine all the locations that the key could be in.
  unsigned location_0 = hash_function(constants, 0, key) % table_size;
  unsigned location_1 = hash_function(constants, 1, key) % table_size;
  unsigned location_2 = hash_function(constants, 2, key) % table_size;
  unsigned stash_loc  = table_size + stash_hash_function(stash_constants, key);

  // Figure out where the key is first located.
  unsigned first_index;
       if (keys[location_0] == key) first_index = location_0;
  else if (keys[location_1] == key) first_index = location_1;
  else if (keys[location_2] == key) first_index = location_2;
  else                              first_index = stash_loc;

  // If this thread got a later copy of the key, remove this thread's copy from the table.
  if (first_index != thread_index || key == kKeyEmpty) {
    keys[thread_index] = kKeyEmpty;
    is_unique[thread_index] = 0;
  } else {
    is_unique[thread_index] = 1;
  }
}	

template <> __global__
void hash_remove_duplicates<4>(const unsigned      table_size,
                               const unsigned      total_table_size,
                               const Functions<4>  constants,
                               const uint2         stash_constants,
                                     unsigned     *keys,
                                     unsigned     *is_unique) {	
  // Read out the key that may be duplicated.
  unsigned int thread_index = threadIdx.x +
                              blockIdx.x * blockDim.x +
                              blockIdx.y * blockDim.x * gridDim.x;
  if (thread_index >= total_table_size)
    return;
  unsigned key = keys[thread_index];

  // Determine all the locations that the key could be in.
  unsigned location_0 = hash_function(constants, 0, key) % table_size;
  unsigned location_1 = hash_function(constants, 1, key) % table_size;
  unsigned location_2 = hash_function(constants, 2, key) % table_size;
  unsigned location_3 = hash_function(constants, 3, key) % table_size;
  unsigned stash_loc  = table_size + stash_hash_function(stash_constants, key);

  // Figure out where the key is first located.
  unsigned first_index;
       if (keys[location_0] == key) first_index = location_0;
  else if (keys[location_1] == key) first_index = location_1;
  else if (keys[location_2] == key) first_index = location_2;
  else if (keys[location_3] == key) first_index = location_3;
  else                              first_index = stash_loc;

  // If this thread got a later copy of the key, remove this thread's copy from the table.
  if (first_index != thread_index || key == kKeyEmpty) {
    keys[thread_index] = kKeyEmpty;
    is_unique[thread_index] = 0;
  } else {
    is_unique[thread_index] = 1;
  }
}	


template <> __global__
void hash_remove_duplicates<5>(const unsigned      table_size,
                               const unsigned      total_table_size,
                               const Functions<5>  constants,
                               const uint2         stash_constants,
                                     unsigned     *keys,
                                     unsigned     *is_unique) {	
  // Read out the key that may be duplicated.
  unsigned int thread_index = threadIdx.x +
                              blockIdx.x * blockDim.x +
                              blockIdx.y * blockDim.x * gridDim.x;
  if (thread_index >= total_table_size)
    return;
  unsigned key = keys[thread_index];

  // Determine all the locations that the key could be in.
  unsigned location_0 = hash_function(constants, 0, key) % table_size;
  unsigned location_1 = hash_function(constants, 1, key) % table_size;
  unsigned location_2 = hash_function(constants, 2, key) % table_size;
  unsigned location_3 = hash_function(constants, 3, key) % table_size;
  unsigned location_4 = hash_function(constants, 4, key) % table_size;
  unsigned stash_loc  = table_size + stash_hash_function(stash_constants, key);

  // Figure out where the key is first located.
  unsigned first_index;
       if (keys[location_0] == key) first_index = location_0;
  else if (keys[location_1] == key) first_index = location_1;
  else if (keys[location_2] == key) first_index = location_2;
  else if (keys[location_3] == key) first_index = location_3;
  else if (keys[location_4] == key) first_index = location_4;
  else                              first_index = stash_loc;

  // If this thread got a later copy of the key, remove this thread's copy from the table.
  if (first_index != thread_index || key == kKeyEmpty) {
    keys[thread_index] = kKeyEmpty;
    is_unique[thread_index] = 0;
  } else {
    is_unique[thread_index] = 1;
  }
}	
/// @}
#endif


//! @name Internal
//! @{

//! Interleave the keys and their unique IDs in the cuckoo hash table, then compact down the keys.
__global__ void hash_compact_down(const unsigned  table_size,
                                        Entry    *table_entry,
                                        unsigned *unique_keys,
                                  const unsigned *table,
                                  const unsigned *indices) {
  // Read out the table entry.
  unsigned int thread_index = threadIdx.x + blockIdx.x*blockDim.x + blockIdx.y*blockDim.x*gridDim.x;
  if (thread_index >= table_size)
    return;
  unsigned key = table[thread_index];
  unsigned index = indices[thread_index] - 1;
  Entry entry = make_entry(key, index);

  // Write the key and value interleaved.  The value for an invalid key doesn't matter.
  table_entry[thread_index] = entry;

  // Compact down the keys.
  if (key != kKeyEmpty) {
    unique_keys[index] = key;
  }
}
//! @}

bool CompactingHashTable::Build(const unsigned  n,
                                const unsigned *d_keys,
                                const unsigned *d_values)
{
  CUT_CHECK_ERROR("Failed before attempting to build.");

  unsigned num_failures = 1;
  unsigned num_attempts = 0;
  unsigned max_iterations = ComputeMaxIterations(n, table_size_, num_hash_functions_);
  unsigned total_table_size = table_size_ + kStashSize;

  while (num_failures && ++num_attempts < kMaxRestartAttempts) {
    if (num_hash_functions_ == 2)
      constants_2_.Generate(n, d_keys, table_size_);
    else if (num_hash_functions_ == 3)
      constants_3_.Generate(n, d_keys, table_size_);
    else if (num_hash_functions_ == 4)
      constants_4_.Generate(n, d_keys, table_size_);
    else
      constants_5_.Generate(n, d_keys, table_size_);

    // Initialize the cuckoo hash table.
    clear_table<<<ComputeGridDim(total_table_size), kBlockSize>>>
               (total_table_size,
                kKeyEmpty,
                d_scratch_cuckoo_keys_);
    num_failures = 0;
    CUDA_SAFE_CALL(cudaMemcpy(d_failures_,
                              &num_failures,
                              sizeof(unsigned),
                              cudaMemcpyHostToDevice));
    unsigned *d_stash_count = NULL;
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_stash_count, sizeof(unsigned)));
    CUDA_SAFE_CALL(cudaMemset(d_stash_count, 0, sizeof(unsigned)));

    if (num_hash_functions_ == 2) {
      hash_build_compacting <<<ComputeGridDim(n), kBlockSize>>>
                            (n, 
                             d_keys,
                             table_size_,
                             constants_2_,
                             stash_constants_,
                             max_iterations,
                             d_scratch_cuckoo_keys_,
                             d_stash_count,
                             d_failures_);
    } else if (num_hash_functions_ == 3) {
      hash_build_compacting <<<ComputeGridDim(n), kBlockSize>>>
                            (n, 
                             d_keys,
                             table_size_,
                             constants_3_,
                             stash_constants_,
                             max_iterations,
                             d_scratch_cuckoo_keys_,
                             d_stash_count,
                             d_failures_);
    } else if (num_hash_functions_ == 4) {
      hash_build_compacting <<<ComputeGridDim(n), kBlockSize>>>
                            (n, 
                             d_keys,
                             table_size_,
                             constants_4_,
                             stash_constants_,
                             max_iterations,
                             d_scratch_cuckoo_keys_,
                             d_stash_count,
                             d_failures_);
    } else {                             
      hash_build_compacting <<<ComputeGridDim(n), kBlockSize>>>
                            (n, 
                             d_keys,
                             table_size_,
                             constants_5_,
                             stash_constants_,
                             max_iterations,
                             d_scratch_cuckoo_keys_,
                             d_stash_count,
                             d_failures_);
    }

    CUDA_SAFE_CALL(cudaMemcpy(&stash_count_, d_stash_count, sizeof(unsigned), cudaMemcpyDeviceToHost));
    if (stash_count_) {
      char buffer[256];
      sprintf(buffer, "Stash count: %u", stash_count_);
      PrintMessage(buffer, true);
    }
    CUDA_SAFE_CALL(cudaFree(d_stash_count));

    CUT_CHECK_ERROR("!!! Failed to cuckoo hash.\n");

    CUDA_SAFE_CALL(cudaMemcpy(&num_failures,
                              d_failures_,
                              sizeof(unsigned),
                              cudaMemcpyDeviceToHost));

#ifdef COUNT_UNINSERTED
    if (num_failures > 0) {
      char buffer[256];
      sprintf(buffer, "Num failures: %u", num_failures);
      PrintMessage(buffer, true);
    }
#endif
  }

  if (num_attempts >= kMaxRestartAttempts) {
    PrintMessage("Completely failed to build.", true);
    return false;
  } else if (num_attempts > 1) {
    char buffer[256];
    sprintf(buffer, "Needed %u attempts", num_attempts);
    PrintMessage(buffer);
  }

  if (num_failures == 0) {
    // Remove any duplicated keys from the hash table and set values to one.
    if (num_hash_functions_ == 2) {
      hash_remove_duplicates <<<ComputeGridDim(total_table_size), kBlockSize>>>
                             (table_size_,
                              total_table_size,
                              constants_2_,
                              stash_constants_,
                              d_scratch_cuckoo_keys_,
                              d_scratch_counts_);
    } else if (num_hash_functions_ == 3) {
      hash_remove_duplicates <<<ComputeGridDim(total_table_size), kBlockSize>>>
                             (table_size_,
                              total_table_size,
                              constants_3_,
                              stash_constants_,
                              d_scratch_cuckoo_keys_,
                              d_scratch_counts_);
    } else if (num_hash_functions_ == 4) {
      hash_remove_duplicates <<<ComputeGridDim(total_table_size), kBlockSize>>>
                             (table_size_,
                              total_table_size,
                              constants_4_,
                              stash_constants_,
                              d_scratch_cuckoo_keys_,
                              d_scratch_counts_);
    } else {                              
      hash_remove_duplicates <<<ComputeGridDim(total_table_size), kBlockSize>>>
                             (table_size_,
                              total_table_size,
                              constants_5_,
                              stash_constants_,
                              d_scratch_cuckoo_keys_,
                              d_scratch_counts_);
    }
    CUT_CHECK_ERROR("!!! Failed to remove duplicates. \n");

    // Do a prefix-sum over the values to assign each key a unique index.
    CUDPPConfiguration config;
    config.op        = CUDPP_ADD;
    config.datatype  = CUDPP_UINT;
    config.algorithm = CUDPP_SCAN;
    config.options   = CUDPP_OPTION_FORWARD | CUDPP_OPTION_INCLUSIVE;

    CUDPPResult result = cudppPlan(&scanplan_, config, total_table_size, 1, 0);
    if (CUDPP_SUCCESS == result) {
      cudppScan(scanplan_, d_scratch_unique_ids_, d_scratch_counts_, total_table_size);
    } else {
      PrintMessage("!!! Failed to create plan.", true);
    }
    CUT_CHECK_ERROR("!!! Scan failed.\n");

    // Determine how many unique values there are.
    CUDA_SAFE_CALL(cudaMemcpy(&unique_keys_size_,
                              d_scratch_unique_ids_ + total_table_size - 1,
                              sizeof(unsigned),
                              cudaMemcpyDeviceToHost));

    // Copy the unique indices back and compact down the neighbors.
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_unique_keys_,
                              sizeof(unsigned) * unique_keys_size_));
    CUDA_SAFE_CALL(cudaMemset(d_unique_keys_,
                              0xff,
                              sizeof(unsigned) * unique_keys_size_));
    hash_compact_down <<<ComputeGridDim(total_table_size), kBlockSize>>>
                      (total_table_size, 
                       d_contents_, 
                       d_unique_keys_, 
                       d_scratch_cuckoo_keys_, 
                       d_scratch_unique_ids_);
  }

  CUT_CHECK_ERROR("Error occurred during hash table build.\n");

  return true;
}

CompactingHashTable::CompactingHashTable() :
    d_unique_keys_(NULL),
    d_scratch_cuckoo_keys_(NULL),
    d_scratch_counts_(NULL),
    d_scratch_unique_ids_(NULL),
    scanplan_(0),
    HashTable()
{
}

bool CompactingHashTable::Initialize(const unsigned   max_table_entries,
                                     const float      space_usage,
                                     const unsigned   num_functions)
{                                    
  bool success = HashTable::Initialize(max_table_entries, space_usage, num_functions);

  unsigned slots_to_allocate = table_size_ + kStashSize;
  CUDA_SAFE_CALL(cudaMalloc( (void**)&d_scratch_cuckoo_keys_, sizeof(unsigned) * slots_to_allocate ));
  CUDA_SAFE_CALL(cudaMalloc( (void**)&d_scratch_counts_,      sizeof(unsigned) * slots_to_allocate ));
  CUDA_SAFE_CALL(cudaMalloc( (void**)&d_scratch_unique_ids_,  sizeof(unsigned) * slots_to_allocate ));

  success &= d_scratch_cuckoo_keys_ != NULL;
  success &= d_scratch_counts_      != NULL;
  success &= d_scratch_unique_ids_  != NULL;

  return success;
}

CompactingHashTable::~CompactingHashTable() {
  Release();
}

void CompactingHashTable::Release() {
  HashTable::Release();

  cudaFree(d_unique_keys_);
  cudaFree(d_scratch_cuckoo_keys_);
  cudaFree(d_scratch_counts_);
  cudaFree(d_scratch_unique_ids_);

  d_unique_keys_         = NULL;
  d_scratch_cuckoo_keys_ = NULL;
  d_scratch_counts_      = NULL;
  d_scratch_unique_ids_  = NULL;

  cudppDestroyPlan(scanplan_);
  scanplan_         = 0;
  unique_keys_size_ = 0;
}

void CompactingHashTable::Retrieve(const unsigned  n_queries,
                                   const unsigned *d_keys,
                                         unsigned *d_values) {
  unsigned *d_retrieval_probes = NULL;
#ifdef TRACK_ITERATIONS
  CUDA_SAFE_CALL(cudaMalloc((void**)&d_retrieval_probes, sizeof(unsigned) * n_queries));
#endif

  if (num_hash_functions_ == 2) {
    hash_retrieve_compacting<<<ComputeGridDim(n_queries), kBlockSize>>>
                            (n_queries,
                             d_keys,
                             table_size_,
                             d_contents_,
                             constants_2_,
                             stash_constants_,
                             stash_count_,
                             d_values,
                             d_retrieval_probes);
  } else if (num_hash_functions_ == 3) {
    hash_retrieve_compacting<<<ComputeGridDim(n_queries), kBlockSize>>>
                            (n_queries,
                             d_keys,
                             table_size_,
                             d_contents_,
                             constants_3_,
                             stash_constants_,
                             stash_count_,
                             d_values,
                             d_retrieval_probes);
  } else if (num_hash_functions_ == 4) {
    hash_retrieve_compacting<<<ComputeGridDim(n_queries), kBlockSize>>>
                            (n_queries,
                             d_keys,
                             table_size_,
                             d_contents_,
                             constants_4_,
                             stash_constants_,
                             stash_count_,
                             d_values,
                             d_retrieval_probes);
  } else {
    hash_retrieve_compacting<<<ComputeGridDim(n_queries), kBlockSize>>>
                            (n_queries,
                             d_keys,
                             table_size_,
                             d_contents_,
                             constants_5_,
                             stash_constants_,
                             stash_count_,
                             d_values,
                             d_retrieval_probes);
  }
  
  CUT_CHECK_ERROR("Retrieval failed.\n");

#ifdef TRACK_ITERATIONS
  OutputRetrievalStatistics(n_queries,
                            d_retrieval_probes,
                            num_hash_functions_);
  CUDA_SAFE_CALL(cudaFree(d_retrieval_probes));
#endif
}




};  // namespace CuckooHashing
};  // namespace CudaHT

#endif
