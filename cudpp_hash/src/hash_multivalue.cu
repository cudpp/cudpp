#ifndef HASH_V2_MULTI__CUH
#define HASH_V2_MULTI__CUH

#include "hash_multivalue.h"
#include "hash_table.cuh"
#include "cuda_util.h"

#include <cudpp.h>
// #include <cutil.h>

// #ifndef USE_DAN_LIB
//  #include <radixsort_single_grid.cu>
//  #include <radixsort_early_exit.cu>
//#else
//  #include <Utilities/utilities.h>
//#endif

namespace CudaHT {
namespace CuckooHashing {

//! @name Internal
/// @{

//! Compacts the unique keys down and stores the location of its values as the value.
__global__ void compact_keys(const unsigned keys[],
                             const unsigned is_unique[],
                             const unsigned locations[],
                             uint2          index_counts[],
                             unsigned       compacted[],
                             size_t         kSize) {
    unsigned index = threadIdx.x +
        blockIdx.x * blockDim.x +
        blockIdx.y * blockDim.x * gridDim.x;
    if (index < kSize && is_unique[index]) {
        unsigned array_index = locations[index] - 1;
        compacted[array_index] = keys[index];
        index_counts[array_index].x = index;
    }
}


//! Finds unique keys by checking neighboring items in a sorted list.
__global__ void check_if_unique(const unsigned keys[],
                                unsigned      is_unique[],
                                size_t        kSize) {
    unsigned id = threadIdx.x +
        blockIdx.x * blockDim.x +
        blockIdx.y * blockDim.x * gridDim.x;
    if (id == 0) {
        is_unique[0] = 1;
    } else if (id < kSize) {
        is_unique[id] = (keys[id] != keys[id - 1] ? 1 : 0);
    }
}


//! Counts how many values each key has.
__global__ void count_values(uint2     index_counts[],
                             unsigned  kSize,
                             unsigned  num_unique) {
    unsigned index = threadIdx.x +
        blockIdx.x * blockDim.x +
        blockIdx.y * blockDim.x * gridDim.x;
    if (index < num_unique - 1) {
        index_counts[index].y = index_counts[index+1].x - index_counts[index].x;
    } else if (index == num_unique-1) {
        index_counts[index].y = kSize - index_counts[index].x;
    }
}

//! Creates an array of values equal to the array index.
__global__ void prepare_indices(const unsigned num_keys,
                                unsigned *data) {
    unsigned index = threadIdx.x +
        blockIdx.x * blockDim.x +
        blockIdx.y * blockDim.x * gridDim.x;
    if (index < num_keys) {
        data[index] = index;
    }
}                            
/// @}


template <unsigned kNumHashFunctions> __global__
void hash_retrieve_multi_sorted(const unsigned   n_queries,
                                const unsigned  *keys_in, 
                                const unsigned   table_size, 
                                const Entry     *table, 
                                const uint2     *index_counts, 
                                const Functions<kNumHashFunctions>  constants,
                                const uint2      stash_constants,
                                const unsigned   stash_count,
                                uint2     *location_count)
{
    // Get the key & perform the query.
    unsigned thread_index = threadIdx.x + blockIdx.x*blockDim.x + 
        blockIdx.y*blockDim.x*gridDim.x;
    if (thread_index >= n_queries)
        return;
    unsigned key = keys_in[thread_index];
    unsigned result = retrieve(key,
                               table_size,
                               table,
                               constants,
                               stash_constants,
                               stash_count,
                               NULL);

    // Return the location of the key's values and the count.
    uint2 index_count;
    if (result == kNotFound) {
        index_count = make_uint2(0, 0);
    } else {
        index_count = index_counts[result];
    }
    location_count[thread_index] = index_count;
}       

bool MultivalueHashTable::Build(const unsigned  n,
                                const unsigned *d_keys,
                                const unsigned *d_vals)
{
    CUDA_CHECK_ERROR("Failed before build.");

    unsigned *d_sorted_keys = NULL;
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_sorted_keys, sizeof(unsigned) * n));
    CUDA_SAFE_CALL(cudaMemcpy(d_sorted_keys, d_keys, sizeof(unsigned) * n, 
                              cudaMemcpyDeviceToDevice));

    unsigned *d_sorted_vals = NULL;
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_sorted_vals, sizeof(unsigned) * n));
    CUDA_SAFE_CALL(cudaMemcpy(d_sorted_vals, d_vals, sizeof(unsigned) * n, 
                              cudaMemcpyDeviceToDevice));
    CUDA_CHECK_ERROR("Failed to allocate.");

    CUDPPConfiguration sort_config;
    sort_config.algorithm = CUDPP_SORT_RADIX;                  
    sort_config.datatype = CUDPP_UINT;
    sort_config.options = CUDPP_OPTION_KEY_VALUE_PAIRS;

    CUDPPHandle sort_plan;
    CUDPPResult sort_result = cudppPlan(theCudpp, &sort_plan, sort_config, n,
                                        1, 0);
    cudppSort(sort_plan, d_sorted_keys, (void*)d_sorted_vals, n);

    if (sort_result != CUDPP_SUCCESS)
    {
        printf("Error in plan creation in MultivalueHashTable::build\n");
        bool retval = false;
        cudppDestroyPlan(sort_plan);
        return retval;
    }
    CUDA_CHECK_ERROR("Failed to sort");

    // Find the first key-value pair for each key.
    dim3 gridDim = ComputeGridDim(n);
    check_if_unique <<<gridDim, kBlockSize>>> (d_sorted_keys,
                                               d_scratch_is_unique_,
                                               n);
    CUDA_CHECK_ERROR("Failed to check uniqueness");


    // Assign a unique index from 0 to k-1 for each of the keys.
    cudppScan(scanplan_, d_scratch_offsets_, d_scratch_is_unique_, n);
    CUDA_CHECK_ERROR("Failed to scan");

    // Check how many unique keys were found.
    unsigned num_unique_keys;
    CUDA_SAFE_CALL(cudaMemcpy(&num_unique_keys, d_scratch_offsets_ + n - 1,
                              sizeof(unsigned), cudaMemcpyDeviceToHost));
    CUDA_CHECK_ERROR("Failed to get # unique keys");

    // Keep a list of the unique keys, and store info on each key's data
    // (location in the values array, how many there are).
    unsigned *d_compacted_keys = NULL;
    uint2 *d_index_counts_tmp = NULL;
    CUDA_SAFE_CALL(cudaMalloc((void**) &d_compacted_keys,
                              sizeof(unsigned) * num_unique_keys));
    CUDA_SAFE_CALL(cudaMalloc((void**) &d_index_counts_tmp,
                              sizeof(uint2) * num_unique_keys));
    compact_keys<<<gridDim, 512>>> (d_sorted_keys,
                                    d_scratch_is_unique_,
                                    d_scratch_offsets_,
                                    d_index_counts_tmp,
                                    d_compacted_keys,
                                    n);
    CUDA_CHECK_ERROR("Failed to compact");

    // Determine the counts.
    count_values<<<ComputeGridDim(num_unique_keys), 512>>>
        (d_index_counts_tmp, n, num_unique_keys);
    CUDA_CHECK_ERROR("Failed to count");

    // Reinitialize the cuckoo hash table using the information we discovered.
    HashTable::Initialize(num_unique_keys,
                          target_space_usage_,
                          num_hash_functions_);

    d_index_counts_  = d_index_counts_tmp;
    d_unique_keys_   = d_compacted_keys;
    d_sorted_values_ = d_sorted_vals;
    sorted_values_size_ = n;

    // Build the cuckoo hash table with each key assigned a unique index.
    // Re-uses the sorted key memory as an array of values from 0 to k-1.
    prepare_indices<<<ComputeGridDim(num_unique_keys), kBlockSize>>>
        (num_unique_keys, d_sorted_keys);
    bool success = HashTable::Build(num_unique_keys, d_unique_keys_,
                                    d_sorted_keys);
    CUDA_SAFE_CALL(cudaFree(d_sorted_keys));
    return success;
}

void MultivalueHashTable::Retrieve(const unsigned  n_queries,
                                   const unsigned *d_keys,
                                   uint2    *d_location_counts)
{
    if (num_hash_functions_ == 2) {
        hash_retrieve_multi_sorted<2> <<<ComputeGridDim(n_queries),kBlockSize>>>
            (n_queries,
             d_keys,
             table_size_,
             d_contents_,
             d_index_counts_,
             constants_2_,
             stash_constants_,
             stash_count_,
             d_location_counts );
    } else if (num_hash_functions_ == 3) {
        hash_retrieve_multi_sorted<3> <<<ComputeGridDim(n_queries),kBlockSize>>>
            (n_queries,
             d_keys,
             table_size_,
             d_contents_,
             d_index_counts_,
             constants_3_,
             stash_constants_,
             stash_count_,
             d_location_counts );
    } else if (num_hash_functions_ == 4) {
        hash_retrieve_multi_sorted<4> <<<ComputeGridDim(n_queries),kBlockSize>>>
            (n_queries,
             d_keys,
             table_size_,
             d_contents_,
             d_index_counts_,
             constants_4_,
             stash_constants_,
             stash_count_,
             d_location_counts );
    } else {
        hash_retrieve_multi_sorted<5> <<<ComputeGridDim(n_queries),kBlockSize>>>
            (n_queries,
             d_keys,
             table_size_,
             d_contents_,
             d_index_counts_,
             constants_5_,
             stash_constants_,
             stash_count_,
             d_location_counts );
    }

    CUDA_CHECK_ERROR("Retrieval failed.\n");
}

bool MultivalueHashTable::Initialize(const unsigned   max_table_entries,
                                     const float      space_usage,
                                     const unsigned   num_hash_functions)
{                                    
    unsigned success = HashTable::Initialize(max_table_entries, space_usage,
                                             num_hash_functions);
    target_space_usage_ = space_usage;

    // + 2N 32-bit entries
    CUDA_SAFE_CALL(cudaMalloc( (void**)&d_scratch_offsets_, 
                               sizeof(unsigned) * max_table_entries ));
    CUDA_SAFE_CALL(cudaMalloc( (void**)&d_scratch_is_unique_,
                               sizeof(unsigned) * max_table_entries ));

    success &= d_scratch_offsets_ != NULL;
    success &= d_scratch_is_unique_ != NULL;

    // Allocate memory for the scan.
    // + Unknown memory usage
    CUDPPConfiguration config;
    config.op            = CUDPP_ADD;
    config.datatype      = CUDPP_UINT;
    config.algorithm     = CUDPP_SCAN;
    config.options       = CUDPP_OPTION_FORWARD | CUDPP_OPTION_INCLUSIVE;
    CUDPPResult result   = cudppPlan(theCudpp, &scanplan_, config, 
                                     max_table_entries, 1, 0);
    if (CUDPP_SUCCESS != result) {
        fprintf(stderr, "Failed to create plan.");
        return false;
    }
    return success;
}

MultivalueHashTable::MultivalueHashTable() :
    d_index_counts_(NULL),
    d_sorted_values_(NULL),
    d_scratch_offsets_(NULL),
    d_scratch_is_unique_(NULL),
    d_unique_keys_(NULL),
    scanplan_(0),
    HashTable()
{
}

void MultivalueHashTable::Release() {
    HashTable::Release();

    if (scanplan_) {
      cudppDestroyPlan(scanplan_);
      scanplan_ = 0;
    }

    cudaFree(d_index_counts_);
    cudaFree(d_sorted_values_);
    cudaFree(d_scratch_offsets_);
    cudaFree(d_scratch_is_unique_);
    cudaFree(d_unique_keys_);

    d_index_counts_      = NULL;
    d_sorted_values_     = NULL;
    d_scratch_offsets_   = NULL;
    d_scratch_is_unique_ = NULL;
    d_unique_keys_       = NULL;
}

};  // namespace CuckooHashing
};  // namespace CudaHT

#endif //  HASH_V2_MULTI__CUH

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
