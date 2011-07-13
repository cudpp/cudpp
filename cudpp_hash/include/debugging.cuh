#ifndef CUDAHT__CUCKOO__SRC__LIBRARY___DEBUGGING__CUH
#define CUDAHT__CUCKOO__SRC__LIBRARY___DEBUGGING__CUH

/*! @file debugging.cuh
 *  @brief Functions for analyzing the hash table's performance.
 */

#include "hash_table.cuh"
#include <algorithm>

namespace CudaHT {
namespace CuckooHashing {

//! @name Debugging functions
/// @{

void TakeHashFunctionStatistics(const unsigned   num_keys,
                                const unsigned  *d_keys,
                                const unsigned   table_size,
                                const uint2     *constants,
                                const unsigned   kNumHashFunctions);


//! Output how many probes were required by each thread to perform the retrieval.
/*! @param[in]  n_queries           Number of queries being performed.
 *  @param[in]  d_retrieval_probes  Device array: the number of probes taken for each thread's retrieval.
 *  @param[in]  n_functions         Number of hash functions used.
 */
void OutputRetrievalStatistics(const unsigned  n_queries,
                               const unsigned *d_retrieval_probes,
                               const unsigned  n_functions);


//! Outputs information about how many iterations threads required to successfully cuckoo hash.
/*! @param[in]  n                       Number of keys in the input.
 *  @param[in]  d_iterations_taken      Device mem: Number of iterations each thread took.
 *  @param[in]  d_max_iterations_taken  Device mem: Largest number of iterations taken by any thread.
 */
void OutputBuildStatistics(const unsigned  n,
                           const unsigned *d_iterations_taken);

//! Prints out the contents of the stash.
void PrintStashContents(const Entry *d_stash);

/// @}

}; // namespace CuckooHashing
}; // namespace CudaHT

#endif

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
