#include "hash_table.h"
#include "debugging.cuh"

#include <cuda_runtime_api.h>
#include <mt19937ar.h>

#include <cassert>

namespace CudaHT {
namespace CuckooHashing {

void GenerateFunctions(const unsigned  N,
                       const unsigned  num_keys,
                       const unsigned *d_keys,
                       const unsigned  table_size,
                             uint2    *constants) {
#ifdef FORCEFULLY_GENERATE_NO_CYCLES
  bool regenerate = true;
  unsigned *d_cycle_exists = NULL;
  uint2 *d_constants = NULL;
  CUDA_SAFE_CALL(cudaMalloc((void**)&d_cycle_exists, sizeof(unsigned)));
  CUDA_SAFE_CALL(cudaMalloc((void**)&d_constants, sizeof(uint2) * N));

  while (regenerate) {
    regenerate = false;
#endif

    // Generate a set of hash function constants for this build attempt.
    for (unsigned i = 0 ; i < N; ++i) {
      unsigned new_a = genrand_int32() % kPrimeDivisor;
      constants[i].x = (1 > new_a ? 1 : new_a);
      constants[i].y = genrand_int32() % kPrimeDivisor;
    }

#ifdef FORCEFULLY_GENERATE_NO_CYCLES
    // Check if all keys were given a full set of N slots by the functions.
    CUDA_SAFE_CALL(cudaMemset(d_cycle_exists, 0, sizeof(unsigned)));
    CUDA_SAFE_CALL(cudaMemcpy(d_constants,
                              constants,
                              sizeof(uint2) * N,
                              cudaMemcpyHostToDevice));

    take_hash_function_statistics<<<ComputeGridDim(num_keys), kBlockSize>>>
                                 (d_keys, num_keys, table_size, d_constants, N,
                                  NULL, NULL, d_cycle_exists);

    unsigned cycle_exists;
    CUDA_SAFE_CALL(cudaMemcpy(&cycle_exists,
                              d_cycle_exists,
                              sizeof(unsigned),
                              cudaMemcpyDeviceToHost));

    if (cycle_exists) {
      regenerate = true;
    }
  }
  CUDA_SAFE_CALL(cudaFree(d_cycle_exists));
  CUDA_SAFE_CALL(cudaFree(d_constants));
#endif

#ifdef TAKE_HASH_FUNCTION_STATISTICS
  // Examine how well distributed the items are.
  TakeHashFunctionStatistics(num_keys, d_keys, table_size, constants, N);
#endif
}

}; // namespace CuckooHashing
}; // namespace CudaHT
