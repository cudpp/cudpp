////////////////////////////////////////////////////
// Include files and defines for b40c radix sort
////////////////////////////////////////////////////
#include <cub/cub.cuh>
#include "radix_sort.h"
using namespace cub;

namespace SA
{
    void CStrRadixSortEngine::KeyValueSort(unsigned int num_elements, unsigned int* d_keys, unsigned int* d_values)
         {
                size_t temp_storage_bytes = 0;
                void *d_temp_storage = NULL;
                DoubleBuffer<unsigned int> d_cub_keys;
                DoubleBuffer<unsigned int> d_cub_values;
                d_cub_keys.d_buffers[d_cub_keys.selector] = d_keys;
                d_cub_values.d_buffers[d_cub_values.selector] = d_values;

                CUDA_SAFE_CALL(cudaMalloc((void**) &d_cub_keys.d_buffers[d_cub_keys.selector ^ 1], sizeof(uint) * num_elements));
                CUDA_SAFE_CALL(cudaMalloc((void**) &d_cub_values.d_buffers[d_cub_values.selector ^ 1], sizeof(uint) * num_elements));
                // Initialize the d_temp_storage
                DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_cub_keys, d_cub_values, num_elements);
                CUDA_SAFE_CALL(cudaMalloc(&d_temp_storage, temp_storage_bytes));
                // Run
                DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_cub_keys, d_cub_values, num_elements);
                                
                CUDA_SAFE_CALL(cudaMemcpy(d_keys, d_cub_keys.d_buffers[d_cub_keys.selector], sizeof(uint) * num_elements, cudaMemcpyDeviceToDevice));
	  	CUDA_SAFE_CALL(cudaMemcpy(d_values, d_cub_values.d_buffers[d_cub_values.selector], sizeof(uint) * num_elements, cudaMemcpyDeviceToDevice));

		// Cleanup "ping-pong" storage
                if (d_cub_values.d_buffers[1]) cudaFree(d_cub_values.d_buffers[1]);
                if (d_cub_keys.d_buffers[1]) cudaFree(d_cub_keys.d_buffers[1]);
         }

}
