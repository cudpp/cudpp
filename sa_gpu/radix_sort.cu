////////////////////////////////////////////////////
// Include files and defines for b40c radix sort
////////////////////////////////////////////////////
#include <b40c/util/error_utils.cuh>
#include <b40c/util/multiple_buffering.cuh>
#include <b40c/radix_sort/enactor.cuh>
#include "radix_sort.h"

using namespace b40c;

namespace SA
{
    void CStrRadixSortEngine::KeyValueSort(unsigned int num_elements, unsigned int* d_keys, unsigned int* d_values)
         {
		radix_sort::Enactor enactor;
		util::DoubleBuffer<unsigned int, unsigned int> double_buffer;
		double_buffer.d_keys[double_buffer.selector] = d_keys;
		double_buffer.d_values[double_buffer.selector] = d_values;
		CUDA_SAFE_CALL(cudaMalloc((void**) &double_buffer.d_keys[double_buffer.selector ^ 1], sizeof(unsigned int) * num_elements));
		CUDA_SAFE_CALL(cudaMalloc((void**) &double_buffer.d_values[double_buffer.selector ^ 1], sizeof(unsigned int) * num_elements));

		enactor.Sort(double_buffer, num_elements);
		//d_keys = double_buffer.d_keys[double_buffer.selector];
		//d_values = double_buffer.d_values[double_buffer.selector];
                CUDA_SAFE_CALL(cudaMemcpy(d_keys, double_buffer.d_keys[double_buffer.selector], sizeof(unsigned int) * num_elements, cudaMemcpyDeviceToDevice));
	  	CUDA_SAFE_CALL(cudaMemcpy(d_values, double_buffer.d_values[double_buffer.selector], sizeof(unsigned int) * num_elements, cudaMemcpyDeviceToDevice));
		// Cleanup "pong" storage
		//if (double_buffer.d_keys[double_buffer.selector ^ 1]) {
		//	cudaFree(double_buffer.d_keys[double_buffer.selector ^ 1]);
		//}
		//if (double_buffer.d_values[double_buffer.selector ^ 1]) {
		//	cudaFree(double_buffer.d_values[double_buffer.selector ^ 1]);
		//}

         }


}
