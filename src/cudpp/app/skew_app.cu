// --------------------------------------------------------------
// CUDPP -- CUDA Data Parallel Primitives library
// --------------------------------------------------------------
// $Revision$
// $Date$
//---------------------------------------------------------------
// This source code is distributed under the terms of license.txt
// in the root directory of this source distribution.
// --------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>

#include "cuda_util.h"
#include "cudpp_globals.h"
#include "cudpp.h"
#include "cudpp_util.h"
#include "cudpp_plan.h"

#include "moderngpu.cuh"
#include "cub/cub.cuh"
#include "kernel/skew_kernel.cuh"

#define SA_BLOCK 128

/*
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

inline void __cudaCheckError( const char *file, const int line )
{

    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
 // More careful checking. However, this will affect performance.
    // Comment away if needed.
    err = cudaDeviceSynchronize();
    if( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
 
    return;
}
*/

/*void ComputeSA(uint* d_str, uint* d_keys_sa, int str_length, mgpu::CudaContext& context, int stage);

// Declare contextPtr and call ComputeSA main function
void runComputeSA(uint* d_str, uint* d_keys_sa, int str_length)
{
	mgpu::ContextPtr context = mgpu::CreateCudaDevice(0);
	ComputeSA(d_str,d_keys_sa, str_length, *context, 0);
 
}*/

void KeyValueSort(unsigned int num_elements, 
                  unsigned int* d_keys, 
                  unsigned int* d_values)
{
    size_t temp_storage_bytes = 0;
    void *d_temp_storage = NULL;
    cub::DoubleBuffer<unsigned int> d_cub_keys;
    cub::DoubleBuffer<unsigned int> d_cub_values;
    d_cub_keys.d_buffers[d_cub_keys.selector] = d_keys;
    d_cub_values.d_buffers[d_cub_values.selector] = d_values;

    CUDA_SAFE_CALL(cudaMalloc((void**) &d_cub_keys.d_buffers[d_cub_keys.selector ^ 1], sizeof(uint) * num_elements));
    CUDA_SAFE_CALL(cudaMalloc((void**) &d_cub_values.d_buffers[d_cub_values.selector ^ 1], sizeof(uint) * num_elements));
    // Initialize the d_temp_storage
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_cub_keys, d_cub_values, num_elements);
    CUDA_SAFE_CALL(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    // Run
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_cub_keys, d_cub_values, num_elements);
                                 
    CUDA_SAFE_CALL(cudaMemcpy(d_keys, d_cub_keys.d_buffers[d_cub_keys.selector], sizeof(uint) * num_elements, cudaMemcpyDeviceToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_values, d_cub_values.d_buffers[d_cub_values.selector], sizeof(uint) * num_elements, cudaMemcpyDeviceToDevice));

    // Cleanup "ping-pong" storage
    if (d_cub_values.d_buffers[1]) cudaFree(d_cub_values.d_buffers[1]);
    if (d_cub_keys.d_buffers[1]) cudaFree(d_cub_keys.d_buffers[1]);

}

template<typename InputIt>
void ScanInc(InputIt data_global, int count, mgpu::CudaContext& context) {
        typedef typename std::iterator_traits<InputIt>::value_type T;
        mgpu::Scan<mgpu::MgpuScanTypeInc>(data_global, count, (T)0, mgpu::plus<T>(), (T*)0,
                (T*)0, data_global, context);
}

template<typename KeysIt1, typename KeysIt2, typename KeysIt3, typename ValsIt1,
        typename ValsIt2, typename ValsIt3>
void Merge(KeysIt1 aKeys_global, ValsIt1 aVals_global, 
        int aCount, KeysIt2 bKeys_global, ValsIt2 bVals_global, int bCount,
        KeysIt3 keys_global, ValsIt3 vals_global, mgpu::CudaContext& context) {

        typedef my_less<typename std::iterator_traits<KeysIt1>::value_type> Comp;
        return mgpu::MergePairs(aKeys_global, aVals_global, aCount, bKeys_global, 
                bVals_global, bCount, keys_global, vals_global, Comp(), context);
}


////////////////////////////////////////////////////////////
// d_str: input the original str
// d_keys_sa: output suffix array
// str_length: original string length without $ sign
// contex used by mgpu functions
/////////////////////////////////////////////////////////////
void ComputeSA(unsigned int* d_str,
               unsigned int* d_keys_sa, 
               size_t str_length, 
               mgpu::CudaContext& context, 
               const CUDPPSkewPlan *plan,
               unsigned int offset,
               unsigned int stage)
{
    size_t mod_1 = (str_length+1)/3 + ((str_length+1)%3 > 0 ? 1:0);
    size_t mod_2 = (str_length+1)/3 + ((str_length+1)%3 > 1 ? 1:0);
    size_t mod_3 = (str_length+1)/3;
    size_t tThreads1 = mod_1+mod_2;
    size_t tThreads2 = mod_3;
    size_t bound = mod_1+mod_2+mod_3;

    bool *unique = new bool[1];
    unique[0] = 1;
    size_t nThreads = SA_BLOCK;
    bool fullBlocks1 = (tThreads1%nThreads==0);
    bool fullBlocks2 = (tThreads2%nThreads==0);
    size_t nBlocks1=(fullBlocks1) ? (tThreads1/nThreads) : (tThreads1/nThreads+1);
    size_t nBlocks2=(fullBlocks2) ? (tThreads2/nThreads) : (tThreads2/nThreads+1);
    dim3 grid_construct1(nBlocks1,1,1);
    dim3 grid_construct2(nBlocks2,1,1);
    dim3 threads_construct(nThreads,1,1);
    
    *plan->m_d_keys_srt_12 = *plan->m_d_keys_srt_12+offset;
    *plan->m_d_new_str = ((stage==0) ? *plan->m_d_new_str : *plan->m_d_new_str+offset+3); 

    CUDA_SAFE_CALL(cudaMemcpy(plan->m_d_unique, unique, sizeof(bool), cudaMemcpyHostToDevice));

   // extract the positions of i%3 != 0 to construct SA12
   // d_str: input,the original string 
   // d_keys_sa: output, extracted string value with SA1 before SA2
   // d_keys_srt_12: output, store the positions of SA12 in original str
   ////////////////////////////////////////////////////////////////////
   sa12_keys_construct<<< grid_construct1, threads_construct >>>
	     (d_str, d_keys_sa, plan->m_d_keys_srt_12, mod_1, tThreads1);
   cudaThreadSynchronize();  
   // LSB radix sort the triplets character by character
   // d_keys_sa store the value of the character from the triplets r->l
   // d_keys_srt_12 store the sorted position of each char from each round
   // 3 round to sort the SA12 triplets
   /////////////////////////////////////////////////////////////////////
   KeyValueSort((mod_1+mod_2), d_keys_sa, plan->m_d_keys_srt_12);
 
   sa12_keys_construct_0<<< grid_construct1, threads_construct >>>
	   (d_str, d_keys_sa, plan->m_d_keys_srt_12, tThreads1);
   cudaThreadSynchronize();

   KeyValueSort((mod_1+mod_2), d_keys_sa, plan->m_d_keys_srt_12);

   sa12_keys_construct_1<<< grid_construct1, threads_construct >>>
	   (d_str, d_keys_sa, plan->m_d_keys_srt_12, tThreads1);
   CUDA_SAFE_CALL(cudaThreadSynchronize());

   KeyValueSort(tThreads1, d_keys_sa, plan->m_d_keys_srt_12);
 
    // Compare each SA12 position's rank to its previous position
    // and  mark 1 if different and 0 for same
    // Input: d_keys_srt_12, first round SA12 (may not fully sorted)
    // Output: d_keys_sa,1 if two position's ranks are the same
    //         d_unique,1 if SA12 are fully sorted
    ////////////////////////////////////////////////////////////////
    compute_rank<<< grid_construct1, threads_construct >>>
          (d_str, plan->m_d_keys_srt_12, d_keys_sa, plan->m_d_unique, tThreads1, str_length);
    CUDA_SAFE_CALL(cudaThreadSynchronize());
 
    CUDA_SAFE_CALL(cudaMemcpy(unique, plan->m_d_unique,  sizeof(bool), cudaMemcpyDeviceToHost));
// If not fully sorted  
if(!unique[0])
 {
   // Inclusive scan to compute the ranks of SA12 
   ScanInc(d_keys_sa, (mod_1+mod_2), context);
 
   // Construct new string with 2/3 str_length of original string
   // Place the ranks of SA1 before SA2 to construct the new str
   ///////////////////////////////////////////////////////////////
   new_str_construct<<< grid_construct1, threads_construct >>>
           (plan->m_d_new_str, plan->m_d_keys_srt_12, d_keys_sa, mod_1, tThreads1);
   CUDA_SAFE_CALL(cudaThreadSynchronize());
 
   ////recursive////
   ComputeSA(plan->m_d_new_str, plan->m_d_keys_srt_12, tThreads1-1, context, plan, tThreads1, stage+1);

   // translate the sorted SA12 to original position and compute ISA12
   // Input: d_keys_srt_12, fully sorted SA12 named by local position
   // Output: d_isa_12, ISA12 to store the rank regard to local position
   // d_keys_srt_12, SA12 with regard to global position
   // d_keys_sa, flag to mark those with i mod 3 = 1 and i > 1
   ////////////////////////////////////////////////////////////////////
   reconstruct<<< grid_construct1, threads_construct >>>
  	   (plan->m_d_keys_srt_12, plan->m_d_isa_12, d_keys_sa, mod_1, tThreads1);
   CUDA_SAFE_CALL(cudaThreadSynchronize());

 }

// SA12 already fully sorted with results stored in d_keys_srt_12
// in their original position, no need to reconstruct, construct ISA12 
// Input: d_keys_`srt_12, fully sorted SA12 named by gloabl position
// Output: d_isa_12, ISA12 to store the rank regard to local position
//         d_keys_sa, flag to mark those with i mod 3 = 1
//////////////////////////////////////////////////////////////////////
else
{
   isa12_construct<<< grid_construct1, threads_construct >>>  
             (plan->m_d_keys_srt_12, plan->m_d_isa_12, d_keys_sa, mod_1, tThreads1);
  
   CUDA_SAFE_CALL(cudaThreadSynchronize());
}

// Exclusive scan to compute the position of SA1
  mgpu::ScanExc(d_keys_sa, (mod_1+mod_2), context);
  // Construct SA3 keys and positions based on SA1's ranks  
  // Input: d_keys_srt_12, sorted SA12
  //        d_keys_sa, positions of sorted SA1
  //        tThreads1, mod_1+mod_2 tThreads2, mod_3
  // Output:d_keys_srt_3, positions of i mod 3 = 3 in the same order of SA1
  //        d_keys_sa, ith character value according to d_keys_srt_3
  ////////////////////////////////////////////////////////////////////////
  sa3_srt_construct<<< grid_construct1, threads_construct >>>
	  (plan->m_d_keys_srt_3, d_str, plan->m_d_keys_srt_12, d_keys_sa, tThreads1, tThreads2, str_length);
  CUDA_SAFE_CALL(cudaThreadSynchronize());
  sa3_keys_construct<<<grid_construct2, threads_construct>>>
          (plan->m_d_keys_srt_3, d_keys_sa, d_str, tThreads2, str_length);
  CUDA_SAFE_CALL(cudaThreadSynchronize());
  // Only one radix sort based on the result of SA1 (induced sorting)
  KeyValueSort(mod_3, d_keys_sa, plan->m_d_keys_srt_3);
//////////////////////////// merge sort//////////////////////////////////

  // Construct SA12 keys in terms of Vector
  // With SA1 composed of 1st char's value, 2nd char's rank, 0 and 1 
  // With SA2 composed of 1st char's value, 2nd char's value, 
  //          3rd char's rank, 0
  // Input: d_keys_srt_12 the order of aKeys
  //        d_isa_12 storing the ranks in sorted SA12 order
  // Output: d_aKeys, storing SA12 keys in Vectors
  //////////////////////////////////////////////////////////////////
  merge_akeys_construct<<< grid_construct1, threads_construct >>>
	(d_str, plan->m_d_keys_srt_12, plan->m_d_isa_12, plan->m_d_aKeys, tThreads1, mod_1, bound, str_length);
  CUDA_SAFE_CALL(cudaThreadSynchronize());

  // Construct SA3 keys in terms of Vector
  // Composed of 1st char's value, 2nd char's value, 2nd char's rank
  // and 3rd char's rank
  // Input: d_keys_srt_3 the order of bKeys
  //        d_isa_12 storing the ranks of chars behind the first char
  // Output:d_bKeys, storing SA3 keys in Vectors 
  ////////////////////////////////////////////////////////////////////
  merge_bkeys_construct<<< grid_construct2, threads_construct >>>
	(d_str, plan->m_d_keys_srt_3, plan->m_d_isa_12, plan->m_d_bKeys, tThreads2, mod_1, bound, str_length);
  CUDA_SAFE_CALL(cudaThreadSynchronize());

  // Merge SA12 and SA3 based on aKeys and bKeys 
  // Output: cKeys storing the merged aKeys and bKeys
  //         d_keys_sa storing the merged SA12 and SA3 (positions)
  /////////////////////////////////////////////////////////////////
  Merge(plan->m_d_aKeys, plan->m_d_keys_srt_12, tThreads1, plan->m_d_bKeys, plan->m_d_keys_srt_3, tThreads2, plan->m_d_cKeys, d_keys_sa, context);
  _SafeDeleteArray(unique);

}

/** @brief Allocate intermediate arrays used by suffix array.
 *
 *
 * @param [in,out] plan Pointer to CUDPPSkewPlan object
 *                      containing options and number of elements,
 *                      which is used to compute storage
 *                      requirements, and within which intermediate
 *                      storage is allocated.
 */

void allocSkewStorage(CUDPPSkewPlan *plan)
{
    size_t str_length = plan->m_numElements;
    size_t mod_1 = (str_length+1)/3 + ((str_length+1)%3 > 0 ? 1:0);
    size_t mod_2 = (str_length+1)/3 + ((str_length+1)%3 > 1 ? 1:0);
    size_t mod_3 = (str_length+1)/3;
    CUDA_SAFE_CALL(cudaMalloc((void**) &(plan->m_d_isa_12), (mod_1+mod_2) * sizeof(uint)));
    CUDA_SAFE_CALL(cudaMalloc((void**) &(plan->m_d_keys_srt_12), 2*str_length * sizeof(uint)));
    CUDA_SAFE_CALL(cudaMalloc((void**) &(plan->m_d_unique), sizeof(bool)));
    CUDA_SAFE_CALL(cudaMalloc((void**) &(plan->m_d_keys_srt_3), mod_3 * sizeof(uint)));
    CUDA_SAFE_CALL(cudaMalloc((void**) &(plan->m_d_new_str), 2*str_length * sizeof(uint)));
    CUDA_SAFE_CALL(cudaMalloc((void**) &(plan->m_d_aKeys), (mod_1+mod_2) * sizeof(Vector)));
    CUDA_SAFE_CALL(cudaMalloc((void**) &(plan->m_d_bKeys), mod_3 * sizeof(Vector)));
    CUDA_SAFE_CALL(cudaMalloc((void**) &(plan->m_d_cKeys), (mod_1+mod_2+mod_3) * sizeof(Vector)));
    CUDA_CHECK_ERROR("allocSkewStorage");
}


/** @brief Deallocate intermediate block arrays in a CUDPPSkewPlan object.
 *
 *
 * @param[in,out] plan Pointer to CUDPPSkewPlan object initialized by allocSkewStorage().
 */
void freeSkewStorage(CUDPPSkewPlan *plan)
{
    CUDA_SAFE_CALL(cudaFree(plan->m_d_isa_12));
    CUDA_SAFE_CALL(cudaFree(plan->m_d_keys_srt_12));
    CUDA_SAFE_CALL(cudaFree(plan->m_d_unique));
    CUDA_SAFE_CALL(cudaFree(plan->m_d_keys_srt_3));
    CUDA_SAFE_CALL(cudaFree(plan->m_d_aKeys));
    CUDA_SAFE_CALL(cudaFree(plan->m_d_bKeys));
    CUDA_SAFE_CALL(cudaFree(plan->m_d_cKeys));
    CUDA_CHECK_ERROR("freeSkewStorage");
}


/** @brief Dispatch function to perform parallel suffix array on a
 *         string with the specified configuration.
 *
 * 
 * @param[in]  d_str input string with three $ 
 * @param[out] d_keys_sa lexicographically sorted suffix position array
 * @param[in]  str_length Number of elements in the string including $
 * @param[in]  plan     Pointer to CUDPPSkewPlan object containing
 *                      suffix_array options and intermediate storage
 */
void cudppSuffixArrayDispatch(void* d_str,
                              void* d_keys_sa,
                              size_t d_str_length,
                              const CUDPPSkewPlan *plan)
{
    mgpu::ContextPtr context = mgpu::CreateCudaDevice(0);
    ComputeSA((unsigned int*)d_str, (unsigned int*)d_keys_sa, d_str_length, *context, plan, 0, 0);
}


