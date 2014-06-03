#include <moderngpu.cuh>
#include "radix_sort.h"
#include "skew.h"
#include "skew_kernel.cuh"

#define SA_BLOCK 128

using namespace SA;

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

void ComputeSA(uint* d_str, uint* d_keys_sa, int str_length, mgpu::CudaContext& context, int stage);

// Declare contextPtr and call ComputeSA main function
void runComputeSA(uint* d_str, uint* d_keys_sa, int str_length)
{
	mgpu::ContextPtr context = mgpu::CreateCudaDevice(0);
	ComputeSA(d_str,d_keys_sa, str_length, *context, 0);
 
}
////////////////////////////////////////////////////////////
// d_str: input the original str
// d_keys_sa: output suffix array
// str_length: original string length without $ sign
// contex used by mgpu functions
/////////////////////////////////////////////////////////////
void ComputeSA(uint* d_str, uint* d_keys_sa, int str_length, mgpu::CudaContext& context, int stage)
{
    int mod_1 = (str_length+1)/3 + ((str_length+1)%3 > 0 ? 1:0);
    int mod_2 = (str_length+1)/3 + ((str_length+1)%3 > 1 ? 1:0);
    int mod_3 = (str_length+1)/3;
    int bound = mod_1+mod_2+mod_3;
    bool *unique = new bool[1];
    unique[0] = 1;

    size_t tThreads1 = mod_1+mod_2;
    size_t tThreads2 = mod_3;
    size_t nThreads = SA_BLOCK;
    bool fullBlocks1 = (tThreads1%nThreads==0);
    bool fullBlocks2 = (tThreads2%nThreads==0);
    uint nBlocks1=(fullBlocks1) ? (tThreads1/nThreads) : (tThreads1/nThreads+1);
    uint nBlocks2=(fullBlocks2) ? (tThreads2/nThreads) : (tThreads2/nThreads+1);
    dim3 grid_construct1(nBlocks1,1,1);
    dim3 grid_construct2(nBlocks2,1,1);
    dim3 threads_construct(nThreads,1,1);

    bool *d_unique;
    uint* d_keys_srt_12;
    uint* d_keys_srt_3;
    Vector* d_aKeys;
    Vector* d_bKeys;
    Vector* d_cKeys;
    uint* d_new_str;
    uint* d_isa_12;
  
   CUDA_SAFE_CALL(cudaMalloc((void**)&d_isa_12, tThreads1 * sizeof(uint)));
   CUDA_SAFE_CALL(cudaMalloc((void**)&d_keys_srt_12, tThreads1 * sizeof(uint)));
   CUDA_SAFE_CALL(cudaMalloc((void**)&d_unique, sizeof(bool)));
   CUDA_SAFE_CALL(cudaMalloc((void**)&d_keys_srt_3, tThreads2 * sizeof(uint)));
   CUDA_SAFE_CALL(cudaMemcpy(d_unique, unique, sizeof(bool), cudaMemcpyHostToDevice));

   // extract the positions of i%3 != 0 to construct SA12
   // d_str: input,the original string 
   // d_keys_sa: output, extracted string value with SA1 before SA2
   // d_keys_srt_12: output, store the positions of SA12 in original str
   ////////////////////////////////////////////////////////////////////
   sa12_keys_construct<<< grid_construct1, threads_construct >>>
	     (d_str, d_keys_sa, d_keys_srt_12, mod_1, tThreads1);
   cudaThreadSynchronize();  
   // LSB radix sort the triplets character by character
   // d_keys_sa store the value of the character from the triplets r->l
   // d_keys_srt_12 store the sorted position of each char from each round
   // 3 round to sort the SA12 triplets
   /////////////////////////////////////////////////////////////////////
   CStrRadixSortEngine sorter;
   sorter.KeyValueSort((mod_1+mod_2), d_keys_sa, d_keys_srt_12);
 
   sa12_keys_construct_0<<< grid_construct1, threads_construct >>>
	   (d_str, d_keys_sa, d_keys_srt_12, tThreads1);
   cudaThreadSynchronize();

   sorter.KeyValueSort((mod_1+mod_2), d_keys_sa, d_keys_srt_12);

   sa12_keys_construct_1<<< grid_construct1, threads_construct >>>
	   (d_str, d_keys_sa, d_keys_srt_12, tThreads1);
   CUDA_SAFE_CALL(cudaThreadSynchronize());

   sorter.KeyValueSort(tThreads1, d_keys_sa, d_keys_srt_12);
 
    // Compare each SA12 position's rank to its previous position
    // and  mark 1 if different and 0 for same
    // Input: d_keys_srt_12, first round SA12 (may not fully sorted)
    // Output: d_keys_sa,1 if two position's ranks are the same
    //         d_unique,1 if SA12 are fully sorted
    ////////////////////////////////////////////////////////////////
    compute_rank<<< grid_construct1, threads_construct >>>
          (d_str, d_keys_srt_12, d_keys_sa, d_unique, tThreads1, str_length);
    CUDA_SAFE_CALL(cudaThreadSynchronize());
 
    CUDA_SAFE_CALL(cudaMemcpy(unique, d_unique,  sizeof(bool), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaFree(d_unique));
// If not fully sorted  
if(!unique[0])
 {
   // Inclusive scan to compute the ranks of SA12 
   mgpu::ScanInc(d_keys_sa, (mod_1+mod_2), context);
 
   CUDA_SAFE_CALL(cudaMalloc((void**)&d_new_str, (tThreads1+3) * sizeof(uint)));
   // Construct new string with 2/3 str_length of original string
   // Place the ranks of SA1 before SA2 to construct the new str
   ///////////////////////////////////////////////////////////////
   new_str_construct<<< grid_construct1, threads_construct >>>
           (d_new_str, d_keys_srt_12, d_keys_sa, mod_1, tThreads1);
   CUDA_SAFE_CALL(cudaThreadSynchronize());
 
   ////recursive////
   ComputeSA(d_new_str, d_keys_srt_12, tThreads1-1, context, stage+1);

   CUDA_SAFE_CALL(cudaFree(d_new_str)); 
   // translate the sorted SA12 to original position and compute ISA12
   // Input: d_keys_srt_12, fully sorted SA12 named by local position
   // Output: d_isa_12, ISA12 to store the rank regard to local position
   // d_keys_srt_12, SA12 with regard to global position
   // d_keys_sa, flag to mark those with i mod 3 = 1 and i > 1
   ////////////////////////////////////////////////////////////////////
   reconstruct<<< grid_construct1, threads_construct >>>
  	   (d_keys_srt_12, d_isa_12, d_keys_sa, mod_1, tThreads1);
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
             (d_keys_srt_12, d_isa_12, d_keys_sa, mod_1, tThreads1);
  
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
	  (d_keys_srt_3, d_str, d_keys_srt_12, d_keys_sa, tThreads1, tThreads2, str_length);
  CUDA_SAFE_CALL(cudaThreadSynchronize());
  sa3_keys_construct<<<grid_construct2, threads_construct>>>
          (d_keys_srt_3, d_keys_sa, d_str, tThreads2, str_length);
  CUDA_SAFE_CALL(cudaThreadSynchronize());
  // Only one radix sort based on the result of SA1 (induced sorting)
  sorter.KeyValueSort(mod_3, d_keys_sa, d_keys_srt_3);
//////////////////////////// merge sort//////////////////////////////////

   CUDA_SAFE_CALL(cudaMalloc((void**)&d_aKeys, (tThreads1) * sizeof(Vector)));
   CUDA_SAFE_CALL(cudaMalloc((void**)&d_bKeys, (tThreads2) * sizeof(Vector)));
  // Construct SA12 keys in terms of Vector
  // With SA1 composed of 1st char's value, 2nd char's rank, 0 and 1 
  // With SA2 composed of 1st char's value, 2nd char's value, 
  //          3rd char's rank, 0
  // Input: d_keys_srt_12 the order of aKeys
  //        d_isa_12 storing the ranks in sorted SA12 order
  // Output: d_aKeys, storing SA12 keys in Vectors
  //////////////////////////////////////////////////////////////////
  merge_akeys_construct<<< grid_construct1, threads_construct >>>
	(d_str, d_keys_srt_12, d_isa_12, d_aKeys, tThreads1, mod_1, bound, str_length);
  CUDA_SAFE_CALL(cudaThreadSynchronize());

  // Construct SA3 keys in terms of Vector
  // Composed of 1st char's value, 2nd char's value, 2nd char's rank
  // and 3rd char's rank
  // Input: d_keys_srt_3 the order of bKeys
  //        d_isa_12 storing the ranks of chars behind the first char
  // Output:d_bKeys, storing SA3 keys in Vectors 
  ////////////////////////////////////////////////////////////////////
  merge_bkeys_construct<<< grid_construct2, threads_construct >>>
	(d_str, d_keys_srt_3, d_isa_12, d_bKeys, tThreads2, mod_1, bound, str_length);
  CUDA_SAFE_CALL(cudaThreadSynchronize());

  CUDA_SAFE_CALL(cudaFree(d_isa_12)); 
  CUDA_SAFE_CALL(cudaMalloc((void**)&d_cKeys, (bound) * sizeof(Vector)));
  // Merge SA12 and SA3 based on aKeys and bKeys 
  // Output: cKeys storing the merged aKeys and bKeys
  //         d_keys_sa storing the merged SA12 and SA3 (positions)
  /////////////////////////////////////////////////////////////////
  mgpu::MergePairs(d_aKeys, d_keys_srt_12, tThreads1, d_bKeys, d_keys_srt_3, tThreads2, d_cKeys, d_keys_sa, context);
CUDA_SAFE_CALL(cudaFree(d_aKeys));
CUDA_SAFE_CALL(cudaFree(d_bKeys));
CUDA_SAFE_CALL(cudaFree(d_cKeys));
CUDA_SAFE_CALL(cudaFree(d_keys_srt_12));
CUDA_SAFE_CALL(cudaFree(d_keys_srt_3));
_SafeDeleteArray(unique);

}

