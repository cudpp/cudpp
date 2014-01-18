#include "skew.h"
#include "radix_sort.h"
#include "merge.cuh"
#include "skew_kernel.cuh"
#include <include/moderngpu.cuh>
#include <iostream>
#include <fstream>
#include <string>
#define SA_BLOCK 128
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

using namespace std;
using namespace SA;
using namespace mgpu;

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

typedef unsigned int uint;

void ComputeSA(uint* d_str, uint* d_keys_sa, int str_length, CudaContext& context, int stage);


void runComputeSA(unsigned char* d_str, unsigned int* d_keys_sa, int str_length)
{
	ContextPtr context = CreateCudaDevice(0);
        size_t nThreads = SA_BLOCK;
        size_t tThreads = str_length+3;
        bool fullBlocks = (tThreads%nThreads==0);
        uint nBlocks = (fullBlocks) ? (tThreads/nThreads) : (tThreads/nThreads+1);
        dim3 grid_construct(nBlocks,1,1);
        dim3 threads_construct(nThreads,1,1);
        uint* d_str_value;
        uint* d_result;
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_str_value, tThreads * sizeof(uint)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_result, (str_length+1) * sizeof(uint)));
        strConstruct<<< grid_construct, threads_construct >>>
               (d_str, d_str_value, str_length);
        CUDA_SAFE_CALL(cudaThreadSynchronize());
	ComputeSA(d_str_value, d_result, str_length, *context, 0);
        resultConstruct<<< grid_construct, threads_construct >>>
               (d_result, d_keys_sa, str_length);
        CUDA_SAFE_CALL(cudaThreadSynchronize());
        CUDA_SAFE_CALL(cudaFree(d_str_value));
        CUDA_SAFE_CALL(cudaFree(d_result));
}


void ComputeSA(uint* d_str, uint* d_keys_sa, int str_length, CudaContext& context, int stage)
{
    int mod_1 = (str_length+1)/3 + ((str_length+1)%3 > 0 ? 1:0);
    int mod_2 = (str_length+1)/3 + ((str_length+1)%3 > 1 ? 1:0);
    int mod_3 = (str_length+1)/3;
    int bound = mod_1+mod_2+mod_3;
    bool *unique = new bool[1];
    unique[0] = 0;
    bool *d_unique;
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

    uint* d_keys_srt_12;
    uint* d_keys_sa_12;
    uint* d_keys_srt_3;
    uint* d_keys_uint_3;
    Vector* d_aKeys;
    Vector* d_bKeys;
    Vector* d_cKeys;
    int* d_aVals;
    int* d_bVals;
    uint* d_new_str;

    
 
   CUDA_SAFE_CALL(cudaMalloc((void**)&d_keys_srt_12, tThreads1 * sizeof(uint)));
   
   sa12_keys_construct<<< grid_construct1, threads_construct >>>
	     (d_str, d_keys_sa, d_keys_srt_12, mod_1, tThreads1);
   CUDA_SAFE_CALL(cudaThreadSynchronize());
   cout << "mod_1=" << mod_1 << "," << "mod_2=" << mod_2 <<endl;
   CStrRadixSortEngine sorter;
   sorter.KeyValueSort((mod_1+mod_2), d_keys_sa, d_keys_srt_12);
   CUDA_SAFE_CALL(cudaThreadSynchronize());
   cout<<"Stage "<< stage << endl;
   sa12_keys_construct_0<<< grid_construct1, threads_construct >>>
	   (d_str, d_keys_sa, d_keys_srt_12, tThreads1, str_length, stage);
   CUDA_SAFE_CALL(cudaThreadSynchronize());
   sorter.KeyValueSort((mod_1+mod_2), d_keys_sa, d_keys_srt_12);
   sa12_keys_construct_1<<< grid_construct1, threads_construct >>>
	   (d_str, d_keys_sa, d_keys_srt_12, tThreads1, str_length);
   CUDA_SAFE_CALL(cudaThreadSynchronize());
   sorter.KeyValueSort(tThreads1, d_keys_sa, d_keys_srt_12);
//cout << "--------------sorter12 completed-------------------------" <<endl;

   CUDA_SAFE_CALL(cudaMalloc((void**)&d_unique, sizeof(bool)));
   compute_rank_1<<< grid_construct1, threads_construct >>>
	 (d_str, d_keys_srt_12, d_keys_sa, d_unique, tThreads1);
   CUDA_SAFE_CALL(cudaThreadSynchronize());

   CUDA_SAFE_CALL(cudaMemcpy(unique, d_unique,  sizeof(bool), cudaMemcpyDeviceToHost));
   CUDA_SAFE_CALL(cudaFree(d_unique));
if (unique[0]!=0) unique[0]=true;

if(!unique[0])
{
   int total = Scan(d_keys_sa, (mod_1+mod_2), context);
   CUDA_SAFE_CALL(cudaMalloc((void**)&d_keys_sa_12, (str_length+2) * sizeof(uint)));
   compute_rank_2<<< grid_construct1, threads_construct >>>
	  (d_keys_srt_12, d_keys_sa, d_keys_sa_12, tThreads1);

   CUDA_SAFE_CALL(cudaMalloc((void**)&d_new_str, (tThreads1+3) * sizeof(uint)));

   new_str_construct<<< grid_construct1, threads_construct >>>
	  (d_new_str, d_keys_srt_12, d_keys_sa_12, mod_1, tThreads1);
   CUDA_SAFE_CALL(cudaThreadSynchronize());

   CUDA_SAFE_CALL(cudaFree(d_keys_sa_12));

////recursive////
ComputeSA(d_new_str, d_keys_srt_12, tThreads1-1, context, stage+1);

  CUDA_SAFE_CALL(cudaFree(d_new_str));
  reconstruct<<< grid_construct1, threads_construct >>>
	   (d_keys_srt_12, mod_1, tThreads1);
  CUDA_SAFE_CALL(cudaThreadSynchronize());

}

//cout << "------unique ranked-------------------" <<endl;

  CUDA_SAFE_CALL(cudaMalloc((void**)&d_keys_sa_12, (str_length+2) * sizeof(uint)));
  SA12_result_store<<< grid_construct1, threads_construct >>>
	  (d_keys_srt_12, d_keys_sa_12, tThreads1, str_length);
  CUDA_SAFE_CALL(cudaThreadSynchronize());

  CUDA_SAFE_CALL(cudaMalloc((void**)&d_keys_srt_3, tThreads2 * sizeof(uint)));
  CUDA_SAFE_CALL(cudaMalloc((void**)&d_keys_uint_3, tThreads2 * sizeof(uint)));

  SA3_keys_construct<<< grid_construct2, threads_construct >>>
	  ( d_str, d_keys_uint_3, d_keys_srt_3,  d_keys_sa_12, bound, tThreads2);
  CUDA_SAFE_CALL(cudaThreadSynchronize());

  sorter.KeyValueSort(mod_3, d_keys_uint_3, d_keys_srt_3);

  SA3_keys_construct_0<<< grid_construct2, threads_construct >>>
	  ( d_str, d_keys_uint_3, d_keys_srt_3, tThreads2);
  CUDA_SAFE_CALL(cudaThreadSynchronize());
  sorter.KeyValueSort(mod_3, d_keys_uint_3, d_keys_srt_3);

// cout << "---------------sorter3 completed------------------------" <<endl;
//////////////////////////// merge sort//////////////////////////////////
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_aKeys, (tThreads1) * sizeof(Vector)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_bKeys, (tThreads2) * sizeof(Vector)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_aVals, (tThreads1) * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_bVals, (tThreads2) * sizeof(int)));

  merge_akeys_construct<<< grid_construct1, threads_construct >>>
	(d_str, d_keys_srt_12, d_keys_sa_12, d_aKeys, d_aVals, tThreads1, bound, str_length);

  CUDA_SAFE_CALL(cudaThreadSynchronize());

  merge_bkeys_construct<<< grid_construct2, threads_construct >>>
	(d_str, d_keys_srt_3, d_keys_sa_12, d_bKeys, d_bVals, tThreads2, bound, str_length);
  CUDA_SAFE_CALL(cudaThreadSynchronize());

  CUDA_SAFE_CALL(cudaFree(d_keys_srt_12));
  CUDA_SAFE_CALL(cudaFree(d_keys_sa_12));
  CUDA_SAFE_CALL(cudaFree(d_keys_srt_3));
  CUDA_SAFE_CALL(cudaMalloc((void**)&d_cKeys, (bound) * sizeof(Vector)));
  MergePairs(d_aKeys, d_aVals, tThreads1, d_bKeys, d_bVals, tThreads2, d_cKeys, d_keys_sa, context);

CUDA_SAFE_CALL(cudaFree(d_aKeys));
CUDA_SAFE_CALL(cudaFree(d_bKeys));
CUDA_SAFE_CALL(cudaFree(d_cKeys));
CUDA_SAFE_CALL(cudaFree(d_aVals));

_SafeDeleteArray(unique);

}


