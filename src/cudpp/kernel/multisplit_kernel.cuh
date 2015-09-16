// -------------------------------------------------------------
// CUDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision$
// $Date$
// ------------------------------------------------------------- 
// This source code is distributed under the terms of license.txt 
// in the root directory of this source distribution.
// ------------------------------------------------------------- 

#include "cudpp_multisplit.h"
#include <cudpp_globals.h>
#include <cudpp_util.h>
#include "sharedmem.h"

/**
 * @file
 * multisplit_kernel.cu
 *   
 * @brief CUDPP kernel-level multisplit routines
 */

/** \addtogroup cudpp_kernel
  * @{
 */

/** @name Multisplit Functions
 * @{
 */



//======================================
template<uint NUM_WARPS, uint NUM_BUCKETS, uint LOG_BUCKETS, uint LOG_WARPS>
__global__ void histogramBallot_Mode13_large(uint* input, uint* bin, uint numElements)
{
  // Block level MS: with more buckets than 32
  // Computing the histogram and local index within each block and storing them in the corresponding localIndex array:
  // we also re-arrange both input elements and their index into the global memory.
  // In this version we remove the localIndex but save two different versions in the bin vector.
  // bin is an array of histograms stored in the following way:
  //                B0                    B1
  //        |w0 + w1 + w2 ... | | w0 + w1 + w2 .... | ... | w0 + w1 + w2 +... |
  // i.e.   sum of the items within each bucket is stored
  // in the shared memory we store elements differently:
  //                w0                w1                    w...
  //        |B0, B1, B2, ...|  |B0, B1, B2, ...| ... |B0, B1, B2, ...|
  // LOG_BUCKETS = ceil(log2(NUM_BUCKETS))

  typedef cub::BlockScan<uint, NUM_BUCKETS> BlockScanT;

  uint  index = threadIdx.x + blockIdx.x * blockDim.x;
  if(index > numElements) return;

  __shared__ union{
    uint temp_storage[NUM_BUCKETS * NUM_WARPS + 32 * NUM_WARPS];
    typename BlockScanT::TempStorage  temp_cub; // being used in CUB's block scan
  }shm;

  uint  *scratchPad = &((shm.temp_storage)[0]);
  uint  *blockMS = &((shm.temp_storage)[NUM_BUCKETS * NUM_WARPS]);

  uint  laneId = threadIdx.x & 0x1F;
  uint  warpId = threadIdx.x >> 5;
  uint  elsPerBucket = (numElements+NUM_BUCKETS-1)/NUM_BUCKETS;
  const uint num_roll = (NUM_BUCKETS + 31)/32; // number of buckets dedicated to each thread (at most)
  uint  bucketId;
  uint  myMask = 0xFFFFFFFF;
  uint  myHisto[num_roll]; // each thread is responsible for multiple histogram values
  uint  scan_temp[num_roll];
  uint  bit;
  uint  rx_buffer;
  uint  item = input[index];
  bucketId = item/elsPerBucket;

  // computing warp-level histogram:
  #pragma unroll
  for(int i = 0; i<num_roll; i++)
    myHisto[i] = 0xFFFFFFFF;

  bit = bucketId;

  #pragma unroll
  for(int i = 0; i<LOG_BUCKETS; i++)
  {
    rx_buffer = __ballot(bit & 0x01);
    myMask  = myMask  & ((bit & 0x01)?rx_buffer:(0xFFFFFFFF ^ rx_buffer));
    #pragma unroll
    for(int k = 0; k<num_roll; k++){
      myHisto[k] = myHisto[k] & ((((laneId + 32*k) >> i) & 0x01)?rx_buffer:(0xFFFFFFFF ^ rx_buffer));
    }
    bit >>= 1;
  }
  // copying back the results into the scratchPad:
  #pragma unroll
  for(int k = 0; k<num_roll; k++)
  {
    myHisto[k] = __popc(myHisto[k]);
    scan_temp[k] = myHisto[k];
    if((laneId + (k<<5)) < NUM_BUCKETS)
    {
      scratchPad[laneId + (k<<5) + warpId*NUM_BUCKETS] = myHisto[k];
    }
  }
  __syncthreads();

  for(int i = 1; i<(1<<LOG_WARPS) ; i<<=1)
  {
    #pragma unroll
    for(int k = 0; k<num_roll; k++){
      if((laneId+(k<<5)) < NUM_BUCKETS)
        scan_temp[k] += ((warpId >= i)?scratchPad[(warpId-i)*NUM_BUCKETS + (k<<5) + laneId]:0);
    }
    __syncthreads();
    #pragma unroll
    for(int kk = 0; kk < num_roll; kk++){
      if((laneId + (kk<<5)) < NUM_BUCKETS)
        scratchPad[warpId*NUM_BUCKETS + (kk<<5) + laneId] = scan_temp[kk];
    }
    __syncthreads();
  }

  // First loading this results into the global memory so that we can use it again by CUB:
  uint block_offset = 0;
  if(threadIdx.x < NUM_BUCKETS)
  {
    block_offset = scratchPad[(NUM_WARPS-1)*NUM_BUCKETS + threadIdx.x];
    bin[(threadIdx.x) * gridDim.x + blockIdx.x] = block_offset;
  }
  __syncthreads();

  // computing block level exlusive scan for having right offsets using CUB's block scan:
  uint temp_results = 0;
  BlockScanT(shm.temp_cub).ExclusiveSum(block_offset, temp_results);
  __syncthreads();
  if(threadIdx.x < NUM_BUCKETS)
  {
    scratchPad[threadIdx.x] = temp_results;
  }
  __syncthreads();

  #pragma unroll
  for(int k = 0; k<num_roll; k++)
    scan_temp[k] -= myHisto[k];

  // we read all those registers because we do not beforehand which ones we need:
  uint myLocalBlockIndex[num_roll];
  #pragma unroll
  for(int k = 0; k<num_roll; k++)
    myLocalBlockIndex[k] = __shfl(scan_temp[k], (bucketId & 0x1F), 32);

  myLocalBlockIndex[(bucketId >> 5)] += (__popc(myMask & (0xFFFFFFFF >> (31-laneId))) - 1);

  // updating the block level index:
  uint myBlockOffset = scratchPad[bucketId] + myLocalBlockIndex[bucketId >> 5];
  blockMS[myBlockOffset] = item;
  __syncthreads();

  input[index] = blockMS[threadIdx.x];

  // storing back the final offsets:
  if(threadIdx.x < NUM_BUCKETS)
  {
    bin[NUM_BUCKETS * (gridDim.x + blockIdx.x) + threadIdx.x] = scratchPad[threadIdx.x];
  }
}
//======================================
template<uint NUM_WARPS, uint NUM_BUCKETS>
__global__ void splitBallot_Mode13_large(unsigned int* input, unsigned int* binOffsets,
  unsigned int* output, unsigned int numElements)
{
  // Performing the splitting proces using the prefixed-sum histograms (binOffsets), and the
  // local warp-level masks (binMask).

  uint index = threadIdx.x + blockIdx.x * blockDim.x;

  if(index > numElements) return;

  __shared__ uint scratchPad[2 * NUM_BUCKETS];
  uint* scanBlock = &scratchPad[NUM_BUCKETS];

  uint elsPerBucket = (numElements+NUM_BUCKETS-1)/NUM_BUCKETS;
  uint item = input[index];
  // uint laneId = threadIdx.x & 0x1F;
  // uint warpId = threadIdx.x >> 5;
  uint bucketId = item/elsPerBucket;

  // Loading all warp indices regarding to each bucket into the shared memory:
  if(threadIdx.x < NUM_BUCKETS)
  {
    scratchPad[threadIdx.x] = binOffsets[threadIdx.x * gridDim.x + blockIdx.x];
    scanBlock[threadIdx.x] = binOffsets[NUM_BUCKETS*(gridDim.x + blockIdx.x) + threadIdx.x];
  }
  __syncthreads();

  // writing back the results:
  output[scratchPad[bucketId] + threadIdx.x - scanBlock[bucketId]] = item;
}


/** @} */ // end Multisplit functions
/** @} */ // end cudpp_kernel

