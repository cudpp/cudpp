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

#define DIST_OPTION 1

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
  uint msb_shift = 32 - LOG_BUCKETS;
  bucketId = item>>msb_shift;//item/elsPerBucket;

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
  uint msb_shift = 32 - ceil(log2(NUM_BUCKETS * 1.0f));
  uint bucketId = item>>msb_shift; //item/elsPerBucket;

  // Loading all warp indices regarding to each bucket into the shared memory:
  if(threadIdx.x < NUM_BUCKETS)
  {
    scratchPad[threadIdx.x] = binOffsets[threadIdx.x * gridDim.x + blockIdx.x];
    scanBlock[threadIdx.x] = binOffsets[NUM_BUCKETS*(gridDim.x + blockIdx.x) + threadIdx.x];
  }
  __syncthreads();

  // writing back the results:
  output[scratchPad[bucketId] + threadIdx.x - scanBlock[bucketId]] = item;
//  if (scratchPad[bucketId] + threadIdx.x - scanBlock[bucketId] == 8)
//    printf("blockidx %u ndx %u ITEM %u\n", blockIdx.x, index, item);
}
//========================================
template<uint NUM_W, uint NUM_B, uint LOG_B, uint DEPTH>
__global__ void histogram_warp(uint* input, uint* bin, uint numElements)
{
  // this kernel just computes warp-level histograms for the pre-scan stage
  // can be used for both direct MS and warp-level MS

  uint  index = threadIdx.x + blockIdx.x * blockDim.x;
  // if(index > numElements) return;

  __shared__ uint scratchPad[NUM_B * NUM_W * DEPTH];

  uint  laneId = threadIdx.x & 0x1F;
  uint  warpId = threadIdx.x >> 5;
 // uint  elsPerBucket = (numElements+NUM_B-1)/NUM_B;
  uint  myInput[DEPTH];
  uint  binCounter[DEPTH];  // results of histograms

  // === Histogram and local index computation
  #pragma unroll
  for(int kk = 0; kk<DEPTH; kk++){
  // for(int kk = 0; kk<DEPTH && (index + kk * gridDim.x * blockDim.x) < numElements; kk++){
    myInput[kk] = input[index + kk * gridDim.x * blockDim.x];
    uint myBucket = myInput[kk] >> (32 - LOG_B); //myInput[kk]/elsPerBucket;
    uint myHisto = 0xFFFFFFFF;
    uint bit = myBucket;
    uint rx_buffer;
    // Computing the histogram and local indices:
    #pragma unroll
    for(int i = 0; i<LOG_B; i++)
    {
      rx_buffer = __ballot(bit & 0x01);
      myHisto = myHisto & (((laneId >> i) & 0x01)?rx_buffer:(0xFFFFFFFF ^ rx_buffer));
      bit >>= 1;
    }
    binCounter[kk] = __popc(myHisto);
    // storing the results into the shared memory
    if(laneId < NUM_B)
    {
      // stored in this hierarchy: Bucket -> Roll -> Warp
      scratchPad[laneId*NUM_W*DEPTH + (kk * NUM_W) + warpId] = binCounter[kk];
    }
  }
  __syncthreads();

  // === storing histogram results from shared memory into global memory:
  uint tid = threadIdx.x;
  #pragma unroll
  while(tid < (NUM_B * NUM_W * DEPTH))
  {
    // storing histogram results:
    uint whatBin = tid / (NUM_W * DEPTH);
    uint whatWarp = tid % (NUM_W * DEPTH);
    uint whatRoll = whatWarp / NUM_W;
    whatWarp = whatWarp % NUM_W;
    // hierarchy in global memory: Bucket -> Roll -> block -> warp
    uint finalIndex = whatRoll*gridDim.x*NUM_W + (blockIdx.x * NUM_W) + whatWarp;
    bin[whatBin * NUM_W * DEPTH * gridDim.x + finalIndex] = scratchPad[tid];
    // updating tid
    tid += blockDim.x;
  }
}
//==============================================
template<uint NUM_W, uint NUM_B, uint LOG_B, uint DEPTH>
__global__ void split_WMS(uint* key_input, uint* warpOffsets, uint* key_output,
    uint numElements) {
  // Warp-level MS, post-scan stage:
  // Histogram and local warp indices are recomputed. Keys are then reordered in shared memory and results are then stored into global memory.
  uint  index = threadIdx.x + blockIdx.x * blockDim.x;

  extern __shared__ uint scratchPad[];
  uint* warp_offsets_smem = scratchPad;
  uint* keys_ms_smem = &warp_offsets_smem[NUM_B * NUM_W * DEPTH];

  uint  laneId = threadIdx.x & 0x1F;
  uint  warpId = threadIdx.x >> 5;
  //uint  elsPerBucket = (numElements+NUM_B-1)/NUM_B;
  uint  myInput[DEPTH];
  uint  myNewIndex[DEPTH];  // warp-level indices
  uint  binCounter[DEPTH];  // results of histograms
  uint  scan_histo[DEPTH];

  // === Histogram and local index computation
  #pragma unroll
  for(int kk = 0; kk<DEPTH; kk++){
    myInput[kk] = key_input[index + kk * gridDim.x * blockDim.x];
    uint myBucket = myInput[kk] >> (32 - LOG_B);//myInput[kk]/elsPerBucket;
    uint myMask = 0xFFFFFFFF;
    uint myHisto = 0xFFFFFFFF;
    uint bit = myBucket;
    uint rx_buffer;
    // Computing the histogram and local indices:
    #pragma unroll
    for(int i = 0; i<LOG_B; i++)
    {
      rx_buffer = __ballot(bit & 0x01);
      myMask  = myMask  & ((bit & 0x01)?rx_buffer:(0xFFFFFFFF ^ rx_buffer));
      myHisto = myHisto & (((laneId >> i) & 0x01)?rx_buffer:(0xFFFFFFFF ^ rx_buffer));
      bit >>= 1;
    }
    // writing back the local masks:
    binCounter[kk] = __popc(myHisto);
    uint n;
    scan_histo[kk] = binCounter[kk];
    #pragma unroll
    for(int i = 1; i<=(1<<LOG_B); i<<=1)
    {
      n = __shfl_up(scan_histo[kk], i, 32);
      if(laneId >= i)
        scan_histo[kk] += n;
    }
    scan_histo[kk] -= binCounter[kk]; //making it exclusive scan.

    // finding its new index within the warp:
    myNewIndex[kk]  = __popc(myMask & (0xFFFFFFFF >> (31-laneId))) - 1;
    myNewIndex[kk] += __shfl(scan_histo[kk], myBucket, 32);
  }

  // ===== Storing the results from global memory:
  // TODO: should make this more general, currently doesn't work for large DEPTH and NUM_BUCKETS
  uint tid = threadIdx.x;
  while(tid < NUM_B*NUM_W*DEPTH)
  {
    uint whatBin = threadIdx.x / (NUM_W * DEPTH);
    uint whatWarp = threadIdx.x % (NUM_W * DEPTH);
    uint whatRoll = whatWarp / NUM_W;
    whatWarp = whatWarp % NUM_W;
    warp_offsets_smem[threadIdx.x] = warpOffsets[(whatBin * DEPTH + whatRoll) * NUM_W * gridDim.x + (blockIdx.x * NUM_W) + whatWarp];
    tid += blockDim.x;
  }

  // Reordering key elements in shared memory:
  #pragma unroll
  for(int kk = 0; kk<DEPTH; kk++)
    keys_ms_smem[threadIdx.x + kk * blockDim.x - laneId + myNewIndex[kk]] = myInput[kk];
  __syncthreads();

  #pragma unroll
  for(int kk = 0; kk<DEPTH; kk++){
    uint myNewKey = keys_ms_smem[threadIdx.x + kk * blockDim.x];
    uint myNewBucket = myNewKey >> (32 - LOG_B);//myNewKey/elsPerBucket;
    uint finalIndex = warp_offsets_smem[NUM_W * DEPTH * myNewBucket + kk * NUM_W + warpId] + laneId;
    finalIndex -= __shfl(scan_histo[kk], myNewBucket, 32);
    // printf("thread %d, finalIndex = %d\n", threadIdx.x, finalIndex);
    key_output[finalIndex] = myNewKey;
  }
}
//====================================
template<uint NUM_W, uint NUM_B, uint LOG_B, uint DEPTH>
__global__ void split_WMS_pairs(uint* key_input, uint* value_input,
    uint* warpOffsets, uint* key_output, uint* value_output, uint numElements) {
  // Warp-level MS, post-scan stage:
  // Histogram and local warp indices are recomputed. Keys are then reordered in shared memory and results are then stored into global memory.
  uint  index = threadIdx.x + blockIdx.x * blockDim.x;

  extern __shared__ uint scratchPad[];
  uint* warp_offsets_smem = scratchPad;
  uint* keys_ms_smem = &warp_offsets_smem[NUM_B * NUM_W * DEPTH];
  uint* values_ms_smem = &keys_ms_smem[32 * NUM_W * DEPTH];

  uint  laneId = threadIdx.x & 0x1F;
  uint  warpId = threadIdx.x >> 5;
 // uint  elsPerBucket = (numElements+NUM_B-1)/NUM_B;
  uint  myInput[DEPTH];
  uint  myValue[DEPTH];
  uint  myNewIndex[DEPTH];  // warp-level indices
  uint  binCounter[DEPTH];  // results of histograms
  uint  scan_histo[DEPTH];

  // === Histogram and local index computation
  #pragma unroll
  for(int kk = 0; kk<DEPTH; kk++){
    myInput[kk] = key_input[index + kk * gridDim.x * blockDim.x];
    myValue[kk] = value_input[index + kk * gridDim.x * blockDim.x];
    uint myBucket = myInput[kk] >> (32 - LOG_B);
    uint myMask = 0xFFFFFFFF;
    uint myHisto = 0xFFFFFFFF;
    uint bit = myBucket;
    uint rx_buffer;
    // Computing the histogram and local indices:
    #pragma unroll
    for(int i = 0; i<LOG_B; i++)
    {
      rx_buffer = __ballot(bit & 0x01);
      myMask  = myMask  & ((bit & 0x01)?rx_buffer:(0xFFFFFFFF ^ rx_buffer));
      myHisto = myHisto & (((laneId >> i) & 0x01)?rx_buffer:(0xFFFFFFFF ^ rx_buffer));
      bit >>= 1;
    }
    // writing back the local masks:
    binCounter[kk] = __popc(myHisto);
    uint n;
    scan_histo[kk] = binCounter[kk];
    #pragma unroll
    for(int i = 1; i<=(1<<LOG_B); i<<=1)
    {
      n = __shfl_up(scan_histo[kk], i, 32);
      if(laneId >= i)
        scan_histo[kk] += n;
    }
    scan_histo[kk] -= binCounter[kk]; //making it exclusive scan.

    // finding its new index within the warp:
    myNewIndex[kk]  = __popc(myMask & (0xFFFFFFFF >> (31-laneId))) - 1;
    myNewIndex[kk] += __shfl(scan_histo[kk], myBucket, 32);
  }

  // ===== Storing the results from global memory:
  uint tid = threadIdx.x;
  while(tid < NUM_B*NUM_W*DEPTH)
  {
    uint whatBin = threadIdx.x / (NUM_W * DEPTH);
    uint whatWarp = threadIdx.x % (NUM_W * DEPTH);
    uint whatRoll = whatWarp / NUM_W;
    whatWarp = whatWarp % NUM_W;
    warp_offsets_smem[threadIdx.x] = warpOffsets[(whatBin * DEPTH + whatRoll) * NUM_W * gridDim.x + (blockIdx.x * NUM_W) + whatWarp];
    tid += blockDim.x;
  }

  // Reordering key elements in shared memory:
  #pragma unroll
  for(int kk = 0; kk<DEPTH; kk++){
    keys_ms_smem[threadIdx.x + kk * blockDim.x - laneId + myNewIndex[kk]] = myInput[kk];
    values_ms_smem[threadIdx.x + kk * blockDim.x - laneId + myNewIndex[kk]] = myValue[kk];
  }
  __syncthreads();

  #pragma unroll
  for(int kk = 0; kk<DEPTH; kk++){
    uint myNewKey = keys_ms_smem[threadIdx.x + kk * blockDim.x];
    uint myNewValue = values_ms_smem[threadIdx.x + kk * blockDim.x];
    uint myNewBucket = myNewKey >> (32 - LOG_B);
    uint finalIndex = warp_offsets_smem[NUM_W * DEPTH * myNewBucket + kk * NUM_W + warpId] + laneId;
    finalIndex -= __shfl(scan_histo[kk], myNewBucket, 32);

    key_output[finalIndex] = myNewKey;
    value_output[finalIndex] = myNewValue;
  }
}
//=====================================
__global__ void histogram_block(uint* input, uint* bin, uint numElements,
    uint numBuckets, uint numWarps, uint depth) {
  // this kernel just computes warp-level histograms for the pre-scan stage
  // can be used for both direct MS and warp-level MS

  uint  index = threadIdx.x + blockIdx.x * blockDim.x;
  uint logBuckets = ceil(log2((float) numBuckets));
  uint logWarps = ceil(log2((float) numWarps));
  // if(index > numElements) return;

  //__shared__ uint scratchPad[NUM_B * NUM_W * DEPTH];
  extern __shared__ uint scratchPad[];

  uint  laneId = threadIdx.x & 0x1F;
  uint  warpId = threadIdx.x >> 5;

  #if DIST_OPTION == 0 // uniform distribution
    uint  elsPerBucket = (numElements+numBuckets-1)/numBuckets;
  #endif

//  uint  myInput[DEPTH];
//  uint  binCounter[DEPTH];  // results of histograms

  // === Histogram and local index computation
  #pragma unroll
  for(int kk = 0; kk<depth; kk++){
    uint  myInput;
    uint  binCounter;  // results of histograms
  // for(int kk = 0; kk<DEPTH && (index + kk * gridDim.x * blockDim.x) < numElements; kk++){
    myInput = input[index + kk * gridDim.x * blockDim.x];
    uint myBucket;
    #if DIST_OPTION == 0 // uniform distribution
      myBucket = myInput/elsPerBucket;
    #elif DIST_OPTION >= 1 // Binomial distribution
      myBucket = myInput >> (32 - logBuckets);
    #endif
    uint myHisto = 0xFFFFFFFF;
    uint bit = myBucket;
    uint rx_buffer;
    // Computing the histogram and local indices:
    #pragma unroll
    for(int i = 0; i<logBuckets; i++)
    {
      rx_buffer = __ballot(bit & 0x01);
      myHisto = myHisto & (((laneId >> i) & 0x01)?rx_buffer:(0xFFFFFFFF ^ rx_buffer));
      bit >>= 1;
    }
    binCounter = __popc(myHisto);
    // storing the results into the shared memory
    if(laneId < numBuckets)
    {
      // // stored in this hierarchy: Bucket -> Roll -> Warp
      // scratchPad[laneId*NUM_W*DEPTH + (kk * NUM_W) + warpId] = binCounter[kk];
      // Hierarchy: Roll -> warp -> bucket
      scratchPad[kk * numWarps * numBuckets + warpId * numBuckets + laneId] = binCounter;
    }
  }
  __syncthreads();
  // if(threadIdx.x < NUM_B * NUM_W * DEPTH)
  //  printf("block = %d, smem[%d] = %d\n", blockIdx.x, threadIdx.x, scratchPad[threadIdx.x]);

  // if (laneId < NUM_B)
  //  printf("block = %d, warp %d, bucket = %d, histo = %d\n", blockIdx.x, warpId, laneId, scratchPad[laneId + warpId * NUM_B]);
  // === storing histogram results from shared memory into global memory:
  #pragma unroll
  for(int i = 1; i <= logWarps; i++)
  {
    // Performing reduction over all elements:
    #pragma unroll
    for(int kk = 0; kk<depth; kk++)
    {
      // uint offset = kk * NUM_B * NUM_W;
      if((warpId & ((1<<i)-1)) == 0)
      {
        if(laneId < numBuckets){
          // printf("block = %d, warpId = %d, 1 = %d, 2 = %d\n", blockIdx.x, warpId, scratchPad[laneId + warpId * NUM_B + offset], scratchPad[laneId + (warpId + (1<<(i-1)))*NUM_B + offset]);
          scratchPad[laneId + warpId * numBuckets + kk * numWarps * numBuckets] += scratchPad[laneId + (warpId + (1<<(i-1)))*numBuckets + kk * numWarps * numBuckets];
        }
      }
    }
    __syncthreads();
  }
  // if(threadIdx.x < NUM_B * NUM_W * DEPTH)
  //  printf("after: block = %d, smem[%d] = %d\n", blockIdx.x, threadIdx.x, scratchPad[threadIdx.x]);

  // #pragma unroll
  // for(int kk = 0; kk<DEPTH; kk++)
  // {
  //  uint offset = kk * NUM_B * NUM_W;
  //  // Performing reduction over all elements:
  //  #pragma unroll
  //  for(int i = 1; i <= LOG_W; i++)
  //  {
  //    if((warpId & ((1<<i)-1)) == 0)
  //    {
  //      if(laneId < NUM_B){
  //        // printf("block = %d, warpId = %d, 1 = %d, 2 = %d\n", blockIdx.x, warpId, scratchPad[laneId + warpId * NUM_B + offset], scratchPad[laneId + (warpId + (1<<(i-1)))*NUM_B + offset]);
  //        scratchPad[laneId + warpId * NUM_B + offset] += scratchPad[laneId + (warpId + (1<<(i-1)))*NUM_B + offset];
  //      }
  //    }
  //    __syncthreads();
  //  }
  // }
  // if(threadIdx.x < NUM_B * NUM_W * DEPTH)
  //  printf("block = %d, smem[%d] = %d\n", blockIdx.x, threadIdx.x, scratchPad[threadIdx.x]);
  // storing the results into global memory:
  // each warp store results of each roll
  if(numWarps >= depth)
  {
    // Global memory hierarchy: Bucket -> Roll -> block
    if((laneId < numBuckets) && (warpId < depth))
      bin[laneId * gridDim.x * depth + warpId * gridDim.x + blockIdx.x] = scratchPad[laneId + warpId * numBuckets * numWarps];
  }
}

template<uint NUM_W, uint LOG_W, uint NUM_B, uint LOG_B, uint DEPTH>
__global__ void split_BMS2(uint* key_input, uint* blockOffsets, uint* key_output, uint numElements)
{
  // Block-level MS, post-scan stage:
  // Histogram and local block indices are recomputed. Keys are then reordered in shared memory and results are then stored into global memory.

  uint  index = threadIdx.x + blockIdx.x * blockDim.x;

  //extern __shared__ uint scratchPad[];
  __shared__ uint scratchPad[2 * NUM_B * DEPTH + 32 * NUM_W * DEPTH + NUM_B * NUM_W * DEPTH];

  uint* block_offsets_smem = scratchPad;
  uint* warp_offsets_smem = &block_offsets_smem[NUM_B * DEPTH];
  uint* keys_ms_smem = &warp_offsets_smem[NUM_B * DEPTH];
  uint* scan_histo_smem = &keys_ms_smem[32 * NUM_W * DEPTH];

  uint  laneId = threadIdx.x & 0x1F;
  uint  warpId = threadIdx.x >> 5;
  #if DIST_OPTION == 0 // uniform distribution
  uint  elsPerBucket = (numElements+NUM_B-1)/NUM_B;
  #endif

  // === Histogram and local index computation
  #pragma unroll
  for(int kk = 0; kk<DEPTH; kk++){
    uint  myInput;
    uint  scan_temp;
    uint  binCounter;

    myInput = key_input[index + kk * gridDim.x * blockDim.x];
    uint myBucket;
    #if DIST_OPTION == 0 // uniform distribution
      myBucket = myInput/elsPerBucket;
    #elif DIST_OPTION >= 1 // Binomial distribution
      myBucket = myInput >> (32 - LOG_B);
    #endif
    uint myMask = 0xFFFFFFFF;
    uint myHisto = 0xFFFFFFFF;
    uint bit = myBucket;
    uint rx_buffer;
    // Computing the histogram and local indices:
    #pragma unroll
    for(int i = 0; i<LOG_B; i++)
    {
      rx_buffer = __ballot(bit & 0x01);
      myMask  = myMask  & ((bit & 0x01)?rx_buffer:(0xFFFFFFFF ^ rx_buffer));
      myHisto = myHisto & (((laneId >> i) & 0x01)?rx_buffer:(0xFFFFFFFF ^ rx_buffer));
      bit >>= 1;
    }
    binCounter = __popc(myHisto);
    // Hieararchy: Roll -> warp -> bucket
    if(laneId < NUM_B)
      scan_histo_smem[laneId + warpId * NUM_B + kk * NUM_B * NUM_W] = binCounter;
    __syncthreads();

    // computing block-wise scan over buckets:
    scan_temp = binCounter;
    for(int i = 1; i<(1<<LOG_W) ; i<<=1)
    {
      if(laneId < NUM_B)
        scan_temp += ((warpId >= i)?scan_histo_smem[kk * NUM_B * NUM_W + (warpId-i)*NUM_B + laneId]:0);
      __syncthreads();
      if(laneId < NUM_B)
        scan_histo_smem[kk * NUM_B * NUM_W + warpId * NUM_B + laneId] = scan_temp;
      __syncthreads();
    }

    // Computing block-level indices:
    scan_temp -= binCounter; // exclusive scan
    uint myLocalBlockIndex = __shfl(scan_temp, myBucket, 32);
    myLocalBlockIndex += __popc(myMask & (0xFFFFFFFF >> (31-laneId))) - 1;

    // Computing warp-level offsets within each block:
    uint block_scan;
    if(warpId == (NUM_W-1))
    {
      block_scan = scan_temp + binCounter;
      uint n;
      #pragma unroll
      for(int i = 1; i<=(1<<LOG_B); i<<=1)
      {
        n = __shfl_up(block_scan, i, 32);
        if(laneId >= i)
          block_scan += n;
      }
      scan_temp += binCounter;
      block_scan -= scan_temp;
      if(laneId < NUM_B){
        warp_offsets_smem[laneId + kk * NUM_B] = block_scan;
      }
    }
    __syncthreads();

    uint myNewBlockIndex = warp_offsets_smem[myBucket + kk * NUM_B] + myLocalBlockIndex;
    // block-level reordering in shared memory
    keys_ms_smem[myNewBlockIndex + kk * blockDim.x] = myInput;

    // loading block offsets from global memory
    if((laneId < NUM_B) && (warpId == 0)){
      block_offsets_smem[laneId + kk * NUM_B] = blockOffsets[laneId * gridDim.x * DEPTH + kk * gridDim.x + blockIdx.x];
      // block_offsets_smem[laneId + kk * NUM_B] = blockOffsets[kk * gridDim.x * NUM_B + laneId * gridDim.x + blockIdx.x];
    }
    __syncthreads();
  }

  // Final position computaiton and storing the results:
  #pragma unroll
  for(int kk = 0; kk<DEPTH; kk++){
    uint myNewKey = keys_ms_smem[threadIdx.x + kk * blockDim.x];
    #if DIST_OPTION == 0 // uniform distribution
      uint myNewBucket = myNewKey/elsPerBucket;
    #elif DIST_OPTION >= 1 // Binomial distribution
      uint myNewBucket = myNewKey >> (32 - LOG_B);
    #endif
    uint finalIndex = block_offsets_smem[NUM_B * kk + myNewBucket] + threadIdx.x;
    finalIndex -= warp_offsets_smem[myNewBucket + kk * NUM_B];
    // global memory write:
    // if(blockIdx.x == 0)
    //  printf("blockIdx.x = %d, thread = %d, finalIndex = %d\n", blockIdx.x, threadIdx.x, finalIndex);
    // if(finalIndex < 128)
    //  printf("block = %d, thread = %d, bucket = %d, item = %d, finalIndex = %d, block_offsets_smem = %d, warp_offset_smem = %d\n", blockIdx.x, threadIdx.x, myNewBucket, myNewKey, finalIndex, block_offsets_smem[NUM_B * kk + myNewBucket], warp_offsets_smem[myNewBucket + kk * NUM_B]);
    key_output[finalIndex] = myNewKey;
  }
}

//===================================
__global__ void split_BMS(uint* key_input, uint* blockOffsets, uint* key_output,
    uint numElements, uint numBuckets, uint numWarps, uint depth) {
  // Block-level MS, post-scan stage:
  // Histogram and local block indices are recomputed. Keys are then reordered in shared memory and results are then stored into global memory.

  uint  index = threadIdx.x + blockIdx.x * blockDim.x;
  uint logBuckets = ceil(log2((float) numBuckets));
  uint logWarps = ceil(log2((float) numWarps));

  extern __shared__ uint scratchPad[];
  uint* block_offsets_smem = scratchPad;
  uint* warp_offsets_smem = &block_offsets_smem[numBuckets * depth];
  uint* keys_ms_smem = &warp_offsets_smem[numBuckets * depth];
  uint* scan_histo_smem = &keys_ms_smem[32 * numWarps * depth];

  uint  laneId = threadIdx.x & 0x1F;
  uint  warpId = threadIdx.x >> 5;
  #if DIST_OPTION == 0 // uniform distribution
  uint  elsPerBucket = (numElements+numBuckets-1)/numBuckets;
  #endif

  // === Histogram and local index computation
  #pragma unroll
  for(int kk = 0; kk<depth; kk++){
    uint  myInput;
    uint  binCounter;  // results of histograms
    uint  scan_temp;

    myInput = key_input[index + kk * gridDim.x * blockDim.x];
    uint myBucket;
    #if DIST_OPTION == 0 // uniform distribution
      myBucket = myInput/elsPerBucket;
    #elif DIST_OPTION >= 1 // Binomial distribution
      myBucket = myInput >> (32 - logBuckets);
    #endif
    uint myMask = 0xFFFFFFFF;
    uint myHisto = 0xFFFFFFFF;
    uint bit = myBucket;
    uint rx_buffer;
    // Computing the histogram and local indices:
    #pragma unroll
    for(int i = 0; i<logBuckets; i++)
    {
      rx_buffer = __ballot(bit & 0x01);
      myMask  = myMask  & ((bit & 0x01)?rx_buffer:(0xFFFFFFFF ^ rx_buffer));
      myHisto = myHisto & (((laneId >> i) & 0x01)?rx_buffer:(0xFFFFFFFF ^ rx_buffer));
      bit >>= 1;
    }
    binCounter = __popc(myHisto);
    // Hieararchy: Roll -> warp -> bucket
    if(laneId < numBuckets)
      scan_histo_smem[laneId + warpId * numBuckets + kk * numBuckets * numWarps] = binCounter;
    __syncthreads();

    // computing block-wise scan over buckets:
    scan_temp = binCounter;
    for(int i = 1; i<(1<<logWarps) ; i<<=1)
    {
      if(laneId < numBuckets)
        scan_temp += ((warpId >= i)?scan_histo_smem[kk * numBuckets * numWarps + (warpId-i)*numBuckets + laneId]:0);
      __syncthreads();
      if(laneId < numBuckets)
        scan_histo_smem[kk * numBuckets * numWarps + warpId * numBuckets + laneId] = scan_temp;
      __syncthreads();
    }

    // Computing block-level indices:
    scan_temp -= binCounter; // exclusive scan
    uint myLocalBlockIndex = __shfl(scan_temp, myBucket, 32);
    myLocalBlockIndex += __popc(myMask & (0xFFFFFFFF >> (31-laneId))) - 1;

    // Computing warp-level offsets within each block:
    uint block_scan;
    if(warpId == (numWarps-1))
    {
      block_scan = scan_temp + binCounter;
      uint n;
      #pragma unroll
      for(int i = 1; i<=(1<<logBuckets); i<<=1)
      {
        n = __shfl_up(block_scan, i, 32);
        if(laneId >= i)
          block_scan += n;
      }
      scan_temp += binCounter;
      block_scan -= scan_temp;
      if(laneId < numBuckets){
        warp_offsets_smem[laneId + kk * numBuckets] = block_scan;
      }
    }
    __syncthreads();

    uint myNewBlockIndex = warp_offsets_smem[myBucket + kk * numBuckets] + myLocalBlockIndex;
    // block-level reordering in shared memory
    keys_ms_smem[myNewBlockIndex + kk * blockDim.x] = myInput;

    // loading block offsets from global memory
    if((laneId < numBuckets) && (warpId == 0)){
      block_offsets_smem[laneId + kk * numBuckets] = blockOffsets[laneId * gridDim.x * depth + kk * gridDim.x + blockIdx.x];
      // block_offsets_smem[laneId + kk * NUM_B] = blockOffsets[kk * gridDim.x * NUM_B + laneId * gridDim.x + blockIdx.x];
    }
    __syncthreads();
  }

  // Final position computaiton and storing the results:
  #pragma unroll
  for(int kk = 0; kk<depth; kk++){
    uint myNewKey = keys_ms_smem[threadIdx.x + kk * blockDim.x];
    #if DIST_OPTION == 0 // uniform distribution
      uint myNewBucket = myNewKey/elsPerBucket;
    #elif DIST_OPTION >= 1 // Binomial distribution
      uint myNewBucket = myNewKey >> (32 - logBuckets);
    #endif
    uint finalIndex = block_offsets_smem[numBuckets * kk + myNewBucket] + threadIdx.x;
    finalIndex -= warp_offsets_smem[myNewBucket + kk * numBuckets];
    // global memory write:
    // if(blockIdx.x == 0)
    //  printf("blockIdx.x = %d, thread = %d, finalIndex = %d\n", blockIdx.x, threadIdx.x, finalIndex);
    // if(finalIndex < 128)
    //  printf("block = %d, thread = %d, bucket = %d, item = %d, finalIndex = %d, block_offsets_smem = %d, warp_offset_smem = %d\n", blockIdx.x, threadIdx.x, myNewBucket, myNewKey, finalIndex, block_offsets_smem[NUM_B * kk + myNewBucket], warp_offsets_smem[myNewBucket + kk * NUM_B]);
    key_output[finalIndex] = myNewKey;
  }
}
//==========================================
__global__ void markBins_general(uint* d_mark, uint* d_elements, uint numElements, uint numBuckets)
{

  unsigned int myId = threadIdx.x + blockIdx.x*blockDim.x;
  unsigned int offset = blockDim.x*gridDim.x;
  unsigned int logBuckets = ceil(log2((float)numBuckets));
  #if DIST_OPTION == 0
    unsigned int elsPerBucket = (numElements + numBuckets - 1)/numBuckets;
  #endif

  for(int i = myId; i < numElements; i+=offset)
  {
    unsigned int myVal = d_elements[i];
    #if DIST_OPTION == 0
      unsigned int myBucket = myVal/elsPerBucket;
    #elif DIST_OPTION >= 1
      unsigned int myBucket = myVal >> (32 - logBuckets);
    #endif
    d_mark[i] = myBucket;
  }
}
//===========================================
__global__ void packingKeyValuePairs(uint64* packed, uint* input_key,
    uint* input_value, uint numElements) {
  uint tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid > numElements)
    return;

  uint myKey = input_key[tid];
  uint myValue = input_value[tid];
  // if(myKey == 130767)
  //  printf("1) thread %d, myKey = %d, myValue = %d\n", tid, myKey, myValue);
  // putting the key as the more significant 32 bits.
  uint64 output = (static_cast<uint64>(myKey) << 32)
      + static_cast<uint>(myValue);
  packed[tid] = output;
}
//===========================================
__global__ void unpackingKeyValuePairs(uint64* packed, uint* out_key,
    uint* out_value, uint numElements) {
  uint tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid > numElements)
    return;

  uint64 myPacked = packed[tid];
  out_value[tid] = static_cast<uint>(myPacked & 0x00000000FFFFFFFF);
  out_key[tid] = static_cast<uint>(myPacked >> 32);
  // if((out_key[tid] > 65536) && (tid < 65536))
  // if(tid == 100)
  //  printf("thread %d: (%d, %d) key-value\n", tid, out_key[tid], out_value[tid]);
}
//===========================================


/** @} */ // end Multisplit functions
/** @} */ // end cudpp_kernel

