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

//==========================================
template<class T>
__global__ void markBins_general(uint* d_mark, uint* d_elements,
    uint numElements, uint numBuckets, T bucketMapper) {

  unsigned int myId = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int offset = blockDim.x * gridDim.x;
  unsigned int logBuckets = ceil(log2((float) numBuckets));

  for (int i = myId; i < numElements; i += offset) {
    unsigned int myVal = d_elements[i];
    unsigned int myBucket = bucketMapper(myVal);
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
}
//=============================
template<uint numWarps, uint numBuckets, uint logBuckets, uint depth, class T>
__global__ void histogram_warp_ver6(uint* input, uint* bin, uint numElements,
    T bucketMapper) {
  // The new warp-level histogram computing kernel:
  uint index = threadIdx.x + blockIdx.x * blockDim.x;
  uint laneId = threadIdx.x & 0x1F;
  uint warpId = threadIdx.x >> 5;
  //uint logBuckets = ceil(log2((float) numBuckets));
  //uint logWarps = ceil(log2((float) numWarps));

  __shared__ uint scratchPad[numBuckets * numWarps * depth];

  if (blockIdx.x == (gridDim.x - 1)) // last block, potentially may try to read invalid inputs
      {
    // === initializing the shared memory results:
    uint k = 0;
#pragma unroll
    while ((threadIdx.x + k * blockDim.x) < numBuckets * numWarps * depth) {
      scratchPad[threadIdx.x + k * blockDim.x] = 0;
      k++;
    }
    __syncthreads();

    // === computing the histogram:
    uint myInput[depth];
    uint binCounter[depth]; // results of histograms

#pragma unroll
    for (int kk = 0; kk < depth; kk++) {
      uint global_index = (index - laneId) * depth + (kk << 5);
      uint myBucket = 0;
      bool valid_input = false;

      // == reading the input only if valid:
      if ((global_index + laneId) < numElements) {
        myInput[kk] = input[global_index + laneId];
        valid_input = true;
        myBucket = bucketMapper(myInput[kk]);
      }
      uint myHisto = 0xFFFFFFFF;
      uint bit = myBucket;
      uint rx_buffer;
      uint mask = __ballot(valid_input);

      // computing histogram
#pragma unroll
      for (int i = 0; i < logBuckets; i++) {
        rx_buffer = __ballot(bit & 0x01);
        myHisto = myHisto
            & (((laneId >> i) & 0x01) ? rx_buffer : (0xFFFFFFFF ^ rx_buffer));
        bit >>= 1;
      }
      binCounter[kk] = __popc(myHisto & mask);
      // === storing the results into the shared memory
      if (laneId < numBuckets) {
        // new hierarchy: Bucket -> warp -> roll
        scratchPad[laneId * numWarps * depth + warpId * depth + kk] =
            binCounter[kk];
      }
    }
    __syncthreads();

    // === storing histogram results from shared memory into global memory:
    uint tid = threadIdx.x;
#pragma unroll
    while (tid < (numBuckets * numWarps * depth)) {
      // storing histogram results:
      uint whatBin = tid / (numWarps * depth);
      uint whatRoll = tid % (numWarps * depth);
      uint whatWarp = whatRoll / depth;
      whatRoll = whatRoll % depth;
      // new hierarchy: Bucket -> block -> warp -> roll:
      uint finalIndex = blockIdx.x * depth * numWarps + whatWarp * depth
          + whatRoll;
      bin[whatBin * numWarps * depth * gridDim.x + finalIndex] =
          scratchPad[tid];
      // updating tid
      tid += blockDim.x;
    }
  } else // all other blocks, that we are sure they are certainly processing valid inputs:
  {
    // === computing the histogram:
    uint myInput[depth];
    uint binCounter[depth]; // results of histograms

#pragma unroll
    for (int kk = 0; kk < depth; kk++) {
      uint global_index = (index - laneId) * depth + (kk << 5);
      uint myBucket = 0;
      myInput[kk] = input[global_index + laneId];
      myBucket = bucketMapper(myInput[kk]);

      uint myHisto = 0xFFFFFFFF;
      uint bit = myBucket;
      uint rx_buffer;

#pragma unroll
      for (int i = 0; i < logBuckets; i++) {
        rx_buffer = __ballot(bit & 0x01);
        myHisto = myHisto
            & (((laneId >> i) & 0x01) ? rx_buffer : (0xFFFFFFFF ^ rx_buffer));
        bit >>= 1;
      }
      binCounter[kk] = __popc(myHisto);

      // === storing the results into the shared memory
      if (laneId < numBuckets) {
        // new hierarchy: Bucket -> warp -> roll
        scratchPad[laneId * numWarps * depth + warpId * depth + kk] =
            binCounter[kk];
      }
      __syncthreads();
    }
    // === storing histogram results from shared memory into global memory:
    uint tid = threadIdx.x;
#pragma unroll
    while (tid < (numBuckets * numWarps * depth)) {
      // storing histogram results:
      uint whatBin = tid / (numWarps * depth);
      uint whatRoll = tid % (numWarps * depth);
      uint whatWarp = whatRoll / depth;
      whatRoll = whatRoll % depth;
      // new hierarchy: Bucket -> block -> warp -> roll:
      uint finalIndex = blockIdx.x * depth * numWarps + whatWarp * depth
          + whatRoll;
      bin[whatBin * numWarps * depth * gridDim.x + finalIndex] =
          scratchPad[tid];
      // updating tid
      tid += blockDim.x;
    }
  }
}
//=============================
template<uint NUM_W, uint NUM_B, uint LOG_B, uint DEPTH, class T>
__global__ void split_WMS_ver6(uint* key_input, uint* warpOffsets,
    uint* key_output, uint numElements, T bucketMapper) {
  // For this kernel, the last CTA follows a different branch than the rest
  // this kernel use different memory hierarchy: bucket -> block -> warp -> roll
  // Warp-level MS, post-scan stage:
  // Histogram and local warp indices are recomputed. Keys are then reordered in shared memory and results are then stored into global memory.
  uint index = threadIdx.x + blockIdx.x * blockDim.x;
  uint laneId = threadIdx.x & 0x1F;
  uint warpId = threadIdx.x >> 5;
  uint warp_global_offset = (index - laneId) * DEPTH;

  __shared__ uint scratchPad[NUM_B * NUM_W * DEPTH + 32 * NUM_W * DEPTH];
  uint* warp_offsets_smem = scratchPad;
  uint* keys_ms_smem = &warp_offsets_smem[NUM_B * NUM_W * DEPTH];

  // ===== Storing the results from global memory:
  // with new hierarchy: bucket -> block -> warp -> roll
  uint tid = threadIdx.x;
  while (tid < NUM_B * NUM_W * DEPTH) {
    uint whatBin = threadIdx.x / (NUM_W * DEPTH);
    uint whatRoll = threadIdx.x % (NUM_W * DEPTH);
    uint whatWarp = whatRoll / DEPTH;
    whatRoll = whatRoll % DEPTH;
    warp_offsets_smem[threadIdx.x] =
        warpOffsets[(whatBin * NUM_W * gridDim.x * DEPTH)
            + (blockIdx.x * NUM_W * DEPTH) + (whatWarp * DEPTH) + whatRoll];
    tid += blockDim.x;
  }
  if ((warp_global_offset >= numElements))
    return;

  uint myInput[DEPTH];
  uint myNewIndex[DEPTH]; // warp-level indices
  uint binCounter[DEPTH]; // results of histograms
  uint scan_histo[DEPTH];

  if (blockIdx.x == (gridDim.x - 1)) {
    // === Histogram and local index computation
#pragma unroll
    for (int kk = 0; kk < DEPTH; kk++) {
      // uint global_index = index + kk * gridDim.x * blockDim.x;
      uint global_index = warp_global_offset + (kk << 5);
      uint myBucket = 0;
      bool valid_input = false;
      if ((global_index + laneId) < numElements) {
        valid_input = true;
        myInput[kk] = key_input[global_index + laneId];
        myBucket = bucketMapper(myInput[kk]);
      }

      uint mask = __ballot(valid_input);
      uint myMask = 0xFFFFFFFF;
      uint myHisto = 0xFFFFFFFF;
      uint bit = myBucket;
      uint rx_buffer;
      // Computing the histogram and local indices:
#pragma unroll
      for (int i = 0; i < LOG_B; i++) {
        rx_buffer = __ballot(bit & 0x01);
        myMask = myMask & ((bit & 0x01) ? rx_buffer : (0xFFFFFFFF ^ rx_buffer));
        myHisto = myHisto
            & (((laneId >> i) & 0x01) ? rx_buffer : (0xFFFFFFFF ^ rx_buffer));
        bit >>= 1;
      }
      // writing back the local masks:
      binCounter[kk] = __popc(myHisto & mask);
      uint n;
      scan_histo[kk] = binCounter[kk];
#pragma unroll
      for (int i = 1; i <= (1 << LOG_B); i <<= 1) {
        n = __shfl_up(scan_histo[kk], i, 32);
        if (laneId >= i)
          scan_histo[kk] += n;
      }
      scan_histo[kk] -= binCounter[kk]; //making it exclusive scan.

      // finding its new index within the warp:
      myNewIndex[kk] = __popc(myMask & (0xFFFFFFFF >> (31 - laneId))) - 1;
      myNewIndex[kk] += __shfl(scan_histo[kk], myBucket, 32);

      myNewIndex[kk] = (valid_input) ? myNewIndex[kk] : 32; // if 32 it means that input was not valid
    }

    // Reordering key elements in shared memory:
#pragma unroll
    for (int kk = 0; kk < DEPTH; kk++) {
      keys_ms_smem[(warpId << 5) * DEPTH + (kk << 5)
          + ((myNewIndex[kk] < 32) ? myNewIndex[kk] : laneId)] = myInput[kk];
    }
    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < DEPTH; kk++) {
      uint global_index = warp_global_offset + (kk << 5);
      bool valid_input = ((global_index + laneId) < numElements) ? true : false;
      uint myNewKey =
          (valid_input) ?
              keys_ms_smem[(warpId << 5) * DEPTH + (kk << 5) + laneId] :
              0xFFFFFFFF;
      uint myNewBucket = bucketMapper(myNewKey);

      uint finalIndex =
          (valid_input) ?
              warp_offsets_smem[NUM_W * DEPTH * myNewBucket + warpId * DEPTH
                  + kk] + laneId :
              0;
      finalIndex -= __shfl(scan_histo[kk], myNewBucket, 32);

      if (valid_input)
        key_output[finalIndex] = myNewKey;
    }
  } else {
    // === Histogram and local index computation
#pragma unroll
    for (int kk = 0; kk < DEPTH; kk++) {
      // uint global_index = index + kk * gridDim.x * blockDim.x;
      uint global_index = warp_global_offset + (kk << 5);

      uint myBucket;
      myInput[kk] = key_input[global_index + laneId];
      myBucket = bucketMapper(myInput[kk]);

      uint myMask = 0xFFFFFFFF;
      uint myHisto = 0xFFFFFFFF;
      uint bit = myBucket;
      uint rx_buffer;
      // Computing the histogram and local indices:
#pragma unroll
      for (int i = 0; i < LOG_B; i++) {
        rx_buffer = __ballot(bit & 0x01);
        myMask = myMask & ((bit & 0x01) ? rx_buffer : (0xFFFFFFFF ^ rx_buffer));
        myHisto = myHisto
            & (((laneId >> i) & 0x01) ? rx_buffer : (0xFFFFFFFF ^ rx_buffer));
        bit >>= 1;
      }
      // writing back the local masks:
      binCounter[kk] = __popc(myHisto);
      uint n;
      scan_histo[kk] = binCounter[kk];
#pragma unroll
      for (int i = 1; i <= (1 << LOG_B); i <<= 1) {
        n = __shfl_up(scan_histo[kk], i, 32);
        if (laneId >= i)
          scan_histo[kk] += n;
      }
      scan_histo[kk] -= binCounter[kk]; //making it exclusive scan.

      // finding its new index within the warp:
      myNewIndex[kk] = __popc(myMask & (0xFFFFFFFF >> (31 - laneId))) - 1;
      myNewIndex[kk] += __shfl(scan_histo[kk], myBucket, 32);
    }

    // Reordering key elements in shared memory:
#pragma unroll
    for (int kk = 0; kk < DEPTH; kk++) {
      keys_ms_smem[(warpId << 5) * DEPTH + (kk << 5)
          + ((myNewIndex[kk] < 32) ? myNewIndex[kk] : laneId)] = myInput[kk];
    }
    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < DEPTH; kk++) {
      uint myNewKey = keys_ms_smem[(warpId << 5) * DEPTH + (kk << 5) + laneId];
      uint myNewBucket = bucketMapper(myNewKey);

      uint finalIndex = warp_offsets_smem[NUM_W * DEPTH * myNewBucket
          + warpId * DEPTH + kk] + laneId;
      finalIndex -= __shfl(scan_histo[kk], myNewBucket, 32);
      key_output[finalIndex] = myNewKey;
    }
  }
}
//=============================================
template<uint numWarps, uint numBuckets, uint logBuckets, uint depth, class T>
__global__ void split_WMS_pairs_ver6(uint* key_input, uint* value_input,
    uint* warpOffsets, uint* key_output, uint* value_output, uint numElements,
    T bucketMapper) {
  // this kernel use different memory hierarchy: bucket -> block -> warp -> roll
  // Warp-level MS, post-scan stage:
  // Histogram and local warp indices are recomputed. Keys are then reordered in shared memory and results are then stored into global memory.
  uint index = threadIdx.x + blockIdx.x * blockDim.x;
  uint laneId = threadIdx.x & 0x1F;
  uint warpId = threadIdx.x >> 5;
  uint warp_global_offset = (index - laneId) * depth;

  __shared__ uint scratchPad[numBuckets * numWarps * depth
      + 64 * numWarps * depth];
  uint* warp_offsets_smem = scratchPad;
  uint* keys_ms_smem = &warp_offsets_smem[numBuckets * numWarps * depth];
  uint* values_ms_smem = &keys_ms_smem[32 * numWarps * depth];

  // ===== Storing the results from global memory:
  // with new hierarchy: bucket -> block -> warp -> roll
  uint tid = threadIdx.x;
  while (tid < numBuckets * numWarps * depth) {
    uint whatBin = threadIdx.x / (numWarps * depth);
    uint whatRoll = threadIdx.x % (numWarps * depth);
    uint whatWarp = whatRoll / depth;
    whatRoll = whatRoll % depth;
    warp_offsets_smem[threadIdx.x] = warpOffsets[(whatBin * numWarps * gridDim.x
        * depth) + (blockIdx.x * numWarps * depth) + (whatWarp * depth)
        + whatRoll];
    tid += blockDim.x;
  }
  if ((warp_global_offset >= numElements))
    return;

  uint myInput[depth];
  uint myValue[depth];
  uint myNewIndex[depth]; // warp-level indices
  uint binCounter[depth]; // results of histograms
  uint scan_histo[depth];

  if (blockIdx.x == (gridDim.x - 1)) {
    // === Histogram and local index computation
#pragma unroll
    for (int kk = 0; kk < depth; kk++) {
      // uint global_index = index + kk * gridDim.x * blockDim.x;
      uint global_index = warp_global_offset + (kk << 5);
      uint myBucket;
      bool valid_input = false;
      if ((global_index + laneId) < numElements) {
        valid_input = true;
        myInput[kk] = key_input[global_index + laneId];
        myValue[kk] = value_input[global_index + laneId];
        myBucket = bucketMapper(myInput[kk]);
      }

      uint mask = __ballot(valid_input);
      uint myMask = 0xFFFFFFFF;
      uint myHisto = 0xFFFFFFFF;
      uint bit = myBucket;
      uint rx_buffer;
      // Computing the histogram and local indices:
#pragma unroll
      for (int i = 0; i < logBuckets; i++) {
        rx_buffer = __ballot(bit & 0x01);
        myMask = myMask & ((bit & 0x01) ? rx_buffer : (0xFFFFFFFF ^ rx_buffer));
        myHisto = myHisto
            & (((laneId >> i) & 0x01) ? rx_buffer : (0xFFFFFFFF ^ rx_buffer));
        bit >>= 1;
      }
      // writing back the local masks:
      binCounter[kk] = __popc(myHisto & mask);
      uint n;
      scan_histo[kk] = binCounter[kk];
#pragma unroll
      for (int i = 1; i <= (1 << logBuckets); i <<= 1) {
        n = __shfl_up(scan_histo[kk], i, 32);
        if (laneId >= i)
          scan_histo[kk] += n;
      }
      scan_histo[kk] -= binCounter[kk]; //making it exclusive scan.

      // finding its new index within the warp:
      myNewIndex[kk] = __popc(myMask & (0xFFFFFFFF >> (31 - laneId))) - 1;
      myNewIndex[kk] += __shfl(scan_histo[kk], myBucket, 32);

      myNewIndex[kk] = (valid_input) ? myNewIndex[kk] : 32; // if 32 it means that input was not valid
    }
    // Reordering key elements in shared memory:
#pragma unroll
    for (int kk = 0; kk < depth; kk++) {
      keys_ms_smem[(warpId << 5) * depth + (kk << 5)
          + ((myNewIndex[kk] < 32) ? myNewIndex[kk] : laneId)] = myInput[kk];
      values_ms_smem[(warpId << 5) * depth + (kk << 5)
          + ((myNewIndex[kk] < 32) ? myNewIndex[kk] : laneId)] = myValue[kk];
    }
    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < depth; kk++) {
      uint global_index = warp_global_offset + (kk << 5);
      // uint global_index = index + kk * gridDim.x * blockDim.x;
      bool valid_input = ((global_index + laneId) < numElements) ? true : false;
      uint myNewKey =
          (valid_input) ?
              keys_ms_smem[(warpId << 5) * depth + (kk << 5) + laneId] :
              0xFFFFFFFF;
      uint myNewValue =
          (valid_input) ?
              values_ms_smem[(warpId << 5) * depth + (kk << 5) + laneId] :
              0xFFFFFFFF;
      uint myNewBucket = bucketMapper(myNewKey);

      uint finalIndex =
          (valid_input) ?
              warp_offsets_smem[numWarps * depth * myNewBucket + warpId * depth
                  + kk] + laneId :
              0;
      finalIndex -= __shfl(scan_histo[kk], myNewBucket, 32);

      if (valid_input) {
        key_output[finalIndex] = myNewKey;
        value_output[finalIndex] = myNewValue;
      }
    }
  } else {
    // === Histogram and local index computation
#pragma unroll
    for (int kk = 0; kk < depth; kk++) {
      // uint global_index = index + kk * gridDim.x * blockDim.x;
      uint global_index = warp_global_offset + (kk << 5);
      uint myBucket;
      myInput[kk] = key_input[global_index + laneId];
      myValue[kk] = value_input[global_index + laneId];
      myBucket = bucketMapper(myInput[kk]);

      uint myMask = 0xFFFFFFFF;
      uint myHisto = 0xFFFFFFFF;
      uint bit = myBucket;
      uint rx_buffer;
      // Computing the histogram and local indices:
#pragma unroll
      for (int i = 0; i < logBuckets; i++) {
        rx_buffer = __ballot(bit & 0x01);
        myMask = myMask & ((bit & 0x01) ? rx_buffer : (0xFFFFFFFF ^ rx_buffer));
        myHisto = myHisto
            & (((laneId >> i) & 0x01) ? rx_buffer : (0xFFFFFFFF ^ rx_buffer));
        bit >>= 1;
      }
      // writing back the local masks:
      binCounter[kk] = __popc(myHisto);
      uint n;
      scan_histo[kk] = binCounter[kk];
#pragma unroll
      for (int i = 1; i <= (1 << logBuckets); i <<= 1) {
        n = __shfl_up(scan_histo[kk], i, 32);
        if (laneId >= i)
          scan_histo[kk] += n;
      }
      scan_histo[kk] -= binCounter[kk]; //making it exclusive scan.

      // finding its new index within the warp:
      myNewIndex[kk] = __popc(myMask & (0xFFFFFFFF >> (31 - laneId))) - 1;
      myNewIndex[kk] += __shfl(scan_histo[kk], myBucket, 32);
    }

    // Reordering key elements in shared memory:
#pragma unroll
    for (int kk = 0; kk < depth; kk++) {
      keys_ms_smem[(warpId << 5) * depth + (kk << 5) + myNewIndex[kk]] =
          myInput[kk];
      values_ms_smem[(warpId << 5) * depth + (kk << 5) + myNewIndex[kk]] =
          myValue[kk];
    }
    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < depth; kk++) {
      uint myNewKey = keys_ms_smem[(warpId << 5) * depth + (kk << 5) + laneId];
      uint myNewValue = values_ms_smem[(warpId << 5) * depth + (kk << 5)
          + laneId];
      uint myNewBucket = bucketMapper(myNewKey);

      uint finalIndex = warp_offsets_smem[numWarps * depth * myNewBucket
          + warpId * depth + kk] + laneId;

      finalIndex -= __shfl(scan_histo[kk], myNewBucket, 32);

      key_output[finalIndex] = myNewKey;
      value_output[finalIndex] = myNewValue;
    }
  }
}
//======================================================================
//======== Block level MS
//======================================================================
template<class T>
__global__ void histogram_block_ver6(uint* input, uint* bin, uint numElements,
    uint numBuckets, uint numWarps, uint depth, T bucketMapper) {
  // this kernel just computes block-level histograms for the pre-scan stage
  // Memory hierarchy which is used to store histogram results: Bucket -> blocks -> roll

  uint laneId = threadIdx.x & 0x1F;
  uint warpId = threadIdx.x >> 5;
  uint logBuckets = ceil(log2((float) numBuckets));
  uint logWarps = ceil(log2((float) numWarps));

  extern __shared__ uint scratchPad[];

  if (blockIdx.x == (gridDim.x - 1)) // last block
      {
    // === initializing the shared memory results:
    uint k = 0;
#pragma unroll
    while ((threadIdx.x + k * blockDim.x) < numBuckets * numWarps * depth) {
      scratchPad[threadIdx.x + k * blockDim.x] = 0;
      k++;
    }
    __syncthreads();

    // === computing the histogram:
#pragma unroll
    for (int kk = 0; kk < depth; kk++) {
      uint myInput;
      uint binCounter; // results of histograms
      uint global_index = blockIdx.x * blockDim.x * depth
          + ((kk * numWarps) << 5) + (warpId << 5) + laneId;

      uint myBucket = 0;
      bool valid_input = false;

      // == reading the input only if valid:
      if (global_index < numElements) {
        myInput = input[global_index];
        valid_input = true;
        myBucket = bucketMapper(myInput);
      }
      uint myHisto = 0xFFFFFFFF;
      uint bit = myBucket;
      uint rx_buffer;
      uint mask = __ballot(valid_input);

      // computing histogram
#pragma unroll
      for (int i = 0; i < logBuckets; i++) {
        rx_buffer = __ballot(bit & 0x01);
        myHisto = myHisto
            & (((laneId >> i) & 0x01) ? rx_buffer : (0xFFFFFFFF ^ rx_buffer));
        bit >>= 1;
      }
      binCounter = __popc(myHisto & mask);
      // === storing the results into the shared memory
      if (laneId < numBuckets) {
        // local hierarchy: roll -> warp -> bucket
        scratchPad[kk * numWarps * numBuckets + warpId * numBuckets + laneId] =
            binCounter;
      }
    }
    __syncthreads();

    // === computing our multi-reduction:
#pragma unroll
    for (int i = 1; i <= logWarps; i++) {
      // Performing reduction over all elements:
#pragma unroll
      for (int kk = 0; kk < depth; kk++) {
        if ((warpId & ((1 << i) - 1)) == 0) {
          if (laneId < numBuckets) {
            scratchPad[laneId + warpId * numBuckets + kk * numWarps * numBuckets] +=
                scratchPad[laneId + (warpId + (1 << (i - 1))) * numBuckets
                    + kk * numWarps * numBuckets];
          }
        }
      }
      __syncthreads();
    }

    // === storing the results back to the global memory with different hierarchy
    uint tid = threadIdx.x;
#pragma unroll
    while (tid < depth * numBuckets) {
      uint bucket_src = tid % numBuckets;
      uint roll_src = tid / numBuckets;

      // global memory hierarchy: bucket -> block -> roll
      bin[bucket_src * gridDim.x * depth + blockIdx.x * depth + roll_src] =
          scratchPad[bucket_src + roll_src * numWarps * numBuckets];
      tid += blockDim.x;
    }
  } else // all other blocks
  {
    // === computing the histogram:
#pragma unroll
    for (int kk = 0; kk < depth; kk++) {
      uint myInput;
      uint binCounter; // results of histograms
      uint global_index = blockIdx.x * blockDim.x * depth
          + ((kk * numWarps) << 5) + (warpId << 5) + laneId;
      uint myBucket = 0;

      // == reading the input only if valid:
      myInput = input[global_index];
      myBucket = bucketMapper(myInput);
      uint myHisto = 0xFFFFFFFF;
      uint bit = myBucket;
      uint rx_buffer;

      // == computing warp-wide histogram
#pragma unroll
      for (int i = 0; i < logBuckets; i++) {
        rx_buffer = __ballot(bit & 0x01);
        myHisto = myHisto
            & (((laneId >> i) & 0x01) ? rx_buffer : (0xFFFFFFFF ^ rx_buffer));
        bit >>= 1;
      }
      binCounter = __popc(myHisto);
      // === storing the results into the shared memory
      if (laneId < numBuckets) {
        // local hierarchy: roll -> warp -> bucket
        scratchPad[kk * numWarps * numBuckets + warpId * numBuckets + laneId] =
            binCounter;
      }
    }
    __syncthreads();

    // === computing our multi-reduction:
#pragma unroll
    for (int i = 1; i <= logWarps; i++) {
      // Performing reduction over all elements:
#pragma unroll
      for (int kk = 0; kk < depth; kk++) {
        if ((warpId & ((1 << i) - 1)) == 0) {
          if (laneId < numBuckets) {
            scratchPad[laneId + warpId * numBuckets + kk * numWarps * numBuckets] +=
                scratchPad[laneId + (warpId + (1 << (i - 1))) * numBuckets
                    + kk * numWarps * numBuckets];
          }
        }
      }
      __syncthreads();
    }

    // Global memory hierarchy: Bucket -> block -> roll
    if (numWarps >= depth) {
      if ((laneId < numBuckets) && (warpId < depth))
        bin[laneId * gridDim.x * depth + blockIdx.x * depth + warpId] =
            scratchPad[laneId + warpId * numBuckets * numWarps];
    }
  }
}
//============================
template<class T>
__global__ void split_BMS_ver6(uint* key_input, uint* blockOffsets,
    uint* key_output, uint numElements, uint numBuckets, uint numWarps,
    uint depth, T bucketMapper) {
  // Block-level MS with post-scan reordering
  // In this kernel, the last thread block follows a different branch than the rest
  // Global memory hierarchy that we use: bucket -> block -> roll

  extern __shared__ uint scratchPad[];
  uint* block_offsets_smem = scratchPad;
  uint* warp_offsets_smem = &block_offsets_smem[numBuckets * depth];
  uint* keys_ms_smem = &warp_offsets_smem[numBuckets * depth];
  uint* scan_histo_smem = &keys_ms_smem[32 * numWarps * depth];
  uint logBuckets = ceil(log2((float) numBuckets));
  uint logWarps = ceil(log2((float) numWarps));

  uint laneId = threadIdx.x & 0x1F;
  uint warpId = threadIdx.x >> 5;

  // === Loading block offset results from global memory:
  uint tid = threadIdx.x;
#pragma unroll
  while (tid < numBuckets * depth) {
    uint bucket_src = tid % numBuckets;
    uint roll_src = tid / numBuckets;
    block_offsets_smem[bucket_src + roll_src * numBuckets] =
        blockOffsets[roll_src + blockIdx.x * depth
            + bucket_src * depth * gridDim.x];
    tid += blockDim.x;
  }
  __syncthreads();

  if (blockIdx.x == (gridDim.x - 1)) // last block
      {
    // === Histogram and local index computation
#pragma unroll
    for (int kk = 0; kk < depth; kk++) {
      uint myInput;
      uint binCounter; // results of histograms
      uint scan_temp;
      uint global_index = blockIdx.x * blockDim.x * depth
          + ((kk * numWarps) << 5) + (warpId << 5) + laneId;
      uint myBucket = 0;
      uint valid_input = false;
      if (global_index < numElements) {
        myInput = key_input[global_index];
        valid_input = true;
        myBucket = bucketMapper(myInput);
      }

      uint myHisto = 0xFFFFFFFF;
      uint myLocalIndex = 0xFFFFFFFF;
      uint bit = myBucket;
      uint rx_buffer;
      uint mask = __ballot(valid_input);

      // computing histogram
#pragma unroll
      for (int i = 0; i < logBuckets; i++) {
        rx_buffer = __ballot(bit & 0x01);
        myLocalIndex = myLocalIndex
            & ((bit & 0x01) ? rx_buffer : (0xFFFFFFFF ^ rx_buffer));
        myHisto = myHisto
            & (((laneId >> i) & 0x01) ? rx_buffer : (0xFFFFFFFF ^ rx_buffer));
        bit >>= 1;
      }
      binCounter = __popc(myHisto & mask);
      // storing the results in smem in order to be scanned:
      // smem hierarchy: roll -> warp -> bucket
      if (laneId < numBuckets)
        scan_histo_smem[laneId + warpId * numBuckets
            + kk * numBuckets * numWarps] = binCounter;
      __syncthreads();
      // computing block-wise scan over buckets:
      scan_temp = binCounter;
      for (int i = 1; i < (1 << logWarps); i <<= 1) {
        if (laneId < numBuckets)
          scan_temp += (
              (warpId >= i) ?
                  scan_histo_smem[kk * numBuckets * numWarps
                      + (warpId - i) * numBuckets + laneId] :
                  0);
        __syncthreads();
        if (laneId < numBuckets)
          scan_histo_smem[kk * numBuckets * numWarps + warpId * numBuckets
              + laneId] = scan_temp;
        __syncthreads();
      }
      // Computing block-level indices:
      scan_temp -= binCounter; // exclusive scan
      uint myLocalBlockIndex = __shfl(scan_temp, myBucket, 32);
      myLocalBlockIndex += __popc(myLocalIndex & (0xFFFFFFFF >> (31 - laneId)))
          - 1;
      myLocalBlockIndex = (valid_input) ? myLocalBlockIndex : blockDim.x;

      // Computing warp-level offsets within each block:
      uint block_scan;
      if (warpId == (numWarps - 1)) // the last warp
          {
        block_scan = scan_temp + binCounter;
        uint n;
#pragma unroll
        for (int i = 1; i <= (1 << logBuckets); i <<= 1) {
          n = __shfl_up(block_scan, i, 32);
          if (laneId >= i)
            block_scan += n;
        }
        scan_temp += binCounter;
        block_scan -= scan_temp;
        if (laneId < numBuckets) {
          warp_offsets_smem[laneId + kk * numBuckets] = block_scan;
        }
      }
      __syncthreads();
      uint myNewBlockIndex = warp_offsets_smem[myBucket + kk * numBuckets]
          + myLocalBlockIndex;
      // block-level reordering in shared memory
      keys_ms_smem[(
          (myLocalBlockIndex < blockDim.x) ? myNewBlockIndex : threadIdx.x)
          + kk * blockDim.x] = myInput;
    }
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < depth; kk++) {
      uint global_index = blockIdx.x * blockDim.x * depth
          + ((kk * numWarps) << 5) + (warpId << 5) + laneId;
      bool valid_input = (global_index < numElements) ? true : false;

      uint myNewKey =
          (valid_input) ?
              keys_ms_smem[threadIdx.x + kk * blockDim.x] : 0xFFFFFFFF;
      uint myNewBucket = bucketMapper(myNewKey);
      uint finalIndex = 0;
      if (valid_input) {
        finalIndex = block_offsets_smem[numBuckets * kk + myNewBucket]
            + threadIdx.x;
        finalIndex -= warp_offsets_smem[myNewBucket + kk * numBuckets];
        key_output[finalIndex] = myNewKey;
      }
    }
  } else // all others
  {
    // === Histogram and local index computation
#pragma unroll
    for (int kk = 0; kk < depth; kk++) {
      uint myInput;
      uint binCounter; // results of histograms
      uint scan_temp;
      uint global_index = blockIdx.x * blockDim.x * depth
          + ((kk * numWarps) << 5) + (warpId << 5) + laneId;
      uint myBucket = 0;
      myInput = key_input[global_index];
      myBucket = bucketMapper(myInput);

      uint myHisto = 0xFFFFFFFF;
      uint myLocalIndex = 0xFFFFFFFF;
      uint bit = myBucket;
      uint rx_buffer;

      // computing histogram
#pragma unroll
      for (int i = 0; i < logBuckets; i++) {
        rx_buffer = __ballot(bit & 0x01);
        myLocalIndex = myLocalIndex
            & ((bit & 0x01) ? rx_buffer : (0xFFFFFFFF ^ rx_buffer));
        myHisto = myHisto
            & (((laneId >> i) & 0x01) ? rx_buffer : (0xFFFFFFFF ^ rx_buffer));
        bit >>= 1;
      }
      binCounter = __popc(myHisto);
      // storing the results in smem in order to be scanned:
      // smem hierarcy: roll -> warp -> bucket
      if (laneId < numBuckets)
        scan_histo_smem[laneId + warpId * numBuckets
            + kk * numBuckets * numWarps] = binCounter;

      __syncthreads();
      // computing block-wise scan over buckets:
      scan_temp = binCounter;
      for (int i = 1; i < (1 << logWarps); i <<= 1) {
        if (laneId < numBuckets)
          scan_temp += (
              (warpId >= i) ?
                  scan_histo_smem[kk * numBuckets * numWarps
                      + (warpId - i) * numBuckets + laneId] :
                  0);
        __syncthreads();
        if (laneId < numBuckets)
          scan_histo_smem[kk * numBuckets * numWarps + warpId * numBuckets
              + laneId] = scan_temp;
        __syncthreads();
      }
      // Computing block-level indices:
      scan_temp -= binCounter; // exclusive scan
      uint myLocalBlockIndex = __shfl(scan_temp, myBucket, 32);
      myLocalBlockIndex += __popc(myLocalIndex & (0xFFFFFFFF >> (31 - laneId)))
          - 1;

      // Computing warp-level offsets within each block:
      uint block_scan;
      if (warpId == (numWarps - 1)) // the last warp
          {
        block_scan = scan_temp + binCounter;
        uint n;
#pragma unroll
        for (int i = 1; i <= (1 << logBuckets); i <<= 1) {
          n = __shfl_up(block_scan, i, 32);
          if (laneId >= i)
            block_scan += n;
        }
        scan_temp += binCounter;
        block_scan -= scan_temp;
        if (laneId < numBuckets) {
          warp_offsets_smem[laneId + kk * numBuckets] = block_scan;
        }
      }
      __syncthreads();
      uint myNewBlockIndex = warp_offsets_smem[myBucket + kk * numBuckets]
          + myLocalBlockIndex;
      // block-level reordering in shared memory
      keys_ms_smem[myNewBlockIndex + kk * blockDim.x] = myInput;
    }
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < depth; kk++) {
      uint myNewKey = keys_ms_smem[threadIdx.x + kk * blockDim.x];
      uint myNewBucket = bucketMapper(myNewKey);
      uint finalIndex = block_offsets_smem[numBuckets * kk + myNewBucket]
          + threadIdx.x;
      finalIndex -= warp_offsets_smem[myNewBucket + kk * numBuckets];
      key_output[finalIndex] = myNewKey;
    }
  }
}
//===============================
template<class T>
__global__ void split_BMS_pairs_ver6(uint* key_input, uint* value_input,
    uint* blockOffsets, uint* key_output, uint* value_output, uint numElements,
    uint numBuckets, uint numWarps, uint depth, T bucketMapper) {
  // Block-level MS with post-scan reordering
  // In this kernel, the last thread block follows a different branch than the rest
  // Global memory hierarchy that we use: bucket -> block -> roll

  extern __shared__ uint scratchPad[];
  uint* block_offsets_smem = scratchPad;
  uint* warp_offsets_smem = &block_offsets_smem[numBuckets * depth];
  uint* keys_ms_smem = &warp_offsets_smem[numBuckets * depth];
  uint* values_ms_smem = &keys_ms_smem[32 * numWarps * depth];
  uint* scan_histo_smem = &keys_ms_smem[64 * numWarps * depth];
  uint logBuckets = ceil(log2((float) numBuckets));
  uint logWarps = ceil(log2((float) numWarps));

  uint laneId = threadIdx.x & 0x1F;
  uint warpId = threadIdx.x >> 5;

  // === Loading block offset results from global memory:
  uint tid = threadIdx.x;
#pragma unroll
  while (tid < numBuckets * depth) {
    uint bucket_src = tid % numBuckets;
    uint roll_src = tid / numBuckets;
    block_offsets_smem[bucket_src + roll_src * numBuckets] =
        blockOffsets[roll_src + blockIdx.x * depth
            + bucket_src * depth * gridDim.x];
    tid += blockDim.x;
  }
  __syncthreads();

  if (blockIdx.x == (gridDim.x - 1)) // last block
      {
    // === Histogram and local index computation
#pragma unroll
    for (int kk = 0; kk < depth; kk++) {
      uint myInput;
      uint myValue;
      uint binCounter; // results of histograms
      uint scan_temp;
      uint global_index = blockIdx.x * blockDim.x * depth
          + ((kk * numWarps) << 5) + (warpId << 5) + laneId;
      uint myBucket = 0;
      uint valid_input = false;
      if (global_index < numElements) {
        myInput = key_input[global_index];
        myValue = value_input[global_index];
        valid_input = true;
        myBucket = bucketMapper(myInput);
      }

      uint myHisto = 0xFFFFFFFF;
      uint myLocalIndex = 0xFFFFFFFF;
      uint bit = myBucket;
      uint rx_buffer;
      uint mask = __ballot(valid_input);

      // computing histogram
#pragma unroll
      for (int i = 0; i < logBuckets; i++) {
        rx_buffer = __ballot(bit & 0x01);
        myLocalIndex = myLocalIndex
            & ((bit & 0x01) ? rx_buffer : (0xFFFFFFFF ^ rx_buffer));
        myHisto = myHisto
            & (((laneId >> i) & 0x01) ? rx_buffer : (0xFFFFFFFF ^ rx_buffer));
        bit >>= 1;
      }
      binCounter = __popc(myHisto & mask);
      // storing the results in smem in order to be scanned:
      // smem hierarcy: roll -> warp -> bucket
      if (laneId < numBuckets)
        scan_histo_smem[laneId + warpId * numBuckets
            + kk * numBuckets * numWarps] = binCounter;
      __syncthreads();
      // computing block-wise scan over buckets:
      scan_temp = binCounter;
      for (int i = 1; i < (1 << logWarps); i <<= 1) {
        if (laneId < numBuckets)
          scan_temp += (
              (warpId >= i) ?
                  scan_histo_smem[kk * numBuckets * numWarps
                      + (warpId - i) * numBuckets + laneId] :
                  0);
        __syncthreads();
        if (laneId < numBuckets)
          scan_histo_smem[kk * numBuckets * numWarps + warpId * numBuckets
              + laneId] = scan_temp;
        __syncthreads();
      }
      // Computing block-level indices:
      scan_temp -= binCounter; // exclusive scan
      uint myLocalBlockIndex = __shfl(scan_temp, myBucket, 32);
      myLocalBlockIndex += __popc(myLocalIndex & (0xFFFFFFFF >> (31 - laneId)))
          - 1;
      myLocalBlockIndex = (valid_input) ? myLocalBlockIndex : blockDim.x;

      // Computing warp-level offsets within each block:
      uint block_scan;
      if (warpId == (numWarps - 1)) // the last warp
          {
        block_scan = scan_temp + binCounter;
        uint n;
#pragma unroll
        for (int i = 1; i <= (1 << logBuckets); i <<= 1) {
          n = __shfl_up(block_scan, i, 32);
          if (laneId >= i)
            block_scan += n;
        }
        scan_temp += binCounter;
        block_scan -= scan_temp;
        if (laneId < numBuckets) {
          warp_offsets_smem[laneId + kk * numBuckets] = block_scan;
        }
      }
      __syncthreads();
      uint myNewBlockIndex = warp_offsets_smem[myBucket + kk * numBuckets]
          + myLocalBlockIndex;
      // block-level reordering in shared memory
      keys_ms_smem[(
          (myLocalBlockIndex < blockDim.x) ? myNewBlockIndex : threadIdx.x)
          + kk * blockDim.x] = myInput;
      values_ms_smem[(
          (myLocalBlockIndex < blockDim.x) ? myNewBlockIndex : threadIdx.x)
          + kk * blockDim.x] = myValue;
    }
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < depth; kk++) {
      uint global_index = blockIdx.x * blockDim.x * depth
          + ((kk * numWarps) << 5) + (warpId << 5) + laneId;
      bool valid_input = (global_index < numElements) ? true : false;

      uint myNewKey =
          (valid_input) ?
              keys_ms_smem[threadIdx.x + kk * blockDim.x] : 0xFFFFFFFF;
      uint myNewValue =
          (valid_input) ?
              values_ms_smem[threadIdx.x + kk * blockDim.x] : 0xFFFFFFFF;
      uint myNewBucket = bucketMapper(myNewKey);

      uint finalIndex = 0;
      if (valid_input) {
        finalIndex = block_offsets_smem[numBuckets * kk + myNewBucket]
            + threadIdx.x;
        finalIndex -= warp_offsets_smem[myNewBucket + kk * numBuckets];
        key_output[finalIndex] = myNewKey;
        value_output[finalIndex] = myNewValue;
      }
    }
  } else // all others
  {
    // === Histogram and local index computation
#pragma unroll
    for (int kk = 0; kk < depth; kk++) {
      uint myInput;
      uint myValue;
      uint binCounter; // results of histograms
      uint scan_temp;
      uint global_index = blockIdx.x * blockDim.x * depth
          + ((kk * numWarps) << 5) + (warpId << 5) + laneId;
      uint myBucket = 0;
      myInput = key_input[global_index];
      myValue = value_input[global_index];
      myBucket = bucketMapper(myInput);

      uint myHisto = 0xFFFFFFFF;
      uint myLocalIndex = 0xFFFFFFFF;
      uint bit = myBucket;
      uint rx_buffer;

      // computing histogram
#pragma unroll
      for (int i = 0; i < logBuckets; i++) {
        rx_buffer = __ballot(bit & 0x01);
        myLocalIndex = myLocalIndex
            & ((bit & 0x01) ? rx_buffer : (0xFFFFFFFF ^ rx_buffer));
        myHisto = myHisto
            & (((laneId >> i) & 0x01) ? rx_buffer : (0xFFFFFFFF ^ rx_buffer));
        bit >>= 1;
      }
      binCounter = __popc(myHisto);
      // storing the results in smem in order to be scanned:
      // smem hierarcy: roll -> warp -> bucket
      if (laneId < numBuckets)
        scan_histo_smem[laneId + warpId * numBuckets
            + kk * numBuckets * numWarps] = binCounter;

      __syncthreads();
      // computing block-wise scan over buckets:
      scan_temp = binCounter;
      for (int i = 1; i < (1 << logWarps); i <<= 1) {
        if (laneId < numBuckets)
          scan_temp += (
              (warpId >= i) ?
                  scan_histo_smem[kk * numBuckets * numWarps
                      + (warpId - i) * numBuckets + laneId] :
                  0);
        __syncthreads();
        if (laneId < numBuckets)
          scan_histo_smem[kk * numBuckets * numWarps + warpId * numBuckets
              + laneId] = scan_temp;
        __syncthreads();
      }
      // Computing block-level indices:
      scan_temp -= binCounter; // exclusive scan
      uint myLocalBlockIndex = __shfl(scan_temp, myBucket, 32);
      myLocalBlockIndex += __popc(myLocalIndex & (0xFFFFFFFF >> (31 - laneId)))
          - 1;

      // Computing warp-level offsets within each block:
      uint block_scan;
      if (warpId == (numWarps - 1)) // the last warp
          {
        block_scan = scan_temp + binCounter;
        uint n;
#pragma unroll
        for (int i = 1; i <= (1 << logBuckets); i <<= 1) {
          n = __shfl_up(block_scan, i, 32);
          if (laneId >= i)
            block_scan += n;
        }
        scan_temp += binCounter;
        block_scan -= scan_temp;
        if (laneId < numBuckets) {
          warp_offsets_smem[laneId + kk * numBuckets] = block_scan;
        }
      }
      __syncthreads();
      uint myNewBlockIndex = warp_offsets_smem[myBucket + kk * numBuckets]
          + myLocalBlockIndex;
      // block-level reordering in shared memory
      keys_ms_smem[myNewBlockIndex + kk * blockDim.x] = myInput;
      values_ms_smem[myNewBlockIndex + kk * blockDim.x] = myValue;
    }
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < depth; kk++) {
      uint myNewKey = keys_ms_smem[threadIdx.x + kk * blockDim.x];
      uint myNewValue = values_ms_smem[threadIdx.x + kk * blockDim.x];
      uint myNewBucket = bucketMapper(myNewKey);

      uint finalIndex = block_offsets_smem[numBuckets * kk + myNewBucket]
          + threadIdx.x;
      finalIndex -= warp_offsets_smem[myNewBucket + kk * numBuckets];
      key_output[finalIndex] = myNewKey;
      value_output[finalIndex] = myNewValue;
    }
  }
}
//======================================
template<uint NUM_WARPS, uint NUM_BUCKETS, uint LOG_BUCKETS, uint LOG_WARPS,
    class T>
__global__ void histogramBallot_Mode13_large(uint* input, uint* bin,
    uint numElements, T bucketMapper) {
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

  uint index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index > numElements)
    return;

  __shared__ union {
    uint temp_storage[NUM_BUCKETS * NUM_WARPS + 32 * NUM_WARPS];
    typename BlockScanT::TempStorage temp_cub; // being used in CUB's block scan
  } shm;

  uint *scratchPad = &((shm.temp_storage)[0]);
  uint *blockMS = &((shm.temp_storage)[NUM_BUCKETS * NUM_WARPS]);

  uint laneId = threadIdx.x & 0x1F;
  uint warpId = threadIdx.x >> 5;
  const uint num_roll = (NUM_BUCKETS + 31) / 32; // number of buckets dedicated to each thread (at most)
  uint bucketId;
  uint myMask = 0xFFFFFFFF;
  uint myHisto[num_roll]; // each thread is responsible for multiple histogram values
  uint scan_temp[num_roll];
  uint bit;
  uint rx_buffer;
  uint item = input[index];
  bucketId = bucketMapper(item);

  // computing warp-level histogram:
#pragma unroll
  for (int i = 0; i < num_roll; i++)
    myHisto[i] = 0xFFFFFFFF;

  bit = bucketId;

#pragma unroll
  for (int i = 0; i < LOG_BUCKETS; i++) {
    rx_buffer = __ballot(bit & 0x01);
    myMask = myMask & ((bit & 0x01) ? rx_buffer : (0xFFFFFFFF ^ rx_buffer));
#pragma unroll
    for (int k = 0; k < num_roll; k++) {
      myHisto[k] = myHisto[k]
          & ((((laneId + 32 * k) >> i) & 0x01) ?
              rx_buffer : (0xFFFFFFFF ^ rx_buffer));
    }
    bit >>= 1;
  }
  // copying back the results into the scratchPad:
#pragma unroll
  for (int k = 0; k < num_roll; k++) {
    myHisto[k] = __popc(myHisto[k]);
    scan_temp[k] = myHisto[k];
    if ((laneId + (k << 5)) < NUM_BUCKETS) {
      scratchPad[laneId + (k << 5) + warpId * NUM_BUCKETS] = myHisto[k];
    }
  }
  __syncthreads();

  for (int i = 1; i < (1 << LOG_WARPS); i <<= 1) {
#pragma unroll
    for (int k = 0; k < num_roll; k++) {
      if ((laneId + (k << 5)) < NUM_BUCKETS)
        scan_temp[k] += (
            (warpId >= i) ?
                scratchPad[(warpId - i) * NUM_BUCKETS + (k << 5) + laneId] : 0);
    }
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < num_roll; kk++) {
      if ((laneId + (kk << 5)) < NUM_BUCKETS)
        scratchPad[warpId * NUM_BUCKETS + (kk << 5) + laneId] = scan_temp[kk];
    }
    __syncthreads();
  }

  // First loading this results into the global memory so that we can use it again by CUB:
  uint block_offset = 0;
  if (threadIdx.x < NUM_BUCKETS) {
    block_offset = scratchPad[(NUM_WARPS - 1) * NUM_BUCKETS + threadIdx.x];
    bin[(threadIdx.x) * gridDim.x + blockIdx.x] = block_offset;
  }
  __syncthreads();

  // computing block level exlusive scan for having right offsets using CUB's block scan:
  uint temp_results = 0;
  BlockScanT(shm.temp_cub).ExclusiveSum(block_offset, temp_results);
  __syncthreads();
  if (threadIdx.x < NUM_BUCKETS) {
    scratchPad[threadIdx.x] = temp_results;
  }
  __syncthreads();

#pragma unroll
  for (int k = 0; k < num_roll; k++)
    scan_temp[k] -= myHisto[k];

  // we read all those registers because we do not beforehand which ones we need:
  uint myLocalBlockIndex[num_roll];
#pragma unroll
  for (int k = 0; k < num_roll; k++)
    myLocalBlockIndex[k] = __shfl(scan_temp[k], (bucketId & 0x1F), 32);

  myLocalBlockIndex[(bucketId >> 5)] += (__popc(
      myMask & (0xFFFFFFFF >> (31 - laneId))) - 1);

  // updating the block level index:
  uint myBlockOffset = scratchPad[bucketId] + myLocalBlockIndex[bucketId >> 5];
  blockMS[myBlockOffset] = item;
  __syncthreads();

  input[index] = blockMS[threadIdx.x];

  // storing back the final offsets:
  if (threadIdx.x < NUM_BUCKETS) {
    bin[NUM_BUCKETS * (gridDim.x + blockIdx.x) + threadIdx.x] =
        scratchPad[threadIdx.x];
  }
}
//======================================
template<uint NUM_WARPS, uint NUM_BUCKETS, class T>
__global__ void splitBallot_Mode13_large(unsigned int* input,
    unsigned int* binOffsets, unsigned int* output, unsigned int numElements,
    T bucketMapper) {
  // Performing the splitting proces using the prefixed-sum histograms (binOffsets), and the
  // local warp-level masks (binMask).

  uint index = threadIdx.x + blockIdx.x * blockDim.x;

  if (index > numElements)
    return;

  __shared__ uint scratchPad[2 * NUM_BUCKETS];
  uint* scanBlock = &scratchPad[NUM_BUCKETS];

  uint item = input[index];
  // uint laneId = threadIdx.x & 0x1F;
  // uint warpId = threadIdx.x >> 5;
  uint bucketId = bucketMapper(item);

  // Loading all warp indices regarding to each bucket into the shared memory:
  if (threadIdx.x < NUM_BUCKETS) {
    scratchPad[threadIdx.x] = binOffsets[threadIdx.x * gridDim.x + blockIdx.x];
    scanBlock[threadIdx.x] = binOffsets[NUM_BUCKETS * (gridDim.x + blockIdx.x)
        + threadIdx.x];
  }
  __syncthreads();

  // writing back the results:
  output[scratchPad[bucketId] + threadIdx.x - scanBlock[bucketId]] = item;
}
//===============================
template<uint NUM_WARPS, uint NUM_BUCKETS, uint LOG_BUCKETS, uint LOG_WARPS,
    class T>
__global__ void histogramBallot_Mode13_large_pairs(uint* key_input,
    uint* value_input, uint* bin, uint numElements, T bucketMapper) {
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

  uint index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index > numElements)
    return;

  __shared__ union {
    uint temp_storage[NUM_BUCKETS * NUM_WARPS + 64 * NUM_WARPS];
    typename BlockScanT::TempStorage temp_cub; // being used in CUB's block scan
  } shm;

  uint *scratchPad = &((shm.temp_storage)[0]);
  uint *blockMS = &((shm.temp_storage)[NUM_BUCKETS * NUM_WARPS]);
  uint *blockMSvalues = &((shm.temp_storage)[NUM_BUCKETS * NUM_WARPS
      + 32 * NUM_WARPS]);

  uint laneId = threadIdx.x & 0x1F;
  uint warpId = threadIdx.x >> 5;
  const uint num_roll = (NUM_BUCKETS + 31) / 32; // number of buckets dedicated to each thread (at most)
  uint bucketId;
  uint myMask = 0xFFFFFFFF;
  uint myHisto[num_roll]; // each thread is responsible for multiple histogram values
  uint scan_temp[num_roll];
  uint bit;
  uint rx_buffer;
  uint item = key_input[index];
  uint myValue = value_input[index];
  bucketId = bucketMapper(item);
  // computing warp-level histogram:
#pragma unroll
  for (int i = 0; i < num_roll; i++)
    myHisto[i] = 0xFFFFFFFF;

  bit = bucketId;

#pragma unroll
  for (int i = 0; i < LOG_BUCKETS; i++) {
    rx_buffer = __ballot(bit & 0x01);
    myMask = myMask & ((bit & 0x01) ? rx_buffer : (0xFFFFFFFF ^ rx_buffer));
#pragma unroll
    for (int k = 0; k < num_roll; k++) {
      myHisto[k] = myHisto[k]
          & ((((laneId + 32 * k) >> i) & 0x01) ?
              rx_buffer : (0xFFFFFFFF ^ rx_buffer));
    }
    bit >>= 1;
  }
  // copying back the results into the scratchPad:
#pragma unroll
  for (int k = 0; k < num_roll; k++) {
    myHisto[k] = __popc(myHisto[k]);
    scan_temp[k] = myHisto[k];
    if ((laneId + (k << 5)) < NUM_BUCKETS) {
      scratchPad[laneId + (k << 5) + warpId * NUM_BUCKETS] = myHisto[k];
    }
  }
  __syncthreads();

  for (int i = 1; i < (1 << LOG_WARPS); i <<= 1) {
#pragma unroll
    for (int k = 0; k < num_roll; k++) {
      if ((laneId + (k << 5)) < NUM_BUCKETS)
        scan_temp[k] += (
            (warpId >= i) ?
                scratchPad[(warpId - i) * NUM_BUCKETS + (k << 5) + laneId] : 0);
    }
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < num_roll; kk++) {
      if ((laneId + (kk << 5)) < NUM_BUCKETS)
        scratchPad[warpId * NUM_BUCKETS + (kk << 5) + laneId] = scan_temp[kk];
    }
    __syncthreads();
  }

  // First loading this results into the global memory so that we can use it again by CUB:
  uint block_offset = 0;
  if (threadIdx.x < NUM_BUCKETS) {
    block_offset = scratchPad[(NUM_WARPS - 1) * NUM_BUCKETS + threadIdx.x];
    bin[(threadIdx.x) * gridDim.x + blockIdx.x] = block_offset;
  }
  __syncthreads();

  // computing block level exlusive scan for having right offsets using CUB's block scan:
  uint temp_results = 0;
  BlockScanT(shm.temp_cub).ExclusiveSum(block_offset, temp_results);
  __syncthreads();
  if (threadIdx.x < NUM_BUCKETS) {
    scratchPad[threadIdx.x] = temp_results;
  }
  __syncthreads();

#pragma unroll
  for (int k = 0; k < num_roll; k++)
    scan_temp[k] -= myHisto[k];

  // we read all those registers because we do not beforehand which ones we need:
  uint myLocalBlockIndex[num_roll];
#pragma unroll
  for (int k = 0; k < num_roll; k++)
    myLocalBlockIndex[k] = __shfl(scan_temp[k], (bucketId & 0x1F), 32);

  myLocalBlockIndex[(bucketId >> 5)] += (__popc(
      myMask & (0xFFFFFFFF >> (31 - laneId))) - 1);

  // updating the block level index:
  uint myBlockOffset = scratchPad[bucketId] + myLocalBlockIndex[bucketId >> 5];
  blockMS[myBlockOffset] = item;
  blockMSvalues[myBlockOffset] = myValue;
  __syncthreads();

  key_input[index] = blockMS[threadIdx.x];
  value_input[index] = blockMSvalues[threadIdx.x];

  // storing back the final offsets:
  if (threadIdx.x < NUM_BUCKETS) {
    bin[NUM_BUCKETS * (gridDim.x + blockIdx.x) + threadIdx.x] =
        scratchPad[threadIdx.x];
  }
}
//======================================
template<uint NUM_WARPS, uint NUM_BUCKETS, class T>
__global__ void splitBallot_Mode13_large_pairs(uint* key_input,
    uint* value_input, unsigned int* binOffsets, uint* key_output,
    uint* value_output, unsigned int numElements, T bucketMapper) {
  // Performing the splitting proces using the prefixed-sum histograms (binOffsets), and the
  // local warp-level masks (binMask).

  uint index = threadIdx.x + blockIdx.x * blockDim.x;
  uint logBuckets = ceil(log2((float) NUM_BUCKETS));

  if (index > numElements)
    return;

  __shared__ uint scratchPad[2 * NUM_BUCKETS];
  uint* scanBlock = &scratchPad[NUM_BUCKETS];

  uint item = key_input[index];
  uint myValue = value_input[index];
  // uint laneId = threadIdx.x & 0x1F;
  // uint warpId = threadIdx.x >> 5;
  uint bucketId;
  bucketId = bucketMapper(item);
  // Loading all warp indices regarding to each bucket into the shared memory:
  if (threadIdx.x < NUM_BUCKETS) {
    scratchPad[threadIdx.x] = binOffsets[threadIdx.x * gridDim.x + blockIdx.x];
    scanBlock[threadIdx.x] = binOffsets[NUM_BUCKETS * (gridDim.x + blockIdx.x)
        + threadIdx.x];
  }
  __syncthreads();

  // writing back the results:
  uint finalIndex = scratchPad[bucketId] + threadIdx.x - scanBlock[bucketId];
  key_output[finalIndex] = item;
  value_output[finalIndex] = myValue;
}
//======================================
//=========================================================================
// Compaction based on Warp-level Multisplit
//=========================================================================
#ifndef __MULTISPLIT_COMPACTION_CUH_
template<uint32_t NUM_W, uint32_t DEPTH, typename bucket_t, typename key_t>
__global__ void histogram_pre_scan_compaction(key_t* input, uint32_t* bin, uint32_t numElements, bucket_t bucket_identifier)
{
  // The new warp-level histogram computing kernel:
  uint32_t  index = threadIdx.x + blockIdx.x * blockDim.x;
  uint32_t  laneId = threadIdx.x & 0x1F;
  uint32_t  warpId = threadIdx.x >> 5;

  __shared__ uint32_t scratchPad[(NUM_W * DEPTH) << 1];

  if(blockIdx.x == (gridDim.x - 1)) // last block, potentially may try to read invalid inputs
  {
    // === initializing the shared memory results:
    uint32_t k = 0;
    #pragma unroll
    while((threadIdx.x + k * blockDim.x) < (2 * NUM_W * DEPTH))
    {
      scratchPad[threadIdx.x + k * blockDim.x] = 0;
      k++;
    }
    __syncthreads();


    // === computing the histogram:
    key_t   myInput[DEPTH];
    uint32_t  binCounter[DEPTH];  // results of histograms

    #pragma unroll
    for(int kk = 0; kk<DEPTH; kk++){
      uint32_t global_index = (index - laneId) * DEPTH + (kk << 5);
      bool valid_input = false;

      // == reading the input only if valid:
      if((global_index + laneId) < numElements)
      {
        myInput[kk] = input[global_index + laneId];
        valid_input = true;
      }
      // masking out those threads which read an invalid key
      uint32_t mask = __ballot(valid_input);

      // computing histogram
      uint32_t rx_buffer = __ballot(bucket_identifier(myInput[kk]));
      uint32_t myHisto = 0xFFFFFFFF & (((laneId) & 0x01)?rx_buffer:(0xFFFFFFFF ^ rx_buffer));
      binCounter[kk] = __popc(myHisto & mask);

      // === storing the results into the shared memory
      if(laneId < 2)
      {
        // new hierarchy: Bucket -> warp -> roll
        scratchPad[laneId * NUM_W * DEPTH + warpId * DEPTH + kk] = binCounter[kk];
      }
    }
    __syncthreads();

    // === storing histogram results from shared memory into global memory:
    uint32_t tid = threadIdx.x;
    #pragma unroll
    while(tid < (2 * NUM_W * DEPTH))
    {
      // storing histogram results:
      uint32_t whatBin = tid / (NUM_W * DEPTH);
      uint32_t whatRoll = tid % (NUM_W * DEPTH);
      uint32_t whatWarp = whatRoll / DEPTH;
      whatRoll = whatRoll % DEPTH;
      // new hierarchy: Bucket -> block -> warp -> roll:
      uint32_t finalIndex = blockIdx.x * DEPTH * NUM_W + whatWarp * DEPTH + whatRoll;
      bin[whatBin * NUM_W * DEPTH * gridDim.x + finalIndex] = scratchPad[tid];
      // updating tid
      tid += blockDim.x;
    }
  }
  else // all other blocks, that we are sure they are certainly processing valid inputs:
  {
    // === computing the histogram:
    key_t   myInput[DEPTH];
    uint32_t  binCounter[DEPTH];  // results of histograms

    #pragma unroll
    for(int kk = 0; kk<DEPTH; kk++){
      uint32_t global_index = (index - laneId) * DEPTH + (kk << 5);

      myInput[kk] = input[global_index + laneId];

      uint32_t rx_buffer = __ballot(bucket_identifier(myInput[kk]));
      uint32_t myHisto = 0xFFFFFFFF & (((laneId) & 0x01)?rx_buffer:(0xFFFFFFFF ^ rx_buffer));
      binCounter[kk] = __popc(myHisto);

      // === storing the results into the shared memory
      if(laneId < 2)
      {
        // new hierarchy: Bucket -> warp -> roll
        scratchPad[laneId * NUM_W * DEPTH + warpId * DEPTH + kk] = binCounter[kk];
      }
      __syncthreads();
    }
    // === storing histogram results from shared memory into global memory:
    uint32_t tid = threadIdx.x;
    #pragma unroll
    while(tid < (2 * NUM_W * DEPTH))
    {
      // storing histogram results:
      uint32_t whatBin = tid / (NUM_W * DEPTH);
      uint32_t whatRoll = tid % (NUM_W * DEPTH);
      uint32_t whatWarp = whatRoll / DEPTH;
      whatRoll = whatRoll % DEPTH;
      // new hierarchy: Bucket -> block -> warp -> roll:
      uint32_t finalIndex = blockIdx.x * DEPTH * NUM_W + whatWarp * DEPTH + whatRoll;
      bin[whatBin * NUM_W * DEPTH * gridDim.x + finalIndex] = scratchPad[tid];
      // updating tid
      tid += blockDim.x;
    }
  }
}
//==================================
template<uint32_t NUM_W, uint32_t DEPTH, typename bucket_t, typename key_t>
__global__ void split_post_scan_compaction(key_t* key_input, uint32_t* warpOffsets, key_t* key_output, uint32_t numElements, bucket_t bucket_identifier)
{
  uint32_t  index = threadIdx.x + blockIdx.x * blockDim.x;
  uint32_t  laneId = threadIdx.x & 0x1F;
  uint32_t  warpId = threadIdx.x >> 5;
  uint32_t  warp_global_offset = (index - laneId) * DEPTH;

  __shared__ uint32_t scratchPad[2 * NUM_W * DEPTH + 32 * NUM_W * DEPTH];
  uint32_t* warp_offsets_smem = scratchPad;
  key_t* keys_ms_smem = &warp_offsets_smem[2 * NUM_W * DEPTH];

  // ===== Storing the results from global memory:
  // memory hierarchy: bucket -> block -> warp -> roll
  uint32_t tid = threadIdx.x;
  #pragma unroll
  while(tid < 2*NUM_W*DEPTH)
  {
    uint32_t whatBin = threadIdx.x / (NUM_W * DEPTH);
    uint32_t whatRoll = threadIdx.x % (NUM_W * DEPTH);
    uint32_t whatWarp = whatRoll / DEPTH;
    whatRoll = whatRoll % DEPTH;
    warp_offsets_smem[threadIdx.x] = warpOffsets[(whatBin * NUM_W * gridDim.x * DEPTH) + (blockIdx.x * NUM_W * DEPTH) + (whatWarp * DEPTH) + whatRoll];
    tid += blockDim.x;
  }
  if((warp_global_offset >= numElements)) return;

  key_t   myInput[DEPTH];
  uint32_t  myNewIndex[DEPTH];  // warp-level indices
  uint32_t  binCounter[DEPTH];  // results of histograms
  uint32_t  scan_histo[DEPTH];

  if(blockIdx.x == (gridDim.x-1))
  {
    // === Histogram and local index computation
    #pragma unroll
    for(int kk = 0; kk<DEPTH; kk++){
      uint32_t global_index = warp_global_offset + (kk << 5);
      uint32_t myBucket = 0;
      bool valid_input = false;
      if((global_index + laneId) < numElements)
      {
        valid_input = true;
        myInput[kk] = key_input[global_index + laneId];
        myBucket = (bucket_identifier(myInput[kk]))?1u:0u;
      }

      uint32_t mask = __ballot(valid_input);

      // Computing the histogram and local indices:
      uint32_t rx_buffer = __ballot(myBucket);
      uint32_t myMask  = 0xFFFFFFFF & ((myBucket)?rx_buffer:(0xFFFFFFFF ^ rx_buffer));
      uint32_t myHisto = 0xFFFFFFFF & (((laneId) & 0x01)?rx_buffer:(0xFFFFFFFF ^ rx_buffer));

      // writing back the local masks:
      binCounter[kk] = __popc(myHisto & mask);
      scan_histo[kk] = binCounter[kk];
      uint32_t n = __shfl(scan_histo[kk], 0, 2);
      if(laneId == 1)
        scan_histo[kk] += n;

      scan_histo[kk] -= binCounter[kk]; //making it exclusive scan.

      // finding its new index within the warp:
      myNewIndex[kk]  = __popc(myMask & (0xFFFFFFFF >> (31-laneId))) - 1;
      myNewIndex[kk] += __shfl(scan_histo[kk], myBucket, 32);

      myNewIndex[kk] = (valid_input)?myNewIndex[kk]:32; // if 32 it means that input was not valid
    }

    // Reordering key elements in shared memory:
    #pragma unroll
    for(int kk = 0; kk<DEPTH; kk++){
        keys_ms_smem[(warpId << 5)*DEPTH + (kk<<5) + ((myNewIndex[kk]<32)?myNewIndex[kk]:laneId)] = myInput[kk];
      }
    __syncthreads();

    #pragma unroll
    for(int kk = 0; kk<DEPTH; kk++){
      uint32_t global_index = warp_global_offset + (kk << 5);
      bool valid_input = ((global_index + laneId) < numElements)?true:false;
      key_t myNewKey = (valid_input)?keys_ms_smem[(warpId << 5)*DEPTH + (kk<<5) + laneId]:0xFFFFFFFF;

      uint32_t finalIndex = (valid_input)?warp_offsets_smem[NUM_W * DEPTH * bucket_identifier(myNewKey) + warpId * DEPTH + kk] + laneId:0;
      finalIndex -= __shfl(scan_histo[kk], bucket_identifier(myNewKey), 32);

      if(valid_input)
        key_output[finalIndex] = myNewKey;
    }
  }
  else{
    // === Histogram and local index computation
    #pragma unroll
    for(int kk = 0; kk<DEPTH; kk++){
      uint32_t global_index = warp_global_offset + (kk << 5);

      myInput[kk] = key_input[global_index + laneId];

      // Computing the histogram and local indices:

      uint32_t myBucket = bucket_identifier(myInput[kk]);
      uint32_t rx_buffer = __ballot(myBucket);
      uint32_t myMask  = 0xFFFFFFFF  & ((myBucket)?rx_buffer:(0xFFFFFFFF ^ rx_buffer));
      uint32_t myHisto = 0xFFFFFFFF & (((laneId) & 0x01)?rx_buffer:(0xFFFFFFFF ^ rx_buffer));

      // writing back the local masks:
      binCounter[kk] = __popc(myHisto);

      scan_histo[kk] = binCounter[kk];

      uint32_t n = __shfl(scan_histo[kk], 0, 2);

      if(laneId == 1)
        scan_histo[kk] += n;

      scan_histo[kk] -= binCounter[kk]; //making it exclusive scan.

      // finding its new index within the warp:
      myNewIndex[kk]  = __popc(myMask & (0xFFFFFFFF >> (31-laneId))) - 1;
      myNewIndex[kk] += __shfl(scan_histo[kk], myBucket, 32);
    }

    // Reordering key elements in shared memory:
    #pragma unroll
    for(int kk = 0; kk<DEPTH; kk++){
        keys_ms_smem[(warpId << 5)*DEPTH + (kk<<5) + myNewIndex[kk]] = myInput[kk];
      }
    __syncthreads();

    #pragma unroll
    for(int kk = 0; kk<DEPTH; kk++){
      key_t myNewKey = keys_ms_smem[(warpId << 5)*DEPTH + (kk<<5) + laneId];

      uint32_t myNewBucket = bucket_identifier(myNewKey);
      uint32_t finalIndex = warp_offsets_smem[NUM_W * DEPTH * myNewBucket + warpId * DEPTH + kk] + laneId;
      finalIndex -= __shfl(scan_histo[kk], myNewBucket, 32);
      key_output[finalIndex] = myNewKey;
    }
  }
}
//=====================================================
template<uint32_t NUM_W, uint32_t DEPTH, typename bucket_t, typename key_t, typename value_t>
__global__ void split_post_scan_pairs_compaction(key_t* key_input, value_t* value_input, uint32_t* warpOffsets, key_t* key_output, value_t* value_output, uint32_t numElements, bucket_t bucket_identifier)
{
  uint32_t  index = threadIdx.x + blockIdx.x * blockDim.x;
  uint32_t  laneId = threadIdx.x & 0x1F;
  uint32_t  warpId = threadIdx.x >> 5;
  uint32_t  warp_global_offset = (index - laneId) * DEPTH;

  __shared__ uint32_t scratchPad[2 * NUM_W * DEPTH + 64 * NUM_W * DEPTH];
  uint32_t* warp_offsets_smem = scratchPad;
  key_t* keys_ms_smem = &warp_offsets_smem[2 * NUM_W * DEPTH];
  value_t* values_ms_smem = &keys_ms_smem[32 * NUM_W * DEPTH];

  // ===== Storing the results from global memory:
  // memory hierarchy: bucket -> block -> warp -> roll
  uint32_t tid = threadIdx.x;
  #pragma unroll
  while(tid < 2*NUM_W*DEPTH)
  {
    uint32_t whatBin = threadIdx.x / (NUM_W * DEPTH);
    uint32_t whatRoll = threadIdx.x % (NUM_W * DEPTH);
    uint32_t whatWarp = whatRoll / DEPTH;
    whatRoll = whatRoll % DEPTH;
    warp_offsets_smem[threadIdx.x] = warpOffsets[(whatBin * NUM_W * gridDim.x * DEPTH) + (blockIdx.x * NUM_W * DEPTH) + (whatWarp * DEPTH) + whatRoll];
    tid += blockDim.x;
  }
  if((warp_global_offset >= numElements)) return;

  key_t   myInput[DEPTH];
  value_t   myValue[DEPTH];
  uint32_t  myNewIndex[DEPTH];  // warp-level indices
  uint32_t  binCounter[DEPTH];  // results of histograms
  uint32_t  scan_histo[DEPTH];

  if(blockIdx.x == (gridDim.x-1))
  {
    // === Histogram and local index computation
    #pragma unroll
    for(int kk = 0; kk<DEPTH; kk++){
      uint32_t global_index = warp_global_offset + (kk << 5);
      uint32_t myBucket = 0;
      bool valid_input = false;
      if((global_index + laneId) < numElements)
      {
        valid_input = true;
        myInput[kk] = key_input[global_index + laneId];
        myValue[kk] = value_input[global_index + laneId];

        myBucket = (bucket_identifier(myInput[kk]))?1u:0u;
      }

      uint32_t mask = __ballot(valid_input);

      // Computing the histogram and local indices:
      uint32_t rx_buffer = __ballot(myBucket);
      uint32_t myMask  = 0xFFFFFFFF & ((myBucket)?rx_buffer:(0xFFFFFFFF ^ rx_buffer));
      uint32_t myHisto = 0xFFFFFFFF & (((laneId) & 0x01)?rx_buffer:(0xFFFFFFFF ^ rx_buffer));

      // writing back the local masks:
      binCounter[kk] = __popc(myHisto & mask);
      scan_histo[kk] = binCounter[kk];
      uint32_t n = __shfl(scan_histo[kk], 0, 2);
      if(laneId == 1)
        scan_histo[kk] += n;

      scan_histo[kk] -= binCounter[kk]; //making it exclusive scan.

      // finding its new index within the warp:
      myNewIndex[kk]  = __popc(myMask & (0xFFFFFFFF >> (31-laneId))) - 1;
      myNewIndex[kk] += __shfl(scan_histo[kk], myBucket, 32);

      myNewIndex[kk] = (valid_input)?myNewIndex[kk]:32; // if 32 it means that input was not valid
    }

    // Reordering key elements in shared memory:
    #pragma unroll
    for(int kk = 0; kk<DEPTH; kk++){
        keys_ms_smem[(warpId << 5)*DEPTH + (kk<<5) + ((myNewIndex[kk]<32)?myNewIndex[kk]:laneId)] = myInput[kk];
        values_ms_smem[(warpId << 5)*DEPTH + (kk<<5) + ((myNewIndex[kk]<32)?myNewIndex[kk]:laneId)] = myValue[kk];
      }
    __syncthreads();

    #pragma unroll
    for(int kk = 0; kk<DEPTH; kk++){
      uint32_t global_index = warp_global_offset + (kk << 5);
      bool valid_input = ((global_index + laneId) < numElements)?true:false;
      key_t myNewKey = (valid_input)?keys_ms_smem[(warpId << 5)*DEPTH + (kk<<5) + laneId]:0xFFFFFFFF;
      value_t myNewValue = (valid_input)?values_ms_smem[(warpId << 5)*DEPTH + (kk<<5) + laneId]:0xFFFFFFFF;

      uint32_t finalIndex = (valid_input)?warp_offsets_smem[NUM_W * DEPTH * bucket_identifier(myNewKey) + warpId * DEPTH + kk] + laneId:0;
      finalIndex -= __shfl(scan_histo[kk], bucket_identifier(myNewKey), 32);

      if(valid_input){
        key_output[finalIndex] = myNewKey;
        value_output[finalIndex] = myNewValue;
      }
    }
  }
  else{
    // === Histogram and local index computation
    #pragma unroll
    for(int kk = 0; kk<DEPTH; kk++){
      uint32_t global_index = warp_global_offset + (kk << 5);

      myInput[kk] = key_input[global_index + laneId];
      myValue[kk] = value_input[global_index + laneId];

      // Computing the histogram and local indices:

      uint32_t myBucket = bucket_identifier(myInput[kk]);
      uint32_t rx_buffer = __ballot(myBucket);
      uint32_t myMask  = 0xFFFFFFFF  & ((myBucket)?rx_buffer:(0xFFFFFFFF ^ rx_buffer));
      uint32_t myHisto = 0xFFFFFFFF & (((laneId) & 0x01)?rx_buffer:(0xFFFFFFFF ^ rx_buffer));

      // writing back the local masks:
      binCounter[kk] = __popc(myHisto);

      scan_histo[kk] = binCounter[kk];

      uint32_t n = __shfl(scan_histo[kk], 0, 2);

      if(laneId == 1)
        scan_histo[kk] += n;

      scan_histo[kk] -= binCounter[kk]; //making it exclusive scan.

      // finding its new index within the warp:
      myNewIndex[kk]  = __popc(myMask & (0xFFFFFFFF >> (31-laneId))) - 1;
      myNewIndex[kk] += __shfl(scan_histo[kk], myBucket, 32);
    }

    // Reordering key elements in shared memory:
    #pragma unroll
    for(int kk = 0; kk<DEPTH; kk++){
        keys_ms_smem[(warpId << 5)*DEPTH + (kk<<5) + myNewIndex[kk]] = myInput[kk];
        values_ms_smem[(warpId << 5)*DEPTH + (kk<<5) + myNewIndex[kk]] = myValue[kk];
      }
    __syncthreads();

    #pragma unroll
    for(int kk = 0; kk<DEPTH; kk++){
      key_t myNewKey = keys_ms_smem[(warpId << 5)*DEPTH + (kk<<5) + laneId];
      value_t myNewValue = values_ms_smem[(warpId << 5)*DEPTH + (kk<<5) + laneId];

      uint32_t myNewBucket = bucket_identifier(myNewKey);
      uint32_t finalIndex = warp_offsets_smem[NUM_W * DEPTH * myNewBucket + warpId * DEPTH + kk] + laneId;
      finalIndex -= __shfl(scan_histo[kk], myNewBucket, 32);
      key_output[finalIndex] = myNewKey;
      value_output[finalIndex] = myNewValue;
    }
  }
}

#endif
//=========================================================================
// Warp-level Multisplit GPU Kernels
//=========================================================================
template<uint32_t NUM_W, uint32_t NUM_B, uint32_t LOG_B, uint32_t DEPTH, typename bucket_t, typename key_type>
__global__ void multisplit_WMS_prescan(key_type* input, uint32_t* bin, uint32_t numElements, bucket_t bucket_identifier)
{
  uint32_t  index = threadIdx.x + blockIdx.x * blockDim.x;
  uint32_t  laneId = threadIdx.x & 0x1F;
  uint32_t  warpId = threadIdx.x >> 5;

  __shared__ uint32_t scratchPad[NUM_B * NUM_W * DEPTH];

  if(blockIdx.x == (gridDim.x - 1)) // last block, potentially may try to read invalid inputs
  {
    // === initializing the shared memory results:
    uint32_t k = 0;
    #pragma unroll
    while((threadIdx.x + k * blockDim.x) < NUM_B * NUM_W * DEPTH)
    {
      scratchPad[threadIdx.x + k * blockDim.x] = 0;
      k++;
    }
    __syncthreads();

    // === computing the histogram:
    key_type  myInput[DEPTH];
    uint32_t  binCounter[DEPTH];  // results of histograms

    #pragma unroll
    for(int kk = 0; kk<DEPTH; kk++){
      uint32_t global_index = (index - laneId) * DEPTH + (kk << 5);
      uint32_t myBucket = 0;
      bool valid_input = false;

      // == reading the input only if valid:
      if((global_index + laneId) < numElements)
      {
        myInput[kk] = input[global_index + laneId];
        valid_input = true;
        myBucket = bucket_identifier(myInput[kk]);
      }

      uint32_t myHisto = 0xFFFFFFFF;
      uint32_t bit = myBucket;
      uint32_t rx_buffer;
      uint32_t mask = __ballot(valid_input);

      // computing histogram
      #pragma unroll
      for(int i = 0; i<LOG_B; i++)
      {
        rx_buffer = __ballot(bit & 0x01);
        myHisto = myHisto & (((laneId >> i) & 0x01)?rx_buffer:(0xFFFFFFFF ^ rx_buffer));
        bit >>= 1;
      }
      binCounter[kk] = __popc(myHisto & mask);
      // === storing the results into the shared memory
      if(laneId < NUM_B)
      {
        // new hierarchy: Bucket -> warp -> roll
        scratchPad[laneId * NUM_W * DEPTH + warpId * DEPTH + kk] = binCounter[kk];
      }
    }
    __syncthreads();

    // === storing histogram results from shared memory into global memory:
    uint32_t tid = threadIdx.x;
    #pragma unroll
    while(tid < (NUM_B * NUM_W * DEPTH))
    {
      // storing histogram results:
      uint32_t whatBin = tid / (NUM_W * DEPTH);
      uint32_t whatRoll = tid % (NUM_W * DEPTH);
      uint32_t whatWarp = whatRoll / DEPTH;
      whatRoll = whatRoll % DEPTH;
      // new hierarchy: Bucket -> block -> warp -> roll:
      uint32_t finalIndex = blockIdx.x * DEPTH * NUM_W + whatWarp * DEPTH + whatRoll;
      bin[whatBin * NUM_W * DEPTH * gridDim.x + finalIndex] = scratchPad[tid];
      // updating tid
      tid += blockDim.x;
    }
  }
  else // all other blocks, that we are sure they are certainly processing valid inputs:
  {
    // === computing the histogram:
    key_type  myInput[DEPTH];
    uint32_t  binCounter[DEPTH];  // results of histograms

    #pragma unroll
    for(int kk = 0; kk<DEPTH; kk++){
      uint32_t global_index = (index - laneId) * DEPTH + (kk << 5);
      uint32_t myBucket = 0;

      myInput[kk] = input[global_index + laneId];
      myBucket = bucket_identifier(myInput[kk]);

      uint32_t myHisto = 0xFFFFFFFF;
      uint32_t bit = myBucket;
      uint32_t rx_buffer;

      #pragma unroll
      for(int i = 0; i<LOG_B; i++)
      {
        rx_buffer = __ballot(bit & 0x01);
        myHisto = myHisto & (((laneId >> i) & 0x01)?rx_buffer:(0xFFFFFFFF ^ rx_buffer));
        bit >>= 1;
      }
      binCounter[kk] = __popc(myHisto);

      // === storing the results into the shared memory
      if(laneId < NUM_B)
      {
        // new hierarchy: Bucket -> warp -> roll
        scratchPad[laneId * NUM_W * DEPTH + warpId * DEPTH + kk] = binCounter[kk];
      }
      __syncthreads();
    }
    // === storing histogram results from shared memory into global memory:
    uint32_t tid = threadIdx.x;
    #pragma unroll
    while(tid < (NUM_B * NUM_W * DEPTH))
    {
      // storing histogram results:
      uint32_t whatBin = tid / (NUM_W * DEPTH);
      uint32_t whatRoll = tid % (NUM_W * DEPTH);
      uint32_t whatWarp = whatRoll / DEPTH;
      whatRoll = whatRoll % DEPTH;
      // new hierarchy: Bucket -> block -> warp -> roll:
      uint32_t finalIndex = blockIdx.x * DEPTH * NUM_W + whatWarp * DEPTH + whatRoll;
      bin[whatBin * NUM_W * DEPTH * gridDim.x + finalIndex] = scratchPad[tid];
      // updating tid
      tid += blockDim.x;
    }
  }
}
//==============================
template<uint32_t NUM_W, uint32_t NUM_B, uint32_t LOG_B, uint32_t DEPTH, typename bucket_t, typename key_type>
__global__ void multisplit_WMS_postscan(key_type* key_input, uint32_t* warpOffsets, key_type* key_output, uint32_t numElements, bucket_t bucket_identifier)
{
  uint32_t  index = threadIdx.x + blockIdx.x * blockDim.x;
  uint32_t  laneId = threadIdx.x & 0x1F;
  uint32_t  warpId = threadIdx.x >> 5;
  uint32_t  warp_global_offset = (index - laneId) * DEPTH;

  __shared__ uint32_t scratchPad[NUM_B * NUM_W * DEPTH + 32 * NUM_W * DEPTH];
  uint32_t* warp_offsets_smem = scratchPad;
  key_type* keys_ms_smem = &warp_offsets_smem[NUM_B * NUM_W * DEPTH];

  // ===== Storing the results from global memory:
  // with new hierarchy: bucket -> block -> warp -> roll
  uint32_t tid = threadIdx.x;
  while(tid < NUM_B*NUM_W*DEPTH)
  {
    uint32_t whatBin = threadIdx.x / (NUM_W * DEPTH);
    uint32_t whatRoll = threadIdx.x % (NUM_W * DEPTH);
    uint32_t whatWarp = whatRoll / DEPTH;
    whatRoll = whatRoll % DEPTH;
    warp_offsets_smem[threadIdx.x] = warpOffsets[(whatBin * NUM_W * gridDim.x * DEPTH) + (blockIdx.x * NUM_W * DEPTH) + (whatWarp * DEPTH) + whatRoll];
    tid += blockDim.x;
  }
  if((warp_global_offset >= numElements)) return;

  uint32_t  myInput[DEPTH];
  uint32_t  myNewIndex[DEPTH];  // warp-level indices
  uint32_t  binCounter[DEPTH];  // results of histograms
  uint32_t  scan_histo[DEPTH];

  if(blockIdx.x == (gridDim.x-1))
  {
    // === Histogram and local index computation
    #pragma unroll
    for(int kk = 0; kk<DEPTH; kk++){
      // uint32_t global_index = index + kk * gridDim.x * blockDim.x;
      uint32_t global_index = warp_global_offset + (kk << 5);
      uint32_t myBucket = 0;
      bool valid_input = false;
      if((global_index + laneId) < numElements)
      {
        valid_input = true;
        myInput[kk] = key_input[global_index + laneId];
        myBucket = bucket_identifier(myInput[kk]);
      }

      uint32_t mask = __ballot(valid_input);
      uint32_t myMask = 0xFFFFFFFF;
      uint32_t myHisto = 0xFFFFFFFF;
      uint32_t bit = myBucket;
      uint32_t rx_buffer;
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
      binCounter[kk] = __popc(myHisto & mask);
      // warp-wide inclusive scan
      uint32_t n;
      scan_histo[kk] = binCounter[kk];
      #pragma unroll
      for(int i = 1; i<(1<<LOG_B); i<<=1)
      {
        n = __shfl_up(scan_histo[kk], i, 32);
        if(laneId >= i)
          scan_histo[kk] += n;
      }
      // making it exclusive scan.
      scan_histo[kk] -= binCounter[kk];

      // finding its new index within the warp:
      myNewIndex[kk]  = __popc(myMask & (0xFFFFFFFF >> (31-laneId))) - 1;
      myNewIndex[kk] += __shfl(scan_histo[kk], myBucket, 32);

      myNewIndex[kk] = (valid_input)?myNewIndex[kk]:32; // if 32 it means that input was not valid
    }

    // Reordering key elements in shared memory:
    #pragma unroll
    for(int kk = 0; kk<DEPTH; kk++){
        keys_ms_smem[(warpId << 5)*DEPTH + (kk<<5) + ((myNewIndex[kk]<32)?myNewIndex[kk]:laneId)] = myInput[kk];
      }
    __syncthreads();

    #pragma unroll
    for(int kk = 0; kk<DEPTH; kk++){
      uint32_t global_index = warp_global_offset + (kk << 5);
      bool valid_input = ((global_index + laneId) < numElements)?true:false;
      key_type myNewKey = (valid_input)?keys_ms_smem[(warpId << 5)*DEPTH + (kk<<5) + laneId]:0xFFFFFFFF;
      uint32_t myNewBucket = bucket_identifier(myNewKey);
      uint32_t finalIndex = (valid_input)?warp_offsets_smem[NUM_W * DEPTH * myNewBucket + warpId * DEPTH + kk] + laneId:0;
      finalIndex -= __shfl(scan_histo[kk], myNewBucket, 32);

      if(valid_input)
        key_output[finalIndex] = myNewKey;
    }
  }
  else{
    // === Histogram and local index computation
    #pragma unroll
    for(int kk = 0; kk<DEPTH; kk++){
      // uint32_t global_index = index + kk * gridDim.x * blockDim.x;
      uint32_t global_index = warp_global_offset + (kk << 5);

      myInput[kk] = key_input[global_index + laneId];
      uint32_t myBucket = bucket_identifier(myInput[kk]);

      uint32_t myMask = 0xFFFFFFFF;
      uint32_t myHisto = 0xFFFFFFFF;
      uint32_t bit = myBucket;
      uint32_t rx_buffer;
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
      // Inclusive scan:
      uint32_t n;
      scan_histo[kk] = binCounter[kk];
      #pragma unroll
      for(int i = 1; i<(1<<LOG_B); i<<=1)
      {
        n = __shfl_up(scan_histo[kk], i, 32);
        if(laneId >= i)
          scan_histo[kk] += n;
      }
      //making it exclusive scan.
      scan_histo[kk] -= binCounter[kk];

      // finding its new index within the warp:
      myNewIndex[kk]  = __popc(myMask & (0xFFFFFFFF >> (31-laneId))) - 1;
      myNewIndex[kk] += __shfl(scan_histo[kk], myBucket, 32);
    }

    // Reordering key elements in shared memory:
    #pragma unroll
    for(int kk = 0; kk<DEPTH; kk++){
        keys_ms_smem[(warpId << 5)*DEPTH + (kk<<5) + ((myNewIndex[kk]<32)?myNewIndex[kk]:laneId)] = myInput[kk];
      }
    __syncthreads();

    #pragma unroll
    for(int kk = 0; kk<DEPTH; kk++){
      key_type myNewKey = keys_ms_smem[(warpId << 5)*DEPTH + (kk<<5) + laneId];
      uint32_t myNewBucket = bucket_identifier(myNewKey);
      uint32_t finalIndex = warp_offsets_smem[NUM_W * DEPTH * myNewBucket + warpId * DEPTH + kk] + laneId;
      finalIndex -= __shfl(scan_histo[kk], myNewBucket, 32);
      key_output[finalIndex] = myNewKey;
    }
  }
}
//=============================
template<uint32_t NUM_W, uint32_t NUM_B, uint32_t LOG_B, uint32_t DEPTH, typename bucket_t, typename key_type, typename value_type>
__global__ void multisplit_WMS_pairs_postscan(key_type* key_input, value_type* value_input, uint32_t* warpOffsets, key_type* key_output, value_type* value_output, uint32_t numElements, bucket_t bucket_identifier)
{
  uint32_t  index = threadIdx.x + blockIdx.x * blockDim.x;
  uint32_t  laneId = threadIdx.x & 0x1F;
  uint32_t  warpId = threadIdx.x >> 5;
  uint32_t  warp_global_offset = (index - laneId) * DEPTH;

  __shared__ uint32_t scratchPad[NUM_B * NUM_W * DEPTH + 64 * NUM_W * DEPTH];
  uint32_t* warp_offsets_smem = scratchPad;
  key_type* keys_ms_smem = &warp_offsets_smem[NUM_B * NUM_W * DEPTH];
  value_type* values_ms_smem = &warp_offsets_smem[32 * NUM_W * DEPTH + NUM_B * NUM_W * DEPTH];

  // ===== Storing the results from global memory:
  // with new hierarchy: bucket -> block -> warp -> roll
  uint32_t tid = threadIdx.x;
  while(tid < NUM_B*NUM_W*DEPTH)
  {
    uint32_t whatBin = threadIdx.x / (NUM_W * DEPTH);
    uint32_t whatRoll = threadIdx.x % (NUM_W * DEPTH);
    uint32_t whatWarp = whatRoll / DEPTH;
    whatRoll = whatRoll % DEPTH;
    warp_offsets_smem[threadIdx.x] = warpOffsets[(whatBin * NUM_W * gridDim.x * DEPTH) + (blockIdx.x * NUM_W * DEPTH) + (whatWarp * DEPTH) + whatRoll];
    tid += blockDim.x;
  }
  if((warp_global_offset >= numElements)) return;

  uint32_t  myInput[DEPTH];
  uint32_t  myValue[DEPTH];
  uint32_t  myNewIndex[DEPTH];  // warp-level indices
  uint32_t  binCounter[DEPTH];  // results of histograms
  uint32_t  scan_histo[DEPTH];

  if(blockIdx.x == (gridDim.x - 1))
  {
    // === Histogram and local index computation
    #pragma unroll
    for(int kk = 0; kk<DEPTH; kk++){
      // uint32_t global_index = index + kk * gridDim.x * blockDim.x;
      uint32_t global_index = warp_global_offset + (kk << 5);
      uint32_t myBucket;
      bool valid_input = false;
      if((global_index + laneId) < numElements)
      {
        valid_input = true;
        myInput[kk] = key_input[global_index + laneId];
        myValue[kk] = value_input[global_index + laneId];
        myBucket = bucket_identifier(myInput[kk]);
      }

      uint32_t mask = __ballot(valid_input);
      uint32_t myMask = 0xFFFFFFFF;
      uint32_t myHisto = 0xFFFFFFFF;
      uint32_t bit = myBucket;
      uint32_t rx_buffer;
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
      binCounter[kk] = __popc(myHisto & mask);

      // Warp-wide Inclusive scan
      uint32_t n;
      scan_histo[kk] = binCounter[kk];
      #pragma unroll
      for(int i = 1; i<(1<<LOG_B); i<<=1)
      {
        n = __shfl_up(scan_histo[kk], i, 32);
        if(laneId >= i)
          scan_histo[kk] += n;
      }
      scan_histo[kk] -= binCounter[kk]; //making it exclusive scan.

      // finding its new index within the warp:
      myNewIndex[kk]  = __popc(myMask & (0xFFFFFFFF >> (31-laneId))) - 1;
      myNewIndex[kk] += __shfl(scan_histo[kk], myBucket, 32);

      myNewIndex[kk] = (valid_input)?myNewIndex[kk]:32; // if 32 it means that input was not valid
    }
    // Reordering key elements in shared memory:
    #pragma unroll
    for(int kk = 0; kk<DEPTH; kk++){
        keys_ms_smem[(warpId << 5)*DEPTH + (kk<<5) + ((myNewIndex[kk]<32)?myNewIndex[kk]:laneId)] = myInput[kk];
        values_ms_smem[(warpId << 5)*DEPTH + (kk<<5) + ((myNewIndex[kk]<32)?myNewIndex[kk]:laneId)] = myValue[kk];
      }
    __syncthreads();

    #pragma unroll
    for(int kk = 0; kk<DEPTH; kk++){
      uint32_t global_index = warp_global_offset + (kk << 5);
      // uint32_t global_index = index + kk * gridDim.x * blockDim.x;
      bool valid_input = ((global_index + laneId) < numElements)?true:false;
      key_type myNewKey = (valid_input)?keys_ms_smem[(warpId << 5)*DEPTH + (kk<<5) + laneId]:0xFFFFFFFF;
      value_type myNewValue = (valid_input)?values_ms_smem[(warpId << 5)*DEPTH + (kk<<5) + laneId]:0xFFFFFFFF;

      uint32_t myNewBucket = bucket_identifier(myNewKey);

      uint32_t finalIndex = (valid_input)?warp_offsets_smem[NUM_W * DEPTH * myNewBucket + warpId * DEPTH + kk] + laneId:0;
      finalIndex -= __shfl(scan_histo[kk], myNewBucket, 32);

      if(valid_input){
        key_output[finalIndex] = myNewKey;
        value_output[finalIndex] = myNewValue;
      }
    }
  }
  else{
    // === Histogram and local index computation
    #pragma unroll
    for(int kk = 0; kk<DEPTH; kk++){
      uint32_t global_index = warp_global_offset + (kk << 5);
      uint32_t myBucket;
      myInput[kk] = key_input[global_index + laneId];
      myValue[kk] = value_input[global_index + laneId];
      myBucket = bucket_identifier(myInput[kk]);

      uint32_t myMask = 0xFFFFFFFF;
      uint32_t myHisto = 0xFFFFFFFF;
      uint32_t bit = myBucket;
      uint32_t rx_buffer;
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
      uint32_t n;
      scan_histo[kk] = binCounter[kk];
      // warp-wide inclusive scan:
      #pragma unroll
      for(int i = 1; i<(1<<LOG_B); i<<=1)
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

    // Reordering key elements in shared memory:
    #pragma unroll
    for(int kk = 0; kk<DEPTH; kk++){
        keys_ms_smem[(warpId << 5)*DEPTH + (kk<<5) + myNewIndex[kk]] = myInput[kk];
        values_ms_smem[(warpId << 5)*DEPTH + (kk<<5) + myNewIndex[kk]] = myValue[kk];
      }
    __syncthreads();

    #pragma unroll
    for(int kk = 0; kk<DEPTH; kk++){
      key_type myNewKey = keys_ms_smem[(warpId << 5)*DEPTH + (kk<<5) + laneId];
      value_type myNewValue = values_ms_smem[(warpId << 5)*DEPTH + (kk<<5) + laneId];
      uint32_t myNewBucket = bucket_identifier(myNewKey);
      uint32_t finalIndex = warp_offsets_smem[NUM_W * DEPTH * myNewBucket + warpId * DEPTH + kk] + laneId;

      finalIndex -= __shfl(scan_histo[kk], myNewBucket, 32);

      key_output[finalIndex] = myNewKey;
      value_output[finalIndex] = myNewValue;
    }
  }
}
//=========================================================================
// Block-level Multisplit GPU Kernels
//=========================================================================
template<uint32_t NUM_W, uint32_t LOG_W, uint32_t NUM_B, uint32_t LOG_B, uint32_t DEPTH, typename bucket_t, typename key_type>
__global__ void multisplit_BMS_prescan(key_type* input, uint32_t* bin, uint32_t numElements, bucket_t bucket_identifier)
{
  uint32_t  laneId = threadIdx.x & 0x1F;
  uint32_t  warpId = threadIdx.x >> 5;

  __shared__ uint32_t scratchPad[NUM_B * NUM_W * DEPTH];

  if(blockIdx.x == (gridDim.x - 1)) // last block
  {
    // === initializing the shared memory results:
    uint32_t k = 0;
    #pragma unroll
    while((threadIdx.x + k * blockDim.x) < NUM_B * NUM_W * DEPTH)
    {
      scratchPad[threadIdx.x + k * blockDim.x] = 0;
      k++;
    }
    __syncthreads();

    // === computing the histogram:
    key_type  myInput[DEPTH];
    uint32_t  binCounter[DEPTH];  // results of histograms

    #pragma unroll
    for(int kk = 0; kk<DEPTH; kk++){
      uint32_t global_index = blockIdx.x * blockDim.x * DEPTH + ((kk * NUM_W) << 5) + (warpId << 5) + laneId;
      uint32_t myBucket = 0;
      bool valid_input = false;

      // == reading the input only if valid:
      if(global_index < numElements)
      {
        myInput[kk] = input[global_index];
        valid_input = true;
        myBucket = bucket_identifier(myInput[kk]);
      }
      uint32_t myHisto = 0xFFFFFFFF;
      uint32_t bit = myBucket;
      uint32_t rx_buffer;
      uint32_t mask = __ballot(valid_input);

      // computing histogram
      #pragma unroll
      for(int i = 0; i<LOG_B; i++)
      {
        rx_buffer = __ballot(bit & 0x01);
        myHisto = myHisto & (((laneId >> i) & 0x01)?rx_buffer:(0xFFFFFFFF ^ rx_buffer));
        bit >>= 1;
      }
      binCounter[kk] = __popc(myHisto & mask);
      // === storing the results into the shared memory
      if(laneId < NUM_B)
      {
        // local hierarchy: roll -> warp -> bucket
        scratchPad[kk * NUM_W * NUM_B + warpId * NUM_B + laneId] = binCounter[kk];
      }
    }
    __syncthreads();

    // === computing our multi-reduction:
    #pragma unroll
    for(int i = 1; i <= LOG_W; i++)
    {
      // Performing reduction over all elements:
      #pragma unroll
      for(int kk = 0; kk<DEPTH; kk++)
      {
        if((warpId & ((1<<i)-1)) == 0)
        {
          if(laneId < NUM_B){
            scratchPad[laneId + warpId * NUM_B + kk * NUM_W * NUM_B] += scratchPad[laneId + (warpId + (1<<(i-1)))*NUM_B + kk * NUM_W * NUM_B];
          }
        }
      }
      __syncthreads();
    }

    // === storing the results back to the global memory with different hierarchy
    uint32_t tid = threadIdx.x;
    #pragma unroll
    while(tid < DEPTH * NUM_B)
    {
      uint32_t bucket_src = tid % NUM_B;
      uint32_t roll_src   = tid / NUM_B;

      // global memory hierarchy: bucket -> block -> roll
      bin[bucket_src * gridDim.x * DEPTH + blockIdx.x * DEPTH + roll_src] = scratchPad[bucket_src + roll_src * NUM_W * NUM_B];
      tid += blockDim.x;
    }
  }
  else // all other blocks
  {
    // === computing the histogram:
    key_type  myInput[DEPTH];
    uint32_t  binCounter[DEPTH];  // results of histograms

    #pragma unroll
    for(int kk = 0; kk<DEPTH; kk++){
      uint32_t global_index = blockIdx.x * blockDim.x * DEPTH + ((kk * NUM_W) << 5) + (warpId << 5) + laneId;
      uint32_t myBucket = 0;

    // == reading the input only if valid:
      myInput[kk] = input[global_index];
      myBucket = bucket_identifier(myInput[kk]);
      uint32_t myHisto = 0xFFFFFFFF;
      uint32_t bit = myBucket;
      uint32_t rx_buffer;

      // == computing warp-wide histogram
      #pragma unroll
      for(int i = 0; i<LOG_B; i++)
      {
        rx_buffer = __ballot(bit & 0x01);
        myHisto = myHisto & (((laneId >> i) & 0x01)?rx_buffer:(0xFFFFFFFF ^ rx_buffer));
        bit >>= 1;
      }
      binCounter[kk] = __popc(myHisto);
      // === storing the results into the shared memory
      if(laneId < NUM_B)
      {
        // local hierarchy: roll -> warp -> bucket
        scratchPad[kk * NUM_W * NUM_B + warpId * NUM_B + laneId] = binCounter[kk];
      }
    }
    __syncthreads();

    // === computing our multi-reduction:
    #pragma unroll
    for(int i = 1; i <= LOG_W; i++)
    {
      // Performing reduction over all elements:
      #pragma unroll
      for(int kk = 0; kk<DEPTH; kk++)
      {
        if((warpId & ((1<<i)-1)) == 0)
        {
          if(laneId < NUM_B){
            scratchPad[laneId + warpId * NUM_B + kk * NUM_W * NUM_B] += scratchPad[laneId + (warpId + (1<<(i-1)))*NUM_B + kk * NUM_W * NUM_B];
          }
        }
      }
      __syncthreads();
    }

    // Global memory hierarchy: Bucket -> block -> roll
    if(NUM_W >= DEPTH)
    {
      if((laneId < NUM_B) && (warpId < DEPTH))
        bin[laneId * gridDim.x * DEPTH + blockIdx.x * DEPTH + warpId] = scratchPad[laneId + warpId * NUM_B * NUM_W];
    }
  }
}
//==================================
template<uint32_t NUM_W, uint32_t LOG_W, uint32_t NUM_B, uint32_t LOG_B, uint32_t DEPTH, typename bucket_t, typename key_type>
__global__ void multisplit_BMS_postscan(key_type* key_input, uint32_t* blockOffsets, key_type* key_output, uint32_t numElements, bucket_t bucket_identifier)
{
  __shared__ uint32_t scratchPad[2 * NUM_B * DEPTH + 32 * NUM_W * DEPTH + NUM_B * NUM_W * DEPTH];
  uint32_t* block_offsets_smem = scratchPad;
  uint32_t* warp_offsets_smem = &block_offsets_smem[NUM_B * DEPTH];
  uint32_t* scan_histo_smem = &warp_offsets_smem[NUM_B * DEPTH];
  key_type* keys_ms_smem = &scan_histo_smem[NUM_B * NUM_W * DEPTH];

  uint32_t  laneId = threadIdx.x & 0x1F;
  uint32_t  warpId = threadIdx.x >> 5;

  uint32_t  myInput[DEPTH];
  uint32_t  binCounter[DEPTH];  // results of histograms
  uint32_t  scan_temp[DEPTH];

  // === Loading block offset results from global memory:
  uint32_t tid = threadIdx.x;
  #pragma unroll
  while(tid < NUM_B * DEPTH)
  {
    uint32_t bucket_src = tid % NUM_B;
    uint32_t roll_src = tid / NUM_B;
    block_offsets_smem[bucket_src + roll_src * NUM_B] = blockOffsets[roll_src + blockIdx.x * DEPTH + bucket_src * DEPTH * gridDim.x];
    tid += blockDim.x;
  }
  __syncthreads();

  if(blockIdx.x == (gridDim.x - 1)) // last block
  {
    // === Histogram and local index computation
    #pragma unroll
    for(int kk = 0; kk<DEPTH; kk++){
      uint32_t global_index = blockIdx.x * blockDim.x * DEPTH + ((kk * NUM_W) << 5) + (warpId << 5) + laneId;
      uint32_t myBucket = 0;
      uint32_t valid_input = false;
      if(global_index < numElements)
      {
        myInput[kk] = key_input[global_index];
        valid_input = true;
        myBucket = bucket_identifier(myInput[kk]);
      }

      uint32_t myHisto = 0xFFFFFFFF;
      uint32_t myLocalIndex = 0xFFFFFFFF;
      uint32_t bit = myBucket;
      uint32_t rx_buffer;
      uint32_t mask = __ballot(valid_input);

      // computing histogram
      #pragma unroll
      for(int i = 0; i<LOG_B; i++)
      {
        rx_buffer = __ballot(bit & 0x01);
        myLocalIndex  = myLocalIndex  & ((bit & 0x01)?rx_buffer:(0xFFFFFFFF ^ rx_buffer));
        myHisto = myHisto & (((laneId >> i) & 0x01)?rx_buffer:(0xFFFFFFFF ^ rx_buffer));
        bit >>= 1;
      }
      binCounter[kk] = __popc(myHisto & mask);

      // storing the results in smem in order to be scanned:
      // smem hierarchy: roll -> warp -> bucket
      if(laneId < NUM_B)
        scan_histo_smem[laneId + warpId * NUM_B + kk * NUM_B * NUM_W] = binCounter[kk];
      __syncthreads();
      // computing block-wise scan over buckets:
      scan_temp[kk] = binCounter[kk];
      for(int i = 1; i<(1<<LOG_W) ; i<<=1)
      {
        if(laneId < NUM_B)
          scan_temp[kk] += ((warpId >= i)?scan_histo_smem[kk * NUM_B * NUM_W + (warpId-i)*NUM_B + laneId]:0);
        __syncthreads();
        if(laneId < NUM_B)
          scan_histo_smem[kk * NUM_B * NUM_W + warpId * NUM_B + laneId] = scan_temp[kk];
        __syncthreads();
      }
      // Computing block-level indices:
      scan_temp[kk] -= binCounter[kk]; // exclusive scan
      uint32_t myLocalBlockIndex = __shfl(scan_temp[kk], myBucket, 32);
      myLocalBlockIndex += __popc(myLocalIndex & (0xFFFFFFFF >> (31-laneId))) - 1;
      myLocalBlockIndex = (valid_input)?myLocalBlockIndex:blockDim.x;

      // Computing warp-level offsets within each block:
      uint32_t block_scan;
      if(warpId == (NUM_W-1)) // the last warp
      {
        block_scan = scan_temp[kk] + binCounter[kk];
        // warp-wide inclusive scan:
        uint32_t n;
        #pragma unroll
        for(int i = 1; i<(1<<LOG_B); i<<=1)
        {
          n = __shfl_up(block_scan, i, 32);
          if(laneId >= i)
            block_scan += n;
        }
        scan_temp[kk] += binCounter[kk];
        // making it exclusive
        block_scan -= scan_temp[kk];
        if(laneId < NUM_B){
          warp_offsets_smem[laneId + kk * NUM_B] = block_scan;
        }
      }
      __syncthreads();
      uint32_t myNewBlockIndex = warp_offsets_smem[myBucket + kk * NUM_B] + myLocalBlockIndex;
      // block-level reordering in shared memory
      keys_ms_smem[((myLocalBlockIndex < blockDim.x)?myNewBlockIndex:threadIdx.x) + kk * blockDim.x] = myInput[kk];
    }
    __syncthreads();
    #pragma unroll
    for(int kk = 0; kk<DEPTH; kk++){
      uint32_t global_index = blockIdx.x * blockDim.x * DEPTH + ((kk * NUM_W) << 5) + (warpId << 5) + laneId;
      bool valid_input = (global_index < numElements)?true:false;

      key_type myNewKey = (valid_input)?keys_ms_smem[threadIdx.x + kk * blockDim.x]:0xFFFFFFFF;
      uint32_t myNewBucket = bucket_identifier(myNewKey);
      uint32_t finalIndex = 0;
      if(valid_input) {
        finalIndex = block_offsets_smem[NUM_B * kk + myNewBucket] + threadIdx.x;
        finalIndex -= warp_offsets_smem[myNewBucket + kk * NUM_B];
        key_output[finalIndex] = myNewKey;
      }
    }
  }
  else // all others
  {
    // === Histogram and local index computation
    #pragma unroll
    for(int kk = 0; kk<DEPTH; kk++){
      uint32_t global_index = blockIdx.x * blockDim.x * DEPTH + ((kk * NUM_W) << 5) + (warpId << 5) + laneId;
      uint32_t myBucket = 0;
      myInput[kk] = key_input[global_index];
      myBucket = bucket_identifier(myInput[kk]);

      uint32_t myHisto = 0xFFFFFFFF;
      uint32_t myLocalIndex = 0xFFFFFFFF;
      uint32_t bit = myBucket;
      uint32_t rx_buffer;

      // computing histogram
      #pragma unroll
      for(int i = 0; i<LOG_B; i++)
      {
        rx_buffer = __ballot(bit & 0x01);
        myLocalIndex  = myLocalIndex  & ((bit & 0x01)?rx_buffer:(0xFFFFFFFF ^ rx_buffer));
        myHisto = myHisto & (((laneId >> i) & 0x01)?rx_buffer:(0xFFFFFFFF ^ rx_buffer));
        bit >>= 1;
      }
      binCounter[kk] = __popc(myHisto);
      // storing the results in smem in order to be scanned:
      // smem hierarcy: roll -> warp -> bucket
      if(laneId < NUM_B)
        scan_histo_smem[laneId + warpId * NUM_B + kk * NUM_B * NUM_W] = binCounter[kk];

      __syncthreads();
      // computing block-wise scan over buckets:
      scan_temp[kk] = binCounter[kk];
      for(int i = 1; i<(1<<LOG_W) ; i<<=1)
      {
        if(laneId < NUM_B)
          scan_temp[kk] += ((warpId >= i)?scan_histo_smem[kk * NUM_B * NUM_W + (warpId-i)*NUM_B + laneId]:0);
        __syncthreads();
        if(laneId < NUM_B)
          scan_histo_smem[kk * NUM_B * NUM_W + warpId * NUM_B + laneId] = scan_temp[kk];
        __syncthreads();
      }
      // Computing block-level indices:
      scan_temp[kk] -= binCounter[kk]; // exclusive scan
      uint32_t myLocalBlockIndex = __shfl(scan_temp[kk], myBucket, 32);
      myLocalBlockIndex += __popc(myLocalIndex & (0xFFFFFFFF >> (31-laneId))) - 1;

      // Computing warp-level offsets within each block:
      uint32_t block_scan;
      if(warpId == (NUM_W-1)) // the last warp
      {
        block_scan = scan_temp[kk] + binCounter[kk];
        // warp-wide inclusive scan
        uint32_t n;
        #pragma unroll
        for(int i = 1; i<(1<<LOG_B); i<<=1)
        {
          n = __shfl_up(block_scan, i, 32);
          if(laneId >= i)
            block_scan += n;
        }
        scan_temp[kk] += binCounter[kk];
        // making it exclusive
        block_scan -= scan_temp[kk];
        if(laneId < NUM_B){
          warp_offsets_smem[laneId + kk * NUM_B] = block_scan;
        }
      }
      __syncthreads();
      uint32_t myNewBlockIndex = warp_offsets_smem[myBucket + kk * NUM_B] + myLocalBlockIndex;
      // block-level reordering in shared memory
      keys_ms_smem[myNewBlockIndex + kk * blockDim.x] = myInput[kk];
    }
    __syncthreads();
    #pragma unroll
    for(int kk = 0; kk<DEPTH; kk++){
      key_type myNewKey = keys_ms_smem[threadIdx.x + kk * blockDim.x];
      uint32_t myNewBucket = bucket_identifier(myNewKey);
      uint32_t finalIndex = block_offsets_smem[NUM_B * kk + myNewBucket] + threadIdx.x;
      finalIndex -= warp_offsets_smem[myNewBucket + kk * NUM_B];
      key_output[finalIndex] = myNewKey;
    }
  }
}
//===============================
template<uint32_t NUM_W, uint32_t LOG_W, uint32_t NUM_B, uint32_t LOG_B, uint32_t DEPTH, typename bucket_t, typename key_type, typename value_type>
__global__ void multisplit_BMS_pairs_postscan(key_type* key_input, value_type* value_input, uint32_t* blockOffsets, key_type* key_output, value_type* value_output, uint32_t numElements, bucket_t bucket_identifier)
{
  __shared__ uint32_t scratchPad[2 * NUM_B * DEPTH + 64 * NUM_W * DEPTH + NUM_B * NUM_W * DEPTH];
  uint32_t* block_offsets_smem = scratchPad;
  uint32_t* warp_offsets_smem = &block_offsets_smem[NUM_B * DEPTH];
  uint32_t* scan_histo_smem = &warp_offsets_smem[NUM_B * DEPTH];
  key_type* keys_ms_smem = &scan_histo_smem[NUM_B * NUM_W * DEPTH];
  value_type* values_ms_smem = &scan_histo_smem[NUM_B * NUM_W * DEPTH + 32 * NUM_W * DEPTH];

  uint32_t  laneId = threadIdx.x & 0x1F;
  uint32_t  warpId = threadIdx.x >> 5;

  uint32_t  myInput[DEPTH];
  uint32_t  myValue[DEPTH];
  uint32_t  binCounter[DEPTH];  // results of histograms
  uint32_t  scan_temp[DEPTH];

  // === Loading block offset results from global memory:
  uint32_t tid = threadIdx.x;
  #pragma unroll
  while(tid < NUM_B * DEPTH)
  {
    uint32_t bucket_src = tid % NUM_B;
    uint32_t roll_src = tid / NUM_B;
    block_offsets_smem[bucket_src + roll_src * NUM_B] = blockOffsets[roll_src + blockIdx.x * DEPTH + bucket_src * DEPTH * gridDim.x];
    tid += blockDim.x;
  }
  __syncthreads();

  if(blockIdx.x == (gridDim.x - 1)) // last block
  {
    // === Histogram and local index computation
    #pragma unroll
    for(int kk = 0; kk<DEPTH; kk++){
      uint32_t global_index = blockIdx.x * blockDim.x * DEPTH + ((kk * NUM_W) << 5) + (warpId << 5) + laneId;
      uint32_t myBucket = 0;
      uint32_t valid_input = false;
      if(global_index < numElements)
      {
        myInput[kk] = key_input[global_index];
        myValue[kk] = value_input[global_index];
        valid_input = true;
        myBucket = bucket_identifier(myInput[kk]);
      }

      uint32_t myHisto = 0xFFFFFFFF;
      uint32_t myLocalIndex = 0xFFFFFFFF;
      uint32_t bit = myBucket;
      uint32_t rx_buffer;
      uint32_t mask = __ballot(valid_input);

      // computing histogram
      #pragma unroll
      for(int i = 0; i<LOG_B; i++)
      {
        rx_buffer = __ballot(bit & 0x01);
        myLocalIndex  = myLocalIndex  & ((bit & 0x01)?rx_buffer:(0xFFFFFFFF ^ rx_buffer));
        myHisto = myHisto & (((laneId >> i) & 0x01)?rx_buffer:(0xFFFFFFFF ^ rx_buffer));
        bit >>= 1;
      }
      binCounter[kk] = __popc(myHisto & mask);
      // storing the results in smem in order to be scanned:
      // smem hierarcy: roll -> warp -> bucket
      if(laneId < NUM_B)
        scan_histo_smem[laneId + warpId * NUM_B + kk * NUM_B * NUM_W] = binCounter[kk];
      __syncthreads();
      // computing block-wise scan over buckets:
      scan_temp[kk] = binCounter[kk];
      for(int i = 1; i<(1<<LOG_W) ; i<<=1)
      {
        if(laneId < NUM_B)
          scan_temp[kk] += ((warpId >= i)?scan_histo_smem[kk * NUM_B * NUM_W + (warpId-i)*NUM_B + laneId]:0);
        __syncthreads();
        if(laneId < NUM_B)
          scan_histo_smem[kk * NUM_B * NUM_W + warpId * NUM_B + laneId] = scan_temp[kk];
        __syncthreads();
      }
      // Computing block-level indices:
      scan_temp[kk] -= binCounter[kk]; // exclusive scan
      uint32_t myLocalBlockIndex = __shfl(scan_temp[kk], myBucket, 32);
      myLocalBlockIndex += __popc(myLocalIndex & (0xFFFFFFFF >> (31-laneId))) - 1;
      myLocalBlockIndex = (valid_input)?myLocalBlockIndex:blockDim.x;

      // Computing warp-level offsets within each block:
      uint32_t block_scan;
      if(warpId == (NUM_W-1)) // the last warp
      {
        block_scan = scan_temp[kk] + binCounter[kk];
        // warp-wide inclusive scan
        uint32_t n;
        #pragma unroll
        for(int i = 1; i<(1<<LOG_B); i<<=1)
        {
          n = __shfl_up(block_scan, i, 32);
          if(laneId >= i)
            block_scan += n;
        }
        scan_temp[kk] += binCounter[kk];
        // making it exclusive
        block_scan -= scan_temp[kk];
        if(laneId < NUM_B){
          warp_offsets_smem[laneId + kk * NUM_B] = block_scan;
        }
      }
      __syncthreads();
      uint32_t myNewBlockIndex = warp_offsets_smem[myBucket + kk * NUM_B] + myLocalBlockIndex;
      // block-level reordering in shared memory
      keys_ms_smem[((myLocalBlockIndex < blockDim.x)?myNewBlockIndex:threadIdx.x) + kk * blockDim.x] = myInput[kk];
      values_ms_smem[((myLocalBlockIndex < blockDim.x)?myNewBlockIndex:threadIdx.x) + kk * blockDim.x] = myValue[kk];
    }
    __syncthreads();
    #pragma unroll
    for(int kk = 0; kk<DEPTH; kk++){
      uint32_t global_index = blockIdx.x * blockDim.x * DEPTH + ((kk * NUM_W) << 5) + (warpId << 5) + laneId;
      bool valid_input = (global_index < numElements)?true:false;

      key_type myNewKey = (valid_input)?keys_ms_smem[threadIdx.x + kk * blockDim.x]:0xFFFFFFFF;
      value_type myNewValue = (valid_input)?values_ms_smem[threadIdx.x + kk * blockDim.x]:0xFFFFFFFF;
      uint32_t myNewBucket = bucket_identifier(myNewKey);
      uint32_t finalIndex = 0;
      if(valid_input) {
        finalIndex = block_offsets_smem[NUM_B * kk + myNewBucket] + threadIdx.x;
        finalIndex -= warp_offsets_smem[myNewBucket + kk * NUM_B];
        key_output[finalIndex] = myNewKey;
        value_output[finalIndex] = myNewValue;
      }
    }
  }
  else // all others
  {
    // === Histogram and local index computation
    #pragma unroll
    for(int kk = 0; kk<DEPTH; kk++){
      uint32_t global_index = blockIdx.x * blockDim.x * DEPTH + ((kk * NUM_W) << 5) + (warpId << 5) + laneId;
      uint32_t myBucket = 0;
      myInput[kk] = key_input[global_index];
      myValue[kk] = value_input[global_index];
      myBucket = bucket_identifier(myInput[kk]);

      uint32_t myHisto = 0xFFFFFFFF;
      uint32_t myLocalIndex = 0xFFFFFFFF;
      uint32_t bit = myBucket;
      uint32_t rx_buffer;

      // computing histogram
      #pragma unroll
      for(int i = 0; i<LOG_B; i++)
      {
        rx_buffer = __ballot(bit & 0x01);
        myLocalIndex  = myLocalIndex  & ((bit & 0x01)?rx_buffer:(0xFFFFFFFF ^ rx_buffer));
        myHisto = myHisto & (((laneId >> i) & 0x01)?rx_buffer:(0xFFFFFFFF ^ rx_buffer));
        bit >>= 1;
      }
      binCounter[kk] = __popc(myHisto);
      // storing the results in smem in order to be scanned:
      // smem hierarcy: roll -> warp -> bucket
      if(laneId < NUM_B)
        scan_histo_smem[laneId + warpId * NUM_B + kk * NUM_B * NUM_W] = binCounter[kk];

      __syncthreads();
      // computing block-wise scan over buckets:
      scan_temp[kk] = binCounter[kk];
      for(int i = 1; i<(1<<LOG_W) ; i<<=1)
      {
        if(laneId < NUM_B)
          scan_temp[kk] += ((warpId >= i)?scan_histo_smem[kk * NUM_B * NUM_W + (warpId-i)*NUM_B + laneId]:0);
        __syncthreads();
        if(laneId < NUM_B)
          scan_histo_smem[kk * NUM_B * NUM_W + warpId * NUM_B + laneId] = scan_temp[kk];
        __syncthreads();
      }
      // Computing block-level indices:
      scan_temp[kk] -= binCounter[kk]; // exclusive scan
      uint32_t myLocalBlockIndex = __shfl(scan_temp[kk], myBucket, 32);
      myLocalBlockIndex += __popc(myLocalIndex & (0xFFFFFFFF >> (31-laneId))) - 1;

      // Computing warp-level offsets within each block:
      uint32_t block_scan;
      if(warpId == (NUM_W-1)) // the last warp
      {
        block_scan = scan_temp[kk] + binCounter[kk];
        uint32_t n;
        #pragma unroll
        for(int i = 1; i<(1<<LOG_B); i<<=1)
        {
          n = __shfl_up(block_scan, i, 32);
          if(laneId >= i)
            block_scan += n;
        }
        scan_temp[kk] += binCounter[kk];
        block_scan -= scan_temp[kk];
        if(laneId < NUM_B){
          warp_offsets_smem[laneId + kk * NUM_B] = block_scan;
        }
      }
      __syncthreads();
      uint32_t myNewBlockIndex = warp_offsets_smem[myBucket + kk * NUM_B] + myLocalBlockIndex;
      // block-level reordering in shared memory
      keys_ms_smem[myNewBlockIndex + kk * blockDim.x] = myInput[kk];
      values_ms_smem[myNewBlockIndex + kk * blockDim.x] = myValue[kk];
    }
    __syncthreads();
    #pragma unroll
    for(int kk = 0; kk<DEPTH; kk++){
      key_type myNewKey = keys_ms_smem[threadIdx.x + kk * blockDim.x];
      value_type myNewValue = values_ms_smem[threadIdx.x + kk * blockDim.x];
      uint32_t myNewBucket = bucket_identifier(myNewKey);
      uint32_t finalIndex = block_offsets_smem[NUM_B * kk + myNewBucket] + threadIdx.x;
      finalIndex -= warp_offsets_smem[myNewBucket + kk * NUM_B];
      key_output[finalIndex] = myNewKey;
      value_output[finalIndex] = myNewValue;
    }
  }
}

/** @} */// end Multisplit functions
/** @} */// end cudpp_kernel
