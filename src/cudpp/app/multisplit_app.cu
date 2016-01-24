// -------------------------------------------------------------
// CUDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision$
// $Date$
// -------------------------------------------------------------
// This source code is distributed under the terms of license.txt
// in the root directory of this source distribution.
// -------------------------------------------------------------

/**
 * @file
 * multisplit_app.cu
 *
 * @brief CUDPP application-level multisplit routines
 */

/** @addtogroup cudpp_app
 * @{
 */

/** @name MultiSplit Functions
 * @{
 */
#include <cub/cub.cuh>
#include "cuda_util.h"
#include "cudpp.h"
#include "cudpp_util.h"
#include "limits.h"
#include "kernel/multisplit_kernel.cuh"


//===============================================
// Global
//===============================================
cub::CachingDeviceAllocator  g_allocator(true);  // Caching allocator for device memory
#define gpuErrCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
typedef unsigned long long int uint64;
//===============================================
// Definitions:
//===============================================
#define NUM_WARPS 8
#define LOG_WARPS 3 // = ceil(log2(NUM_WARPS))
#define SMEM_BUCK_SIZE (1536/(NUM_BUCKETS*NUM_WARPS))
#define PACK_DEPTH 2
#define PACK_PRE 8
#define PACK_POST 2

#define BLOCKSORT_SIZE 1024
#define DEPTH 8

/** @brief Performs merge sort utilizing 3 stages:
 * (1) Blocksort, (2) simple merge and (3) multi merge
 *
 *
 * @param[in,out] pkeys Keys to be sorted.
 * @param[in,out] pvals Associated values to be sorted
 * @param[in] numElements Number of elements in the sort.
 * @param[in] plan Configuration information for mergesort.
 **/
void runMultiSplit(unsigned int *d_inp, uint numElements, uint numBuckets, const CUDPPMultiSplitPlan *plan) {
  unsigned int numBlocks = ceil(numElements / (NUM_WARPS * 32));
  unsigned int numThreads = NUM_WARPS * 32;
  void     *d_temp_storage = NULL;
  size_t   temp_storage_bytes = 0;

  if (numBuckets == 1)
    return;

  if (numBuckets == 2) {
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, plan->m_d_histo, plan->m_d_histo, numBuckets * numBlocks * NUM_WARPS * PACK_DEPTH);
  } else if (numBuckets <= 32) {
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, plan->m_d_histo, plan->m_d_histo, numBuckets * numBlocks * PACK_DEPTH);
  } else if (numBuckets > 96) {
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, plan->m_d_mask, plan->m_d_out, d_inp, plan->m_d_fin,
        numElements, 0, plan->m_logBuckets);
  } else if (numBuckets <= 96){
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, plan->m_d_histo, plan->m_d_histo, numBuckets * numBlocks);
  } else {
    printf("Bad number of buckets: %u\n", numBuckets);
    return;
  }
  g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes);

  if (numBuckets == 2) {
    histogram_warp<NUM_WARPS, 2, 1, PACK_PRE> <<<numBlocks / PACK_PRE, numThreads>>>(d_inp,
        plan->m_d_histo, numElements);
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, plan->m_d_histo, plan->m_d_histo, 2 * numBlocks * NUM_WARPS * PACK_DEPTH);
    split_WMS<NUM_WARPS, 2, 1, PACK_POST><<<numBlocks/PACK_POST, numThreads>>>(d_inp, plan->m_d_histo, plan->m_d_fin, numElements);
  } else if (numBuckets <= 32) {
    histogram_block<<<numBlocks / PACK_PRE, numThreads,
        NUM_WARPS * numBuckets * DEPTH * sizeof(uint)>>>(d_inp, plan->m_d_histo,
        numElements, numBuckets, NUM_WARPS, PACK_PRE);
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, plan->m_d_histo, plan->m_d_histo, numBuckets * numBlocks * PACK_DEPTH);
    split_BMS<<<numBlocks / PACK_POST, numThreads,
        (2 * numBuckets * PACK_POST + 32 * NUM_WARPS * PACK_POST
            + numBuckets * NUM_WARPS * PACK_POST) * sizeof(uint)>>>(d_inp, plan->m_d_histo, plan->m_d_fin,
        numElements, numBuckets, NUM_WARPS, PACK_POST);
  } else if (numBuckets > 96) {
    markBins_general<<<numBlocks, numThreads>>>(plan->m_d_mask, d_inp, numElements, numBuckets);
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, plan->m_d_mask,
        plan->m_d_out, d_inp, plan->m_d_fin, numElements, 0,
        int(ceil(log2(float(numBuckets)))));
  } else if (numBuckets <= 96) {
    switch(numBuckets){
      case 33:
        histogramBallot_Mode13_large<NUM_WARPS, 33, 6, LOG_WARPS> <<<
          numBlocks, numThreads>>>(d_inp, plan->m_d_histo, numElements);
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, plan->m_d_histo,
          plan->m_d_histo, 33 * numBlocks);
        splitBallot_Mode13_large<NUM_WARPS, 33> <<<numBlocks, numThreads>>>(d_inp, plan->m_d_histo,
          plan->m_d_fin, numElements);
      break;
      case 34:
        histogramBallot_Mode13_large<NUM_WARPS, 34, 6, LOG_WARPS> <<<
          numBlocks, numThreads>>>(d_inp, plan->m_d_histo, numElements);
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, plan->m_d_histo,
          plan->m_d_histo, 34 * numBlocks);
        splitBallot_Mode13_large<NUM_WARPS, 34> <<<numBlocks, numThreads>>>(d_inp, plan->m_d_histo,
          plan->m_d_fin, numElements);
      break;
      case 35:
        histogramBallot_Mode13_large<NUM_WARPS, 35, 6, LOG_WARPS> <<<
          numBlocks, numThreads>>>(d_inp, plan->m_d_histo, numElements);
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, plan->m_d_histo,
          plan->m_d_histo, 35 * numBlocks);
        splitBallot_Mode13_large<NUM_WARPS, 35> <<<numBlocks, numThreads>>>(d_inp, plan->m_d_histo,
          plan->m_d_fin, numElements);
      break;
      case 36:
        histogramBallot_Mode13_large<NUM_WARPS, 36, 6, LOG_WARPS> <<<
          numBlocks, numThreads>>>(d_inp, plan->m_d_histo, numElements);
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, plan->m_d_histo,
          plan->m_d_histo, 36 * numBlocks);
        splitBallot_Mode13_large<NUM_WARPS, 36> <<<numBlocks, numThreads>>>(d_inp, plan->m_d_histo,
          plan->m_d_fin, numElements);
      break;
      case 37:
        histogramBallot_Mode13_large<NUM_WARPS, 37, 6, LOG_WARPS> <<<
          numBlocks, numThreads>>>(d_inp, plan->m_d_histo, numElements);
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, plan->m_d_histo,
          plan->m_d_histo, 37 * numBlocks);
        splitBallot_Mode13_large<NUM_WARPS, 37> <<<numBlocks, numThreads>>>(d_inp, plan->m_d_histo,
          plan->m_d_fin, numElements);
      break;
      case 38:
        histogramBallot_Mode13_large<NUM_WARPS, 38, 6, LOG_WARPS> <<<
          numBlocks, numThreads>>>(d_inp, plan->m_d_histo, numElements);
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, plan->m_d_histo,
          plan->m_d_histo, 38 * numBlocks);
        splitBallot_Mode13_large<NUM_WARPS, 38> <<<numBlocks, numThreads>>>(d_inp, plan->m_d_histo,
          plan->m_d_fin, numElements);
      break;
      case 39:
        histogramBallot_Mode13_large<NUM_WARPS, 39, 6, LOG_WARPS> <<<
          numBlocks, numThreads>>>(d_inp, plan->m_d_histo, numElements);
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, plan->m_d_histo,
          plan->m_d_histo, 39 * numBlocks);
        splitBallot_Mode13_large<NUM_WARPS, 39> <<<numBlocks, numThreads>>>(d_inp, plan->m_d_histo,
          plan->m_d_fin, numElements);
      break;
      case 40:
        histogramBallot_Mode13_large<NUM_WARPS, 40, 6, LOG_WARPS> <<<
          numBlocks, numThreads>>>(d_inp, plan->m_d_histo, numElements);
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, plan->m_d_histo,
          plan->m_d_histo, 40 * numBlocks);
        splitBallot_Mode13_large<NUM_WARPS, 40> <<<numBlocks, numThreads>>>(d_inp, plan->m_d_histo,
          plan->m_d_fin, numElements);
      break;
      case 41:
        histogramBallot_Mode13_large<NUM_WARPS, 41, 6, LOG_WARPS> <<<
          numBlocks, numThreads>>>(d_inp, plan->m_d_histo, numElements);
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, plan->m_d_histo,
          plan->m_d_histo, 41 * numBlocks);
        splitBallot_Mode13_large<NUM_WARPS, 41> <<<numBlocks, numThreads>>>(d_inp, plan->m_d_histo,
          plan->m_d_fin, numElements);
      break;
      case 42:
        histogramBallot_Mode13_large<NUM_WARPS, 42, 6, LOG_WARPS> <<<
          numBlocks, numThreads>>>(d_inp, plan->m_d_histo, numElements);
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, plan->m_d_histo,
          plan->m_d_histo, 42 * numBlocks);
        splitBallot_Mode13_large<NUM_WARPS, 42> <<<numBlocks, numThreads>>>(d_inp, plan->m_d_histo,
          plan->m_d_fin, numElements);
      break;
      case 43:
        histogramBallot_Mode13_large<NUM_WARPS, 43, 6, LOG_WARPS> <<<
          numBlocks, numThreads>>>(d_inp, plan->m_d_histo, numElements);
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, plan->m_d_histo,
          plan->m_d_histo, 43 * numBlocks);
        splitBallot_Mode13_large<NUM_WARPS, 43> <<<numBlocks, numThreads>>>(d_inp, plan->m_d_histo,
          plan->m_d_fin, numElements);
      break;
      case 44:
        histogramBallot_Mode13_large<NUM_WARPS, 44, 6, LOG_WARPS> <<<
          numBlocks, numThreads>>>(d_inp, plan->m_d_histo, numElements);
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, plan->m_d_histo,
          plan->m_d_histo, 44 * numBlocks);
        splitBallot_Mode13_large<NUM_WARPS, 44> <<<numBlocks, numThreads>>>(d_inp, plan->m_d_histo,
          plan->m_d_fin, numElements);
      break;
      case 45:
        histogramBallot_Mode13_large<NUM_WARPS, 45, 6, LOG_WARPS> <<<
          numBlocks, numThreads>>>(d_inp, plan->m_d_histo, numElements);
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, plan->m_d_histo,
          plan->m_d_histo, 45 * numBlocks);
        splitBallot_Mode13_large<NUM_WARPS, 45> <<<numBlocks, numThreads>>>(d_inp, plan->m_d_histo,
          plan->m_d_fin, numElements);
      break;
      case 46:
        histogramBallot_Mode13_large<NUM_WARPS, 46, 6, LOG_WARPS> <<<
          numBlocks, numThreads>>>(d_inp, plan->m_d_histo, numElements);
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, plan->m_d_histo,
          plan->m_d_histo, 46 * numBlocks);
        splitBallot_Mode13_large<NUM_WARPS, 46> <<<numBlocks, numThreads>>>(d_inp, plan->m_d_histo,
          plan->m_d_fin, numElements);
      break;
      case 47:
        histogramBallot_Mode13_large<NUM_WARPS, 47, 6, LOG_WARPS> <<<
          numBlocks, numThreads>>>(d_inp, plan->m_d_histo, numElements);
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, plan->m_d_histo,
          plan->m_d_histo, 47 * numBlocks);
        splitBallot_Mode13_large<NUM_WARPS, 47> <<<numBlocks, numThreads>>>(d_inp, plan->m_d_histo,
          plan->m_d_fin, numElements);
      break;
      case 48:
        histogramBallot_Mode13_large<NUM_WARPS, 48, 6, LOG_WARPS> <<<
          numBlocks, numThreads>>>(d_inp, plan->m_d_histo, numElements);
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, plan->m_d_histo,
          plan->m_d_histo, 48 * numBlocks);
        splitBallot_Mode13_large<NUM_WARPS, 48> <<<numBlocks, numThreads>>>(d_inp, plan->m_d_histo,
          plan->m_d_fin, numElements);
      break;
      case 49:
        histogramBallot_Mode13_large<NUM_WARPS, 49, 6, LOG_WARPS> <<<
          numBlocks, numThreads>>>(d_inp, plan->m_d_histo, numElements);
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, plan->m_d_histo,
          plan->m_d_histo, 49 * numBlocks);
        splitBallot_Mode13_large<NUM_WARPS, 49> <<<numBlocks, numThreads>>>(d_inp, plan->m_d_histo,
          plan->m_d_fin, numElements);
      break;
      case 50:
        histogramBallot_Mode13_large<NUM_WARPS, 50, 6, LOG_WARPS> <<<
          numBlocks, numThreads>>>(d_inp, plan->m_d_histo, numElements);
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, plan->m_d_histo,
          plan->m_d_histo, 50 * numBlocks);
        splitBallot_Mode13_large<NUM_WARPS, 50> <<<numBlocks, numThreads>>>(d_inp, plan->m_d_histo,
          plan->m_d_fin, numElements);
      break;
      case 51:
        histogramBallot_Mode13_large<NUM_WARPS, 51, 6, LOG_WARPS> <<<
          numBlocks, numThreads>>>(d_inp, plan->m_d_histo, numElements);
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, plan->m_d_histo,
          plan->m_d_histo, 51 * numBlocks);
        splitBallot_Mode13_large<NUM_WARPS, 51> <<<numBlocks, numThreads>>>(d_inp, plan->m_d_histo,
          plan->m_d_fin, numElements);
      break;
      case 52:
        histogramBallot_Mode13_large<NUM_WARPS, 52, 6, LOG_WARPS> <<<
          numBlocks, numThreads>>>(d_inp, plan->m_d_histo, numElements);
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, plan->m_d_histo,
          plan->m_d_histo, 52 * numBlocks);
        splitBallot_Mode13_large<NUM_WARPS, 52> <<<numBlocks, numThreads>>>(d_inp, plan->m_d_histo,
          plan->m_d_fin, numElements);
      break;
      case 53:
        histogramBallot_Mode13_large<NUM_WARPS, 53, 6, LOG_WARPS> <<<
          numBlocks, numThreads>>>(d_inp, plan->m_d_histo, numElements);
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, plan->m_d_histo,
          plan->m_d_histo, 53 * numBlocks);
        splitBallot_Mode13_large<NUM_WARPS, 53> <<<numBlocks, numThreads>>>(d_inp, plan->m_d_histo,
          plan->m_d_fin, numElements);
      break;
      case 54:
        histogramBallot_Mode13_large<NUM_WARPS, 54, 6, LOG_WARPS> <<<
          numBlocks, numThreads>>>(d_inp, plan->m_d_histo, numElements);
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, plan->m_d_histo,
          plan->m_d_histo, 54 * numBlocks);
        splitBallot_Mode13_large<NUM_WARPS, 54> <<<numBlocks, numThreads>>>(d_inp, plan->m_d_histo,
          plan->m_d_fin, numElements);
      break;
      case 55:
        histogramBallot_Mode13_large<NUM_WARPS, 55, 6, LOG_WARPS> <<<
          numBlocks, numThreads>>>(d_inp, plan->m_d_histo, numElements);
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, plan->m_d_histo,
          plan->m_d_histo, 55 * numBlocks);
        splitBallot_Mode13_large<NUM_WARPS, 55> <<<numBlocks, numThreads>>>(d_inp, plan->m_d_histo,
          plan->m_d_fin, numElements);
      break;
      case 56:
        histogramBallot_Mode13_large<NUM_WARPS, 56, 6, LOG_WARPS> <<<
          numBlocks, numThreads>>>(d_inp, plan->m_d_histo, numElements);
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, plan->m_d_histo,
          plan->m_d_histo, 56 * numBlocks);
        splitBallot_Mode13_large<NUM_WARPS, 56> <<<numBlocks, numThreads>>>(d_inp, plan->m_d_histo,
          plan->m_d_fin, numElements);
      break;
      case 57:
        histogramBallot_Mode13_large<NUM_WARPS, 57, 6, LOG_WARPS> <<<
          numBlocks, numThreads>>>(d_inp, plan->m_d_histo, numElements);
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, plan->m_d_histo,
          plan->m_d_histo, 57 * numBlocks);
        splitBallot_Mode13_large<NUM_WARPS, 57> <<<numBlocks, numThreads>>>(d_inp, plan->m_d_histo,
          plan->m_d_fin, numElements);
      break;
      case 58:
        histogramBallot_Mode13_large<NUM_WARPS, 58, 6, LOG_WARPS> <<<
          numBlocks, numThreads>>>(d_inp, plan->m_d_histo, numElements);
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, plan->m_d_histo,
          plan->m_d_histo, 58 * numBlocks);
        splitBallot_Mode13_large<NUM_WARPS, 58> <<<numBlocks, numThreads>>>(d_inp, plan->m_d_histo,
          plan->m_d_fin, numElements);
      break;
      case 59:
        histogramBallot_Mode13_large<NUM_WARPS, 59, 6, LOG_WARPS> <<<
          numBlocks, numThreads>>>(d_inp, plan->m_d_histo, numElements);
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, plan->m_d_histo,
          plan->m_d_histo, 59 * numBlocks);
        splitBallot_Mode13_large<NUM_WARPS, 59> <<<numBlocks, numThreads>>>(d_inp, plan->m_d_histo,
          plan->m_d_fin, numElements);
      break;
      case 60:
        histogramBallot_Mode13_large<NUM_WARPS, 60, 6, LOG_WARPS> <<<
          numBlocks, numThreads>>>(d_inp, plan->m_d_histo, numElements);
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, plan->m_d_histo,
          plan->m_d_histo, 60 * numBlocks);
        splitBallot_Mode13_large<NUM_WARPS, 60> <<<numBlocks, numThreads>>>(d_inp, plan->m_d_histo,
          plan->m_d_fin, numElements);
      break;
      case 61:
        histogramBallot_Mode13_large<NUM_WARPS, 61, 6, LOG_WARPS> <<<
          numBlocks, numThreads>>>(d_inp, plan->m_d_histo, numElements);
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, plan->m_d_histo,
          plan->m_d_histo, 61 * numBlocks);
        splitBallot_Mode13_large<NUM_WARPS, 61> <<<numBlocks, numThreads>>>(d_inp, plan->m_d_histo,
          plan->m_d_fin, numElements);
      break;
      case 62:
        histogramBallot_Mode13_large<NUM_WARPS, 62, 6, LOG_WARPS> <<<
          numBlocks, numThreads>>>(d_inp, plan->m_d_histo, numElements);
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, plan->m_d_histo,
          plan->m_d_histo, 62 * numBlocks);
        splitBallot_Mode13_large<NUM_WARPS, 62> <<<numBlocks, numThreads>>>(d_inp, plan->m_d_histo,
          plan->m_d_fin, numElements);
      break;
      case 63:
        histogramBallot_Mode13_large<NUM_WARPS, 63, 6, LOG_WARPS> <<<
          numBlocks, numThreads>>>(d_inp, plan->m_d_histo, numElements);
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, plan->m_d_histo,
          plan->m_d_histo, 63 * numBlocks);
        splitBallot_Mode13_large<NUM_WARPS, 63> <<<numBlocks, numThreads>>>(d_inp, plan->m_d_histo,
          plan->m_d_fin, numElements);
      break;
      case 64:
        histogramBallot_Mode13_large<NUM_WARPS, 64, 6, LOG_WARPS> <<<
          numBlocks, numThreads>>>(d_inp, plan->m_d_histo, numElements);
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, plan->m_d_histo,
          plan->m_d_histo, 64 * numBlocks);
        splitBallot_Mode13_large<NUM_WARPS, 64> <<<numBlocks, numThreads>>>(d_inp, plan->m_d_histo,
          plan->m_d_fin, numElements);
      break;
      case 65:
        histogramBallot_Mode13_large<NUM_WARPS, 65, 7, LOG_WARPS> <<<numBlocks, numThreads>>>(
            d_inp, plan->m_d_histo, numElements);
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
            plan->m_d_histo, plan->m_d_histo, 65 * numBlocks);
        splitBallot_Mode13_large<NUM_WARPS, 65> <<<numBlocks, numThreads>>>(d_inp,
            plan->m_d_histo, plan->m_d_fin, numElements);
        break;
      case 66:
        histogramBallot_Mode13_large<NUM_WARPS, 66, 7, LOG_WARPS> <<<numBlocks, numThreads>>>(
            d_inp, plan->m_d_histo, numElements);
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
            plan->m_d_histo, plan->m_d_histo, 66 * numBlocks);
        splitBallot_Mode13_large<NUM_WARPS, 66> <<<numBlocks, numThreads>>>(d_inp,
            plan->m_d_histo, plan->m_d_fin, numElements);
        break;
      case 67:
        histogramBallot_Mode13_large<NUM_WARPS, 67, 7, LOG_WARPS> <<<numBlocks, numThreads>>>(
            d_inp, plan->m_d_histo, numElements);
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
            plan->m_d_histo, plan->m_d_histo, 67 * numBlocks);
        splitBallot_Mode13_large<NUM_WARPS, 67> <<<numBlocks, numThreads>>>(d_inp,
            plan->m_d_histo, plan->m_d_fin, numElements);
        break;
      case 68:
        histogramBallot_Mode13_large<NUM_WARPS, 68, 7, LOG_WARPS> <<<numBlocks, numThreads>>>(
            d_inp, plan->m_d_histo, numElements);
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
            plan->m_d_histo, plan->m_d_histo, 68 * numBlocks);
        splitBallot_Mode13_large<NUM_WARPS, 68> <<<numBlocks, numThreads>>>(d_inp,
            plan->m_d_histo, plan->m_d_fin, numElements);
        break;
      case 69:
        histogramBallot_Mode13_large<NUM_WARPS, 69, 7, LOG_WARPS> <<<numBlocks, numThreads>>>(
            d_inp, plan->m_d_histo, numElements);
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
            plan->m_d_histo, plan->m_d_histo, 69 * numBlocks);
        splitBallot_Mode13_large<NUM_WARPS, 69> <<<numBlocks, numThreads>>>(d_inp,
            plan->m_d_histo, plan->m_d_fin, numElements);
        break;
      case 70:
        histogramBallot_Mode13_large<NUM_WARPS, 70, 7, LOG_WARPS> <<<numBlocks, numThreads>>>(
            d_inp, plan->m_d_histo, numElements);
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
            plan->m_d_histo, plan->m_d_histo, 70 * numBlocks);
        splitBallot_Mode13_large<NUM_WARPS, 70> <<<numBlocks, numThreads>>>(d_inp,
            plan->m_d_histo, plan->m_d_fin, numElements);
        break;
      case 71:
        histogramBallot_Mode13_large<NUM_WARPS, 71, 7, LOG_WARPS> <<<numBlocks, numThreads>>>(
            d_inp, plan->m_d_histo, numElements);
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
            plan->m_d_histo, plan->m_d_histo, 71 * numBlocks);
        splitBallot_Mode13_large<NUM_WARPS, 71> <<<numBlocks, numThreads>>>(d_inp,
            plan->m_d_histo, plan->m_d_fin, numElements);
        break;
      case 72:
        histogramBallot_Mode13_large<NUM_WARPS, 72, 7, LOG_WARPS> <<<numBlocks, numThreads>>>(
            d_inp, plan->m_d_histo, numElements);
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
            plan->m_d_histo, plan->m_d_histo, 72 * numBlocks);
        splitBallot_Mode13_large<NUM_WARPS, 72> <<<numBlocks, numThreads>>>(d_inp,
            plan->m_d_histo, plan->m_d_fin, numElements);
        break;
      case 73:
        histogramBallot_Mode13_large<NUM_WARPS, 73, 7, LOG_WARPS> <<<numBlocks, numThreads>>>(
            d_inp, plan->m_d_histo, numElements);
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
            plan->m_d_histo, plan->m_d_histo, 73 * numBlocks);
        splitBallot_Mode13_large<NUM_WARPS, 73> <<<numBlocks, numThreads>>>(d_inp,
            plan->m_d_histo, plan->m_d_fin, numElements);
        break;
      case 74:
        histogramBallot_Mode13_large<NUM_WARPS, 74, 7, LOG_WARPS> <<<numBlocks, numThreads>>>(
            d_inp, plan->m_d_histo, numElements);
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
            plan->m_d_histo, plan->m_d_histo, 74 * numBlocks);
        splitBallot_Mode13_large<NUM_WARPS, 74> <<<numBlocks, numThreads>>>(d_inp,
            plan->m_d_histo, plan->m_d_fin, numElements);
        break;
      case 75:
        histogramBallot_Mode13_large<NUM_WARPS, 75, 7, LOG_WARPS> <<<numBlocks, numThreads>>>(
            d_inp, plan->m_d_histo, numElements);
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
            plan->m_d_histo, plan->m_d_histo, 75 * numBlocks);
        splitBallot_Mode13_large<NUM_WARPS, 75> <<<numBlocks, numThreads>>>(d_inp,
            plan->m_d_histo, plan->m_d_fin, numElements);
        break;
      case 76:
        histogramBallot_Mode13_large<NUM_WARPS, 76, 7, LOG_WARPS> <<<numBlocks, numThreads>>>(
            d_inp, plan->m_d_histo, numElements);
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
            plan->m_d_histo, plan->m_d_histo, 76 * numBlocks);
        splitBallot_Mode13_large<NUM_WARPS, 76> <<<numBlocks, numThreads>>>(d_inp,
            plan->m_d_histo, plan->m_d_fin, numElements);
        break;
      case 77:
        histogramBallot_Mode13_large<NUM_WARPS, 77, 7, LOG_WARPS> <<<numBlocks, numThreads>>>(
            d_inp, plan->m_d_histo, numElements);
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
            plan->m_d_histo, plan->m_d_histo, 77 * numBlocks);
        splitBallot_Mode13_large<NUM_WARPS, 77> <<<numBlocks, numThreads>>>(d_inp,
            plan->m_d_histo, plan->m_d_fin, numElements);
        break;
      case 78:
        histogramBallot_Mode13_large<NUM_WARPS, 78, 7, LOG_WARPS> <<<numBlocks, numThreads>>>(
            d_inp, plan->m_d_histo, numElements);
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
            plan->m_d_histo, plan->m_d_histo, 78 * numBlocks);
        splitBallot_Mode13_large<NUM_WARPS, 78> <<<numBlocks, numThreads>>>(d_inp,
            plan->m_d_histo, plan->m_d_fin, numElements);
        break;
      case 79:
        histogramBallot_Mode13_large<NUM_WARPS, 79, 7, LOG_WARPS> <<<numBlocks, numThreads>>>(
            d_inp, plan->m_d_histo, numElements);
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
            plan->m_d_histo, plan->m_d_histo, 79 * numBlocks);
        splitBallot_Mode13_large<NUM_WARPS, 79> <<<numBlocks, numThreads>>>(d_inp,
            plan->m_d_histo, plan->m_d_fin, numElements);
        break;
      case 80:
        histogramBallot_Mode13_large<NUM_WARPS, 80, 7, LOG_WARPS> <<<numBlocks, numThreads>>>(
            d_inp, plan->m_d_histo, numElements);
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
            plan->m_d_histo, plan->m_d_histo, 80 * numBlocks);
        splitBallot_Mode13_large<NUM_WARPS, 80> <<<numBlocks, numThreads>>>(d_inp,
            plan->m_d_histo, plan->m_d_fin, numElements);
        break;
      case 81:
        histogramBallot_Mode13_large<NUM_WARPS, 81, 7, LOG_WARPS> <<<numBlocks, numThreads>>>(
            d_inp, plan->m_d_histo, numElements);
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
            plan->m_d_histo, plan->m_d_histo, 81 * numBlocks);
        splitBallot_Mode13_large<NUM_WARPS, 81> <<<numBlocks, numThreads>>>(d_inp,
            plan->m_d_histo, plan->m_d_fin, numElements);
        break;
      case 82:
        histogramBallot_Mode13_large<NUM_WARPS, 82, 7, LOG_WARPS> <<<numBlocks, numThreads>>>(
            d_inp, plan->m_d_histo, numElements);
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
            plan->m_d_histo, plan->m_d_histo, 82 * numBlocks);
        splitBallot_Mode13_large<NUM_WARPS, 82> <<<numBlocks, numThreads>>>(d_inp,
            plan->m_d_histo, plan->m_d_fin, numElements);
        break;
      case 83:
        histogramBallot_Mode13_large<NUM_WARPS, 83, 7, LOG_WARPS> <<<numBlocks, numThreads>>>(
            d_inp, plan->m_d_histo, numElements);
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
            plan->m_d_histo, plan->m_d_histo, 83 * numBlocks);
        splitBallot_Mode13_large<NUM_WARPS, 83> <<<numBlocks, numThreads>>>(d_inp,
            plan->m_d_histo, plan->m_d_fin, numElements);
        break;
      case 84:
        histogramBallot_Mode13_large<NUM_WARPS, 84, 7, LOG_WARPS> <<<numBlocks, numThreads>>>(
            d_inp, plan->m_d_histo, numElements);
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
            plan->m_d_histo, plan->m_d_histo, 84 * numBlocks);
        splitBallot_Mode13_large<NUM_WARPS, 84> <<<numBlocks, numThreads>>>(d_inp,
            plan->m_d_histo, plan->m_d_fin, numElements);
        break;
      case 85:
        histogramBallot_Mode13_large<NUM_WARPS, 85, 7, LOG_WARPS> <<<numBlocks, numThreads>>>(
            d_inp, plan->m_d_histo, numElements);
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
            plan->m_d_histo, plan->m_d_histo, 85 * numBlocks);
        splitBallot_Mode13_large<NUM_WARPS, 85> <<<numBlocks, numThreads>>>(d_inp,
            plan->m_d_histo, plan->m_d_fin, numElements);
        break;
      case 86:
        histogramBallot_Mode13_large<NUM_WARPS, 86, 7, LOG_WARPS> <<<numBlocks, numThreads>>>(
            d_inp, plan->m_d_histo, numElements);
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
            plan->m_d_histo, plan->m_d_histo, 86 * numBlocks);
        splitBallot_Mode13_large<NUM_WARPS, 86> <<<numBlocks, numThreads>>>(d_inp,
            plan->m_d_histo, plan->m_d_fin, numElements);
        break;
      case 87:
        histogramBallot_Mode13_large<NUM_WARPS, 87, 7, LOG_WARPS> <<<numBlocks, numThreads>>>(
            d_inp, plan->m_d_histo, numElements);
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
            plan->m_d_histo, plan->m_d_histo, 87 * numBlocks);
        splitBallot_Mode13_large<NUM_WARPS, 87> <<<numBlocks, numThreads>>>(d_inp,
            plan->m_d_histo, plan->m_d_fin, numElements);
        break;
      case 88:
        histogramBallot_Mode13_large<NUM_WARPS, 88, 7, LOG_WARPS> <<<numBlocks, numThreads>>>(
            d_inp, plan->m_d_histo, numElements);
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
            plan->m_d_histo, plan->m_d_histo, 88 * numBlocks);
        splitBallot_Mode13_large<NUM_WARPS, 88> <<<numBlocks, numThreads>>>(d_inp,
            plan->m_d_histo, plan->m_d_fin, numElements);
        break;
      case 89:
        histogramBallot_Mode13_large<NUM_WARPS, 89, 7, LOG_WARPS> <<<numBlocks, numThreads>>>(
            d_inp, plan->m_d_histo, numElements);
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
            plan->m_d_histo, plan->m_d_histo, 89 * numBlocks);
        splitBallot_Mode13_large<NUM_WARPS, 89> <<<numBlocks, numThreads>>>(d_inp,
            plan->m_d_histo, plan->m_d_fin, numElements);
        break;
      case 90:
        histogramBallot_Mode13_large<NUM_WARPS, 90, 7, LOG_WARPS> <<<numBlocks, numThreads>>>(
            d_inp, plan->m_d_histo, numElements);
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
            plan->m_d_histo, plan->m_d_histo, 90 * numBlocks);
        splitBallot_Mode13_large<NUM_WARPS, 90> <<<numBlocks, numThreads>>>(d_inp,
            plan->m_d_histo, plan->m_d_fin, numElements);
        break;
      case 91:
        histogramBallot_Mode13_large<NUM_WARPS, 91, 7, LOG_WARPS> <<<numBlocks, numThreads>>>(
            d_inp, plan->m_d_histo, numElements);
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
            plan->m_d_histo, plan->m_d_histo, 91 * numBlocks);
        splitBallot_Mode13_large<NUM_WARPS, 91> <<<numBlocks, numThreads>>>(d_inp,
            plan->m_d_histo, plan->m_d_fin, numElements);
        break;
      case 92:
        histogramBallot_Mode13_large<NUM_WARPS, 92, 7, LOG_WARPS> <<<numBlocks, numThreads>>>(
            d_inp, plan->m_d_histo, numElements);
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
            plan->m_d_histo, plan->m_d_histo, 92 * numBlocks);
        splitBallot_Mode13_large<NUM_WARPS, 92> <<<numBlocks, numThreads>>>(d_inp,
            plan->m_d_histo, plan->m_d_fin, numElements);
        break;
      case 93:
        histogramBallot_Mode13_large<NUM_WARPS, 93, 7, LOG_WARPS> <<<numBlocks, numThreads>>>(
            d_inp, plan->m_d_histo, numElements);
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
            plan->m_d_histo, plan->m_d_histo, 93 * numBlocks);
        splitBallot_Mode13_large<NUM_WARPS, 93> <<<numBlocks, numThreads>>>(d_inp,
            plan->m_d_histo, plan->m_d_fin, numElements);
        break;
      case 94:
        histogramBallot_Mode13_large<NUM_WARPS, 94, 7, LOG_WARPS> <<<numBlocks, numThreads>>>(
            d_inp, plan->m_d_histo, numElements);
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
            plan->m_d_histo, plan->m_d_histo, 94 * numBlocks);
        splitBallot_Mode13_large<NUM_WARPS, 94> <<<numBlocks, numThreads>>>(d_inp,
            plan->m_d_histo, plan->m_d_fin, numElements);
        break;
      case 95:
        histogramBallot_Mode13_large<NUM_WARPS, 95, 7, LOG_WARPS> <<<numBlocks, numThreads>>>(
            d_inp, plan->m_d_histo, numElements);
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
            plan->m_d_histo, plan->m_d_histo, 95 * numBlocks);
        splitBallot_Mode13_large<NUM_WARPS, 95> <<<numBlocks, numThreads>>>(d_inp,
            plan->m_d_histo, plan->m_d_fin, numElements);
        break;
      case 96:
        histogramBallot_Mode13_large<NUM_WARPS, 96, 7, LOG_WARPS> <<<numBlocks, numThreads>>>(
            d_inp, plan->m_d_histo, numElements);
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
            plan->m_d_histo, plan->m_d_histo, 96 * numBlocks);
        splitBallot_Mode13_large<NUM_WARPS, 96> <<<numBlocks, numThreads>>>(d_inp,
            plan->m_d_histo, plan->m_d_fin, numElements);
        break;
      default:
        break;
    }
  }

  CUDA_SAFE_CALL(cudaMemcpy(d_inp, plan->m_d_fin, numElements*sizeof(unsigned int), cudaMemcpyDeviceToDevice));

  if(d_temp_storage)
    CubDebugExit(g_allocator.DeviceFree(d_temp_storage));
}

/** @brief Performs merge sort utilizing 3 stages:
 * (1) Blocksort, (2) simple merge and (3) multi merge
 *
 *
 * @param[in,out] pkeys Keys to be sorted.
 * @param[in,out] pvals Associated values to be sorted
 * @param[in] numElements Number of elements in the sort.
 * @param[in] plan Configuration information for mergesort.
 **/
void runMultiSplit(unsigned int *d_keys, unsigned int *d_values,
    unsigned int numElements, unsigned int numBuckets,
    const CUDPPMultiSplitPlan *plan) {
  unsigned int numBlocks = ceil(numElements / (NUM_WARPS * 32));
  unsigned int numThreads = NUM_WARPS * 32;
  void     *d_temp_storage = NULL;
  size_t   temp_storage_bytes = 0;

  if (numBuckets == 1)
    return;

  if (numBuckets == 2) {
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, plan->m_d_histo, plan->m_d_histo, numBuckets * numBlocks * NUM_WARPS * PACK_DEPTH);
  } else if (numBuckets <= 32) {
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, plan->m_d_histo, plan->m_d_histo, numBuckets * numBlocks * PACK_DEPTH);
  } else if (numBuckets > 96) {
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, plan->m_d_mask, plan->m_d_out, d_keys, plan->m_d_fin,
        numElements, 0, plan->m_logBuckets);
  } else if (numBuckets <= 96){
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, plan->m_d_histo, plan->m_d_histo, numBuckets * numBlocks);
  } else {
    printf("Bad number of buckets: %u\n", numBuckets);
    return;
  }
  g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes);

  if (numBuckets == 2) {
    histogram_warp<NUM_WARPS, 2, 1, PACK_PRE> <<<numBlocks / PACK_PRE, numThreads>>>(d_keys,
        plan->m_d_histo, numElements);
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, plan->m_d_histo, plan->m_d_histo, 2 * numBlocks * NUM_WARPS * PACK_DEPTH);
    split_WMS_pairs<NUM_WARPS, 2, 1, PACK_POST><<<numBlocks/PACK_POST,
      numThreads>>>(d_keys, d_values, plan->m_d_histo, plan->m_d_temp_keys,
          plan->m_d_temp_values, numElements);
  } else if (numBuckets <= 32) {
    histogram_block<<<numBlocks / PACK_PRE, numThreads,
        NUM_WARPS * numBuckets * DEPTH * sizeof(uint)>>>(d_keys, plan->m_d_histo,
        numElements, numBuckets, NUM_WARPS, PACK_PRE);
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, plan->m_d_histo, plan->m_d_histo, numBuckets * numBlocks * PACK_DEPTH);
    split_BMS<<<numBlocks / PACK_POST, numThreads,
        (2 * numBuckets * PACK_POST + 32 * NUM_WARPS * PACK_POST
            + numBuckets * NUM_WARPS * PACK_POST) * sizeof(uint)>>>(d_keys, plan->m_d_histo, plan->m_d_fin,
        numElements, numBuckets, NUM_WARPS, PACK_POST);
  } else if (numBuckets > 96) {
    markBins_general<<<numBlocks, numThreads>>>(plan->m_d_mask, d_keys, numElements, numBuckets);
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, plan->m_d_mask,
        plan->m_d_out, d_keys, plan->m_d_fin, numElements, 0,
        int(ceil(log2(float(numBuckets)))));
  } else if (numBuckets <= 96) {
    switch(numBuckets){
      default:
        break;
    }
  }

  CUDA_SAFE_CALL(cudaMemcpy(d_keys, plan->m_d_temp_keys, numElements*sizeof(unsigned int), cudaMemcpyDeviceToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(d_values, plan->m_d_temp_values, numElements*sizeof(unsigned int), cudaMemcpyDeviceToDevice));
  
  if(d_temp_storage)
    CubDebugExit(g_allocator.DeviceFree(d_temp_storage));
}

#ifdef __cplusplus
extern "C"
{
#endif

/**
 * @brief From the programmer-specified sort configuration,
 *        creates internal memory for performing the sort.
 *
 * @param[in] plan Pointer to CUDPPMergeSortPlan object
 **/
void allocMultiSplitStorage(CUDPPMultiSplitPlan *plan)
{
  unsigned int nB = ceil(plan->m_numElements / (NUM_WARPS * 32));

/*
  printf("NUM ALLOCATED BYTES: %u\n",
      (plan->m_numElements + 1)
          * sizeof(unsigned int) + plan->m_numElements * sizeof(unsigned int)+
          sizeof(unsigned int) * plan->m_numBuckets * nB * NUM_WARPS * 2 * PACK_DEPTH + plan->m_numElements*sizeof(unsigned int));
*/
  if (plan->m_config.options & CUDPP_OPTION_KEY_VALUE_PAIRS) {
    CUDA_SAFE_CALL(cudaMalloc((void**) &plan->m_d_temp_keys, plan->m_numElements*sizeof(unsigned int))); // gpu output
    CUDA_SAFE_CALL(cudaMalloc((void**) &plan->m_d_temp_values, plan->m_numElements*sizeof(unsigned int))); // gpu output
  }

  if (plan->m_numBuckets > 96) {
    CUDA_SAFE_CALL(cudaMalloc((void**) &plan->m_d_mask, (plan->m_numElements+1)*sizeof(unsigned int)));  // mask verctor, +1 added only for the near-far implementation
    CUDA_SAFE_CALL(cudaMalloc((void**) &plan->m_d_out, plan->m_numElements*sizeof(unsigned int))); // gpu output
  }
  CUDA_SAFE_CALL(
      cudaMalloc((void**) &plan->m_d_histo, sizeof(unsigned int) * plan->m_numBuckets * nB * NUM_WARPS * 2 * PACK_DEPTH)); //
  CUDA_SAFE_CALL(cudaMalloc((void**) &plan->m_d_fin, plan->m_numElements*sizeof(unsigned int))); // final masks (used for reduced bit method, etc.)

  if (plan->m_numBuckets > 96) {
    CUDA_SAFE_CALL(cudaMemset(plan->m_d_mask, 0, sizeof(unsigned int)*(plan->m_numElements+1)));
    CUDA_SAFE_CALL(cudaMemset(plan->m_d_out, 0, sizeof(unsigned int)*plan->m_numElements));
  }
  CUDA_SAFE_CALL(cudaMemset(plan->m_d_histo, 0, sizeof(unsigned int) * plan->m_numBuckets * nB * NUM_WARPS * 2
      * PACK_DEPTH));
  CUDA_SAFE_CALL(cudaMemset(plan->m_d_fin, 0, sizeof(unsigned int)*plan->m_numElements));
}

/** @brief Deallocates intermediate memory from allocRadixSortStorage.
 *
 *
 * @param[in] plan Pointer to CUDPPMergeSortPlan object
 **/

void freeMultiSplitStorage(CUDPPMultiSplitPlan* plan)
{
  if (plan->m_config.options & CUDPP_OPTION_KEY_VALUE_PAIRS) {
    cudaFree (plan->m_d_temp_keys);
    cudaFree (plan->m_d_temp_values);
  }
  if (plan->m_numBuckets > 96) {
    cudaFree (plan->m_d_mask);
    cudaFree (plan->m_d_out);
  }
  cudaFree(plan->m_d_histo);
  cudaFree(plan->m_d_fin);
}

/** @brief Dispatch function to perform a sort on an array with
 * a specified configuration.
 *
 * This is the dispatch routine which calls mergeSort...() with
 * appropriate template parameters and arguments as specified by
 * the plan.
 * Currently only sorts keys of type int, unsigned int, and float.
 * @param[in,out] keys Keys to be sorted.
 * @param[in,out] values Associated values to be sorted (through keys).
 * @param[in] numElements Number of elements in the sort.
 * @param[in] plan Configuration information for mergeSort.
 **/
void cudppMultiSplitDispatch(unsigned int *d_keys,
                             unsigned int *d_values,
                            size_t numElements,
                            size_t numBuckets,
                            const CUDPPMultiSplitPlan *plan)
{
  cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);

  if (plan->m_config.options & CUDPP_OPTION_KEY_VALUE_PAIRS) {
    runMultiSplit(d_keys, d_values, numElements, numBuckets, plan);
  } else {
    runMultiSplit(d_keys, numElements, numBuckets, plan);
  }
}

#ifdef __cplusplus
}
#endif

/** @} */ // end mergesort functions
/** @} */ // end cudpp_app
