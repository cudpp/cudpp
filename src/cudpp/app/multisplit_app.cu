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
#define PACK_DEPTH 4
#define PACK_PRE 8
#define PACK_POST 4

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
void runMultiSplit(uint* d_inp, uint* d_histo, uint* d_fin, uint* d_mask, uint* d_out, void *d_temp_storage,
    size_t temp_storage_bytes, uint numElements, uint numBuckets) {
  unsigned int nB = ceil(numElements / (NUM_WARPS * 32));
  unsigned int NT = NUM_WARPS*32;

  switch(numBuckets){
    case 1:
    break;
    case 2:
/*
      histogram_warp<NUM_WARPS, 2, 1, PACK_PRE><<<nB/PACK_PRE, NT>>>(d_inp, d_histo, numElements);
      cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_histo, d_histo, 2 * nB * NUM_WARPS * PACK_DEPTH);
      split_WMS<NUM_WARPS, 2, 1, PACK_POST><<<nB/PACK_POST, NT>>>(d_inp, d_histo, d_fin, numElements);
*/
      histogramBallot_Mode13_large<NUM_WARPS, 2, 2, LOG_WARPS> <<<
        nB, NT>>>(d_inp, d_histo, numElements);
      cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_histo,
        d_histo, 2 * nB);
      splitBallot_Mode13_large<NUM_WARPS, 2> <<<nB, NT>>>(d_inp, d_histo,
        d_fin, numElements);
    break;
    case 3:
      histogramBallot_Mode13_large<NUM_WARPS, 3, 2, LOG_WARPS> <<<
        nB, NT>>>(d_inp, d_histo, numElements);
      cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_histo,
        d_histo, 3 * nB);
      splitBallot_Mode13_large<NUM_WARPS, 3> <<<nB, NT>>>(d_inp, d_histo,
        d_fin, numElements);
    break;
    case 4:
      histogramBallot_Mode13_large<NUM_WARPS, 4, 2, LOG_WARPS> <<<
        nB, NT>>>(d_inp, d_histo, numElements);
      cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_histo,
        d_histo, 4 * nB);
      splitBallot_Mode13_large<NUM_WARPS, 4> <<<nB, NT>>>(d_inp, d_histo,
        d_fin, numElements);
    break;
    case 5:
      histogramBallot_Mode13_large<NUM_WARPS, 5, 3, LOG_WARPS> <<<
        nB, NT>>>(d_inp, d_histo, numElements);
      cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_histo,
        d_histo, 5 * nB);
      splitBallot_Mode13_large<NUM_WARPS, 5> <<<nB, NT>>>(d_inp, d_histo,
        d_fin, numElements);
    break;
    case 6:
      histogramBallot_Mode13_large<NUM_WARPS, 6, 3, LOG_WARPS> <<<
        nB, NT>>>(d_inp, d_histo, numElements);
      cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_histo,
        d_histo, 6 * nB);
      splitBallot_Mode13_large<NUM_WARPS, 6> <<<nB, NT>>>(d_inp, d_histo,
        d_fin, numElements);
    break;
    case 7:
      histogramBallot_Mode13_large<NUM_WARPS, 7, 3, LOG_WARPS> <<<
        nB, NT>>>(d_inp, d_histo, numElements);
      cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_histo,
        d_histo, 7 * nB);
      splitBallot_Mode13_large<NUM_WARPS, 7> <<<nB, NT>>>(d_inp, d_histo,
        d_fin, numElements);
    break;
    case 8:
      histogramBallot_Mode13_large<NUM_WARPS, 8, 3, LOG_WARPS> <<<
        nB, NT>>>(d_inp, d_histo, numElements);
      cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_histo,
        d_histo, 8 * nB);
      splitBallot_Mode13_large<NUM_WARPS, 8> <<<nB, NT>>>(d_inp, d_histo,
        d_fin, numElements);
    break;
    case 9:
      histogramBallot_Mode13_large<NUM_WARPS, 9, 4, LOG_WARPS> <<<
        nB, NT>>>(d_inp, d_histo, numElements);
      cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_histo,
        d_histo, 9 * nB);
      splitBallot_Mode13_large<NUM_WARPS, 9> <<<nB, NT>>>(d_inp, d_histo,
        d_fin, numElements);
    break;
    case 10:
      histogramBallot_Mode13_large<NUM_WARPS, 10, 4, LOG_WARPS> <<<
        nB, NT>>>(d_inp, d_histo, numElements);
      cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_histo,
        d_histo, 10 * nB);
      splitBallot_Mode13_large<NUM_WARPS, 10> <<<nB, NT>>>(d_inp, d_histo,
        d_fin, numElements);
    break;
    case 11:
      histogramBallot_Mode13_large<NUM_WARPS, 11, 4, LOG_WARPS> <<<
        nB, NT>>>(d_inp, d_histo, numElements);
      cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_histo,
        d_histo, 11 * nB);
      splitBallot_Mode13_large<NUM_WARPS, 11> <<<nB, NT>>>(d_inp, d_histo,
        d_fin, numElements);
    break;
    case 12:
      histogram_block<NUM_WARPS, LOG_WARPS, 12, 4, PACK_PRE><<<nB/PACK_PRE, NT>>>(d_inp, d_histo, numElements);
      cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_histo, d_histo, 12 * nB * PACK_DEPTH);
      split_BMS<NUM_WARPS, LOG_WARPS, 12, 4, PACK_POST><<<nB/PACK_POST, NT>>>(d_inp, d_histo, d_fin, numElements);
    break;
    case 13:
      histogram_block<NUM_WARPS, LOG_WARPS, 13, 4, PACK_PRE><<<nB/PACK_PRE, NT>>>(d_inp, d_histo, numElements);
      cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_histo, d_histo, 13 * nB * PACK_DEPTH);
      split_BMS<NUM_WARPS, LOG_WARPS, 13, 4, PACK_POST><<<nB/PACK_POST, NT>>>(d_inp, d_histo, d_fin, numElements);
    break;
    case 14:
      histogramBallot_Mode13_large<NUM_WARPS, 14, 4, LOG_WARPS> <<<
        nB, NT>>>(d_inp, d_histo, numElements);
      cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_histo,
        d_histo, 14 * nB);
      splitBallot_Mode13_large<NUM_WARPS, 14> <<<nB, NT>>>(d_inp, d_histo,
        d_fin, numElements);
    break;
    case 15:
      histogramBallot_Mode13_large<NUM_WARPS, 15, 4, LOG_WARPS> <<<
        nB, NT>>>(d_inp, d_histo, numElements);
      cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_histo,
        d_histo, 15 * nB);
      splitBallot_Mode13_large<NUM_WARPS, 15> <<<nB, NT>>>(d_inp, d_histo,
        d_fin, numElements);
    break;
    case 16:
      histogramBallot_Mode13_large<NUM_WARPS, 16, 4, LOG_WARPS> <<<
        nB, NT>>>(d_inp, d_histo, numElements);
      cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_histo,
        d_histo, 16 * nB);
      splitBallot_Mode13_large<NUM_WARPS, 16> <<<nB, NT>>>(d_inp, d_histo,
        d_fin, numElements);
    break;
    case 17:
      histogramBallot_Mode13_large<NUM_WARPS, 17, 5, LOG_WARPS> <<<
        nB, NT>>>(d_inp, d_histo, numElements);
      cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_histo,
        d_histo, 17 * nB);
      splitBallot_Mode13_large<NUM_WARPS, 17> <<<nB, NT>>>(d_inp, d_histo,
        d_fin, numElements);
    break;
    case 18:
      histogramBallot_Mode13_large<NUM_WARPS, 18, 5, LOG_WARPS> <<<
        nB, NT>>>(d_inp, d_histo, numElements);
      cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_histo,
        d_histo, 18 * nB);
      splitBallot_Mode13_large<NUM_WARPS, 18> <<<nB, NT>>>(d_inp, d_histo,
        d_fin, numElements);
    break;
    case 19:
      histogramBallot_Mode13_large<NUM_WARPS, 19, 5, LOG_WARPS> <<<
        nB, NT>>>(d_inp, d_histo, numElements);
      cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_histo,
        d_histo, 19 * nB);
      splitBallot_Mode13_large<NUM_WARPS, 19> <<<nB, NT>>>(d_inp, d_histo,
        d_fin, numElements);
    break;
    case 20:
      histogramBallot_Mode13_large<NUM_WARPS, 20, 5, LOG_WARPS> <<<
        nB, NT>>>(d_inp, d_histo, numElements);
      cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_histo,
        d_histo, 20 * nB);
      splitBallot_Mode13_large<NUM_WARPS, 20> <<<nB, NT>>>(d_inp, d_histo,
        d_fin, numElements);
    break;
    case 21:
      histogramBallot_Mode13_large<NUM_WARPS, 21, 5, LOG_WARPS> <<<
        nB, NT>>>(d_inp, d_histo, numElements);
      cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_histo,
        d_histo, 21 * nB);
      splitBallot_Mode13_large<NUM_WARPS, 21> <<<nB, NT>>>(d_inp, d_histo,
        d_fin, numElements);
    break;
    case 22:
      histogramBallot_Mode13_large<NUM_WARPS, 22, 5, LOG_WARPS> <<<
        nB, NT>>>(d_inp, d_histo, numElements);
      cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_histo,
        d_histo, 22 * nB);
      splitBallot_Mode13_large<NUM_WARPS, 22> <<<nB, NT>>>(d_inp, d_histo,
        d_fin, numElements);
    break;
    case 23:
      histogramBallot_Mode13_large<NUM_WARPS, 23, 5, LOG_WARPS> <<<
        nB, NT>>>(d_inp, d_histo, numElements);
      cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_histo,
        d_histo, 23 * nB);
      splitBallot_Mode13_large<NUM_WARPS, 23> <<<nB, NT>>>(d_inp, d_histo,
        d_fin, numElements);
    break;
    case 24:
      histogramBallot_Mode13_large<NUM_WARPS, 24, 5, LOG_WARPS> <<<
        nB, NT>>>(d_inp, d_histo, numElements);
      cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_histo,
        d_histo, 24 * nB);
      splitBallot_Mode13_large<NUM_WARPS, 24> <<<nB, NT>>>(d_inp, d_histo,
        d_fin, numElements);
    break;
    case 25:
      histogramBallot_Mode13_large<NUM_WARPS, 25, 5, LOG_WARPS> <<<
        nB, NT>>>(d_inp, d_histo, numElements);
      cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_histo,
        d_histo, 25 * nB);
      splitBallot_Mode13_large<NUM_WARPS, 25> <<<nB, NT>>>(d_inp, d_histo,
        d_fin, numElements);
    break;
    case 26:
      histogramBallot_Mode13_large<NUM_WARPS, 26, 5, LOG_WARPS> <<<
        nB, NT>>>(d_inp, d_histo, numElements);
      cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_histo,
        d_histo, 26 * nB);
      splitBallot_Mode13_large<NUM_WARPS, 26> <<<nB, NT>>>(d_inp, d_histo,
        d_fin, numElements);
    break;
    case 27:
      histogramBallot_Mode13_large<NUM_WARPS, 27, 5, LOG_WARPS> <<<
        nB, NT>>>(d_inp, d_histo, numElements);
      cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_histo,
        d_histo, 27 * nB);
      splitBallot_Mode13_large<NUM_WARPS, 27> <<<nB, NT>>>(d_inp, d_histo,
        d_fin, numElements);
    break;
    case 28:
      histogramBallot_Mode13_large<NUM_WARPS, 28, 5, LOG_WARPS> <<<
        nB, NT>>>(d_inp, d_histo, numElements);
      cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_histo,
        d_histo, 28 * nB);
      splitBallot_Mode13_large<NUM_WARPS, 28> <<<nB, NT>>>(d_inp, d_histo,
        d_fin, numElements);
    break;
    case 29:
      histogramBallot_Mode13_large<NUM_WARPS, 29, 5, LOG_WARPS> <<<
        nB, NT>>>(d_inp, d_histo, numElements);
      cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_histo,
        d_histo, 29 * nB);
      splitBallot_Mode13_large<NUM_WARPS, 29> <<<nB, NT>>>(d_inp, d_histo,
        d_fin, numElements);
    break;
    case 30:
      histogramBallot_Mode13_large<NUM_WARPS, 30, 5, LOG_WARPS> <<<
        nB, NT>>>(d_inp, d_histo, numElements);
      cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_histo,
        d_histo, 30 * nB);
      splitBallot_Mode13_large<NUM_WARPS, 30> <<<nB, NT>>>(d_inp, d_histo,
        d_fin, numElements);
    break;
    case 31:
      histogramBallot_Mode13_large<NUM_WARPS, 31, 5, LOG_WARPS> <<<
        nB, NT>>>(d_inp, d_histo, numElements);
      cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_histo,
        d_histo, 31 * nB);
      splitBallot_Mode13_large<NUM_WARPS, 31> <<<nB, NT>>>(d_inp, d_histo,
        d_fin, numElements);
    break;
    case 32:
      histogram_block<NUM_WARPS, LOG_WARPS, 32, 5, PACK_PRE><<<nB/PACK_PRE, NT>>>(d_inp, d_histo, numElements);
      cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_histo, d_histo, 32 * nB * PACK_DEPTH);
      split_BMS<NUM_WARPS, LOG_WARPS, 32, 5, PACK_POST><<<nB/PACK_POST, NT>>>(d_inp, d_histo, d_fin, numElements);
    break;
    case 33:
      histogramBallot_Mode13_large<NUM_WARPS, 33, 6, LOG_WARPS> <<<
        nB, NT>>>(d_inp, d_histo, numElements);
      cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_histo,
        d_histo, 33 * nB);
      splitBallot_Mode13_large<NUM_WARPS, 33> <<<nB, NT>>>(d_inp, d_histo,
        d_fin, numElements);
    break;
    case 34:
      histogramBallot_Mode13_large<NUM_WARPS, 34, 6, LOG_WARPS> <<<
        nB, NT>>>(d_inp, d_histo, numElements);
      cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_histo,
        d_histo, 34 * nB);
      splitBallot_Mode13_large<NUM_WARPS, 34> <<<nB, NT>>>(d_inp, d_histo,
        d_fin, numElements);
    break;
    case 35:
      histogramBallot_Mode13_large<NUM_WARPS, 35, 6, LOG_WARPS> <<<
        nB, NT>>>(d_inp, d_histo, numElements);
      cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_histo,
        d_histo, 35 * nB);
      splitBallot_Mode13_large<NUM_WARPS, 35> <<<nB, NT>>>(d_inp, d_histo,
        d_fin, numElements);
    break;
    case 36:
      histogramBallot_Mode13_large<NUM_WARPS, 36, 6, LOG_WARPS> <<<
        nB, NT>>>(d_inp, d_histo, numElements);
      cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_histo,
        d_histo, 36 * nB);
      splitBallot_Mode13_large<NUM_WARPS, 36> <<<nB, NT>>>(d_inp, d_histo,
        d_fin, numElements);
    break;
    case 37:
      histogramBallot_Mode13_large<NUM_WARPS, 37, 6, LOG_WARPS> <<<
        nB, NT>>>(d_inp, d_histo, numElements);
      cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_histo,
        d_histo, 37 * nB);
      splitBallot_Mode13_large<NUM_WARPS, 37> <<<nB, NT>>>(d_inp, d_histo,
        d_fin, numElements);
    break;
    case 38:
      histogramBallot_Mode13_large<NUM_WARPS, 38, 6, LOG_WARPS> <<<
        nB, NT>>>(d_inp, d_histo, numElements);
      cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_histo,
        d_histo, 38 * nB);
      splitBallot_Mode13_large<NUM_WARPS, 38> <<<nB, NT>>>(d_inp, d_histo,
        d_fin, numElements);
    break;
    case 39:
      histogramBallot_Mode13_large<NUM_WARPS, 39, 6, LOG_WARPS> <<<
        nB, NT>>>(d_inp, d_histo, numElements);
      cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_histo,
        d_histo, 39 * nB);
      splitBallot_Mode13_large<NUM_WARPS, 39> <<<nB, NT>>>(d_inp, d_histo,
        d_fin, numElements);
    break;
    case 40:
      histogramBallot_Mode13_large<NUM_WARPS, 40, 6, LOG_WARPS> <<<
        nB, NT>>>(d_inp, d_histo, numElements);
      cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_histo,
        d_histo, 40 * nB);
      splitBallot_Mode13_large<NUM_WARPS, 40> <<<nB, NT>>>(d_inp, d_histo,
        d_fin, numElements);
    break;
    case 41:
      histogramBallot_Mode13_large<NUM_WARPS, 41, 6, LOG_WARPS> <<<
        nB, NT>>>(d_inp, d_histo, numElements);
      cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_histo,
        d_histo, 41 * nB);
      splitBallot_Mode13_large<NUM_WARPS, 41> <<<nB, NT>>>(d_inp, d_histo,
        d_fin, numElements);
    break;
    case 42:
      histogramBallot_Mode13_large<NUM_WARPS, 42, 6, LOG_WARPS> <<<
        nB, NT>>>(d_inp, d_histo, numElements);
      cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_histo,
        d_histo, 42 * nB);
      splitBallot_Mode13_large<NUM_WARPS, 42> <<<nB, NT>>>(d_inp, d_histo,
        d_fin, numElements);
    break;
    case 43:
      histogramBallot_Mode13_large<NUM_WARPS, 43, 6, LOG_WARPS> <<<
        nB, NT>>>(d_inp, d_histo, numElements);
      cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_histo,
        d_histo, 43 * nB);
      splitBallot_Mode13_large<NUM_WARPS, 43> <<<nB, NT>>>(d_inp, d_histo,
        d_fin, numElements);
    break;
    case 44:
      histogramBallot_Mode13_large<NUM_WARPS, 44, 6, LOG_WARPS> <<<
        nB, NT>>>(d_inp, d_histo, numElements);
      cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_histo,
        d_histo, 44 * nB);
      splitBallot_Mode13_large<NUM_WARPS, 44> <<<nB, NT>>>(d_inp, d_histo,
        d_fin, numElements);
    break;
    case 45:
      histogramBallot_Mode13_large<NUM_WARPS, 45, 6, LOG_WARPS> <<<
        nB, NT>>>(d_inp, d_histo, numElements);
      cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_histo,
        d_histo, 45 * nB);
      splitBallot_Mode13_large<NUM_WARPS, 45> <<<nB, NT>>>(d_inp, d_histo,
        d_fin, numElements);
    break;
    case 46:
      histogramBallot_Mode13_large<NUM_WARPS, 46, 6, LOG_WARPS> <<<
        nB, NT>>>(d_inp, d_histo, numElements);
      cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_histo,
        d_histo, 46 * nB);
      splitBallot_Mode13_large<NUM_WARPS, 46> <<<nB, NT>>>(d_inp, d_histo,
        d_fin, numElements);
    break;
    case 47:
      histogramBallot_Mode13_large<NUM_WARPS, 47, 6, LOG_WARPS> <<<
        nB, NT>>>(d_inp, d_histo, numElements);
      cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_histo,
        d_histo, 47 * nB);
      splitBallot_Mode13_large<NUM_WARPS, 47> <<<nB, NT>>>(d_inp, d_histo,
        d_fin, numElements);
    break;
    case 48:
      histogramBallot_Mode13_large<NUM_WARPS, 48, 6, LOG_WARPS> <<<
        nB, NT>>>(d_inp, d_histo, numElements);
      cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_histo,
        d_histo, 48 * nB);
      splitBallot_Mode13_large<NUM_WARPS, 48> <<<nB, NT>>>(d_inp, d_histo,
        d_fin, numElements);
    break;
    case 49:
      histogramBallot_Mode13_large<NUM_WARPS, 49, 6, LOG_WARPS> <<<
        nB, NT>>>(d_inp, d_histo, numElements);
      cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_histo,
        d_histo, 49 * nB);
      splitBallot_Mode13_large<NUM_WARPS, 49> <<<nB, NT>>>(d_inp, d_histo,
        d_fin, numElements);
    break;
    case 50:
      histogramBallot_Mode13_large<NUM_WARPS, 50, 6, LOG_WARPS> <<<
        nB, NT>>>(d_inp, d_histo, numElements);
      cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_histo,
        d_histo, 50 * nB);
      splitBallot_Mode13_large<NUM_WARPS, 50> <<<nB, NT>>>(d_inp, d_histo,
        d_fin, numElements);
    break;
    case 51:
      histogramBallot_Mode13_large<NUM_WARPS, 51, 6, LOG_WARPS> <<<
        nB, NT>>>(d_inp, d_histo, numElements);
      cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_histo,
        d_histo, 51 * nB);
      splitBallot_Mode13_large<NUM_WARPS, 51> <<<nB, NT>>>(d_inp, d_histo,
        d_fin, numElements);
    break;
    case 52:
      histogramBallot_Mode13_large<NUM_WARPS, 52, 6, LOG_WARPS> <<<
        nB, NT>>>(d_inp, d_histo, numElements);
      cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_histo,
        d_histo, 52 * nB);
      splitBallot_Mode13_large<NUM_WARPS, 52> <<<nB, NT>>>(d_inp, d_histo,
        d_fin, numElements);
    break;
    case 53:
      histogramBallot_Mode13_large<NUM_WARPS, 53, 6, LOG_WARPS> <<<
        nB, NT>>>(d_inp, d_histo, numElements);
      cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_histo,
        d_histo, 53 * nB);
      splitBallot_Mode13_large<NUM_WARPS, 53> <<<nB, NT>>>(d_inp, d_histo,
        d_fin, numElements);
    break;
    case 54:
      histogramBallot_Mode13_large<NUM_WARPS, 54, 6, LOG_WARPS> <<<
        nB, NT>>>(d_inp, d_histo, numElements);
      cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_histo,
        d_histo, 54 * nB);
      splitBallot_Mode13_large<NUM_WARPS, 54> <<<nB, NT>>>(d_inp, d_histo,
        d_fin, numElements);
    break;
    case 55:
      histogramBallot_Mode13_large<NUM_WARPS, 55, 6, LOG_WARPS> <<<
        nB, NT>>>(d_inp, d_histo, numElements);
      cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_histo,
        d_histo, 55 * nB);
      splitBallot_Mode13_large<NUM_WARPS, 55> <<<nB, NT>>>(d_inp, d_histo,
        d_fin, numElements);
    break;
    case 56:
      histogramBallot_Mode13_large<NUM_WARPS, 56, 6, LOG_WARPS> <<<
        nB, NT>>>(d_inp, d_histo, numElements);
      cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_histo,
        d_histo, 56 * nB);
      splitBallot_Mode13_large<NUM_WARPS, 56> <<<nB, NT>>>(d_inp, d_histo,
        d_fin, numElements);
    break;
    case 57:
      histogramBallot_Mode13_large<NUM_WARPS, 57, 6, LOG_WARPS> <<<
        nB, NT>>>(d_inp, d_histo, numElements);
      cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_histo,
        d_histo, 57 * nB);
      splitBallot_Mode13_large<NUM_WARPS, 57> <<<nB, NT>>>(d_inp, d_histo,
        d_fin, numElements);
    break;
    case 58:
      histogramBallot_Mode13_large<NUM_WARPS, 58, 6, LOG_WARPS> <<<
        nB, NT>>>(d_inp, d_histo, numElements);
      cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_histo,
        d_histo, 58 * nB);
      splitBallot_Mode13_large<NUM_WARPS, 58> <<<nB, NT>>>(d_inp, d_histo,
        d_fin, numElements);
    break;
    case 59:
      histogramBallot_Mode13_large<NUM_WARPS, 59, 6, LOG_WARPS> <<<
        nB, NT>>>(d_inp, d_histo, numElements);
      cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_histo,
        d_histo, 59 * nB);
      splitBallot_Mode13_large<NUM_WARPS, 59> <<<nB, NT>>>(d_inp, d_histo,
        d_fin, numElements);
    break;
    case 60:
      histogramBallot_Mode13_large<NUM_WARPS, 60, 6, LOG_WARPS> <<<
        nB, NT>>>(d_inp, d_histo, numElements);
      cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_histo,
        d_histo, 60 * nB);
      splitBallot_Mode13_large<NUM_WARPS, 60> <<<nB, NT>>>(d_inp, d_histo,
        d_fin, numElements);
    break;
    case 61:
      histogramBallot_Mode13_large<NUM_WARPS, 61, 6, LOG_WARPS> <<<
        nB, NT>>>(d_inp, d_histo, numElements);
      cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_histo,
        d_histo, 61 * nB);
      splitBallot_Mode13_large<NUM_WARPS, 61> <<<nB, NT>>>(d_inp, d_histo,
        d_fin, numElements);
    break;
    case 62:
      histogramBallot_Mode13_large<NUM_WARPS, 62, 6, LOG_WARPS> <<<
        nB, NT>>>(d_inp, d_histo, numElements);
      cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_histo,
        d_histo, 62 * nB);
      splitBallot_Mode13_large<NUM_WARPS, 62> <<<nB, NT>>>(d_inp, d_histo,
        d_fin, numElements);
    break;
    case 63:
      histogramBallot_Mode13_large<NUM_WARPS, 63, 6, LOG_WARPS> <<<
        nB, NT>>>(d_inp, d_histo, numElements);
      cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_histo,
        d_histo, 63 * nB);
      splitBallot_Mode13_large<NUM_WARPS, 63> <<<nB, NT>>>(d_inp, d_histo,
        d_fin, numElements);
    break;
    case 64:
      histogramBallot_Mode13_large<NUM_WARPS, 64, 6, LOG_WARPS> <<<
        nB, NT>>>(d_inp, d_histo, numElements);
      cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_histo,
        d_histo, 64 * nB);
      splitBallot_Mode13_large<NUM_WARPS, 64> <<<nB, NT>>>(d_inp, d_histo,
        d_fin, numElements);
    break;
    default:
    break;
  }
  if (numBuckets > 96) {
     markBins_general<<<nB, NT>>>(d_mask, d_inp, numElements, 97);
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_mask,
        d_out, d_inp, d_fin, numElements, 0, int(ceil(log2(float(97)))));
  }
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
  /*  CUDA_SAFE_CALL(cudaMalloc((void**)&plan->m_tempValues,    sizeof(unsigned int)*plan->m_numElements));
    CUDA_SAFE_CALL(cudaMalloc((void**)&plan->m_partitionBeginA, plan->m_swapPoint*plan->m_subPartitions*sizeof(unsigned int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&plan->m_partitionSizeA, plan->m_swapPoint*plan->m_subPartitions*sizeof(unsigned int)));
    switch(plan->m_config.datatype)
    {
    case CUDPP_CHAR:
        CUDA_SAFE_CALL(cudaMalloc((void**)&plan->m_tempKeys,    sizeof(char)*plan->m_numElements));
        break;
    case CUDPP_UCHAR:
        CUDA_SAFE_CALL(cudaMalloc((void**)&plan->m_tempKeys,    sizeof(unsigned char)*plan->m_numElements));
        break;
    case CUDPP_SHORT:
        CUDA_SAFE_CALL(cudaMalloc((void**)&plan->m_tempKeys,    sizeof(short)*plan->m_numElements));
        break;
    case CUDPP_USHORT:
        CUDA_SAFE_CALL(cudaMalloc((void**)&plan->m_tempKeys,    sizeof(unsigned short)*plan->m_numElements));
        break;
    case CUDPP_INT:
        CUDA_SAFE_CALL(cudaMalloc((void**)&plan->m_tempKeys,    sizeof(int)*plan->m_numElements));
        break;
    case CUDPP_UINT:
        CUDA_SAFE_CALL(cudaMalloc((void**)&plan->m_tempKeys,    sizeof(unsigned int)*plan->m_numElements));

        break;
    case CUDPP_FLOAT:
        CUDA_SAFE_CALL(cudaMalloc((void**)&plan->m_tempKeys,    sizeof(float)*plan->m_numElements));
        break;
    case CUDPP_DOUBLE:
        CUDA_SAFE_CALL(cudaMalloc((void**)&plan->m_tempKeys,    sizeof(double)*plan->m_numElements));
        break;
    case CUDPP_LONGLONG:
        CUDA_SAFE_CALL(cudaMalloc((void**)&plan->m_tempKeys,    sizeof(long long)*plan->m_numElements));

        break;
    case CUDPP_ULONGLONG:
        CUDA_SAFE_CALL(cudaMalloc((void**)&plan->m_tempKeys,    sizeof(unsigned long long)*plan->m_numElements));

        break;
    default:
        break;
    }
*/
}

/** @brief Deallocates intermediate memory from allocRadixSortStorage.
 *
 *
 * @param[in] plan Pointer to CUDPPMergeSortPlan object
 **/

void freeMultiSplitStorage(CUDPPMultiSplitPlan* plan)
{
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

void cudppMultiSplitDispatch(unsigned int *elements,
                            size_t numElements,
                            size_t numBuckets,
                            const CUDPPMultiSplitPlan *plan)
{
  unsigned int logBuckets =ceil(log2((double) numBuckets));
  unsigned int *d_mask, *d_out, *d_fin, *d_histo, *d_value_inp, *d_value_out;
  uint64      *d_key_value;
  unsigned int nB = ceil(numElements / (NUM_WARPS * 32));

  CUDA_SAFE_CALL(cudaMalloc((void**) &d_value_inp, numElements*sizeof(unsigned int)));   // gpu value inputs
  CUDA_SAFE_CALL(cudaMalloc((void**) &d_value_out, numElements*sizeof(unsigned int)));   // gpu value outputs
  CUDA_SAFE_CALL(cudaMalloc((void**) &d_fin, numElements*sizeof(unsigned int))); // final masks (used for reduced bit method, etc.)

  if (numBuckets > 96) {
    CUDA_SAFE_CALL(cudaMalloc((void**) &d_out, numElements*sizeof(unsigned int))); // gpu output
    CUDA_SAFE_CALL(cudaMalloc((void**) &d_mask, (numElements+1)*sizeof(unsigned int)));  // mask verctor, +1 added only for the near-far implementation
  }
  if (numBuckets < 512)
    CUDA_SAFE_CALL(
        cudaMalloc((void**) &d_histo,
            sizeof(unsigned int) * numBuckets * nB * NUM_WARPS * 2
                * PACK_DEPTH)); //
  CUDA_SAFE_CALL(cudaMalloc((void**) &d_key_value, numElements*sizeof(uint64))); // key value pair intermediate vector.

#if NUM_BUCKETS < 512
  CUDA_SAFE_CALL(cudaMemset(d_histo, 0, sizeof(unsigned int)*numBuckets * nB * NUM_WARPS * 2));
#endif

  if (numBuckets > 96) {
    CUDA_SAFE_CALL(cudaMemset(d_mask, 0, sizeof(unsigned int)*(numElements+1)));
    CUDA_SAFE_CALL(cudaMemset(d_out, 0, sizeof(unsigned int)*numElements));
  }
  
  CUDA_SAFE_CALL(cudaMemset(d_fin, 0, sizeof(unsigned int)*numElements));
  CUDA_SAFE_CALL(cudaMemset(d_key_value, 0, sizeof(uint64)*numElements));

  cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);

  void     *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;

  //cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_histo,
  //    d_histo, numBuckets * nB);
  if (numBuckets > 96) {
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_mask, d_out, elements, d_fin,
      numElements, 0, ceil(log2((float)numBuckets)));
  } else {
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_histo, d_histo, numBuckets * nB * NUM_WARPS * PACK_DEPTH);
  }

  g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes);

  runMultiSplit(elements, d_histo, d_fin, d_mask, d_out, d_temp_storage, temp_storage_bytes, numElements, numBuckets);

  CUDA_SAFE_CALL(cudaMemcpy(elements, d_fin, numElements*sizeof(unsigned int), cudaMemcpyDeviceToDevice));

  if(d_temp_storage)
    CubDebugExit(g_allocator.DeviceFree(d_temp_storage));
  if(d_value_inp)   cudaFree(d_value_inp);
  if(d_value_out)   cudaFree(d_value_out);
  if(d_key_value)   cudaFree(d_key_value);
  // g_allocator.DeviceFree(d_inp);
  if(d_fin)   cudaFree(d_fin);
  // g_allocator.DeviceFree(d_fin);
  if (numBuckets > 96) {
    if (d_mask)
      cudaFree(d_mask);
    // g_allocator.DeviceFree(d_mask);
    if (d_out)
      cudaFree(d_out);
  }
  // g_allocator.DeviceFree(d_out);
  if(d_histo) cudaFree(d_histo);

  /*    switch(plan->m_config.datatype)
    {
    case CUDPP_INT:
        runMergeSort<int>((int*)keys, (unsigned int*)values, numElements, plan);
        break;
    case CUDPP_UINT:
        runMergeSort<unsigned int>((unsigned int*)keys, (unsigned int*)values, numElements, plan);
        break;
    case CUDPP_FLOAT:
        runMergeSort<float>((float*)keys, (unsigned int*)values, numElements, plan);
        break;
    default:
         do nothing, not handled
        break;
    }*/
}

#ifdef __cplusplus
}
#endif

/** @} */ // end mergesort functions
/** @} */ // end cudpp_app
