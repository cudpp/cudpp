// -------------------------------------------------------------
// cuDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision: 5636 $
// $Date: 2009-07-02 13:39:38 +1000 (Thu, 02 Jul 2009) $
// ------------------------------------------------------------- 
// This source code is distributed under the terms of license.txt in
// the root directory of this source distribution.
// ------------------------------------------------------------- 

/**
 * @file
 * cudpp_globals.h
 *
 * @brief Global declarations defining machine characteristics of GPU target
 * These are currently set for best performance on G8X GPUs.  The optimal 
 * parameters may change on future GPUs. In the future, we hope to make
 * CUDPP a self-tuning library.
 */

#ifndef __CUDPP_GLOBALS_H__
#define __CUDPP_GLOBALS_H__

const int SORT_CTA_SIZE = 256;                   /**< Number of threads per CTA for radix sort. Must equal 16 * number of radices */
const int SCAN_CTA_SIZE = 128;                   /**< Number of threads in a CTA */
const int REDUCE_CTA_SIZE = 256;                 /**< Number of threads in a CTA */

const int LOG_SCAN_CTA_SIZE = 7;                 /**< log_2(CTA_SIZE) */

const int WARP_SIZE = 32;                        /**< Number of threads in a warp */

const int LOG_WARP_SIZE = 5;                     /**< log_2(WARP_SIZE) */
const int LOG_SIZEOF_FLOAT = 2;                  /**< log_2(sizeof(float)) */

const int SCAN_ELTS_PER_THREAD = 8;              /**< Number of elements per scan thread */
const int SEGSCAN_ELTS_PER_THREAD = 8;           /**< Number of elements per segmented scan thread */

#endif // __CUDPP_GLOBALS_H__

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
