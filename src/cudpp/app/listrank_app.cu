// -------------------------------------------------------------
// CUDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision$
// $Date$
// ------------------------------------------------------------- 
// This source code is distributed under the terms of license.txt 
// in the root directory of this source distribution.
// ------------------------------------------------------------- 

#include <stdio.h>
#include <stdlib.h>

#include "cuda_util.h"
#include "cudpp_globals.h"
#include "cudpp.h"
#include "cudpp_util.h"
#include "cudpp_plan.h"

#include "kernel/listrank_kernel.cuh"

/**
 * @file
 * listrank_app.cu
 * 
 * @brief CUDPP application-level listrank routines
 */

/** \addtogroup cudpp_app 
 * @{
 */

/** @name ListRank Functions
 * @{
 */

#ifdef __cplusplus
extern "C" 
{
#endif

/** @brief Allocate intermediate arrays used by ListRank.
 *
 * @todo
 *
 * @param [in,out] plan Pointer to CUDPPListRankPlan object containing
 *                      options and number of elements, which is used
 *                      to compute storage requirements, and within
 *                      which intermediate storage is allocated.
 */
void allocListRankStorage(CUDPPListRankPlan *plan)
{
    size_t numElts = plan->m_numElements;
}

/** @brief Deallocate intermediate block arrays in a CUDPPListRankPlan object.
 *
 * @todo 
 *
 * @param[in,out] plan Pointer to CUDPPListRankPlan object initialized by allocListRankStorage().
 */
void freeListRankStorage(CUDPPListRankPlan *plan)
{
}


/** @brief Dispatch function to perform parallel list ranking on a
 * linked-list with the specified configuration.
 *
 * @todo
 * 
 * @param[out] d_ranked_values Ranked values array
 * @param[in]  d_unranked_values Unranked values array
 * @param[in]  d_next_indices Next indices array
 * @param[in]  head Head pointer index
 * @param[in]  numElements Number of nodes values to rank
 * @param[in]  plan     Pointer to CUDPPListRankPlan object containing
 *                      list ranking options and intermediate storage
 */
void cudppListRankDispatch(void *d_ranked_values,
                           void *d_unranked_values,
                           void *d_next_indices,
                           size_t *head,
                           size_t numElements,
                           const CUDPPListRankPlan *plan)
{
}


#ifdef __cplusplus
}
#endif


/** @} */ // end listrank functions
/** @} */ // end cudpp_app