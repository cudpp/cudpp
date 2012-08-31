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

/** @brief Perform Huffman encoding
 * 
 * @todo
 *
 */
void listRank(int                       *d_ranked_values,
              int                       *d_unranked_values,
              int                       *d_next_indices,
              size_t                    head,
              size_t                    numElements,
              const CUDPPListRankPlan   *plan)
{
    int step = 1;
    int cnt = 1;
    int* d_tmp = d_next_indices;

    // thread info -- kernel1
    int nThreads = LISTRANK_CTA_BLOCK;
    int tThreads = LISTRANK_TOTAL;
    int nBlocks  = tThreads/LISTRANK_CTA_BLOCK;

    dim3 grid_construct   (nBlocks,  1, 1);
    dim3 threads_construct(nThreads, 1, 1);

    // thread info -- kernel2
    tThreads = LISTRANK_MAX;
    nBlocks = tThreads/LISTRANK_CTA_BLOCK;
    dim3 grid_construct2   (nBlocks,  1, 1);
    dim3 threads_construct2(nThreads, 1, 1);


    while(step<LISTRANK_MAX)
    {
        // Each step doubles the number of threads added to pointer "chase"
        if(cnt%2 == 1)
        {
            // ping
            list_rank_kernel_soa_1<<< grid_construct, threads_construct >>>
                (d_ranked_values, d_unranked_values, d_tmp,
                plan->m_d_tmp1, plan->m_d_tmp2, step, head, numElements);
            d_tmp = plan->m_d_tmp3;
        }
        else
        {
            // pong
            list_rank_kernel_soa_1<<< grid_construct, threads_construct >>>
                (d_ranked_values, d_unranked_values, plan->m_d_tmp1,
                d_tmp, plan->m_d_tmp2, step, head, numElements);
        }
        step *= 2;
        cnt++;
    }

    // Out of threads to dispatch, each thread now keeps chasing pointer until
    // all lists are ranked
    if(LISTRANK_MAX < numElements)
    {
        if(cnt%2 == 0)
        {
            list_rank_kernel_soa_2<<< grid_construct2, threads_construct2 >>>
                (d_ranked_values, d_unranked_values, plan->m_d_tmp1, plan->m_d_tmp2, head, numElements);
            CUDA_SAFE_CALL(cudaThreadSynchronize());
        }
        else
        {
            list_rank_kernel_soa_2<<< grid_construct2, threads_construct2 >>>
                (d_ranked_values, d_unranked_values, d_tmp, plan->m_d_tmp2, head, numElements);
            CUDA_SAFE_CALL(cudaThreadSynchronize());
        }
    }
}

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

    CUDA_SAFE_CALL(cudaMalloc( (void**) &(plan->m_d_tmp1),     numElts*sizeof(int) ));
    CUDA_SAFE_CALL(cudaMalloc( (void**) &(plan->m_d_tmp2),     numElts*sizeof(int) ));
    CUDA_SAFE_CALL(cudaMalloc( (void**) &(plan->m_d_tmp3),     numElts*sizeof(int) ));
}

/** @brief Deallocate intermediate block arrays in a CUDPPListRankPlan object.
 *
 * @todo 
 *
 * @param[in,out] plan Pointer to CUDPPListRankPlan object initialized by allocListRankStorage().
 */
void freeListRankStorage(CUDPPListRankPlan *plan)
{
    if(plan->m_d_tmp1 != NULL) CUDA_SAFE_CALL(cudaFree(plan->m_d_tmp1));
    if(plan->m_d_tmp2 != NULL) CUDA_SAFE_CALL(cudaFree(plan->m_d_tmp2));
    if(plan->m_d_tmp3 != NULL) CUDA_SAFE_CALL(cudaFree(plan->m_d_tmp3));
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
                           size_t head,
                           size_t numElements,
                           const CUDPPListRankPlan *plan)
{
    // Call to list ranker
    // TODO - template to allow other value types
    listRank((int*) d_ranked_values, (int*) d_unranked_values,
             (int*) d_next_indices, head, numElements, plan);
}


#ifdef __cplusplus
}
#endif


/** @} */ // end listrank functions
/** @} */ // end cudpp_app