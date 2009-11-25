// -------------------------------------------------------------
// CUDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision: 5636 $
// $Date: 2009-07-02 13:39:38 +1000 (Thu, 02 Jul 2009) $
// ------------------------------------------------------------- 
// This source code is distributed under the terms of license.txt 
// in the root directory of this source distribution.
// ------------------------------------------------------------- 

/**
 * @file
 * reduce_app.cu
 *
 * @brief CUDPP application-level reduction routines
 */

#include "kernel/reduce_kernel.cu"
#include "cudpp_plan.h"
#include "cudpp_util.h"
#include <cutil.h>

/** \addtogroup cudpp_app
  *
  */

/** @name Reduce Functions
 * @{
 */

/**
  * @brief Per-block reduction function
  *
  * This function dispatches the appropriate reduction kernel given the size of the blocks.
  *
  * @param d_odata The output data pointer.  Each block writes a single output element.
  * @param d_idata The input data pointer.  
  * @param numElements The number of elements to be reduced.
  * @param plan A pointer to the plan structure for the reduction.
*/
template <class T, class Oper>
void reduceBlocks(T *d_odata, const T *d_idata, size_t numElements, CUDPPReducePlan *plan)
{
    unsigned int numThreads = (numElements > 2 * plan->m_threadsPerBlock) ?
        plan->m_threadsPerBlock : ceilPow2((numElements + 1) / 2);
    dim3 dimBlock(numThreads, 1, 1);
    unsigned int numBlocks = min(plan->m_maxBlocks, 
        (numElements + plan->m_threadsPerBlock - 1) / plan->m_threadsPerBlock);

    dim3 dimGrid(numBlocks, 1, 1);
    int smemSize = REDUCE_CTA_SIZE * sizeof(T);

    // choose which of the optimized versions of reduction to launch
    
    if (isPowerOfTwo(numElements))
    {
        switch (dimBlock.x)
        {
        case 512:
            reduce<T, Oper, 512, true><<< dimGrid, dimBlock, smemSize >>>(d_odata, d_idata, numElements); break;
        case 256:
            reduce<T, Oper, 256, true><<< dimGrid, dimBlock, smemSize >>>(d_odata, d_idata, numElements); break;
        case 128:
            reduce<T, Oper, 128, true><<< dimGrid, dimBlock, smemSize >>>(d_odata, d_idata, numElements); break;
        case 64:
            reduce<T, Oper, 64, true><<< dimGrid, dimBlock, smemSize >>>(d_odata, d_idata, numElements); break;
        case 32:
            reduce<T, Oper, 32, true><<< dimGrid, dimBlock, smemSize >>>(d_odata, d_idata, numElements); break;
        case 16:
            reduce<T, Oper, 16, true><<< dimGrid, dimBlock, smemSize >>>(d_odata, d_idata, numElements); break;
        case  8:
            reduce<T, Oper,  8, true><<< dimGrid, dimBlock, smemSize >>>(d_odata, d_idata, numElements); break;
        case  4:
            reduce<T, Oper,  4, true><<< dimGrid, dimBlock, smemSize >>>(d_odata, d_idata, numElements); break;
        case  2:
            reduce<T, Oper,  2, true><<< dimGrid, dimBlock, smemSize >>>(d_odata, d_idata, numElements); break;
        case  1:
            reduce<T, Oper,  1, true><<< dimGrid, dimBlock, smemSize >>>(d_odata, d_idata, numElements); break;
        }
    }
    else
    {
        switch (dimBlock.x)
        {
        case 512:
            reduce<T, Oper, 512, false><<< dimGrid, dimBlock, smemSize >>>(d_odata, d_idata, numElements); break;
        case 256:
            reduce<T, Oper, 256, false><<< dimGrid, dimBlock, smemSize >>>(d_odata, d_idata, numElements); break;
        case 128:
            reduce<T, Oper, 128, false><<< dimGrid, dimBlock, smemSize >>>(d_odata, d_idata, numElements); break;
        case 64:
            reduce<T, Oper,  64, false><<< dimGrid, dimBlock, smemSize >>>(d_odata, d_idata, numElements); break;
        case 32:
            reduce<T, Oper,  32, false><<< dimGrid, dimBlock, smemSize >>>(d_odata, d_idata, numElements); break;
        case 16:
            reduce<T, Oper,  16, false><<< dimGrid, dimBlock, smemSize >>>(d_odata, d_idata, numElements); break;
        case  8:
            reduce<T, Oper,   8, false><<< dimGrid, dimBlock, smemSize >>>(d_odata, d_idata, numElements); break;
        case  4:
            reduce<T, Oper,   4, false><<< dimGrid, dimBlock, smemSize >>>(d_odata, d_idata, numElements); break;
        case  2:
            reduce<T, Oper,   2, false><<< dimGrid, dimBlock, smemSize >>>(d_odata, d_idata, numElements); break;
        case  1:
            reduce<T, Oper,   1, false><<< dimGrid, dimBlock, smemSize >>>(d_odata, d_idata, numElements); break;
        }
    }
}
/**
  * @brief Array reduction function.
  *
  * Performs multi-level reduction on large arrays using reduceBlocks().  
  *
  * @param d_odata The output data pointer.  This is a pointer to a single element.
  * @param d_idata The input data pointer.  
  * @param numElements The number of elements to be reduced.
  * @param plan A pointer to the plan structure for the reduction.
*/
template <class Oper, class T>
void reduceArray(T *d_odata, const T *d_idata, size_t numElements, CUDPPReducePlan *plan)
{
    unsigned int numBlocks = min(plan->m_maxBlocks, 
        (numElements + plan->m_threadsPerBlock - 1) / plan->m_threadsPerBlock);

    if (numBlocks > 1)
    {
        reduceBlocks<T, Oper>((T*)plan->m_blockSums, d_idata, numElements, plan);
        reduceBlocks<T, Oper>(d_odata, (const T*)plan->m_blockSums, numBlocks, plan);
    }
    else
    {
        reduceBlocks<T, Oper>(d_odata, d_idata, numElements, plan);
    }
}

/** @brief Allocate intermediate arrays used by reductions.
  *
  * Reductions of large arrays must be split into multiple blocks, 
  * where each block is reduced by a single CUDA thread block.  
  * Each block writes its partial sum to global memory where it is reduced
  * to a single element in a second pass.
  *
  * @param plan Pointer to CUDPPReducePlan object containing options and number 
  *             of elements, which is used to compute storage requirements, and
  *             within which intermediate storage is allocated.
  */
void allocReduceStorage(CUDPPReducePlan *plan)
{
    unsigned int blocks = min(plan->m_maxBlocks, 
        (plan->m_numElements + plan->m_threadsPerBlock - 1) / plan->m_threadsPerBlock);
  
    switch (plan->m_config.datatype)
    {
    case CUDPP_INT:
        cudaMalloc(&plan->m_blockSums, blocks * sizeof(int));
        break;
    case CUDPP_UINT:
        cudaMalloc(&plan->m_blockSums, blocks * sizeof(unsigned int));
        break;
    case CUDPP_FLOAT:
        cudaMalloc(&plan->m_blockSums, blocks * sizeof(float));
        break;
    case CUDPP_DOUBLE:
        cudaMalloc(&plan->m_blockSums, blocks * sizeof(double));
        break;
    default:
        break;
    }
   
    CUT_CHECK_ERROR("allocReduceStorage");
}

/** @brief Deallocate intermediate block sums arrays in a CUDPPReducePlan object.
  *
  * These arrays must have been allocated by allocScanStorage(), which is called
  * by the constructor of cudppReducePlan().  
  *
  * @param plan Pointer to CUDPPReducePlan object initialized by allocScanStorage().
  */
void freeReduceStorage(CUDPPReducePlan *plan)
{
    cudaFree(plan->m_blockSums);

    CUT_CHECK_ERROR("freeReduceStorage");

    plan->m_blockSums = 0;
}

/** @brief Dispatch function to perform a parallel reduction on an
  * array with the specified configuration.
  *
  * This is the dispatch routine which calls reduceArray() with 
  * appropriate template parameters and arguments to achieve the scan as 
  * specified in \a plan. 
  * 
  * @param[out] d_odata     The output array of scan results
  * @param[in]  d_idata     The input array
  * @param[in]  numElements The number of elements to scan
  * @param[in]  plan     Pointer to CUDPPReducePlan object containing reduce options
  *                      and intermediate storage
  */
void cudppReduceDispatch(void *d_odata, const void *d_idata, size_t numElements, CUDPPReducePlan *plan)
{
    switch (plan->m_config.datatype)
    {
    case CUDPP_INT:
        switch (plan->m_config.op)
        {
        case CUDPP_ADD:
        default:
            reduceArray< OperatorAdd<int> >((int*)d_odata, (int*)d_idata, numElements, plan);
            break;
        case CUDPP_MULTIPLY:
            reduceArray< OperatorMultiply<int> >((int*)d_odata, (int*)d_idata, numElements, plan);
            break;
        case CUDPP_MAX:
            reduceArray< OperatorMax<int> >((int*)d_odata, (int*)d_idata, numElements, plan);
            break;
        case CUDPP_MIN:
            reduceArray< OperatorMin<int> >((int*)d_odata, (int*)d_idata, numElements, plan);
            break;
        }
        break;
    case CUDPP_UINT:
        switch (plan->m_config.op)
        {
        case CUDPP_ADD:
        default:
            reduceArray< OperatorAdd<unsigned int> >((unsigned int*)d_odata, (unsigned int*)d_idata, numElements, plan);
            break;
        case CUDPP_MULTIPLY:
            reduceArray< OperatorMultiply<unsigned int> >((unsigned int*)d_odata, (unsigned int*)d_idata, numElements, plan);
            break;
        case CUDPP_MAX:
            reduceArray< OperatorMax<unsigned int> >((unsigned int*)d_odata, (unsigned int*)d_idata, numElements, plan);
            break;
        case CUDPP_MIN:
            reduceArray< OperatorMin<unsigned int> >((unsigned int*)d_odata, (unsigned int*)d_idata, numElements, plan);
            break;
        }
        break;
    case CUDPP_FLOAT:
        switch (plan->m_config.op)
        {
        case CUDPP_ADD:
        default:
            reduceArray< OperatorAdd<float> >((float*)d_odata, (float*)d_idata, numElements, plan);
            break;
        case CUDPP_MULTIPLY:
            reduceArray< OperatorMultiply<float> >((float*)d_odata, (float*)d_idata, numElements, plan);
            break;
        case CUDPP_MAX:
            reduceArray< OperatorMax<float> >((float*)d_odata, (float*)d_idata, numElements, plan);
            break;
        case CUDPP_MIN:
            reduceArray< OperatorMin<float> >((float*)d_odata, (float*)d_idata, numElements, plan);
            break;
        }
        break;
    case CUDPP_DOUBLE:
        switch (plan->m_config.op)
        {
        case CUDPP_ADD:
        default:
            reduceArray< OperatorAdd<double> >((double*)d_odata, (double*)d_idata, numElements, plan);
            break;
        case CUDPP_MULTIPLY:
            reduceArray< OperatorMultiply<double> >((double*)d_odata, (double*)d_idata, numElements, plan);
            break;
        case CUDPP_MAX:
            reduceArray< OperatorMax<double> >((double*)d_odata, (double*)d_idata, numElements, plan);
            break;
        case CUDPP_MIN:
            reduceArray< OperatorMin<double> >((double*)d_odata, (double*)d_idata, numElements, plan);
            break;
        }
        break;
    default:
        break;
    }
}

/** @} */ // end reduce functions
/** @} */ // end cudpp_app
