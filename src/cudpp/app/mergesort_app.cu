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
 * mergesort_app.cu
 *   
 * @brief CUDPP application-level merge sorting routines
 */

/** @addtogroup cudpp_app 
 * @{
 */

/** @name MergeSort Functions
 * @{
 */
 
#include "cuda_util.h"
#include "cudpp.h"
#include "cudpp_util.h"
#include "cudpp_mergesort.h"
#include "kernel/mergesort_kernel.cuh"
#include "limits.h"


#define BLOCKSORT_SIZE 1024

/** @brief Performs merge sor utilzing three stages. 
* (1) Blocksort, (2) simple merge and (3) multi merge
* 
* 
* @param[in,out] pkeys Keys to be sorted.
* @param[in,out] pvals Associated values to be sorted
* @param[in] numElements Number of elements in the sort.
* @param[in] plan Configuration information for mergesort.
**/
template<typename T, T INV_VAL, T MIN_VAL>
void runMergeSort(T *pkeys, 
             unsigned int *pvals,
             size_t numElements, 
             const CUDPPMergeSortPlan *plan)
{

	int numPartitions = (numElements+BLOCKSORT_SIZE-1)/BLOCKSORT_SIZE;
	int numBlocks = numPartitions/2;
	int partitionSize = BLOCKSORT_SIZE;
	int subPartitions = 4;
	
	
	T* temp_keys;
	unsigned int* temp_vals;

	CUDA_SAFE_CALL( cudaMalloc((void **) &temp_keys, sizeof(T)*numElements));
	CUDA_SAFE_CALL( cudaMalloc((void **) &temp_vals, sizeof(unsigned int)*numElements));

	int *partitionSizeA, *partitionBeginA;
	unsigned int swapPoint = 32;
	int blockLimit = swapPoint*subPartitions;	

	cudaMalloc((void**)&partitionBeginA, blockLimit*sizeof(unsigned int)); 
	cudaMalloc((void**)&partitionSizeA, blockLimit*sizeof(unsigned int));

	int numThreads = 128;	
#define DEPTH 8
	blockWiseSort<T, DEPTH, INV_VAL>
	<<<numPartitions, BLOCKSORT_SIZE/DEPTH, (BLOCKSORT_SIZE)*sizeof(T) + (BLOCKSORT_SIZE)*sizeof(unsigned int)>>>(pkeys, pvals, BLOCKSORT_SIZE, numElements);

	int mult = 1; int count = 0;

	//we run p stages of simpleMerge until numBlocks <= some Critical level
	while(numPartitions > 32 )
	{				
		if(count%2 == 0)
		{ 				
			simpleMerge_lower<T, 2, INV_VAL, MIN_VAL>
				<<<numBlocks, CTASIZE_simple, sizeof(T)*(INTERSECT_B_BLOCK_SIZE_simple+4)>>>
				(pkeys, pvals, temp_keys, temp_vals, partitionSize*mult, (int)numElements);				
			simpleMerge_higher<T, 2, INV_VAL, MIN_VAL>
				<<<numBlocks, CTASIZE_simple, sizeof(T)*(INTERSECT_B_BLOCK_SIZE_simple+4)>>>
				(pkeys, pvals, temp_keys, temp_vals, partitionSize*mult, (int)numElements);		
			if(numPartitions%2 == 1)
			{			
				
				int offset = (partitionSize*mult*(numPartitions-1));
				int numElementsToCopy = numElements-offset;												
				simpleCopy<T>
					<<<(numElementsToCopy+numThreads-1)/numThreads, numThreads>>>(pkeys, pvals, temp_keys, temp_vals, offset, numElementsToCopy);
			}
		}
		else
		{			
			simpleMerge_lower<T, 2, INV_VAL, MIN_VAL>
				<<<numBlocks, CTASIZE_simple, sizeof(T)*(INTERSECT_B_BLOCK_SIZE_simple+4)>>>
				(temp_keys, temp_vals, pkeys, pvals, partitionSize*mult, numElements);				
			simpleMerge_higher<T, 2, INV_VAL, MIN_VAL>
				<<<numBlocks, CTASIZE_simple, sizeof(T)*(INTERSECT_B_BLOCK_SIZE_simple+4)>>>
				(temp_keys, temp_vals, pkeys, pvals, partitionSize*mult, numElements);	
			if(numPartitions%2 == 1)
			{			
				int offset = (partitionSize*mult*(numPartitions-1));
				int numElementsToCopy = numElements-offset;						
				simpleCopy<T>
					<<<(numElementsToCopy+numThreads-1)/numThreads, numThreads>>>(temp_keys, temp_vals, pkeys, pvals, offset, numElementsToCopy);
			}
		}
			
		mult*=2;
		count++;
		numPartitions = (numPartitions+1)/2;
		numBlocks=numPartitions/2;				
	}				
	
	

	//End of simpleMerge, now blocks cooperate to merge partitions
	while (numPartitions > 1)
	{		
		int secondBlocks = (numBlocks*subPartitions+numThreads-1)/numThreads;			
		if(count%2 == 1)
		{								
			findMultiPartitions<T, INV_VAL><<<secondBlocks, numThreads>>>(temp_keys, subPartitions, numBlocks*2, 
															partitionSize*mult, partitionBeginA, partitionSizeA, numElements);						
			mergeMulti_lower<T, 4, INV_VAL, MIN_VAL>
				<<<numBlocks*subPartitions, CTASIZE_multi, (INTERSECT_B_BLOCK_SIZE_multi+3)*sizeof(T)>>>
				(pkeys, pvals,temp_keys, temp_vals, subPartitions, numBlocks, partitionBeginA, partitionSizeA, mult*partitionSize, numElements);
			
			
			mergeMulti_higher<T, 4, INV_VAL, MIN_VAL>
				<<<numBlocks*subPartitions, CTASIZE_multi, (INTERSECT_B_BLOCK_SIZE_multi+3)*sizeof(T)>>>
				(pkeys, pvals, temp_keys, temp_vals, subPartitions, numBlocks, partitionBeginA, partitionSizeA, mult*partitionSize, numElements);
			
			if(numPartitions%2 == 1)
			{			
				int offset = (partitionSize*mult*(numPartitions-1));
				int numElementsToCopy = numElements-offset;				
				simpleCopy<T>
					<<<(numElementsToCopy+numThreads-1)/numThreads, numThreads>>>(temp_keys, temp_vals, pkeys, pvals, offset, numElementsToCopy);
			}
		
		}
		else
		{
				
			findMultiPartitions <T, INV_VAL> <<<secondBlocks, numThreads>>>(pkeys, subPartitions, numBlocks*2, partitionSize*mult, partitionBeginA, partitionSizeA, numElements);
				
			
			mergeMulti_lower<T, 4, INV_VAL, MIN_VAL>
				<<<numBlocks*subPartitions, CTASIZE_multi, (INTERSECT_B_BLOCK_SIZE_multi+3)*sizeof(T)>>>
				(temp_keys, temp_vals, pkeys, pvals, subPartitions, numBlocks, partitionBeginA, partitionSizeA, mult*partitionSize, numElements);
			
			mergeMulti_higher<T, 4, INV_VAL, MIN_VAL>
				<<<numBlocks*subPartitions, CTASIZE_multi, (INTERSECT_B_BLOCK_SIZE_multi+3)*sizeof(T)>>>
				(temp_keys, temp_vals, pkeys, pvals, subPartitions, numBlocks, partitionBeginA, partitionSizeA, mult*partitionSize, numElements);
			
			if(numPartitions%2 == 1)
			{			
				int offset = (partitionSize*mult*(numPartitions-1));
				int numElementsToCopy = numElements-offset;				
				simpleCopy<T>
					<<<(numElementsToCopy+numThreads-1)/numThreads, numThreads>>>(pkeys, pvals, temp_keys, temp_vals, offset, numElementsToCopy);
			}
		
		}

			
		count++;
		mult*=2;
		numBlocks/=2;
		subPartitions*=2;
		numPartitions = (numPartitions+1)/2;		
		numBlocks = numPartitions/2;
	}	
	
	/*int * tempPartSize, *tempPartStart;
	tempPartSize = (int*)malloc(sizeof(int)*numPartitions*subPartitions);
	tempPartStart = (int*)malloc(sizeof(int)*numPartitions*subPartitions);
	cudaMemcpy(tempPartSize, partitionSizeA, numPartitions*subPartitions*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(tempPartStart, partitionBeginA, numPartitions*subPartitions*sizeof(int), cudaMemcpyDeviceToHost);

	for(int i =0;i<numPartitions*subPartitions; i++)
		printf("%d ", tempPartSize[i]);
	printf("\n");
	for(int i =0;i<numPartitions*subPartitions; i++)
		printf("%d ", tempPartStart[i]);
	printf("\n");
	for(int i =0;i<numPartitions*subPartitions; i++)
		printf("%d ", tempPartStart[i]+tempPartSize[i]);
	printf("\n");*/
	if(count%2==1)
	{
		cudaMemcpy(pkeys, temp_keys, numElements*sizeof(T), cudaMemcpyDeviceToDevice);
		cudaMemcpy(pvals, temp_vals, numElements*sizeof(unsigned int), cudaMemcpyDeviceToDevice);
	}
	
	CUDA_SAFE_CALL(cudaFree(temp_keys));
	CUDA_SAFE_CALL(cudaFree(temp_vals));	
	
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
void allocMergeSortStorage(CUDPPMergeSortPlan *plan)
{               
}

/** @brief Deallocates intermediate memory from allocRadixSortStorage.
 *
 *
 * @param[in] plan Pointer to CUDPPMergeSortPlan object
**/

void freeMergeSortStorage(CUDPPMergeSortPlan* plan)
{
}

/** @brief Dispatch function to perform a sort on an array with 
 * a specified configuration.
 *
 * This is the dispatch routine which calls mergeSort...() with 
 * appropriate template parameters and arguments as specified by 
 * the plan.
 * @param[in,out] keys Keys to be sorted.
 * @param[in,out] values Associated values to be sorted (through keys).
 * @param[in] numElements Number of elements in the sort.
 * @param[in] plan Configuration information for mergeSort.
**/

void cudppMergeSortDispatch(void  *keys,
                            void  *values,
                            size_t numElements,
                            const CUDPPMergeSortPlan *plan)
{
    switch(plan->m_config.datatype)
    {
   // case CUDPP_CHAR:
     //   runSort<char>((char*)keys, (unsigned int*)values, numElements, plan);
     //   break;
//    case CUDPP_UCHAR:
  //      runSort<unsigned char>((unsigned char*)keys, (unsigned int*)values, numElements, plan);
    //    break;
    case CUDPP_INT:
        runMergeSort<int, INT_MAX, INT_MIN>((int*)keys, (unsigned int*)values, numElements, plan);
        break;
    case CUDPP_UINT:		
        runMergeSort<unsigned int, UINT_MAX, 0>((unsigned int*)keys, (unsigned int*)values, numElements, plan);
        break;
    case CUDPP_FLOAT:
        runMergeSort<float, FLT_MAX, -FLT_MAX>((float*)keys, (unsigned int*)values, numElements, plan);
        break;
    //case CUDPP_DOUBLE:
     //   runMergeSort<double, DBL_MAX>((double*)keys, (unsigned int*)values, numElements, plan);
     //   break;
    //case CUDPP_LONGLONG:
      //  runSort<long long, LLONG_MAX>((long long*)keys, (unsigned int*)values, numElements, plan);        
        //break;
    //case CUDPP_ULONGLONG:
      //  runSort<unsigned long long, ULLONG_MAX>((unsigned long long*)keys, (unsigned int*)values, numElements, plan);
        //break;
    }    
}                            

#ifdef __cplusplus
}
#endif






/** @} */ // end mergesort functions
/** @} */ // end cudpp_app
