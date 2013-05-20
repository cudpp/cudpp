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
* stringsort_app.cu
*   
* @brief CUDPP application-level merge sorting routines
*/

/** @addtogroup cudpp_app 
* @{
*/

/** @name StringSort Functions
* @{
*/

#include "cuda_util.h"
#include "cudpp.h"
#include "cudpp_util.h"
#include "cudpp_stringsort.h"
#include "kernel/stringsort_kernel.cuh"
#include "limits.h"


#define BLOCKSORT_SIZE 1024
#define DEPTH 8

/** @brief Performs merge sor utilzing three stages. 
* (1) Blocksort, (2) simple merge and (3) multi merge on a 
* set of strings 
* 
* @param[in,out] pkeys Keys (first four characters of string) to be sorted.
* @param[in,out] pvals Addresses of string locations for tie-breaks
* @param[out] stringVals global string value array (four characters stuffed into a uint)
* @param[in] numElements Number of elements in the sort.
* @param[in] stringArrayLength The size of our string array in uints (4 chars per uint)
* @param[in] plan Configuration information for mergesort.
**/
void runStringSort(unsigned int *pkeys, 
				   unsigned int *pvals,
				   unsigned int *stringVals,
				   size_t numElements,
				   size_t stringArrayLength,
				   const CUDPPStringSortPlan *plan)
{

	//printf("start\n");
	int numPartitions = (numElements+BLOCKSORT_SIZE-1)/BLOCKSORT_SIZE;
	int numBlocks = numPartitions/2;
	int partitionSize = BLOCKSORT_SIZE;
	int subPartitions = 4;


	unsigned int* temp_keys;
	unsigned int* temp_vals;

	CUDA_SAFE_CALL( cudaMalloc((void **) &temp_keys, sizeof(unsigned int)*numElements));
	CUDA_SAFE_CALL( cudaMalloc((void **) &temp_vals, sizeof(unsigned int)*numElements));


	unsigned int *partitionSizeA, *partitionBeginA, *partitionSizeB, *partitionBeginB;
	unsigned int swapPoint = 32;
	int blockLimit = swapPoint*subPartitions;	

	cudaMalloc((void**)&partitionBeginA, blockLimit*sizeof(unsigned int)); 
	cudaMalloc((void**)&partitionSizeA, blockLimit*sizeof(unsigned int));
	cudaMalloc((void**)&partitionBeginB, blockLimit*sizeof(unsigned int)); 
	cudaMalloc((void**)&partitionSizeB, blockLimit*sizeof(unsigned int));

	int numThreads = 128;	

	blockWiseStringSort<unsigned int, DEPTH>
		<<<numPartitions, BLOCKSORT_SIZE/DEPTH, 2*(BLOCKSORT_SIZE)*sizeof(unsigned int)>>>(pkeys, pvals, stringVals, BLOCKSORT_SIZE, numElements, stringArrayLength);

	int mult = 1; int count = 0;

	CUDA_SAFE_CALL(cudaThreadSynchronize());
	//we run p stages of simpleMerge until numBlocks <= some Critical level
	while(numPartitions > 32 || (partitionSize*mult < 16384 && numPartitions > 1))
	{	
		//printf("Running simple merge for %d partitions of size %d\n", numPartitions, partitionSize*mult);
		numBlocks = (numPartitions&0xFFFE);	    
		if(count%2 == 0)
		{ 				
			simpleStringMerge<unsigned int, 2>
				<<<numBlocks, CTASIZE_simple, sizeof(unsigned int)*(2*INTERSECT_B_BLOCK_SIZE_simple+4)>>>(pkeys, temp_keys, 				
				pvals, temp_vals, stringVals, partitionSize*mult, numElements, count, stringArrayLength);		

			if(numPartitions%2 == 1)
			{			

				int offset = (partitionSize*mult*(numPartitions-1));
				int numElementsToCopy = numElements-offset;												
				simpleCopy<unsigned int>
					<<<(numElementsToCopy+numThreads-1)/numThreads, numThreads>>>(pkeys, pvals, temp_keys, temp_vals, offset, numElementsToCopy);
			}
		}
		else
		{			
			simpleStringMerge<unsigned int, 2>
				<<<numBlocks, CTASIZE_simple, sizeof(unsigned int)*(2*INTERSECT_B_BLOCK_SIZE_simple+4)>>>(temp_keys, pkeys, 				
				temp_vals, pvals, stringVals, partitionSize*mult, numElements, count, stringArrayLength);		
			
			if(numPartitions%2 == 1)
			{			
				int offset = (partitionSize*mult*(numPartitions-1));
				int numElementsToCopy = numElements-offset;						
				simpleCopy<unsigned int>
					<<<(numElementsToCopy+numThreads-1)/numThreads, numThreads>>>(temp_keys, temp_vals, pkeys, pvals, offset, numElementsToCopy);
			}
		}

		mult*=2;
		count++;
		numPartitions = (numPartitions+1)/2;
		
	}				


	
	
	//End of simpleMerge, now blocks cooperate to merge partitions
	while (numPartitions > 1)
	{		
		//printf("Running multi merge for %d partitions of size %d\n", numPartitions, partitionSize*mult);
		numBlocks = (numPartitions&0xFFFE);	 
		int secondBlocks = ((numBlocks)*subPartitions+numThreads-1)/numThreads;			
		if(count%2 == 1)
		{								
			findMultiPartitions<unsigned int>
				<<<secondBlocks, numThreads>>>(temp_keys, temp_vals, stringVals, subPartitions, numBlocks, partitionSize*mult, partitionBeginA, partitionSizeA, 
				partitionBeginB, partitionSizeB, numElements, stringArrayLength);			
			

			//int lastSubPart = getLastSubPart(numBlocks, subPartitions, partitionSize, mult, numElements);
			CUDA_SAFE_CALL(cudaThreadSynchronize());
			stringMergeMulti<unsigned int, DEPTH_multi>
				<<<numBlocks*subPartitions, CTASIZE_multi, (2*INTERSECT_B_BLOCK_SIZE_multi+4)*sizeof(unsigned int)>>>(temp_keys, pkeys, temp_vals, 
				pvals, stringVals, subPartitions, numBlocks, partitionBeginA, partitionSizeA, partitionBeginB, partitionSizeB, mult*partitionSize, count, numElements, stringArrayLength);
			CUDA_SAFE_CALL(cudaThreadSynchronize());
			if(numPartitions%2 == 1)
			{			
				int offset = (partitionSize*mult*(numPartitions-1));
				int numElementsToCopy = numElements-offset;				
				simpleCopy<unsigned int>
					<<<(numElementsToCopy+numThreads-1)/numThreads, numThreads>>>(temp_keys, temp_vals, pkeys, pvals, offset, numElementsToCopy);
			}

		}
		else
		{

			findMultiPartitions<unsigned int>
				<<<secondBlocks, numThreads>>>(pkeys, pvals, stringVals, subPartitions, numBlocks, partitionSize*mult, partitionBeginA, partitionSizeA, 
				partitionBeginB, partitionSizeB, numElements, stringArrayLength);											
			CUDA_SAFE_CALL(cudaThreadSynchronize());
			//int lastSubPart = getLastSubPart(numBlocks, subPartitions, partitionSize, mult, numElements);
			stringMergeMulti<unsigned int, DEPTH_multi>
				<<<numBlocks*subPartitions, CTASIZE_multi, (2*INTERSECT_B_BLOCK_SIZE_multi+4)*sizeof(unsigned int)>>>(pkeys, temp_keys, pvals, 
				temp_vals, stringVals, subPartitions, numBlocks, partitionBeginA, partitionSizeA, partitionBeginB, partitionSizeB, mult*partitionSize, count, numElements, stringArrayLength);

			CUDA_SAFE_CALL(cudaThreadSynchronize());
			if(numPartitions%2 == 1)
			{			
				int offset = (partitionSize*mult*(numPartitions-1));
				int numElementsToCopy = numElements-offset;				
				simpleCopy<unsigned int>
					<<<(numElementsToCopy+numThreads-1)/numThreads, numThreads>>>(pkeys, pvals, temp_keys, temp_vals, offset, numElementsToCopy);
			}

		}


		count++;
		mult*=2;		
		subPartitions*=2;
		numPartitions = (numPartitions+1)/2;				
	}	

	if(count%2==1)
	{
		CUDA_SAFE_CALL(cudaMemcpy(pkeys, temp_keys, numElements*sizeof(unsigned int), cudaMemcpyDeviceToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(pvals, temp_vals, numElements*sizeof(unsigned int), cudaMemcpyDeviceToDevice));
	}

	CUDA_SAFE_CALL(cudaFree(partitionBeginA));
	CUDA_SAFE_CALL(cudaFree(partitionBeginB));
	CUDA_SAFE_CALL(cudaFree(partitionSizeA));
	CUDA_SAFE_CALL(cudaFree(partitionSizeB));

	CUDA_SAFE_CALL(cudaFree(temp_keys));
	CUDA_SAFE_CALL(cudaFree(temp_vals));	

	//printf("end\n");
}

#ifdef __cplusplus
extern "C" 
{
#endif


	/**
	* @brief From the programmer-specified sort configuration, 
	*        creates internal memory for performing the sort.
	* 
	* @param[in] plan Pointer to CUDPPStringSortPlan object
	**/
	void allocStringSortStorage(CUDPPStringSortPlan *plan)
	{               
	}

	/** @brief Deallocates intermediate memory from allocStringSortStorage.
	*
	*
	* @param[in] plan Pointer to CUDPStringSortPlan object
	**/

	void freeStringSortStorage(CUDPPStringSortPlan* plan)
	{
	}

	/** @brief Dispatch function to perform a sort on an array with 
	* a specified configuration.
	*
	* This is the dispatch routine which calls stringSort...() with 
	* appropriate template parameters and arguments as specified by 
	* the plan.
	* @param[in,out] keys Keys (first four chars of string) to be sorted.
	* @param[in,out] values Address of string values in array of null terminated strings
	* @param[in] stringVals Global string array
	* @param[in] numElements Number of elements in the sort.
	* @param[in] stringArrayLength The size of our string array in uints (4 chars per uint)
	* @param[in] plan Configuration information for mergeSort.
	**/

	void cudppStringSortDispatch(void  *keys,
		                         void  *values,
		                         void  *stringVals,
		                         size_t numElements,
								 size_t stringArrayLength,
		                         const CUDPPStringSortPlan *plan)
	{
		runStringSort((unsigned int*)keys, (unsigned int*)values, (unsigned int*) stringVals, numElements, stringArrayLength, plan);
	}                            

#ifdef __cplusplus
}
#endif






/** @} */ // end stringsort functions
/** @} */ // end cudpp_app
