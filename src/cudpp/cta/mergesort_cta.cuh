// -------------------------------------------------------------
// CUDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision$
// $Date$
// ------------------------------------------------------------- 
// This source code is distributed under the terms of license.txt 
// in the root directory of this source distribution.
// ------------------------------------------------------------- 
#include <cudpp_globals.h>
#include "cudpp_mergesort.h"
#include <cudpp.h>
#include <stdio.h>

#include <cudpp_util.h>
#include <math.h>
#include "sharedmem.h"

/**
 * @file
 * sort_cta.cu
 * 
 * @brief CUDPP CTA-level sort routines
 */

/** \addtogroup cudpp_cta 
* @{
*/

/** @name Merge Sort Functions
* @{
*/

#define BLOCKSORT_SIZE 1024
#define CTA_BLOCK 128
#define DEPTH_simple 2
#define DEPTH_multi 4
#define CTASIZE_simple 256
#define CTASIZE_multi 128

#define INTERSECT_A_BLOCK_SIZE_simple DEPTH_simple*CTASIZE_simple
#define INTERSECT_B_BLOCK_SIZE_simple 2*DEPTH_simple*CTASIZE_simple

#define INTERSECT_A_BLOCK_SIZE_multi DEPTH_multi*CTASIZE_multi
#define INTERSECT_B_BLOCK_SIZE_multi 2*DEPTH_multi*CTASIZE_multi
typedef unsigned int uint;

//does a portion of binary search
template<class T, int depth>
__device__ void bin_search_block(T &cmpValue, T tmpVal, T* in, unsigned int & j, unsigned int bump, unsigned int addPart)
{

	cmpValue = in[j]; 

    j = ((cmpValue < tmpVal || cmpValue == tmpVal && addPart == 1) ? j + bump : j - bump);  
    __syncthreads();

}
//linear search to find next index
//addPart is needed to determine if you are the "right" block or "left" block

//lin_search_block<T, depth>(cmpValue, myKey[1], myAddress[1], scratchPad, addressPad, j, 1, last, startAddress, addPart);				
template<class T, int depth>
__device__ void lin_search_block(T &cmpValue, T mVal, unsigned int &tmpVal, T* in, unsigned int* addressPad, unsigned int &j, 
								 unsigned int offset, unsigned int last, unsigned int startAddress, unsigned int addPart)
{			
	
	while (cmpValue < mVal && j < last)		
		cmpValue = in[++j];			
	while (cmpValue == mVal && j < last && addPart == 1)
		cmpValue = in[++j];	
	
	//Corner case to handle being at the edge of our shared memory search
    j = (j==last && (cmpValue < mVal || (cmpValue == mVal && addPart == 1)) ? j+1 : j);	
	
    tmpVal = j+startAddress+offset;
}

//Key-Value compare and swap
template<class T>
__device__ void compareSwapVal(T &A1, T &A2, unsigned int& ref1, unsigned int& ref2)
{
    if(A1 > A2)
    {
        T tmp = A1;
        A1 = A2;
        A2 = tmp;

        tmp = ref1;
        ref1 = ref2;
        ref2 = tmp;
    }   
}

template<class T>
__device__ 
inline void  binSearch_fragment_lower(T* binArray, int offset, int &mid, T testValue)
{	 mid = (binArray[mid] >= testValue ? mid-offset : mid+offset);  }
//Binary Search fragment for later block
template<class T>
__device__ 
inline void  binSearch_fragment_higher(T* binArray, int offset, int &mid, T testValue)
{	 mid = (binArray[mid] > testValue ? mid-offset : mid+offset); }


template<class T>
__device__
inline void binSearch_whole_lower(T* BKeys, int &index, T myKey)
{	
	index = (INTERSECT_B_BLOCK_SIZE_simple/2)-1;
	binSearch_fragment_lower<T> (BKeys, 256, index, myKey);
	binSearch_fragment_lower<T> (BKeys, 128, index, myKey);
	binSearch_fragment_lower<T> (BKeys, 64,  index, myKey);
	binSearch_fragment_lower<T> (BKeys, 32,  index, myKey);
	binSearch_fragment_lower<T> (BKeys, 16,  index, myKey);
	binSearch_fragment_lower<T> (BKeys, 8,   index, myKey);
	binSearch_fragment_lower<T> (BKeys, 4,   index, myKey);
	binSearch_fragment_lower<T> (BKeys, 2,   index, myKey);
	binSearch_fragment_lower<T> (BKeys, 1,   index, myKey);								
}

template<class T>
__device__
inline void binSearch_whole_higher(T* BKeys, int &index, T myKey)
{
	index = (INTERSECT_B_BLOCK_SIZE_simple/2)-1;		
	binSearch_fragment_higher<T> (BKeys, 256, index, myKey);		
	binSearch_fragment_higher<T> (BKeys, 128, index, myKey);		
	binSearch_fragment_higher<T> (BKeys, 64,  index, myKey);
	binSearch_fragment_higher<T> (BKeys, 32,  index, myKey);
	binSearch_fragment_higher<T> (BKeys, 16,  index, myKey);
	binSearch_fragment_higher<T> (BKeys, 8,   index, myKey);
	binSearch_fragment_higher<T> (BKeys, 4,   index, myKey);
	binSearch_fragment_higher<T> (BKeys, 2,   index, myKey);
	binSearch_fragment_higher<T> (BKeys, 1,   index, myKey);							
}

#define DVAL1  1043863958
#define DVAL2  1043863958

template<class T, int depth>
__device__
inline void linearMerge_lower(T* searchArray, T myKey, unsigned int myVal, int &index, T* saveGlobalArray, unsigned int* saveValueArray, int myStartIdxC, 
					    T nextMaxB, int localAPartSize, int localBPartSize, T localMaxB, T localMinB, int aIndex, int bIndex, int offset)
{		
	
	
	while(searchArray[index] < myKey && index < INTERSECT_B_BLOCK_SIZE_multi )
		index++;
	
	int globalCAddress = myStartIdxC + index + bIndex + aIndex + offset + threadIdx.x*depth; 
	


	//Save Key-Val Pair
	if(((myKey <=  nextMaxB || myKey <= localMaxB) && myKey > localMinB)  && offset+threadIdx.x*depth+aIndex < localAPartSize)			
	{ 
		saveGlobalArray[globalCAddress] =  myKey;   saveValueArray[globalCAddress] = myVal;			
	}
		
				
}

template<class T, int depth>
__device__
inline void linearMerge_higher(T* searchArray, T myKey, unsigned int myVal, int &index, T* saveGlobalArray, unsigned int* saveValueArray, int myStartIdxC, 
															T localMinB, T nextMaxB, int aIndex, int bIndex, int offset, int localAPartSize, int localBPartSize)
{		
	
	while(searchArray[index] <= myKey && index < INTERSECT_B_BLOCK_SIZE_multi && index < (localBPartSize-bIndex) )
		index++;

	int globalCAddress = myStartIdxC + index + bIndex + aIndex + offset + threadIdx.x*depth;

	//Save Key-Val Pair
	if((myKey <= nextMaxB && myKey >= localMinB /*|| bIndex + index >= localBPartSize*/)&& offset+threadIdx.x*depth+aIndex < localAPartSize)	
	{ saveGlobalArray[globalCAddress] =  myKey;	saveValueArray [globalCAddress] =  myVal;	}
			
}



/** @} */ // end merte  sort functions
/** @} */ // end cudpp_cta
