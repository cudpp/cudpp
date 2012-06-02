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
#include "sharedmem.h"
#include <stdio.h>
#include "cta/compress_cta.cuh"

/**
* @file
* compact_kernel.cu
* 
* @brief CUDPP kernel-level compact routines
*/

/** \addtogroup cudpp_kernel
* @{
*/

/** @name Compress Functions
* @{
*/

typedef unsigned int uint;
typedef unsigned char uchar;

/** @brief Compute final BWT
* @todo
**/
__global__ void
bwt_compute_final_kernel(uchar *d_bwtIn,
                         uint *d_values,
                         int *d_bwtIndex,
                         uchar *d_bwtOut,
                         uint numElements,
                         uint tThreads)
{
    // Global, local IDs
    uint idx = threadIdx.x + (blockIdx.x * blockDim.x);

    for(int i = idx; i < numElements; i += tThreads)
    {
        uint val = d_values[i];

        if(val == 0) *d_bwtIndex = i;
        d_bwtOut[i] = (val == 0) ? d_bwtIn[numElements-1] : d_bwtIn[val-1];
    }

}

/** @brief Multi merge
* @todo
**/
template<class T, int depth>
__global__ void
stringMergeMulti(T      *A_keys,
                 T      *A_keys_out,
                 T      *A_values,
                 T      *A_values_out,
                 T      *stringValues,
                 int    subPartitions,
                 int    numBlocks, 
                 int    *partitionBeginA,
                 int    *partitionSizeA,
                 int    *partitionBeginB,
                 int    *partitionSizeB,
                 int    entirePartitionSize,
                 int    step,
                 size_t numElements)
{
    int myId = blockIdx.x;
    int myStartId = (myId%subPartitions) + (myId/(2*subPartitions))*2*subPartitions;
    int myStartIdxA, myStartIdxB, localAPartSize, localBPartSize, localCPartSize;

    T finalMaxB;
    myStartIdxA = partitionBeginA[myId];
    myStartIdxB = partitionBeginB[myId];
    localAPartSize = partitionSizeA[myId];
    localBPartSize = partitionSizeB[myId];

    int myStartIdxC;			
    myStartIdxC = myStartIdxA + myStartIdxB - ((myStartId+subPartitions)/(subPartitions))*entirePartitionSize;	
    localCPartSize = localAPartSize + localBPartSize;	

    if(myId%subPartitions != subPartitions-1 && myStartIdxB + localBPartSize < (myId/subPartitions)*entirePartitionSize+2*entirePartitionSize)
        finalMaxB = A_keys[myStartIdxB+localBPartSize+1];
    else
        finalMaxB = UINT_MAX-1;

    //Now we have the beginning and end points of our subpartitions, merge the two together
    T cmpValue;
    int mid, index;
    int bIndex = 0; int aIndex = 0;	

    __shared__ T      BValues[2*BWT_INTERSECT_B_BLOCK_SIZE_multi+3];
    T * BKeys =      &BValues[BWT_INTERSECT_B_BLOCK_SIZE_multi];
    T * BMax =       &BValues[2*BWT_INTERSECT_B_BLOCK_SIZE_multi];
    T * lastAIndex = &BValues[2*BWT_INTERSECT_B_BLOCK_SIZE_multi+3];

    bool breakout = false;
    int tid = threadIdx.x;

    T localMaxB, localMaxA;			
    T localMinB = 0;	

    T myKey[depth];
    T myValue[depth];

#pragma unroll
    for(int i =0; i <depth; i++)
    {
        myKey[i] =   (depth*tid + i < localAPartSize ? A_keys  [myStartIdxA + depth*tid + i]   : UINT_MAX-3);		
        myValue[i] = (depth*tid + i < localAPartSize ? A_values[myStartIdxA + depth*tid + i]   : UINT_MAX-3);		
    }

    if(bIndex + BWT_INTERSECT_B_BLOCK_SIZE_multi < localBPartSize) 
    {
        int bi = tid;					
#pragma unroll
        for(int i = 0;i < BWT_INTERSECT_B_BLOCK_SIZE_multi/BWT_CTASIZE_multi; i++, bi+=BWT_CTASIZE_multi) 
        {
            BKeys[bi] =   A_keys  [myStartIdxB + bi];
            BValues[bi] = A_values[myStartIdxB + bi];
        }
    }
    else {
        int bi = tid;
#pragma unroll
        for(int i = 0;i < BWT_INTERSECT_B_BLOCK_SIZE_multi/BWT_CTASIZE_multi; i++, bi+=BWT_CTASIZE_multi)
        {
            BKeys[bi] =   ((bIndex + bi < localBPartSize) ? A_keys  [myStartIdxB + bi]   : UINT_MAX-1);
            BValues[bi] = ((bIndex + bi < localBPartSize) ? A_values[myStartIdxB + bi]   : UINT_MAX-1);
        }
    }

    if(tid == BWT_CTASIZE_multi-1)
    {
        BMax[1] =  myKey[depth-1];
        BMax[0] =  (BWT_INTERSECT_B_BLOCK_SIZE_multi < localBPartSize ? A_keys  [myStartIdxB + BWT_INTERSECT_B_BLOCK_SIZE_multi-1] : UINT_MAX-1);
    }	

    __syncthreads();

    localMaxB = BMax[0];
    localMaxA = BMax[1];

    do 
    {
        index = 0;

        if(1)
        {
            index = -1;
            int cumulativeAddress = myStartIdxA+aIndex+threadIdx.x*depth;
            mid = (BWT_INTERSECT_B_BLOCK_SIZE_multi/2)-1;

            if(BWT_INTERSECT_B_BLOCK_SIZE_multi >= 1024)
                binSearch_frag_mult<T, depth> (BKeys, BValues, 256, mid, cmpValue, myKey[0], myValue[0], cumulativeAddress, A_values, stringValues, myStartIdxB+bIndex+index, numElements);

            if(BWT_INTERSECT_B_BLOCK_SIZE_multi>= 512)
                binSearch_frag_mult<T, depth> (BKeys, BValues, 128, mid, cmpValue, myKey[0], myValue[0], cumulativeAddress, A_values, stringValues, myStartIdxB+bIndex+index, numElements);

            if(BWT_INTERSECT_B_BLOCK_SIZE_multi >= 256)
                binSearch_frag_mult<T, depth> (BKeys, BValues, 64, mid, cmpValue, myKey[0], myValue[0], cumulativeAddress, A_values, stringValues, myStartIdxB+bIndex+index, numElements);

            binSearch_frag_mult<T, depth> (BKeys, BValues, 32, mid, cmpValue, myKey[0], myValue[0], cumulativeAddress, A_values, stringValues, myStartIdxB+bIndex+index, numElements);						
            binSearch_frag_mult<T, depth> (BKeys, BValues, 16, mid, cmpValue, myKey[0], myValue[0], cumulativeAddress, A_values, stringValues, myStartIdxB+bIndex+index, numElements);			
            binSearch_frag_mult<T, depth> (BKeys, BValues,  8, mid, cmpValue, myKey[0], myValue[0], cumulativeAddress, A_values, stringValues, myStartIdxB+bIndex+index, numElements);						
            binSearch_frag_mult<T, depth> (BKeys, BValues,  4, mid, cmpValue, myKey[0], myValue[0], cumulativeAddress, A_values, stringValues, myStartIdxB+bIndex+index, numElements);			
            binSearch_frag_mult<T, depth> (BKeys, BValues,  2, mid, cmpValue, myKey[0], myValue[0], cumulativeAddress, A_values, stringValues, myStartIdxB+bIndex+index, numElements);			
            binSearch_frag_mult<T, depth> (BKeys, BValues,  1, mid, cmpValue, myKey[0], myValue[0], cumulativeAddress, A_values, stringValues, myStartIdxB+bIndex+index, numElements);

            index = mid;			
            cmpValue = BKeys[index];
            if(cmpValue < myKey[0] && index < (localBPartSize-bIndex) && index < BWT_INTERSECT_B_BLOCK_SIZE_multi)				
                cmpValue = BKeys[++index];

            if(cmpValue == myKey[0] && index < BWT_INTERSECT_B_BLOCK_SIZE_multi && index < (localBPartSize-bIndex))
            {
                int count = 1;
                T tmpKey, cmpKey;
                tmpKey = myKey[0];
                cmpKey = cmpValue;

                while(tmpKey == cmpKey )
                {
                    tmpKey = (myValue[0]+4*count > numElements-1) ? stringValues[myValue[0] + 4*count - numElements] : stringValues[myValue[0] + 4*count];
                    cmpKey = (BValues[index]+4*count > numElements-1) ? stringValues[BValues[index] + 4*count - numElements] : stringValues[BValues[index] + 4*count];

                    if(cmpKey < tmpKey)
                    {	cmpValue = BKeys[++index];	break; }	

                    count++;				
                }				
            }


            if(cmpValue < myKey[0])
                index++;

            if(cmpValue == myKey[0]) 
            {
                int count = 1;
                T tmpKey, cmpKey;
                tmpKey = myKey[0];
                cmpKey = cmpValue;

                while(tmpKey == cmpKey)
                {
                    tmpKey = (myValue[0]+4*count > numElements-1) ? stringValues[myValue[0] + 4*count - numElements] : stringValues[myValue[0] + 4*count];
                    cmpKey = (A_values[myStartIdxB+bIndex+index]+4*count > numElements-1) ?
                        stringValues[A_values[myStartIdxB+bIndex+index] + 4*count - numElements] : stringValues[A_values[myStartIdxB+bIndex+index] + 4*count];

                    if(cmpKey < tmpKey)
                    {index++;	break; }		
                    count++;		
                }							
            }

            int globalCAddress = (myStartIdxC + index + bIndex + aIndex + tid*depth);

            if(((myKey[0] < localMaxB && myKey[0] > localMinB) || (bIndex+index) >= (localBPartSize) || 
                (index > 0 && index <BWT_INTERSECT_B_BLOCK_SIZE_multi)) && globalCAddress < (myStartIdxC+localCPartSize) && myKey[0] < finalMaxB)
            {
                A_keys_out  [globalCAddress] = myKey[0];											
                A_values_out[globalCAddress] = myValue[0];
            }
            else if((myKey[0] == localMaxB && myKey[0] <= finalMaxB && index > 0 && index <=1024) && globalCAddress < (myStartIdxC+localCPartSize))
            {
                //tie break
                unsigned int tmpAdd = A_values[myStartIdxA+aIndex+depth*tid];
                unsigned int cmpAdd = A_values[myStartIdxB+bIndex+index];
                int count = 1;
                unsigned int tmpKey = (tmpAdd+count > numElements-1) ? stringValues[tmpAdd + count - numElements] : stringValues[tmpAdd + count];
                unsigned int cmpKey = (cmpAdd+count > numElements-1) ? stringValues[cmpAdd + count - numElements] : stringValues[cmpAdd + count];

                while(tmpKey == cmpKey )
                {
                    count++;

                    tmpKey = (tmpAdd+count > numElements-1) ? stringValues[tmpAdd + count - numElements] : stringValues[tmpAdd + count];
                    cmpKey = (cmpAdd+count > numElements-1) ? stringValues[cmpAdd + count - numElements] : stringValues[cmpAdd + count];
                }
                if(tmpKey < cmpKey)
                {
                    A_keys_out  [myStartIdxC + bIndex + aIndex+depth*tid+index] = myKey[0];	
                    A_values_out[myStartIdxC + bIndex + aIndex+depth*tid+index] = myValue[0];	
                }
            }
            else if(myKey[0] == localMinB && globalCAddress < (myStartIdxC+localCPartSize))
            {
                unsigned int tmpAdd = A_values[myStartIdxA+aIndex+depth*tid];
                unsigned int cmpAdd = A_values[myStartIdxB+bIndex+index];

                int count = 1;

                unsigned int tmpKey = (tmpAdd+4*count > numElements-1) ? stringValues[tmpAdd + 4*count - numElements] : stringValues[tmpAdd + 4*count];
                unsigned int cmpKey = (cmpAdd+4*count > numElements-1) ? stringValues[cmpAdd + 4*count - numElements] : stringValues[cmpAdd + 4*count];

                while(tmpKey == cmpKey)
                {
                    count++;

                    tmpKey = (tmpAdd+4*count > numElements-1) ? stringValues[tmpAdd + 4*count - numElements] : stringValues[tmpAdd + 4*count];
                    cmpKey = (cmpAdd+4*count > numElements-1) ? stringValues[cmpAdd + 4*count - numElements] : stringValues[cmpAdd + 4*count];
                }	
                if(tmpKey > cmpKey)
                {
                    A_keys_out  [myStartIdxC + bIndex + aIndex+depth*tid+index] = myKey[0];	
                    A_values_out[myStartIdxC + bIndex + aIndex+depth*tid+index] = myValue[0];	
                }
            }

            if(myKey[1] <= localMaxB)
                linearStringMerge<T, depth>(BKeys, BValues, A_values, myKey[1], myValue[1], index, cmpValue, A_keys_out, A_values_out, stringValues, 
                myStartIdxC, myStartIdxA, myStartIdxB, localAPartSize, localBPartSize, localCPartSize, localMaxB, finalMaxB, localMinB, tid, aIndex, bIndex, 
                1, subPartitions, numElements);
        }

        if(threadIdx.x == blockDim.x - 1) { *lastAIndex = index; }

        bool reset = false;
        __syncthreads();
        if(localMaxA == localMaxB)
        {	
            //Break the tie
            if(tid == (blockDim.x-1))
            {
                unsigned int tmpAdd = myValue[depth-1]; 
                unsigned int cmpAdd = BValues[BWT_INTERSECT_B_BLOCK_SIZE_multi-1];

                int count = 1;

                unsigned int tmpKey = (tmpAdd+4*count > numElements-1) ? stringValues[tmpAdd + 4*count - numElements] : stringValues[tmpAdd + 4*count];
                unsigned int cmpKey = (cmpAdd+4*count > numElements-1) ? stringValues[cmpAdd + 4*count - numElements] : stringValues[cmpAdd + 4*count];

                while(tmpKey == cmpKey)
                {
                    count++;
                    tmpKey = (tmpAdd+4*count > numElements-1) ? stringValues[tmpAdd + 4*count - numElements] : stringValues[tmpAdd + 4*count];
                    cmpKey = (cmpAdd+4*count > numElements-1) ? stringValues[cmpAdd + 4*count - numElements] : stringValues[cmpAdd + 4*count];
                }

                if(tmpKey > cmpKey)
                    BMax[1]++;
                else
                    BMax[0]++;				

            }
            __syncthreads();
            localMaxB = BMax[0];
            localMaxA = BMax[1];
            reset = true;		
        }

        __syncthreads();		
        __threadfence();
        if((localMaxA < localMaxB || (bIndex+BWT_INTERSECT_B_BLOCK_SIZE_multi-1) >= localBPartSize) && (aIndex+BWT_INTERSECT_A_BLOCK_SIZE_multi)< localAPartSize)
        {

            aIndex += BWT_INTERSECT_A_BLOCK_SIZE_multi;

            if(aIndex + BWT_INTERSECT_A_BLOCK_SIZE_multi < localAPartSize) 
            {		
#pragma unroll
                for(int i = 0;i < depth; i++) 
                { myKey[i] = A_keys[myStartIdxA + aIndex + depth*tid + i]; myValue[i] = A_values[myStartIdxA + aIndex + depth*tid + i]; }
            }
            else 
            {

#pragma unroll
                for(int i = 0;i < depth; i++) 
                { myKey[i] =   (aIndex+depth*tid + i < localAPartSize ? A_keys[myStartIdxA + aIndex+ depth*tid + i]   : UINT_MAX-3); 
                myValue[i] = (aIndex+depth*tid + i < localAPartSize ? A_values[myStartIdxA + aIndex+ depth*tid + i]   : UINT_MAX-3);}
            }

            if(tid == BWT_CTASIZE_multi-1)		
            {
                BMax[1] = myKey[depth-1];		
                if(reset)
                    BMax[0]--;			
            }
            reset = false;
        }			
        else if(localMaxB < localMaxA && (bIndex+BWT_INTERSECT_B_BLOCK_SIZE_multi-1) < localBPartSize)
        {				
            bIndex += BWT_INTERSECT_B_BLOCK_SIZE_multi-1;	
            if(bIndex + BWT_INTERSECT_B_BLOCK_SIZE_multi < localBPartSize) 
            {
                int bi = tid;					
#pragma unroll
                for(int i = 0;i < BWT_INTERSECT_B_BLOCK_SIZE_multi/BWT_CTASIZE_multi; i++, bi+=BWT_CTASIZE_multi) 
                {
                    BKeys[bi] =   A_keys[myStartIdxB + bIndex + bi];
                    BValues[bi] = A_values[myStartIdxB + bIndex + bi];
                }
            }
            else {
                int bi = tid;
#pragma unroll
                for(int i = 0;i < BWT_INTERSECT_B_BLOCK_SIZE_multi/BWT_CTASIZE_multi; i++, bi+=BWT_CTASIZE_multi) 
                {
                    BKeys[bi] =   ((bIndex + bi < localBPartSize) ? A_keys  [myStartIdxB + bIndex + bi]   : UINT_MAX-1);
                    BValues[bi] = ((bIndex + bi < localBPartSize)? A_values[myStartIdxB + bIndex + bi]   : UINT_MAX-1);
                }
            }

            if(tid ==BWT_CTASIZE_multi-1)
            {
                BMax[0] =  (bIndex + BWT_INTERSECT_B_BLOCK_SIZE_multi < localBPartSize ? A_keys[myStartIdxB + bIndex + BWT_INTERSECT_B_BLOCK_SIZE_multi-1] : UINT_MAX-1);
                if(reset)
                    BMax[1]--;
            }
            reset = false;
            __syncthreads();
            localMinB = BKeys[0];
        }
        else
            breakout = true;	
        __syncthreads();
        __threadfence();

        localMaxB = BMax[0];
        localMaxA = BMax[1];
    }
    while(!breakout);
}

/** @brief Multi merge -- find partitions
* @todo
**/
template<class T>
__global__ void
findMultiPartitions(T       *A,
                    int     splitsPP,
                    int     numPartitions,
                    int     partitionSize,
                    int     *partitionBeginA,
                    int     *partitionSizesA,
                    int     *partitionBeginB,
                    int     *partitionSizesB,
                    int     sizeA)
{
    int myId = threadIdx.x + blockIdx.x*blockDim.x;
    if (myId >= (numPartitions*splitsPP))
        return;

    int myStartA, myEndA;
    T testSample, myStartSample, myEndSample;
    int testIdx;
    int subPartitionSize = partitionSize/splitsPP;
    int myPartitionId = myId/splitsPP;
    int mySubPartitionId = myId%splitsPP;

    myStartA = (myPartitionId)*partitionSize + (mySubPartitionId)*subPartitionSize; // we are at the beginning of a partition
    T mySample = A[myStartA];

    if(mySubPartitionId != 0)
    {
        //need to ensure that we don't start inbetween duplicates
        // we have sampled in the middle of a repeated sequence search until we are at a new sequence
        if(threadIdx.x%2 == 1)
        {
            testSample = (myId == 0 ? 0 : A[myStartA-1]);
            int count = 1; testIdx = myStartA;
            if(testSample == mySample)
            {
                while(testSample == mySample && (testIdx+count) < (myPartitionId)*partitionSize+partitionSize) 	
                    testSample = A[testIdx + (count++)];
                myStartA = (testIdx + count-1);
            }
        }
        else
        {
            testSample = (myId == 0 ? 0 : A[myStartA-1]);
            int count = 1; testIdx = myStartA;

            if(testSample == mySample)
            {
                while(testSample == mySample && (testIdx+count) < (myPartitionId)*partitionSize+partitionSize) 	
                    testSample = A[testIdx + (count++)];
                myStartA = (testIdx + count-1);
            }			
        }	    		
    }


    partitionBeginA[myId] = myStartA; //partitionBegin found for first set
    myStartSample = mySample;
    myEndA = ((myId+1)/splitsPP)*partitionSize+((myId+1)%splitsPP)*subPartitionSize;

    if(mySubPartitionId!= splitsPP-1 )
    {
        mySample = A[myEndA];	
        //need to ensure that we don't start inbetween duplicates

        if(threadIdx.x%2 == 0)
        {
            testSample = A[myEndA-1];			
            int count = 1; testIdx = myEndA;

            if(testSample == mySample)
            {
                while(testSample == mySample && (testIdx+count) < (myPartitionId)*partitionSize+partitionSize) 	
                    testSample = A[testIdx + (count++)];
                myEndA = (testIdx + count-1);
            }
        }
        else
        {
            testSample = A[myEndA-1];			
            int count = 1; testIdx = myEndA;

            if(testSample == mySample)
            {
                while(testSample == mySample && (testIdx+count) < (myPartitionId)*partitionSize+partitionSize) 	
                    testSample = A[testIdx + (count++)];
                myEndA = (testIdx + count-1);
            }
        }
        myEndSample = A[(myEndA < (myPartitionId+1)*partitionSize && myEndA < sizeA) ? myEndA : myEndA];


    }
    else
    {
        myEndA = (myPartitionId)*partitionSize + partitionSize;			
        myEndSample = A[myEndA-1];

    }

    partitionSizesA[myId] = myEndA-myStartA ;

    int myStartRange = (myPartitionId)*partitionSize + partitionSize - 2*(myPartitionId%2)*partitionSize;
    int myEndRange = myStartRange + partitionSize;
    int first = myStartRange;
    int last = myEndRange;
    int mid = (first + last)/2;
    testSample = A[mid];

    while(testSample != myStartSample)
    {	
        if(testSample < myStartSample)		
            first = mid;					
        else		
            last = mid;

        mid = (first+last)/2;		
        testSample = A[mid];
        if(mid == last || mid == first )
            break;	
    }

    while (testSample >= myStartSample && mid > myStartRange)	
        testSample = A[--mid];

    myStartA = mid;	
    first = myStartA;
    last = myEndRange;
    mid = (first + last)/2;	
    testSample = A[mid];

    while(testSample != myEndSample)
    {
        if(testSample <= myEndSample)		
            first = mid;					
        else		
            last = mid;

        mid = (first+last)/2;					
        testSample = A[mid];
        if(mid == last || mid == first )
            break;
    }

    while (myEndSample >= testSample && mid < myEndRange)
        testSample = A[++mid];

    myEndA = mid;

    if(mySubPartitionId  == splitsPP-1)
        myEndA = myStartRange + partitionSize;

    partitionBeginB[myId] = myStartA;
    partitionSizesB[myId] = myEndA-myStartA;
}

/** @brief Simple merge
* @todo
**/
template<class T, int depth>
__global__ void
simpleStringMerge(T         *A_keys,
                  T         *A_keys_out,
                  T         *A_values,
                  T         *A_values_out,
                  T         *stringValues,
                  int       sizePerPartition,
                  int       size,
                  T         *stringValues2,
                  size_t    numElements)
{
    //each block will be responsible for a submerge
    int myStartIdxA, myStartIdxB, myStartIdxC;
    int myId = blockIdx.x;

    int totalSize;

    //Slight difference in loading if we are an odd or even block
    if(myId%2 == 0)
    {
        myStartIdxA = (myId/2)*2*sizePerPartition; myStartIdxB = myStartIdxA+sizePerPartition; myStartIdxC = myStartIdxA;
        totalSize = myStartIdxB + sizePerPartition;
    }
    else
    {
        myStartIdxB = (myId/2)*2*sizePerPartition; myStartIdxA = myStartIdxB + sizePerPartition; myStartIdxC = myStartIdxB;
        totalSize = myStartIdxA + sizePerPartition;
    }

    T cmpValue;
    int mid, index;     int bIndex = 0; int aIndex = 0;

    //Shared Memory pool
    __shared__ T BValues[BWT_INTERSECT_B_BLOCK_SIZE_simple*2+2];
    T* BKeys = (T*) &BValues[BWT_INTERSECT_B_BLOCK_SIZE_simple];
    T* BMax = (T*) &BValues[2*BWT_INTERSECT_B_BLOCK_SIZE_simple];


    bool breakout = false;
    int tid = threadIdx.x;

    T localMaxB, localMaxA;
    T myKey[depth];
    T myValue[depth];

    //Load Registers
    if(aIndex + BWT_INTERSECT_A_BLOCK_SIZE_simple < sizePerPartition)
    {
#pragma unroll
        for(int i = 0;i < depth; i++)
        {
            myKey[i]   = A_keys  [myStartIdxA + aIndex+ depth*tid + i];
            myValue[i] = A_values[myStartIdxA + aIndex+ depth*tid + i];
        }

    }

    else
    {
#pragma unroll
        for(int i = 0;i < depth; i++)
        {
            myKey[i] =   (aIndex+depth*tid + i < sizePerPartition ? A_keys  [myStartIdxA + aIndex+ depth*tid + i]   : UINT_MAX-1); // ==ADDED==
            myValue[i] = (aIndex+depth*tid + i < sizePerPartition ? A_values[myStartIdxA + aIndex+ depth*tid + i]   : UINT_MAX-1); // ==ADDED==
        }
    }

    //Load Shared-Memory
    if(bIndex + BWT_INTERSECT_B_BLOCK_SIZE_simple < sizePerPartition)
    {
        int bi = tid;
#pragma unroll
        for(int i = 0;i < BWT_INTERSECT_B_BLOCK_SIZE_simple/BWT_CTASIZE_simple; i++, bi+=BWT_CTASIZE_simple)
        {
            BKeys[bi] =   A_keys[myStartIdxB + bIndex + bi];
            BValues[bi] = A_values[myStartIdxB + bIndex + bi];
        }

    }
    else
    {
        int bi = tid;
#pragma unroll
        for(int i = 0;i < BWT_INTERSECT_B_BLOCK_SIZE_simple/BWT_CTASIZE_simple; i++, bi+=BWT_CTASIZE_simple)
        {
            BKeys[bi] =   (bIndex + bi < sizePerPartition ? A_keys  [myStartIdxB + bIndex + bi] : UINT_MAX);
            BValues[bi] = (bIndex + bi < sizePerPartition ? A_values[myStartIdxB + bIndex + bi] : UINT_MAX);
        }
    }

    //Save localMaxA and localMaxB
    if(tid == BWT_CTASIZE_simple-1)
        BMax[1] = myKey[depth-1];
    if(tid == 0)
        BMax[0] =  (bIndex + BWT_INTERSECT_B_BLOCK_SIZE_simple - 1 < sizePerPartition ?
        A_keys[myStartIdxB + bIndex + BWT_INTERSECT_B_BLOCK_SIZE_simple - 1] : UINT_MAX);

    __syncthreads();

    //Maximum values for B and A in this stream
    localMaxB = BMax[0];
    localMaxA = BMax[1];

    T localMinB = 0;

    __syncthreads();
    __threadfence(); //Extra Added

    do
    {
        __syncthreads();

        index = 0;

        int cumulativeAddress = myStartIdxA+aIndex+threadIdx.x*depth;			
        if((myKey[0] <= localMaxB && myKey[depth-1] >= localMinB) ||  (bIndex+BWT_INTERSECT_B_BLOCK_SIZE_simple-1) >= (sizePerPartition) && cumulativeAddress < sizePerPartition) // ==ADDED==
        {
            index = -1;

            mid = (BWT_INTERSECT_B_BLOCK_SIZE_simple/2)-1;
            if(BWT_INTERSECT_B_BLOCK_SIZE_simple >= 1024)
                binSearch_fragment<T,depth> (BKeys, BValues, 256, mid, cmpValue, myKey[0], myValue[0], stringValues, stringValues2, numElements);
            if(BWT_INTERSECT_B_BLOCK_SIZE_simple >= 512)
                binSearch_fragment<T,depth> (BKeys, BValues, 128, mid, cmpValue, myKey[0], myValue[0], stringValues, stringValues2, numElements);
            if(BWT_INTERSECT_B_BLOCK_SIZE_simple >= 256)
                binSearch_fragment<T,depth> (BKeys, BValues, 64, mid, cmpValue, myKey[0], myValue[0], stringValues, stringValues2, numElements);

            binSearch_fragment<T,depth> (BKeys, BValues, 32, mid, cmpValue, myKey[0], myValue[0], stringValues, stringValues2, numElements);
            binSearch_fragment<T,depth> (BKeys, BValues, 16, mid, cmpValue, myKey[0], myValue[0], stringValues, stringValues2, numElements);
            binSearch_fragment<T,depth> (BKeys, BValues,  8, mid, cmpValue, myKey[0], myValue[0], stringValues, stringValues2, numElements);
            binSearch_fragment<T,depth> (BKeys, BValues,  4, mid, cmpValue, myKey[0], myValue[0], stringValues, stringValues2, numElements);
            binSearch_fragment<T,depth> (BKeys, BValues,  2, mid, cmpValue, myKey[0], myValue[0], stringValues, stringValues2, numElements);
            binSearch_fragment<T,depth> (BKeys, BValues,  1, mid, cmpValue, myKey[0], myValue[0], stringValues, stringValues2, numElements);

            index = mid;
            cmpValue = BKeys[index];

            //correct search if needed
            if(cmpValue < myKey[0] && index < BWT_INTERSECT_B_BLOCK_SIZE_simple)
                cmpValue = BKeys[++index];

            //Tied version of previous if statement
            if(cmpValue == myKey[0] && index < BWT_INTERSECT_B_BLOCK_SIZE_simple)
            {
                int count = 1;
                T tmpKey, cmpKey;
                tmpKey = myKey[0];
                cmpKey = cmpValue;

                while(tmpKey == cmpKey)
                {
                    tmpKey = (myValue[0]+4*count > numElements-1) ? stringValues2[myValue[0] + 4*count - numElements] : stringValues2[myValue[0] + 4*count];
                    cmpKey = (BValues[index]+4*count > numElements-1) ? stringValues2[BValues[index] + 4*count - numElements] : stringValues2[BValues[index] + 4*count];

                    if(cmpKey < tmpKey)
                    {
                        cmpValue = BKeys[++index];
                        break;
                    }
                    count++;
                }
            }


            if(cmpValue < myKey[0] && (bIndex+index) < sizePerPartition)
            {
                index++;
                cmpValue =  A_keys[myStartIdxB+bIndex + (index)];
            }

            //Tied version of previous if statement
            if(cmpValue == myKey[0])
            {
                int count = 1;
                T tmpKey, cmpKey;
                tmpKey = myKey[0];
                cmpKey = cmpValue;

                while(tmpKey == cmpKey && (bIndex+index) < sizePerPartition)
                {
                    tmpKey = (myValue[0]+4*count > numElements-1) ? stringValues2[myValue[0] + 4*count - numElements] : stringValues2[myValue[0] + 4*count];
                    cmpKey = (A_values[myStartIdxB+bIndex+index]+4*count > numElements-1) ?
                        stringValues2[A_values[myStartIdxB+bIndex+index] + 4*count - numElements] : stringValues2[A_values[myStartIdxB+bIndex+index] + 4*count];

                    if(cmpKey < tmpKey)
                    {
                        index++;
                        break;
                    }
                    count++;
                }
            }

            //End Binary Search
            //binary search done for first element in our set (A_0)

            //Save Value if it is valid (correct window)
            //If we are on the edge of a window, and we are tied with the localMax or localMin value
            //we must go to global memory to find out if we are valid
            if((myKey[0] < localMaxB && myKey[0] > localMinB) || (index==BWT_INTERSECT_B_BLOCK_SIZE_simple && (bIndex+index)>=sizePerPartition) || (index > 0 && index <BWT_INTERSECT_B_BLOCK_SIZE_simple))
            {
                A_keys_out[myStartIdxC + bIndex + aIndex + depth*tid + index] = myKey[0];
                A_values_out[myStartIdxC + bIndex + aIndex + depth*tid + index] = myValue[0];

            }
            else if(myKey[0] == localMaxB && index == BWT_INTERSECT_B_BLOCK_SIZE_simple)
            {
                //tie break
                unsigned int tmpAdd = myValue[0];
                unsigned int cmpAdd = A_values[myStartIdxB+bIndex+index];

                int count = 1;

                T tmpKey = (tmpAdd+4*count > numElements-1) ? stringValues2[tmpAdd + 4*count - numElements] : stringValues2[tmpAdd + 4*count];
                T cmpKey = (cmpAdd+4*count > numElements-1) ? stringValues2[cmpAdd + 4*count - numElements] : stringValues2[cmpAdd + 4*count];

                while(tmpKey == cmpKey)
                {
                    count++;
                    tmpKey = (tmpAdd+4*count > numElements-1) ? stringValues2[tmpAdd + 4*count - numElements] : stringValues2[tmpAdd + 4*count];
                    cmpKey = (cmpAdd+4*count > numElements-1) ? stringValues2[cmpAdd + 4*count - numElements] : stringValues2[cmpAdd + 4*count];
                }

                if(tmpKey < cmpKey)
                {
                    A_keys_out[myStartIdxC + bIndex + aIndex + depth*tid + index] = myKey[0];
                    A_values_out[myStartIdxC + bIndex + aIndex + depth*tid + index] = myValue[0];
                }

            }
            else if(myKey[0] == localMinB && index == 0)
            {
                unsigned int tmpAdd = myValue[0];
                unsigned int cmpAdd = BValues[0];

                int count = 1;

                unsigned int tmpKey = (tmpAdd+4*count > numElements-1) ? stringValues2[tmpAdd + 4*count - numElements] : stringValues2[tmpAdd + 4*count];
                unsigned int cmpKey = (cmpAdd+4*count > numElements-1) ? stringValues2[cmpAdd + 4*count - numElements] : stringValues2[cmpAdd + 4*count];

                while(tmpKey == cmpKey)
                {
                    count++;
                    tmpKey = (tmpAdd+4*count > numElements-1) ? stringValues2[tmpAdd + 4*count - numElements] : stringValues2[tmpAdd + 4*count];
                    cmpKey = (cmpAdd+4*count > numElements-1) ? stringValues2[cmpAdd + 4*count - numElements] : stringValues2[cmpAdd + 4*count];
                }

                if(tmpKey > cmpKey)
                {
                    A_keys_out[myStartIdxC + bIndex + aIndex+depth*tid+index] = myKey[0];
                    A_values_out[myStartIdxC + bIndex + aIndex+depth*tid+index] = myValue[0];
                }
            }

            //After binary search, linear merge
            lin_merge_simple<T, depth>(cmpValue, myKey[1], myValue[1], index, BKeys, BValues, stringValues, A_values, A_keys_out, A_values_out,
                myStartIdxA, myStartIdxB, myStartIdxC, localMinB, localMaxB, aIndex+tid*depth, bIndex, totalSize, sizePerPartition, 1, stringValues2, numElements);

        }

        bool reset = false;
        __syncthreads();
        if(localMaxA == localMaxB)
        {

            //Break the tie
            if(tid == (blockDim.x-1))
            {
                unsigned int tmpAdd = myValue[1];
                unsigned int cmpAdd = BValues[BWT_INTERSECT_B_BLOCK_SIZE_simple-1];

                int count = 1;

                unsigned int tmpKey = (tmpAdd+4*count > numElements-1) ? stringValues2[tmpAdd + 4*count - numElements] : stringValues2[tmpAdd + 4*count];
                unsigned int cmpKey = (cmpAdd+4*count > numElements-1) ? stringValues2[cmpAdd + 4*count - numElements] : stringValues2[cmpAdd + 4*count];

                while(tmpKey == cmpKey)
                {
                    count++;
                    tmpKey = (tmpAdd+4*count > numElements-1) ? stringValues2[tmpAdd + 4*count - numElements] : stringValues2[tmpAdd + 4*count];
                    cmpKey = (cmpAdd+4*count > numElements-1) ? stringValues2[cmpAdd + 4*count - numElements] : stringValues2[cmpAdd + 4*count];
                }

                if(tmpKey > cmpKey)
                    BMax[1]++;
                else
                    BMax[0]++;
            }
            __syncthreads();
            localMaxB = BMax[0];
            localMaxA = BMax[1];
            reset = true;

        }

        __syncthreads();
        if((localMaxA < localMaxB || (bIndex+BWT_INTERSECT_B_BLOCK_SIZE_simple-1) >= sizePerPartition) && (aIndex+BWT_INTERSECT_A_BLOCK_SIZE_simple)< sizePerPartition)
        {
            __syncthreads();
            __threadfence();

            aIndex += BWT_INTERSECT_A_BLOCK_SIZE_simple;

            if(aIndex + BWT_INTERSECT_A_BLOCK_SIZE_simple < sizePerPartition)
            {
#pragma unroll
                for(int i = 0;i < depth; i++)
                {
                    myKey[i] = A_keys[myStartIdxA + aIndex + depth*tid + i];
                    myValue[i] = A_values[myStartIdxA + aIndex + depth*tid + i];

                }
            }
            else
            {

#pragma unroll
                for(int i = 0;i < depth; i++)
                {
                    myKey[i] = (aIndex+depth*tid + i < sizePerPartition ? A_keys[myStartIdxA + aIndex + depth*tid + i]   : UINT_MAX-1);
                    myValue[i] = (aIndex+depth*tid + i < sizePerPartition ? A_values[myStartIdxA + aIndex + depth*tid + i]   : UINT_MAX-1);

                }
            }
            if(tid == BWT_CTASIZE_simple-1)
            {
                BMax[1] = myKey[depth-1];
                if(reset)
                    BMax[0]--;
            }
            reset = false;

        }
        else if(localMaxB < localMaxA && (bIndex+BWT_INTERSECT_B_BLOCK_SIZE_simple-1) < sizePerPartition)
        {

            //Use UINT_MAX as an "invalid/no-value" type in case the streaming window cannot be filled
            bIndex += BWT_INTERSECT_B_BLOCK_SIZE_simple-1;

            if(bIndex + BWT_INTERSECT_B_BLOCK_SIZE_simple < sizePerPartition)
            {
                int bi = tid;
#pragma unroll
                for(int i = 0;i < BWT_INTERSECT_B_BLOCK_SIZE_simple/BWT_CTASIZE_simple; i++, bi+=BWT_CTASIZE_simple)
                {
                    BKeys[bi] =   A_keys[myStartIdxB + bIndex + bi];
                    BValues[bi] = A_values[myStartIdxB + bIndex + bi];

                }
            }
            else
            {
                int bi = tid;
#pragma unroll
                for(int i = 0;i < BWT_INTERSECT_B_BLOCK_SIZE_simple/BWT_CTASIZE_simple; i++, bi+=BWT_CTASIZE_simple)
                {
                    BKeys[bi] =   (bIndex + bi < sizePerPartition ? A_keys[myStartIdxB + bIndex + bi]   : UINT_MAX);
                    BValues[bi] = (bIndex + bi < sizePerPartition ? A_values[myStartIdxB + bIndex + bi] : UINT_MAX);

                }
            }

            if(tid == 0)
            {
                BMax[0] =  (bIndex + BWT_INTERSECT_B_BLOCK_SIZE_simple < sizePerPartition ? A_keys[myStartIdxB + bIndex + BWT_INTERSECT_B_BLOCK_SIZE_simple - 1] : UINT_MAX);
                if(reset)
                    BMax[1]--;
            }
            __syncthreads();
            localMinB = BKeys[0];

            __syncthreads();
            __threadfence(); //Extra Added
            reset = false;
        }
        else
            breakout = true;
        __syncthreads();



        localMaxB = BMax[0];
        localMaxA = BMax[1];

        __syncthreads();
        __threadfence(); //Extra Added

    }
    while(!breakout);

    __syncthreads();

}

/** @brief Block sort
* @todo
**/
template<class T, int depth>
__global__ void blockWiseStringSort(T*      A_keys,
                                    T*      A_address,
                                    const   T* stringVals,
                                    T*      stringVals2,
                                    int     blockSize,
                                    size_t  numElements)
{
    //load into registers
    T Aval[depth]; // keys
    T saveValue[depth];

    __shared__ T scratchPad[2*BWT_BLOCKSORT_SIZE];

    // half of scratch pad is taken up by addresses
    // there are BWT_BLOCKSORT_SIZE addresses
    T* addressPad = (T*) &scratchPad[BWT_BLOCKSORT_SIZE];


    int bid = blockIdx.x;
    int tid = threadIdx.x;


    for(int i = 0; i < depth; i++)
    {
        // Brining keys into registers
        Aval[i] = A_keys[bid*blockSize+tid*depth+i];
    }

    // bringing adressess into shared mem (coalesced reads)
    for(int i = tid; i < blockSize; i+=BWT_CTA_BLOCK)
        addressPad[i] = A_address[blockSize*bid+i];


    __syncthreads();

    //Sort first 8 values
    // Bitonic sort -- each thread sorts 8 string using bitonc
    int offset = tid*depth;

    compareSwapVal<T>(Aval[0], Aval[1], offset, offset+1, addressPad, stringVals, stringVals2, numElements);
    compareSwapVal<T>(Aval[2], Aval[3], offset+2, offset+3, addressPad, stringVals, stringVals2, numElements);
    compareSwapVal<T>(Aval[0], Aval[2], offset+0, offset+2, addressPad, stringVals, stringVals2, numElements);
    compareSwapVal<T>(Aval[1], Aval[3], offset+1, offset+3, addressPad, stringVals, stringVals2, numElements);
    compareSwapVal<T>(Aval[1], Aval[2], offset+1, offset+2, addressPad, stringVals, stringVals2, numElements);
    //4-way sort on second set of values
    compareSwapVal<T>(Aval[4], Aval[5], offset+4, offset+5, addressPad, stringVals, stringVals2, numElements);
    compareSwapVal<T>(Aval[6], Aval[7], offset+6, offset+7, addressPad, stringVals, stringVals2, numElements);
    compareSwapVal<T>(Aval[4], Aval[6], offset+4, offset+6, addressPad, stringVals, stringVals2, numElements);
    compareSwapVal<T>(Aval[5], Aval[7], offset+5, offset+7, addressPad, stringVals, stringVals2, numElements);
    compareSwapVal<T>(Aval[5], Aval[6], offset+5, offset+6, addressPad, stringVals, stringVals2, numElements);
    compareSwapVal<T>(Aval[0], Aval[4], offset+0, offset+4, addressPad, stringVals, stringVals2, numElements);
    compareSwapVal<T>(Aval[1], Aval[5], offset+1, offset+5, addressPad, stringVals, stringVals2, numElements);
    compareSwapVal<T>(Aval[2], Aval[6], offset+2, offset+6, addressPad, stringVals, stringVals2, numElements);
    compareSwapVal<T>(Aval[3], Aval[7], offset+3, offset+7, addressPad, stringVals, stringVals2, numElements);
    compareSwapVal<T>(Aval[2], Aval[4], offset+2, offset+4, addressPad, stringVals, stringVals2, numElements);
    compareSwapVal<T>(Aval[3], Aval[5], offset+3, offset+5, addressPad, stringVals, stringVals2, numElements);
    compareSwapVal<T>(Aval[1], Aval[2], offset+1, offset+2, addressPad, stringVals, stringVals2, numElements);
    compareSwapVal<T>(Aval[3], Aval[4], offset+3, offset+4, addressPad, stringVals, stringVals2, numElements);
    compareSwapVal<T>(Aval[5], Aval[6], offset+5, offset+6, addressPad, stringVals, stringVals2, numElements);

    __syncthreads();

    int j;
#pragma unroll
    // loading all keys into shared mem., used to find where to merge into
    for(int i=0;i<depth;i++)
        scratchPad[tid*depth+i] = Aval[i];

    __syncthreads();

    // 1st half of scratch pad has keys (first 4 chars of each 1024 strings)
    // 2nd half of scratch pad has values

    T * in = scratchPad;

    int mult = 1;
    int count = 0;
    int steps = 128;

    while (mult < steps)
    {
        // What is first, last, midpoint?
        int first, last;
        first = (tid>>(count+1))*depth*2*mult;
        int midPoint = first+mult*depth;

        T cmpValue;
        T tmpVal;

        //first half or second half
        int addPart = threadIdx.x%(mult<<1) >= mult ? 1 : 0;

        if(addPart == 0)
            first += depth*mult;
        last = first+depth*mult-1;

        j = (first+last)/2;

        int startAddress = threadIdx.x*depth-midPoint;

        int range = last-first;

        __syncthreads();
        tmpVal = Aval[0];

        //Begin binary search
        switch(range)
        {
        case 1023: bin_search_block<T, depth>(cmpValue, tmpVal, in, addressPad, stringVals, j, 256, stringVals2, numElements);
        case 511: bin_search_block<T, depth>(cmpValue, tmpVal, in, addressPad, stringVals, j, 128, stringVals2, numElements);
        case 255: bin_search_block<T, depth>(cmpValue, tmpVal, in, addressPad, stringVals, j, 64, stringVals2, numElements);
        case 127: bin_search_block<T, depth>(cmpValue, tmpVal, in, addressPad, stringVals, j, 32, stringVals2, numElements);
        case 63: bin_search_block<T, depth>(cmpValue, tmpVal, in, addressPad, stringVals, j, 16, stringVals2, numElements);
        case 31: bin_search_block<T, depth>(cmpValue, tmpVal, in, addressPad, stringVals, j, 8, stringVals2, numElements);
        case 15: bin_search_block<T, depth>(cmpValue, tmpVal, in, addressPad, stringVals, j, 4, stringVals2, numElements);
        case 7: bin_search_block<T, depth>(cmpValue, tmpVal, in,  addressPad,stringVals, j, 2, stringVals2, numElements);
        case 3: bin_search_block<T, depth>(cmpValue, tmpVal, in, addressPad, stringVals, j, 1, stringVals2, numElements);
        }

        cmpValue = in[j];
        if(cmpValue == tmpVal)
        {
            T tmp = (addressPad[depth*tid]+4*1 > numElements-1) ?
                stringVals2[addressPad[depth*tid]+4*1-numElements] : stringVals2[addressPad[depth*tid]+4*1];
            T tmp2 = (addressPad[j]+4*1 > numElements-1) ? stringVals2[addressPad[j]+4*1-numElements] : stringVals2[addressPad[j]+4*1];

            int i = 2;
            while(tmp == tmp2)
            {
                tmp = (addressPad[depth*tid]+4*i > numElements-1) ?
                    stringVals2[addressPad[depth*tid]+4*i-numElements] : stringVals2[addressPad[depth*tid]+4*i];
                tmp2 = (addressPad[j]+4*i > numElements-1) ? stringVals2[addressPad[j]+4*i-numElements] : stringVals2[addressPad[j]+4*i];

                i++;
            }
            j = (tmp2 < tmp ? j +1 : j);
            cmpValue = in[j];
        }
        else if(cmpValue < tmpVal)
            cmpValue = in[++j];

        if(cmpValue == tmpVal && j == last)
        {
            T tmp = (addressPad[depth*tid]+4*1 > numElements-1) ?
                stringVals2[addressPad[depth*tid]+4*1-numElements] : stringVals2[addressPad[depth*tid]+4*1];
            T tmp2 = (addressPad[j]+4*1 > numElements-1) ? stringVals2[addressPad[j]+4*1-numElements] : stringVals2[addressPad[j]+4*1];

            int i = 2;
            while(tmp == tmp2)
            {
                tmp = (addressPad[depth*tid]+4*i > numElements-1) ?
                    stringVals2[addressPad[depth*tid]+4*i-numElements] : stringVals2[addressPad[depth*tid]+4*i];
                tmp2 = (addressPad[j]+4*i > numElements-1) ? stringVals2[addressPad[j]+4*i-numElements] : stringVals2[addressPad[j]+4*i];

                i++;
            }
            j = (tmp2 < tmp ? j +1 : j);
        }
        else if(cmpValue < tmpVal && j == last)
            j++;

        Aval[0] = j+startAddress;
        lin_search_block<T, depth>(cmpValue,  Aval[1], in, addressPad, stringVals, j, 1, last, startAddress, 0, stringVals2, numElements);
        lin_search_block<T, depth>(cmpValue,  Aval[2], in, addressPad, stringVals, j, 2, last, startAddress, 0, stringVals2, numElements);
        lin_search_block<T, depth>(cmpValue,  Aval[3], in, addressPad, stringVals, j, 3, last, startAddress, 0, stringVals2, numElements);
        lin_search_block<T, depth>(cmpValue,  Aval[4], in, addressPad, stringVals, j, 4, last, startAddress, 0, stringVals2, numElements);
        lin_search_block<T, depth>(cmpValue,  Aval[5], in, addressPad, stringVals, j, 5, last, startAddress, 0, stringVals2, numElements);
        lin_search_block<T, depth>(cmpValue,  Aval[6], in, addressPad, stringVals, j, 6, last, startAddress, 0, stringVals2, numElements);
        lin_search_block<T, depth>(cmpValue,  Aval[7], in, addressPad, stringVals, j, 7, last, startAddress, 0, stringVals2, numElements);
        __syncthreads();
        saveValue[0] = in[tid*depth];
        saveValue[1] = in[tid*depth+1];
        saveValue[2] = in[tid*depth+2];
        saveValue[3] = in[tid*depth+3];
        saveValue[4] = in[tid*depth+4];
        saveValue[5] = in[tid*depth+5];
        saveValue[6] = in[tid*depth+6];
        saveValue[7] = in[tid*depth+7];
        __syncthreads();
        in[Aval[0]] = saveValue[0];
        in[Aval[1]] = saveValue[1];
        in[Aval[2]] = saveValue[2];
        in[Aval[3]] = saveValue[3];
        in[Aval[4]] = saveValue[4];
        in[Aval[5]] = saveValue[5];
        in[Aval[6]] = saveValue[6];
        in[Aval[7]] = saveValue[7];
        __syncthreads();
        saveValue[0] = addressPad[tid*depth];
        saveValue[1] = addressPad[tid*depth+1];
        saveValue[2] = addressPad[tid*depth+2];
        saveValue[3] = addressPad[tid*depth+3];
        saveValue[4] = addressPad[tid*depth+4];
        saveValue[5] = addressPad[tid*depth+5];
        saveValue[6] = addressPad[tid*depth+6];
        saveValue[7] = addressPad[tid*depth+7];
        __syncthreads();
        addressPad[Aval[0]] = saveValue[0];
        addressPad[Aval[1]] = saveValue[1];
        addressPad[Aval[2]] = saveValue[2];
        addressPad[Aval[3]] = saveValue[3];
        addressPad[Aval[4]] = saveValue[4];
        addressPad[Aval[5]] = saveValue[5];
        addressPad[Aval[6]] = saveValue[6];
        addressPad[Aval[7]] = saveValue[7];
        __syncthreads();



        mult*=2;
        count++;

        if(mult < steps)
        {
            __syncthreads();

#pragma unroll
            for(int i=0;i<depth;i++)
                Aval[i] = in[tid*depth+i];
        }
        __syncthreads();
    }

    __syncthreads();
#pragma unroll
    for(int i=tid;i<blockSize;i+= BWT_CTA_BLOCK)
    {
        A_keys[bid*blockSize+i] = in[i];
        A_address[bid*blockSize+i] = addressPad[i];
        __syncthreads();
    }
}

/** @brief Massage input to set up for merge sort
* @todo
**/
__global__ void
bwt_keys_construct_kernel(uchar4    *d_bwtIn,
                          uint      *d_bwtInRef,
                          uint      *d_keys,
                          uint      *d_values,
                          uint      *d_bwtInRef2,
                          uint      tThreads)
{
    // Global, local IDs
    uint idx = threadIdx.x + (blockIdx.x * blockDim.x);

    if(idx < tThreads)
    {
        uint start = idx*4;
        uchar4 prefix = d_bwtIn[idx];
        uchar4 prefix_end;

        uint keys[4];

        if(idx == (tThreads-1))
            prefix_end = d_bwtIn[0];
        else
            prefix_end = d_bwtIn[idx+1];

        // Manipulate ordering of d_bwtIn for when we typecast
        // if 1st four chars of d_bwtIn[0,1,2,3] = [a][b][c][d]
        // then 1st int of d_bwtInRef[0] = [abcd] instead of [dcba]
        uint word = 0;
        word |= (uint)prefix.x<<24;
        word |= (uint)prefix.y<<16;
        word |= (uint)prefix.z<<8;
        word |= (uint)prefix.w;
        d_bwtInRef[idx] = word;

        // key0
        keys[0] = (uint)prefix.x << 24;
        keys[0] |= (uint)prefix.y << 16;
        keys[0] |= (uint)prefix.z << 8;
        keys[0] |= (uint)prefix.w;
        d_keys[start] = keys[0];
        d_values[start] = start;

        // key1
        keys[1] = (uint)prefix.y << 24;
        keys[1] |= (uint)prefix.z << 16;
        keys[1] |= (uint)prefix.w << 8;
        keys[1] |= (uint)prefix_end.x;;
        d_keys[start+1] = keys[1];
        d_values[start+1] = start+1;

        // key2
        keys[2] = (uint)prefix.z << 24;
        keys[2] |= (uint)prefix.w << 16;
        keys[2] |= (uint)prefix_end.x << 8;
        keys[2] |= (uint)prefix_end.y;
        d_keys[start+2] = keys[2];
        d_values[start+2] = start+2;

        // key3
        keys[3] = (uint)prefix.w << 24;
        keys[3] |= (uint)prefix_end.x << 16;
        keys[3] |= (uint)prefix_end.y << 8;
        keys[3] |= (uint)prefix_end.z;
        d_keys[start+3] = keys[3];
        d_values[start+3] = start+3;

        d_bwtInRef2[start] = keys[0];
        d_bwtInRef2[start+1] = keys[1];
        d_bwtInRef2[start+2] = keys[2];
        d_bwtInRef2[start+3] = keys[3];
    }
}