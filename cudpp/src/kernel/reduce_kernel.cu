//#include "reduce_cta.cu"
#include "sharedmem.h"
#include <stdlib.h>
#include <stdio.h>



template <unsigned int blockSize>
__global__ 
void gridReduce(float* d_in, float* d_out, unsigned int numEls)
{  
    SharedMemory<float> smem;    
	float *sdata = smem.getPointer();
	int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockSize+threadIdx.x;// + blockIdx.x*(blockSize*2);
    float res = 0;
	int gridJump = gridDim.x*blockSize;
    //unsigned int gridSize = blockSize//*2*gridDim.x;//*gridDim.x;	
    //sdata[tid] = 0;//d_in[i];
	//use register instead of shared memory
    
    while (i < numEls)
    {	        
        res += d_in[i];// + d_in[i+blockSize];  		
		i += gridJump;        		
    } 
    sdata[tid] = res;
    __syncthreads();

    // do reduction in shared mem
   
    if (blockDim.x >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockDim.x >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockDim.x >= 128) { if (tid <  64) { sdata[tid] += sdata[tid +  64]; } __syncthreads(); }
    
    #ifndef __DEVICE_EMULATION__
    if (tid < 32)    
    {
    #endif
        if (blockDim.x >=  64) { sdata[tid] += sdata[tid + 32];  }
        if (blockDim.x >=  32) { sdata[tid] += sdata[tid + 16];  }
        if (blockDim.x >=  16) { sdata[tid] += sdata[tid +  8];  }
        if (blockDim.x >=   8) { sdata[tid] += sdata[tid +  4];  }
        if (blockDim.x >=   4) { sdata[tid] += sdata[tid +  2];  }
        if (blockDim.x >=   2) { sdata[tid] += sdata[tid +  1];  }
    #ifndef __DEVICE_EMULATION__
    }
    #endif


    if (tid == 0) d_out[blockIdx.x] = sdata[0];
}
