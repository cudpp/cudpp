#include "reduce_kernel.cu"
#include "globals.h"
#include "cudpp_tune.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <memory.h>
#include <cstdio>
#include <math.h>
#include <limits>
//#include <cstdlib>
//#include <assert.h>

#include <time.h>
#include "cutil.h"



float reduceGold(float *din, unsigned int numElements)
{
  float result = din[0];
  int x;
  for(x=1;x<numElements;x++)
    result += din[x];
  return result;
}


void tuneReduce(CUDPPTuneReduce &tuneConfig)
{
    if(TUNE)
        return;  //already tuned, no need to tune again <TODO: a "forced" retune if user is unhappy with results
    
    CUT_DEVICE_INIT(argc, argv);
    unsigned int timer;    
    int i, j, y, z;	
    
    FILE *out = fopen("cudpp_tune.h", "w");
    CUT_DEVICE_INIT(argc, argv);
    CUT_SAFE_CALL(cutCreateTimer(&timer));
	
    int numIterations = 5;
    
    float* d_idata;
    float *d_out;    
    
    float cpuResult = 0;

    int bestBlocks, numBlocks, maxBlocks;     
    unsigned long x;
    int maxThreadsPerProc;	     
    unsigned int maxElements = 37000100; //pre-determined maximum elements for this test
    // device input array
    float *h_result = (float*) malloc(sizeof(float)*maxElements);
    float* datain= (float*) malloc(sizeof(float)*maxElements);       
	
    //Fill in Data.
    for(y=0;y<maxElements;y++)  
    {
        datain[y] =.00001;// float((rand()%maxa)/1000.); Temporarily using just a constant in order to make the CPU check easier and faster (numElements*C)
    }  

    CUDA_SAFE_CALL(cudaMalloc( (void**) &d_idata, maxElements*sizeof(float))); 
    CUDA_SAFE_CALL(cudaMalloc( (void**) &d_out, maxElements*sizeof(float))); 
    cudaGetDeviceProperties(&deviceProp, 0);	    
    
    #if DEBUG	        
    warpsize = deviceProp.warpSize;
    maxblocksize = deviceProp.maxThreadsPerBlock; 
    printf("%s\n", deviceProp.name);


    printf("Major revision number:         %d\n",  deviceProp.major);
    printf("Minor revision number:         %d\n",  deviceProp.minor);
    printf("Name:                          %s\n",  deviceProp.name);
    printf("Total global memory:           %u\n",  deviceProp.totalGlobalMem);
    printf("Total shared memory per block: %u\n",  deviceProp.sharedMemPerBlock);
    printf("Total registers per block:     %d\n",  deviceProp.regsPerBlock);
    printf("Warp size:                     %d\n",  deviceProp.warpSize);
    printf("Maximum memory pitch:          %u\n",  deviceProp.memPitch);
    printf("Maximum threads per block:     %d\n",  deviceProp.maxThreadsPerBlock);
    for (int i = 0; i < 3; ++i)
    printf("Maximum dimension %d of block:  %d\n", i, deviceProp.maxThreadsDim[i]);
    for (int i = 0; i < 3; ++i)
    printf("Maximum dimension %d of grid:   %d\n", i, deviceProp.maxGridSize[i]);
    printf("Clock rate:                    %d\n",  deviceProp.clockRate);
    printf("Total constant memory:         %u\n",  deviceProp.totalConstMem);
    printf("Texture alignment:             %u\n",  deviceProp.textureAlignment);
    printf("Concurrent copy and execution: %s\n",  (deviceProp.deviceOverlap ? "Yes" : "No"));
    printf("Number of multiprocessors:     %d\n",  deviceProp.multiProcessorCount);
    printf("Kernel execution timeout:      %s\n",  (deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
    #endif

    CUDA_SAFE_CALL(cudaMemcpy(d_idata, datain, sizeof(float)*maxElements, cudaMemcpyHostToDevice)); 
    int testPoint = 64*deviceProp.numProcessors;

    int bestGPUWin,  threadCount, smemSize, threadsperBlock, totalCount, switchPoint, currentGPUWin, bestThreadsPerBlock, cLast;
    int last5[5];
    float cpuTArray[30];
    float jump;
    float avgTime, gTime, bestTime;
    int bestNumBlocks;
    cLast = 0;
    numProcessors = 30;

    last5[0] = 0; last5[1] = maxElements; last5[2] = maxElements; 
    last5[3] = maxElements; last5[4] = maxElements; //this is a kind of hacked/ugly method of finding GPU-CPU switchpoint


    for(x=testPoint;x<maxElements && abs(last5[4]-last5[0])>0 &&(float(abs(last5[4]-last5[0]))/float(last5[4])) > .01;)
    {     
        bestGPUWin = 0;
	    bestTime = FLT_MAX;
        printf("Test for %d\n", x);
        avgTime = 0;
  
        for(z=0;z<30;z++) //Various 'close' combinations to combat noise and possible bad combinations
        {	
            cutResetTimer(timer);	
            cutStartTimer(timer);	     
            for(i=0;i<numIterations;i++) 
            { 	            
                CUDA_SAFE_CALL(cudaMemcpy(h_result, d_idata, sizeof(float)*(x-15+z), cudaMemcpyDeviceToHost));	
                for(j = 1;j<(x-15+z);j++)
                    h_result[0] += h_result[j];
            }
	        
            cudaThreadSynchronize();
            cutStopTimer(timer);	        
            avgTime += cutGetTimerValue(timer);
	        
        }       
        printf("CPU time %f\n", avgTime/30.);   
        //as a memory bound process 128 TPB works best for a large number of elements
        //but for smaller sizes 64 or sometimes even 32 TPB is preferred
        for(threadCount=32;threadCount<=256;threadCount*=2) 
        {                              
            smemSize =  threadCount* sizeof(float);
            for(numBlocks = 1; (numBlocks)*threadCount < 24576 
                && (numBlocks)*threadCount < x; numBlocks += 1)
            {
                currentGPUWin = 0; 
	        for(z=0;z<=30;z++)
	        {			
                    cutResetTimer(timer);	
                    cutStartTimer(timer);	 
                    for(i=0;i<numIterations;i++)
                    {    
                        switch(threadCount)
                        {
                            case 16:
                                gridReduce<16><<<numBlocks, threadCount, smemSize>>>(d_idata, d_out, (x-15+z));
                                break;
                            case 32:
                                gridReduce<32><<<numBlocks, threadCount, smemSize>>>(d_idata, d_out, (x-15+z));
                                break;
                            case 64:
                                gridReduce<64><<<numBlocks, threadCount, smemSize>>>(d_idata, d_out, (x-15+z));
                                break;
                            case 128:
                                gridReduce<128><<<numBlocks, threadCount, smemSize>>>(d_idata, d_out, (x-15+z));
                                break;
                            case 256:
                                gridReduce<256><<<numBlocks, threadCount, smemSize>>>(d_idata, d_out, (x-15+z));
                                break;
                            case 512:
                                gridReduce<512><<<numBlocks, threadCount, smemSize>>>(d_idata, d_out, (x-15+z));
	                        break;
                        }
                        CUDA_SAFE_CALL(cudaMemcpy(h_result, d_out, sizeof(float)*numBlocks, cudaMemcpyDeviceToHost));
                        for(j = 1;j<numBlocks;j++)
                            h_result[0] += h_result[j];
                     }
                     cudaThreadSynchronize();
                     cutStopTimer(timer);
                     gTime = cutGetTimerValue(timer);
                    //	printf("gtime %f, avgTime/30 %f\n", gTime, avgTime/30.);
                    if(gTime<avgTime)
                        currentGPUWin +=1;
                    if(gTime<bestTime)
                        bestTime = gTime;
                } 	
	  
                if(currentGPUWin>bestGPUWin)
                {
                    bestGPUWin = currentGPUWin;	  	
                    bestThreadsPerBlock = threadCount;	
                    bestNumBlocks = numBlocks;
                    bestTime = gTime;
                }	  
            }
        }
        jump = (15-bestGPUWin)*(x*.001);
        printf("Jump by %f", jump);
        jump = (abs(jump) >= 1 ? jump : jump/(abs(jump)));
        printf("Best GPU Time %f\n", bestTime);
        printf("%d/30 were faster than the CPU runs\n", bestGPUWin);		
        printf("Jump by %f", jump);
  
        if(bestGPUWin < 5 || bestGPUWin>20)
        {	
            x+=jump;
        }
        else if(bestGPUWin !=15 && bestGPUWin !=16)
            x+= jump/abs(jump);
		
			
			/*if(bestGPUWin > 15)
	    {		
			 if((float(bestGPUWin-15)*x*.05)>1)
	              x = x-(float(15-bestGPUWin)*.1*x);
			 else
				 x=x-1;
	    }
	    else 
	    {	
			if((float(15-bestGPUWin)*x*.05)>1)
		        x= x+(float(15-bestGPUWin)*.05*x);
			else
				x=x-1;
	    }*/
       last5[cLast] = x;
       cLast = (cLast+1)%5;
	        
        printf("%d with threadCount %d, %d CPUtime %f GPUtime %f\n",  x, bestThreadsPerBlock, bestNumBlocks, avgTime/30., bestTime);
        switchPoint = x;
    }

    printf("Switch Point %d with %d threadsPerBlock\n", x, bestThreadsPerBlock);	
    printf("MaxTHreadsPerProc %d numProcessors %d\n", maxThreadsPerProc, numProcessors);	
    tuneConfig.switchOverPoint = switchPoint;
    tuneConfig.minThreads = bestThreadsPerBlock*numBlocks);
    int smem;	
    threadsperBlock = 128;	
    int maxThreads;
    maxThreads = 24576;//maxThreadsPerProc*numProcessors;
    maxBlocks = ceil(float(maxThreads)/float(threadsperBlock));
    smem =sizeof(float)*threadsperBlock;
    numIterations = 25;
    //TODO: Benchmark for maxThreads since it cannot usually be queried
  
    int  prevPoint = 0;

    testPoint = maxThreads;
    float maxTest = FLT_MAX;
    float minTest = 0;
    while(maxTest >= minTest)
    {
        maxBlocks = maxThreads/threadsperBlock;
        prevPoint = testPoint;
        testPoint*=1.4;
        cpuResult = testPoint*.0001;	    
        minTest = FLT_MAX;
        for(numBlocks=maxBlocks/2;numBlocks<maxBlocks && numBlocks*threadsperBlock < testPoint;numBlocks= numBlocks+1)
        {    	    
            gridReduce<128><<<numBlocks, threadsperBlock, smem>>>(d_idata, d_out, testPoint);	 
            cutResetTimer(timer);	
            cutStartTimer(timer);	  
            for(i=0;i<numIterations;i++)
            {    
                gridReduce<128><<<numBlocks, threadsperBlock, smem>>>(d_idata, d_out, testPoint-numIterations/2 + i); //smooth out noisy cases	 
                CUDA_SAFE_CALL(cudaMemcpy(h_result, d_out, sizeof(float)*numBlocks, cudaMemcpyDeviceToHost));	   	  	  
                for(j=1;j<numBlocks;j++)
                    h_result[0] += h_result[j];
            }	    		
            cutStopTimer(timer); 
            time1 = cutGetTimerValue(timer);
            if((cpuResult-h_result[0])/cpuResult > .01)
            {
                printf("%d Elements: Error check failed for %d blocks (%f, %f)\n", x, numBlocks, cpuResult, h_result[0]);
                system("PAUSE");
            }
            if(time1<minTest)
            {
                minTest = time1;
                bestBlocks = numBlocks;	        
            }	
            gridReduce<128><<<maxBlocks, threadsperBlock, smem>>>(d_idata, d_out, x);	 
            cutResetTimer(timer);	
            cutStartTimer(timer);	  
            for(i=0;i<numIterations;i++)
            {    
                gridReduce<128><<<maxBlocks, threadsperBlock, smem>>>(d_idata, d_out, x-numIterations/2 + i); //smooth out noisy cases	 
                CUDA_SAFE_CALL(cudaMemcpy(h_result, d_out, sizeof(float)*numBlocks, cudaMemcpyDeviceToHost));	   	  	  
                for(j=1;j<numBlocks;j++)
                    h_result[0] += h_result[j];
            }	    	
            cutStopTimer(timer); 
            maxTest = cutGetTimerValue(timer);
        }
    }

    printf("Threshold point is in between %d and %d\n", prevPoint, testPoint);
    float nTslope = float(2*maxThreads)/float(testPoint+prevPoint);  
    printf("threadSlope chosen as %f\n", nTslope);  

    fprintf(out, "#define THRESHOLD %d\n", ceil(float(testPoint+prevPoint)/2.));
    fprintf(out, "define SLOPE %d\n", nTslope);

    minElements = 100;   
    //benchmark

    if(tuneConfig.showResults)
    {
        smem =sizeof(float)*threadsperBlock;
        for(x=minElements;x<37000000;x*=1.05)
        {
            if(x<switchPoint)
            {
                cutResetTimer(timer);	
                cutStartTimer(timer);	  
                for(i=0;i<numIterations;i++)
                {
                    CUDA_SAFE_CALL(cudaMemcpy(h_result, d_out, sizeof(float)*x, cudaMemcpyDeviceToHost));	   	  	  
                    for(j=1;j<x;j++)
                        h_result[0] += h_result[j];
                }
                cutStopTimer(timer);
                time1=cutGetTimerValue(timer);
                bestBlocks = 0;
            }
            else
            {
                threadCount = nTslope*x;
                numBlocks = ceil(float(threadCount)/float(threadsperBlock)) + 1;
                //numBlocks = numBlocks <= 0 ? 2 : numBlocks;
                numBlocks = numBlocks > maxBlocks ? maxBlocks : numBlocks;
                cpuResult =x*.0001; //made all values .0001 to make it a faster calculation
                //for(z=0;z<x;z++)
                // cpuResult += datain[z];
                //printf("Test for %d elements\n", x); //Old way to do it (may revert back as it is mor correct)
                printf("Test on %d\n", x);

              
                gridReduce<128><<<numBlocks, threadsperBlock, smem>>>(d_idata, d_out, x);	 //get GPU warmed up.
                cudaThreadSynchronize();
                cutResetTimer(timer);	
                cutStartTimer(timer);	  
                for(i=0;i<numIterations;i++)
                {    
                    gridReduce<128><<<numBlocks, threadsperBlock, smem>>>(d_idata, d_out, x-numIterations/2+i);	 
                    CUDA_SAFE_CALL(cudaMemcpy(h_result, d_out, sizeof(float)*numBlocks, cudaMemcpyDeviceToHost));	   	  	  
                    for(j=1;j<numBlocks;j++) //grab leftoverblocks
                        h_result[0] += h_result[j];
                }
                cudaThreadSynchronize();	
                cutStopTimer(timer); 
                time1 = cutGetTimerValue(timer);
                if((cpuResult-h_result[0])/cpuResult > .01) //error check method (cpuResult is sometimes more off for some reason)
                {
                    printf("%d Elements: Error check failed for %d blocks (%f, %f)\n", x, numBlocks, cpuResult, h_result[0]);
                    system("PAUSE");
                }
                bestBlocks = numBlocks;
            }
       	
            fprintf(out, "%d %d %f %d \n", x, bestBlocks, time1,  bestBlocks*threadsperBlock);
        }
    }

    return 0;         
}

