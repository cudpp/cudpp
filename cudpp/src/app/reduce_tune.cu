
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
* reduce_tune.cu
*
* @brief CUDPP application-level tunes the parameters in CUDPPReducePlan for optimum results
* to match the parameters set in the tuning configuration
*/

#include "cudpp_plan.h"
#include "cudpp_util.h"
#include "kernel/reduce_kernel.cu"

#include <stdio.h>
#include <cutil.h>
#include "cudpp_reduce.h"


#define DEBUG 0
 
/** @brief Perform tuning for the reduction primitive.
  *
  * Either grabs parameters from a previous test, or performs a new test that identifies key
  * characteristics for tuning the reduction primitive
  *
  * Modifies data within the plan to run near-optimum thread parameters
  * 
  *
  * @param[in]  plan         Plan Configuration, holds the variables to be modified.
  * @param[in]  tuneConfig   Tune configuration, with timers, ability to "retune", name of tuning path (to save tuning data)
  
  */
void tuneReduce(CUDPPReducePlan *plan, CUDPPTuneConfig *tuneConfig)
{           
    srand(95123);
  
    unsigned int timer;    
    int i, x;	
    unsigned int j, y;
    int nThreads;
    int maxa = 100;
    struct cudaDeviceProp deviceProp;
    float time1, nTslope;
    unsigned int maxElements = 30000000; //very large default maximum elements for test runs (should always be less than threshold maxElements)
    unsigned int switchPoint, minThreads, maxThreads;

    CUT_SAFE_CALL(cutCreateTimer(&timer));    
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
   
    FILE *grabParameters;
   
    //Checks to see if a file already exists with tuning information
    //to easily set the plan parameters
    //Can be reset by setting the reTune flag to true
   
    if((grabParameters = fopen(tuneConfig->tuneFilePath, "r")) && !tuneConfig->reTune  )
    {
        fscanf(grabParameters,"%f ", &nTslope);
        fscanf(grabParameters,"%d ", &switchPoint);
        fscanf(grabParameters,"%d ", &minThreads);
        fscanf(grabParameters,"%d ", &maxThreads);
        fscanf(grabParameters,"%d ", &maxElements);
#if DEBUG
        printf("%f %d %d %d %d\n", nTslope, switchPoint, minThreads, maxThreads, maxElements);
#endif
        if(tuneConfig->numElements >= maxElements)
        {
            plan->m_threadsPerBlock = 128;            
            nThreads = maxThreads;
        }
        else if(tuneConfig->numElements <= switchPoint)
        {
            plan->m_threadsPerBlock = (minThreads > 512 ?  128: 64);
            nThreads = minThreads;
        }        
        else
        {
            plan->m_threadsPerBlock = 128;
            nThreads = minThreads + int(nTslope*float(tuneConfig->numElements-switchPoint));
        }
        plan->m_maxBlocks = unsigned int(ceil(float(nThreads)/float(plan->m_threadsPerBlock)));              
#if DEBUG
        printf("Selected %d maxBlocks and %d threadsPerBlock\n", plan->m_maxBlocks, plan->m_threadsPerBlock);
#endif
        
        fclose(grabParameters);
        return;
    }

    
   //Tuned Parameters are not saved.. Run Simulations and save the results
   //TODO: have a float timeCap for the tuning run
    FILE *out = fopen(tuneConfig->tuneFilePath, "w");      

    int numIterations = 50;
    
    float* d_idata;
    float *d_out;    
    
    float cpuResult, gpuResult;
    
    int numBlocks, maxBlocks;     
    int z;
        
    // device input array
    float *h_result = (float*) malloc(sizeof(float)*maxElements);
    float* datain= (float*) malloc(sizeof(float)*maxElements);       
	
    //Fill in Data.
    for(y=0;y<maxElements;y++)  
    {
        datain[y] = pow(-1, (float)y) * (1.0 * (rand() / (float)RAND_MAX));; 
    }  

    CUDA_SAFE_CALL(cudaMalloc( (void**) &d_idata, maxElements*sizeof(float))); 
    CUDA_SAFE_CALL(cudaMalloc( (void**) &d_out, maxElements*sizeof(float))); 
    cudaGetDeviceProperties(&deviceProp, 0);    
    
    
#if DEBUG	                   
    //Print device properties
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

    printf("Beginning tuning process\n");
    int testPoint = 64*deviceProp.multiProcessorCount;

    CUDA_SAFE_CALL(cudaMemcpy(d_idata, datain, sizeof(float)*maxElements, cudaMemcpyHostToDevice)); 
 
    //TODO: get rid of the last5 implementation as it is ugly and old, use the cont == true flag 
    //to break out of the loop
    int   threadCount, smemSize, threadsperBlock, currentGPUWin, bestThreadsPerBlock, cLast;
    int last5[5];
    int bestGPUWin;
    float jump;
    float avgTime, gTime, bestTime, tTime, cpuTime; //timing variables
    bool isPow2;
    cLast = 0;


    last5[0] = 0; last5[1] = maxElements; last5[2] = maxElements; 
    last5[3] = maxElements; last5[4] = maxElements; //this is a kind of hacked/ugly method of finding GPU-CPU switchpoint


#if DEBUG
    printf("Starting switchpoint test\n");
#endif
    bool cont = true;
    
    numIterations = 1; //Don't need multiple iterations for this section as we vary 'z' s.t. multiple gputests are performed per point
    for(x=testPoint;x<maxElements && cont == true && abs(last5[4]-last5[0])>0 &&(float(abs(last5[4]-last5[0]))/float(last5[4])) > .01;)
    {     
        bestGPUWin = 0;
        bestTime = FLT_MAX;       
        avgTime = 0; 
        tTime = 0;
        cpuTime = 0;
        CUDA_SAFE_CALL(cudaMemcpy(d_idata, datain, sizeof(float)*(x+30), cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(h_result, d_idata, sizeof(float)*(x), cudaMemcpyDeviceToHost));	
        cudaThreadSynchronize();



        for(i=0;i<numIterations;i++) 
        { 	            
            CUDA_SAFE_CALL( cudaEventRecord(startEvent, 0) );                 
            CUDA_SAFE_CALL(cudaMemcpy(h_result, d_idata, sizeof(float)*(x), cudaMemcpyDeviceToHost));	
            cudaThreadSynchronize();
            CUDA_SAFE_CALL( cudaEventSynchronize(stopEvent) );
            CUDA_SAFE_CALL( cudaEventRecord(stopEvent, 0) );
            
            CUDA_SAFE_CALL( cudaEventElapsedTime(&avgTime, startEvent, stopEvent));
            tTime += avgTime;
            cutResetTimer(timer);
            cutStartTimer(timer);
            for(j = 1; j<x;j++)
                h_result[0] += h_result[j];
            cutStopTimer(timer);
            cpuTime += cutGetTimerValue(timer);

        }	    
        cpuResult = h_result[0];
#if DEBUG
        printf("%d elements CPU time: %f \n" , x, tTime + cpuTime); 
#endif
       
        //note: on one machine, the time reporting for the CPU Time was incredibly flawed, works on others
        //TODO, find out what caused such a bug.
      
        //as a memory bound process 128 TPB works best for a large number of elements
        //but for smaller sizes 64 or sometimes even 32 TPB is preferred
        avgTime = tTime + cpuTime;
        for(threadCount=32;threadCount<=256;threadCount*=2) 
        {                              
            smemSize =  threadCount* sizeof(float);
            for(numBlocks = 4; (numBlocks)*threadCount < 24576 
               && (numBlocks)*threadCount < x-15; numBlocks += 4)
            {
                currentGPUWin = 0; 
                
	            for(z=0;z<=30;z++)
	            {	
                    CUDA_SAFE_CALL(cudaMemcpy(d_idata, datain, sizeof(float)*(x-15+z), cudaMemcpyHostToDevice));
                    isPow2 = isPowerOfTwo(x-15+z);
                    CUDA_SAFE_CALL( cudaEventRecord(startEvent, 0) );	         
                    for(i=0;i<numIterations;i++)
                    {    
                        CUDA_SAFE_CALL(cudaMemcpy(h_result, d_idata, sizeof(float)*numBlocks, cudaMemcpyDeviceToHost));                        
                        switch(threadCount)
                        {
                           
                            case 32:
                                if(isPow2)
                                    reduce<float, Operator<float, CUDPP_ADD>, 32, true><<<numBlocks, threadCount, smemSize>>>(d_out, d_idata, (x-15+z));
                                else
                                    reduce<float, Operator<float, CUDPP_ADD>, 32, false><<<numBlocks, threadCount, smemSize>>>(d_out, d_idata, (x-15+z));
                                break;
                            case 64:
                                if(isPow2)
                                    reduce<float,Operator<float, CUDPP_ADD>, 64, true><<<numBlocks, threadCount, smemSize>>>(d_out, d_idata, (x-15+z));
                                else
                                    reduce<float, Operator<float, CUDPP_ADD>, 64, false><<<numBlocks, threadCount, smemSize>>>(d_out, d_idata, (x-15+z));
                                break;
                            case 128:
                                if(isPow2)
                                    reduce<float, Operator<float, CUDPP_ADD>, 128, true><<<numBlocks, threadCount, smemSize>>>(d_out, d_idata, (x-15+z));
                                else
                                    reduce<float, Operator<float, CUDPP_ADD>, 128, false><<<numBlocks, threadCount, smemSize>>>(d_out, d_idata, (x-15+z));
                                break;
                            case 256:
                               if(isPow2)
                                    reduce<float, Operator<float, CUDPP_ADD>, 256, true><<<numBlocks, threadCount, smemSize>>>(d_out, d_idata, (x-15+z));
                                else
                                    reduce<float, Operator<float, CUDPP_ADD>, 256, false><<<numBlocks, threadCount, smemSize>>>(d_out, d_idata, (x-15+z));
                                break;                           
                        }
                        CUDA_SAFE_CALL(cudaMemcpy(h_result, d_out, sizeof(float)*numBlocks, cudaMemcpyDeviceToHost));                        

                        gpuResult = 0;
                        for(j = 0;j<numBlocks;j++)
                            gpuResult += h_result[j];
                     }
                     cudaThreadSynchronize();
                     CUDA_SAFE_CALL( cudaEventRecord(stopEvent, 0) );
                     CUDA_SAFE_CALL( cudaEventSynchronize(stopEvent) );
                     
                     
                     CUDA_SAFE_CALL( cudaEventElapsedTime(&gTime, startEvent, stopEvent));

                    if(gTime<avgTime)
                        currentGPUWin +=1;
                     if(gTime<bestTime)
                        bestTime = gTime;
                } 	
	  
                if(currentGPUWin>bestGPUWin)
                {
                    bestGPUWin = currentGPUWin;	  	
                    bestThreadsPerBlock = threadCount;	
                    bestTime = gTime;
                }	  
            }
        }
        jump = float(15-bestGPUWin)*(float(x*.001));
        jump = (abs(jump) >= 1 ? jump : jump/(abs(jump)));
       
#if DEBUG
        printf("(%f, %f)\n", cpuResult, gpuResult);
        printf("testPoint: %d gtime %f, avgTime %f\n", x, bestTime, avgTime);          
        printf("Jump by %f", jump);
        printf("Best GPU Time %f\n", bestTime);
        printf("%d/30 were faster than the CPU runs\n", bestGPUWin);		
        //system("PAUSE");
#endif
  
        //checks to see if we are near the switchpoint (GPU vs CPU)
        if(bestGPUWin == 0 || bestGPUWin == 31)
        {	
           x += int(jump);
        }
        else 
            cont = false;
			
        last5[cLast] = x;
        cLast = (cLast+1)%5;
	        

        switchPoint = x;
    }


    minThreads = bestThreadsPerBlock*numBlocks;
    int smem;	
    threadsperBlock = 128;	
    int bestThreads;


    //find MaxThreads
    float maxTime = FLT_MAX;
    for(maxThreads = 1512; maxThreads < 65536; maxThreads += 1512)
    {
        threadCount = 128;
        smemSize =  threadCount* sizeof(float);
        isPow2 = isPowerOfTwo(maxElements);
        numBlocks = int(ceil(float(maxThreads)/128.));
        if(isPow2)
            reduce<float, Operator<float, CUDPP_ADD>, 128, true> <<<numBlocks, threadCount, smemSize>>>(d_out, d_idata, (maxElements));
        else
            reduce<float, Operator<float, CUDPP_ADD>, 128, false> <<<numBlocks, threadCount, smemSize>>>(d_out, d_idata, (maxElements));  

        CUDA_SAFE_CALL( cudaEventRecord(startEvent, 0) );
        for(i=0;i<numIterations;i++)
        {                
           if(isPow2)
               reduce<float, Operator<float, CUDPP_ADD>, 128, true> <<<numBlocks, threadCount, smemSize>>>(d_out, d_idata, (maxElements));
           else
               reduce<float, Operator<float, CUDPP_ADD>, 128, false> <<<numBlocks, threadCount, smemSize>>>(d_out, d_idata, (maxElements));                  
           CUDA_SAFE_CALL(cudaMemcpy(h_result, d_out, sizeof(float)*numBlocks, cudaMemcpyDeviceToHost));
           for(j = 1;j<numBlocks;j++)
               h_result[0] += h_result[j];
        }
        CUDA_SAFE_CALL( cudaEventRecord(stopEvent, 0) );
        CUDA_SAFE_CALL( cudaEventSynchronize(stopEvent) );
        CUDA_SAFE_CALL( cudaEventElapsedTime(&gTime, startEvent, stopEvent));
        
#if DEBUG
        printf("gTime: %f %d %d\n", gTime, numBlocks, maxThreads);
#endif
        if(gTime < maxTime)
        {
            maxTime = gTime;
            bestThreads = maxThreads;      
        }        
    }
    
    maxThreads = bestThreads;
#if DEBUG
    printf("MaxThreads: %d\n", maxThreads);
    //system("PAUSE");
#endif

    threadsperBlock = 128;
    smem =sizeof(float)*128;
    numIterations = 25;
  
    unsigned int  prevPoint = 0;
    testPoint = maxThreads;
    float maxTest = FLT_MAX;
    float minTest = 0;
    int bestBlocks;
    while(maxTest >= minTest*1.01)
    {
        maxBlocks = maxThreads/threadsperBlock;
        bestBlocks = maxBlocks/2;
        prevPoint = testPoint;
        testPoint = (int(testPoint*float(1.4))>testPoint ? int(testPoint*float(1.4)) : testPoint + 1);
        cpuResult = 0;
        for(x=0;x<testPoint;x++)            
            cpuResult += datain[x];
        minTest = FLT_MAX;
        isPow2 = isPowerOfTwo(testPoint);
        for(numBlocks=bestBlocks;numBlocks<maxBlocks && numBlocks*threadsperBlock < testPoint;numBlocks= numBlocks+4)
        {    
        
            if(isPow2)
                reduce<float, Operator<float, CUDPP_ADD>, 128, true> <<<numBlocks, threadsperBlock, smem>>>(d_out, d_idata, testPoint);	 
            else
                reduce<float, Operator<float, CUDPP_ADD>, 128, false> <<<numBlocks, threadsperBlock, smem>>>(d_out, d_idata, testPoint);	 

            CUDA_SAFE_CALL( cudaEventRecord(startEvent, 0) );
            for(i=0;i<numIterations;i++)
            {                 
                if(isPow2)
                    reduce<float, Operator<float, CUDPP_ADD>, 128, true> <<<numBlocks, threadsperBlock, smem>>>(d_out, d_idata, testPoint); 
                else
                    reduce<float, Operator<float, CUDPP_ADD>, 128, false> <<<numBlocks, threadsperBlock, smem>>>(d_out, d_idata, testPoint); 	 
                CUDA_SAFE_CALL(cudaMemcpy(h_result, d_out, sizeof(float)*numBlocks, cudaMemcpyDeviceToHost));	   	  	  
                for(j=1;j<numBlocks;j++)
                    h_result[0] += h_result[j];
            }	    		
            CUDA_SAFE_CALL( cudaEventRecord(stopEvent, 0) );
            CUDA_SAFE_CALL( cudaEventSynchronize(stopEvent) );
            CUDA_SAFE_CALL( cudaEventElapsedTime(&time1, startEvent, stopEvent));
            if(abs((cpuResult-h_result[0])/cpuResult) > .01)
            {
                printf("%d Elements: Error check failed for %d blocks (%f, %f)\n", testPoint, numBlocks, cpuResult, h_result[0]);
               // system("PAUSE");
            }
            if(time1<minTest)
            {
                minTest = time1;  
                bestBlocks = numBlocks;	        
            }	
            isPow2 = isPowerOfTwo(x);
            if(isPow2)
                reduce<float, Operator<float, CUDPP_ADD>, 128, true><<<maxBlocks, threadsperBlock, smem>>>(d_out, d_idata, x);	 
            else
                reduce<float, Operator<float, CUDPP_ADD>, 128, false><<<maxBlocks, threadsperBlock, smem>>>(d_out, d_idata, x);	 
            CUDA_SAFE_CALL( cudaEventRecord(startEvent, 0) );	  
            for(i=0;i<numIterations;i++)
            {    
                isPow2 = isPowerOfTwo(x - numIterations/2 + 1);
                if(isPow2)
                    reduce<float, Operator<float, CUDPP_ADD>, 128, true><<<maxBlocks, threadsperBlock, smem>>>(d_out, d_idata, x-numIterations/2 + i); //smooth out noisy cases	 
                else 
                    reduce<float, Operator<float, CUDPP_ADD>, 128, false><<<maxBlocks, threadsperBlock, smem>>>(d_out, d_idata, x-numIterations/2 + i); //smooth out noisy cases	 
                CUDA_SAFE_CALL(cudaMemcpy(h_result, d_out, sizeof(float)*numBlocks, cudaMemcpyDeviceToHost));	   	  	  
                for(j=1;j<numBlocks;j++)
                    h_result[0] += h_result[j];
            }	    	
            CUDA_SAFE_CALL( cudaEventRecord(stopEvent, 0) );
            CUDA_SAFE_CALL( cudaEventSynchronize(stopEvent) );
            CUDA_SAFE_CALL( cudaEventElapsedTime(&maxTest, startEvent, stopEvent));
        }
    }

    //save results to an output .h file (will change to a better storage format like JSON)
    printf("Threshold point is in between %d and %d\n", prevPoint, testPoint);
    nTslope = float(2*maxThreads)/float(testPoint+prevPoint);  
    printf("ThreadSlope chosen as %f\n", nTslope);  
    
    fprintf(out,  "%f ", nTslope);
    fprintf(out, "%d ", switchPoint);
    fprintf(out, "%d ", minThreads);
    fprintf(out, "%d ", maxThreads);
    fprintf(out, "%d ", (testPoint+prevPoint)/2);
    fclose(out);
 
    //currently using a single output file to hold parameters without any "markers" -- Will move to a JSON type format.
    //not including CPU-Switchover point. Just a minimum threads point

    
    //benchmark
#if 0
    printf("Tuning Complete. Will now benchmark performance.\n");
    minElements = 100;   
    if(tuneConfig.showResults)
    {
        smem =sizeof(float)*threadsperBlock;
        for(x=minElements;x<maxElements;x*=1.05)
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
                cpuResult = 0;
                for(j=0;j<x;j++)
                    cpuResult += datain[x];
                //for(z=0;z<x;z++)
                // cpuResult += datain[z];
                //printf("Test for %d elements\n", x); //Old way to do it (may revert back as it is mor correct)
                printf("Test on %d\n", x);

              
                reduce<float,Operator<float, CUDPP_ADD>, 128, isPow2><<<numBlocks, threadsperBlock, smem>>>(d_idata, d_out, x);	 //get GPU warmed up.
                cudaThreadSynchronize();
                cutResetTimer(timer);	
                cutStartTimer(timer);	  
                for(i=0;i<numIterations;i++)
                {    
                    reduce<float, Operator<float, CUDPP_ADD>, 128, isPow2><<<numBlocks, threadsperBlock, smem>>>(d_idata, d_out, x-numIterations/2+i);	 
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
        }
    }
#endif 
}







