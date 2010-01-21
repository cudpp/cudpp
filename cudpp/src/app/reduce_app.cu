//For future use to add device functions and templated structs, etc.
//#include <stdlib.h>
#include "cudpp.h"
#include "cudpp_util.h"
#include "cudpp_plan.h"
#include "cudpp_tune.h"
#include "reduce_kernel.cu"
#include "cudpp_reduce.h"
#include <math.h>


//TODO: Templates for operand and type
void reduce(int                   *d_out, 
            const int             *d_in,                    
            size_t              numElements,                        
            CUDPPTuneReduce           tuneConfig)
{

    int traitsCode = 0x000F;
    int nThreads, nBlocks;

    if(tuneConfig == NULL || tuneConfig.tuneEnabled == false) //default
    {
        nThreads = 128;
        nBlocks = 128;
    }
    else if(tuneConfig.tuned == true)
    {
   //     if(numElements > tuneConfig.threshold)
     //   {
     //   }
     //   else
      //  {
     //   }
        nThreads = 128;
        nBlocks = 128;

    }
   else (tuneConfig.tuneEnabled == true)
    {
        tuneReduce(tuneConfig);        
    }


    gridreduce<<<nThreads, nBlocks>>>(din, dout, numElements);    

}

