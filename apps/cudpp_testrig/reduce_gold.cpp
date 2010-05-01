// -------------------------------------------------------------
// cuDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision$
// $Date$
// ------------------------------------------------------------- 
// This source code is distributed under the terms of license.txt
// in the root directory of this source distribution.
// ------------------------------------------------------------- 

#include <cudpp.h>
#include <stdio.h>
#include <limits.h>
#include <float.h>
#include <algorithm>

extern "C" 
void computeSumReduceGold( float &out, const float* idata, const unsigned int len);

extern "C" 
void computeMultiplyReduceGold( float &out, const float* idata, const unsigned int len);

extern "C" 
void computeMaxReduceGold( float &out, const float* idata, const unsigned int len);

extern "C" 
void computeMinReduceGold( float &out, const float* idata, const unsigned int len);


void computeSumReduceGold( float &out, const float* idata, const unsigned int len)
{   
    float sum = 0;
    
    for (unsigned int i = 0; i < len; i++)
    {
        sum = sum + idata[i];
    }
    out = sum;
}


void computeMultiplyReduceGold( float &out, const float* idata, const unsigned int len)
{
    
    float prod = 1;
    
    for (unsigned int i = 0; i < len; i++)
    {
        prod = prod *idata[i];
    }
    out = prod;
}



void computeMaxReduceGold( float &out, const float* idata, const unsigned int len)
{
 
    float maxi = FLT_MIN;
    
    for (unsigned int i = 0; i < len; i++)
    {
        maxi = std::max(maxi, idata[i]);
    }
    out = maxi;
}



void computeMinReduceGold( float &out, const float* idata, const unsigned int len)
{ 
    float mini = FLT_MAX;
    
    for (unsigned int i = 0; i < len; i++)
    {
        mini = std::min(mini,idata[i]);
    }
    out = mini;
}

