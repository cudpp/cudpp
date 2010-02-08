// -------------------------------------------------------------
// cuDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision$
// $Date$
// ------------------------------------------------------------- 
// This source code is distributed under the terms of license.txt
// in the root directory of this source distribution.
// ------------------------------------------------------------- 

#ifndef __REDUCE_GOLD_H__
#define __REDUCE_GOLD_H__

#include <cudpp.h>
#include <stdio.h>
#include <limits.h>
#include <float.h>
#include <algorithm>
#include "cudpp_testrig_utils.h"

template <class Oper, typename T>
void computeReduceGold( T* out, const T* idata, const unsigned int len)
{
    Oper op;
    T sum = op.identity();
    
    for (unsigned int i = 0; i < len; i++)
    {
        sum = op(sum, idata[i]);
    }
    *out = sum;
}

#endif // __REDUCE_GOLD_H__