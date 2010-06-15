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
	T sum = 0;//op.identity();
    
    for (unsigned int i = 0; i < len; i++)
    {
        sum = op(sum, idata[i]);
    }
    *out = sum;
}