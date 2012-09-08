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

////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set for list ranking
//! Each element is the sum of the elements before it in the array.
//! @param reference  reference data, computed but preallocated
//! @param idata      const input data as provided to device
//! @param len        number of elements in reference / idata
////////////////////////////////////////////////////////////////////////////////

void listRankGold( int* reference, const int* ivalues, 
                  const int* inextindices, const unsigned int head,
                  const unsigned int count) 
{
    int cur_id = head;
    for(unsigned int i=0; i<count; i++){
        reference[i] = ivalues[cur_id];
        cur_id = inextindices[cur_id];
    }
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End: