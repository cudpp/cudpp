//////////////////////////////////////////////////////////////////////
// cc_util.h
//
// Yangzihao Wang
// yzhwang@ucdavis.edu
//
//////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////
//  Usage:
//  
//	This is needed for most of the suffix array cuda files.
//
//////////////////////////////////////////////////////////////////////
#pragma once

// c library includes
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// CUDA library includes
#include <cuda.h>

// CUDA-5.0 Samples Helper includes
#include <helper_cuda.h>
#include <helper_string.h>

// C++ includes
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>

#define BLOCK_THREADS 16
namespace SA
{

#define DEBUG

#ifndef _SafeDeleteArray
#define _SafeDeleteArray(x) { if(x) { delete [](x); (x)=0; } }
#endif

#ifndef _SafeDelete
#define _SafeDelete(x) { if(x) { delete (x); (x)=0; } }
#endif

#ifndef _in_
#define _in_
#endif

#ifndef _out_
#define _out_
#endif

#ifndef _inout_
#define _inout_
#endif
// vector type of keys
/*struct Vector 
{
    unsigned long long x;
    unsigned long long y;
};*/

struct Vector 
{
    unsigned int a;
    unsigned int b;
    unsigned int c;
    unsigned int d;
};

struct str_tuple
{
    char tuple[3];
    int idx; //starting index in the original string
};

//GpuTimer struct from b40c
struct GpuTimer
{
	cudaEvent_t start;
	cudaEvent_t stop;

	GpuTimer()
	{
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
	}

	~GpuTimer()
	{
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}

	void Start()
	{
		cudaEventRecord(start, 0);
	}

	void Stop()
	{
		cudaEventRecord(stop, 0);
	}

	float ElapsedMillis()
	{
		float elapsed;
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsed, start, stop);
		return elapsed;
	}
};

//Other utility data structures

}
