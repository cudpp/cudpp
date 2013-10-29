#include "sa_util.h"

__global__ void AssignSample( _in_ char *str, _out_ int* sample_set, _in_ size_t str_length )
{
    __shared__ char s_str[BLOCK_THREADS];
    __shared__ int s_set[BLOCK_THREADS];

    //copy to shared mem, the last char of the block
    //overlaps with the next block
    int start_idx = blockIdx.x * (BLOCK_THREADS-1);
    int end_idx = ((blockIdx.x+1) * BLOCK_THREADS < str_length) ? (blockIdx.x+1)*BLOCK_THREADS : str_length;
    if (start_idx + threadIdx.x < end_idx)
        s_str[threadIdx.x] = str[start_idx+threadIdx.x];

    s_set[threadIdx.x] = -1;
    __syncthreads();

    if (threadIdx.x < end_idx-1)
    {
        if (s_str[threadIdx.x] < s_str[threadIdx.x+1])
            s_set[threadIdx.x] = 1; //U set
        else 
        {
            if (s_str[threadIdx.x] > s_str[threadIdx.x+1])
        {
            s_set[threadIdx.x] = 2; //V set
        }
        else
            s_set[threadIdx.x] = 0;
        }
    }
    __syncthreads();
    int region = 0;
    while (s_set[threadIdx.x+region] == 0)
    {
        region++;
    }
    region-=1;
    for ( int i = 0; i < region; ++i )
    {
        s_set[threadIdx.x+i] = s_set[threadIdx.x+region];
    }
    __syncthreads();

    //copy back to global mem
    if (threadIdx.x == end_idx)
    {
        for (int i = 0; i < end_idx-1; ++i)
        {
            sample_set[start_idx+i] = s_set[i];
        }
        if (end_idx == str_length)
            sample_set[end_idx-1] = 2;
    }

    return;
}

__global__ void AssignSample2( _in_ char *str, _out_ int* sample_set, _in_ size_t str_length )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < str_length-1)
    {
        if (str[idx] < str[idx+1])
            sample_set[idx] = 1;
        else
        {
            if (str[idx] > str[idx+1])
                sample_set[idx] = 2;
            else
                sample_set[idx] = 0;
        }
    }
    else
    {
        if (idx == str_length-1)
            sample_set[idx] = 2;
        else
            ;
    }
    return;
}
