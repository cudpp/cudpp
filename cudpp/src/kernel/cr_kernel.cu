// -------------------------------------------------------------
// cuDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
//  $Revision:
//  $Date:
// ------------------------------------------------------------- 
// This source code is distributed under the terms of license.txt 
// in the root directory of this source distribution.
// ------------------------------------------------------------- 

/**
 * @file
 * cr_kernel.cu
 *
 * @brief CUDPP kernel-level tridiagonal routines
 */

/** \addtogroup cudpp_kernel
  * @{
  */
/** @name tridiagonal Functions
 * @{
 */

template <class T>
__global__ void crKernel(T *a_d, T *b_d, T *c_d, T *d_d, T *x_d)
{
    int thid = threadIdx.x;
    int blid = blockIdx.x;

    int stride = 1;

    int numThreads = blockDim.x;
    const unsigned int systemSize = blockDim.x * 2;
   
    int iteration = (int)log2(T(systemSize/2));

    __syncthreads();

    extern __shared__ char shared[];

    T* a = (T*)shared;
    T* b = (T*)&a[systemSize];
    T* c = (T*)&b[systemSize];
    T* d = (T*)&c[systemSize];
    T* x = (T*)&d[systemSize];

    a[thid] = a_d[thid + blid * systemSize];
    a[thid + blockDim.x] = a_d[thid + blockDim.x + blid * systemSize];

    b[thid] = b_d[thid + blid * systemSize];
    b[thid + blockDim.x] = b_d[thid + blockDim.x + blid * systemSize];

    c[thid] = c_d[thid + blid * systemSize];
    c[thid + blockDim.x] = c_d[thid + blockDim.x + blid * systemSize];

    d[thid] = d_d[thid + blid * systemSize];
    d[thid + blockDim.x] = d_d[thid + blockDim.x + blid * systemSize];

    __syncthreads();


    //forward elimination
    for (int j = 0; j <iteration; j++)
    {
        __syncthreads();
        stride *= 2;
        int delta = stride/2;
        if (thid < numThreads)
        {
            int i = stride * thid + stride - 1;

            if(i == systemSize - 1)
            {
                T tmp = a[i] / b[i-delta];
                b[i] = b[i] - c[i-delta] * tmp;
                d[i] = d[i] - d[i-delta] * tmp;
                a[i] = -a[i-delta] * tmp;
                c[i] = 0;
            }
            else
            {
                T tmp1 = a[i] / b[i-delta];
                T tmp2 = c[i] / b[i+delta];
                b[i] = b[i] - c[i-delta] * tmp1 - a[i+delta] * tmp2;
                d[i] = d[i] - d[i-delta] * tmp1 - d[i+delta] * tmp2;
                a[i] = -a[i-delta] * tmp1;
                c[i] = -c[i+delta] * tmp2;
            }
        }
        numThreads /= 2;
    }

    if (thid < 2)
    {
      int addr1 = stride - 1;
      int addr2 = 2 * stride - 1;
      T tmp3 = b[addr2]*b[addr1]-c[addr1]*a[addr2];
      x[addr1] = (b[addr2]*d[addr1]-c[addr1]*d[addr2])/tmp3;
      x[addr2] = (d[addr2]*b[addr1]-d[addr1]*a[addr2])/tmp3;
    }

    //backward substitution
    numThreads = 2;
    for (int j = 0; j <iteration; j++)
    {
        int delta = stride/2;
        __syncthreads();
        if (thid < numThreads)
        {
            int i = stride * thid + stride/2 - 1;
            if(i == delta - 1)
                  x[i] = (d[i] - c[i]*x[i+delta])/b[i];
            else
                  x[i] = (d[i] - a[i]*x[i-delta] - c[i]*x[i+delta])/b[i];
         }
         stride /= 2;
         numThreads *= 2;
      }

    __syncthreads();

    x_d[thid + blid * systemSize] = x[thid];
    x_d[thid + blockDim.x + blid * systemSize] = x[thid + blockDim.x];

}

template <class T>
void cr(T *a, T *b, T *c, T *d, T *x, int systemSize, int numSystems)
{
    const unsigned int num_threads_block = systemSize/2;
    const unsigned int memSize = sizeof(T)*numSystems*systemSize;

    // allocate device memory input and output arrays
    T* d_a;
    T* d_b;
    T* d_c;
    T* d_d;
    T* d_x;

    unsigned int timer;
    CUT_SAFE_CALL(cutCreateTimer(&timer));
    cutStartTimer(timer);

    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_a,memSize));
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_b,memSize));
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_c,memSize));
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_d,memSize));
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_x,memSize));

    cutStopTimer(timer);
    printf("GPU cudaMalloc time: %f ms\n", cutGetTimerValue(timer));

   // copy host memory to device input array
    CUDA_SAFE_CALL( cudaMemcpy( d_a, a,memSize, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL( cudaMemcpy( d_b, b,memSize, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL( cudaMemcpy( d_c, c,memSize, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL( cudaMemcpy( d_d, d,memSize, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL( cudaMemcpy( d_x, x,memSize, cudaMemcpyHostToDevice));

    // setup execution parameters
    dim3  grid(numSystems, 1, 1);
    dim3  threads(num_threads_block, 1, 1);

    crKernel<<< grid, threads,systemSize*5*sizeof(T)>>>(d_a, d_b, d_c, d_d, d_x);
    //cudaThreadSynchronize();

    // copy result from device to host
    CUDA_SAFE_CALL( cudaMemcpy(x, d_x,memSize, cudaMemcpyDeviceToHost));

    // cleanup memory
    CUDA_SAFE_CALL(cudaFree(d_a));
    CUDA_SAFE_CALL(cudaFree(d_b));
    CUDA_SAFE_CALL(cudaFree(d_c));
    CUDA_SAFE_CALL(cudaFree(d_d));
    CUDA_SAFE_CALL(cudaFree(d_x));
}
/** @} */ // end tridiagonal functions
/** @} */ // end cudpp_kernel

