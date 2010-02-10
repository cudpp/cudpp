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
 * pcr_kernel.cu
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
__global__ void pcrKernel(T *a_d, T *b_d, T *c_d, T *d_d, T *x_d)
{
    int thid = threadIdx.x;
    int blid = blockIdx.x;
    int delta = 1;
    const unsigned int systemSize = blockDim.x;
    int iteration = (int)log2(T(systemSize/2));

    __syncthreads();

    extern __shared__ char shared[];

    T* a = (T*)shared;
    T* b = (T*)&a[systemSize];//
    T* c = (T*)&b[systemSize];
    T* d = (T*)&c[systemSize];
    T* x = (T*)&d[systemSize];

    a[thid] = a_d[thid + blid * systemSize];
    b[thid] = b_d[thid + blid * systemSize];
    c[thid] = c_d[thid + blid * systemSize];
    d[thid] = d_d[thid + blid * systemSize];
  
    T aNew, bNew, cNew, dNew;
  
    __syncthreads();

    //parallel cyclic reduction
    for (int j = 0; j <iteration; j++)
    {
        int i = thid;
        if(i < delta)
        {
            T tmp2 = c[i] / b[i+delta];
            bNew = b[i] - a[i+delta] * tmp2;
            dNew = d[i] - d[i+delta] * tmp2;
            aNew = 0;
            cNew = -c[i+delta] * tmp2;
        }
        else 
        {
            if((systemSize-i-1) < delta)
            {
                T tmp = a[i] / b[i-delta];
                bNew = b[i] - c[i-delta] * tmp;
                dNew = d[i] - d[i-delta] * tmp;
                aNew = -a[i-delta] * tmp;
                cNew = 0;
            }
            else
            {
                T tmp1 = a[i] / b[i-delta];
                T tmp2 = c[i] / b[i+delta];
             bNew = b[i] - c[i-delta] * tmp1 - a[i+delta] * tmp2;
               dNew = d[i] - d[i-delta] * tmp1 - d[i+delta] * tmp2;
               aNew = -a[i-delta] * tmp1;
               cNew = -c[i+delta] * tmp2;
           }
        }

        __syncthreads();

        b[i] = bNew;
        d[i] = dNew;
        a[i] = aNew;
        c[i] = cNew;
    
        delta *=2;
        __syncthreads();
    }

    if (thid < delta)
    {
        int addr1 = thid;
        int addr2 = thid+delta;
        T tmp3 = b[addr2]*b[addr1]-c[addr1]*a[addr2];
        x[addr1] = (b[addr2]*d[addr1]-c[addr1]*d[addr2])/tmp3;
        x[addr2] = (d[addr2]*b[addr1]-d[addr1]*a[addr2])/tmp3;
    }

    __syncthreads();
    x_d[thid + blid * systemSize] = x[thid];
}

template <class T>
__global__ void pcrKernelBranchFree(T *a_d, T *b_d, T *c_d, T *d_d, T *x_d)
{
    int thid = threadIdx.x;
    int blid = blockIdx.x;
    int delta = 1;
    const unsigned int systemSize = blockDim.x;
    int iteration = (int)log2(T(systemSize/2));

    __syncthreads();

    extern __shared__ char shared[];

    T* a = (T*)shared;
    T* b = (T*)&a[systemSize+1];
    T* c = (T*)&b[systemSize+1];
    T* d = (T*)&c[systemSize+1];
    T* x = (T*)&d[systemSize+1];

    a[thid] = a_d[thid + blid * systemSize];
    b[thid] = b_d[thid + blid * systemSize];
    c[thid] = c_d[thid + blid * systemSize];
    d[thid] = d_d[thid + blid * systemSize];
  
    T aNew, bNew, cNew, dNew;
  
    __syncthreads();

    //parallel cyclic reduction
    for (int j = 0; j <iteration; j++)
    {
        int i = thid;

        int iRight = i+delta;
        iRight = iRight%systemSize;

        int iLeft = i-delta;
        iLeft = iLeft%systemSize;

        T tmp1 = a[i] / b[iLeft];
        T tmp2 = c[i] / b[iRight];

        bNew = b[i] - c[iLeft] * tmp1 - a[iRight] * tmp2;
        dNew = d[i] - d[iLeft] * tmp1 - d[iRight] * tmp2;
        aNew = -a[iLeft] * tmp1;
        cNew = -c[iRight] * tmp2;

        __syncthreads();

        b[i] = bNew;
        d[i] = dNew;
        a[i] = aNew;
        c[i] = cNew;

        delta *=2;
        __syncthreads();
    }

    if (thid < delta)
    {
        int addr1 = thid;
        int addr2 = thid+delta;
        T tmp3 = b[addr2]*b[addr1]-c[addr1]*a[addr2];
        x[addr1] = (b[addr2]*d[addr1]-c[addr1]*d[addr2])/tmp3;
        x[addr2] = (d[addr2]*b[addr1]-d[addr1]*a[addr2])/tmp3;
    }

    __syncthreads();

    x_d[thid + blid * systemSize] = x[thid];
}

template <class T>
void pcr(T *a, T *b, T *c, T *d, T *x, int systemSize, int numSystems)
{
    const unsigned int num_threads_block = systemSize;
    const unsigned int memSize = sizeof(T)*numSystems*systemSize;

    // allocate device memory input and output arrays
    T* d_a;
    T* d_b;
    T* d_c;
    T* d_d;
    T* d_x;

    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_a,memSize));
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_b,memSize));
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_c,memSize));
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_d,memSize));
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_x,memSize));

    // copy host memory to device input array
    CUDA_SAFE_CALL( cudaMemcpy( d_a, a,memSize, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL( cudaMemcpy( d_b, b,memSize, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL( cudaMemcpy( d_c, c,memSize, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL( cudaMemcpy( d_d, d,memSize, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL( cudaMemcpy( d_x, x,memSize, cudaMemcpyHostToDevice));

    // setup execution parameters
    dim3  grid(numSystems, 1, 1);
    dim3  threads(num_threads_block, 1, 1);

    pcrKernel<<< grid, threads,(systemSize+1)*5*sizeof(T)>>>(d_a, d_b, d_c, d_d, d_x);
    //pcrKernelBranchFree<<< grid, threads,(systemSize+1)*5*sizeof(T)>>>(d_a, d_b, d_c, d_d, d_x);

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

