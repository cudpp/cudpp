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
__global__ void crpcrKernel(T *a_d, T *b_d, T *c_d, T *d_d, T *x_d)
{
    int thid = threadIdx.x;
    int blid = blockIdx.x;
    int stride = 1;
    int numThreads = blockDim.x;
    const unsigned int systemSize = blockDim.x * 2;
    int iteration = (int)log2(T(systemSize/2));
    int restSystemSize = systemSize/2;
    int restIteration = (int)log2(T(restSystemSize/2));

    __syncthreads();

    extern __shared__ char shared[];
    T* a = (T*)shared;
    T* b = (T*)&a[systemSize+1];
    T* c = (T*)&b[systemSize+1];
    T* d = (T*)&c[systemSize+1];
    T* x = (T*)&d[systemSize+1];

    a[thid] = a_d[thid + blid * systemSize];
    a[thid + blockDim.x] = a_d[thid + blockDim.x + blid * systemSize];

    b[thid] = b_d[thid + blid * systemSize];
    b[thid + blockDim.x] = b_d[thid + blockDim.x + blid * systemSize];

    c[thid] = c_d[thid + blid * systemSize];
    c[thid + blockDim.x] = c_d[thid + blockDim.x + blid * systemSize];

    d[thid] = d_d[thid + blid * systemSize];
    d[thid + blockDim.x] = d_d[thid + blockDim.x + blid * systemSize];

    __syncthreads();

    for (int j = 0; j <(iteration-restIteration); j++)
    {
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
        __syncthreads();    
    }
    
    T* aa = (T*)&x[systemSize+1];
    T* bb = (T*)&aa[restSystemSize];
    T* cc = (T*)&bb[restSystemSize];
    T* dd = (T*)&cc[restSystemSize];
    T* xx = (T*)&dd[restSystemSize];

    if(thid<restSystemSize)
    {
        aa[thid] = a[thid*stride+stride-1];
        bb[thid] = b[thid*stride+stride-1];
        cc[thid] = c[thid*stride+stride-1];
        dd[thid] = d[thid*stride+stride-1];

        T aNew, bNew, cNew, dNew;
        int delta = 1;

        __syncthreads();

        //parallel cyclic reduction
        for (int j = 0; j <restIteration; j++)
        {
            int i = thid;
            if(i < delta)
            {
                T tmp2 = cc[i] / bb[i+delta];
                bNew = bb[i] - aa[i+delta] * tmp2;
                dNew = dd[i] - dd[i+delta] * tmp2;
                aNew = 0;
                cNew = -cc[i+delta] * tmp2;
            }
            else if((restSystemSize-i-1) < delta)
            {
                T tmp = aa[i] / bb[i-delta];
                bNew = bb[i] - cc[i-delta] * tmp;
                dNew = dd[i] - dd[i-delta] * tmp;
                aNew = -aa[i-delta] * tmp;
                cNew = 0;
            }
            else
            {
                T tmp1 = aa[i] / bb[i-delta];
                T tmp2 = cc[i] / bb[i+delta];
                bNew = bb[i] - cc[i-delta] * tmp1 - aa[i+delta] * tmp2;
                dNew = dd[i] - dd[i-delta] * tmp1 - dd[i+delta] * tmp2;
                aNew = -aa[i-delta] * tmp1;
                cNew = -cc[i+delta] * tmp2;
            }
            __syncthreads();

            bb[i] = bNew;
            dd[i] = dNew;
            aa[i] = aNew;
            cc[i] = cNew;
            delta *=2;
            __syncthreads();
        }

        if (thid < delta)
        {
            int addr1 = thid;
            int addr2 = thid+delta;
            T tmp3 = bb[addr2]*bb[addr1]-cc[addr1]*aa[addr2];
            xx[addr1] = (bb[addr2]*dd[addr1]-cc[addr1]*dd[addr2])/tmp3;
            xx[addr2] = (dd[addr2]*bb[addr1]-dd[addr1]*aa[addr2])/tmp3;
        }
    __syncthreads(); 
    x[thid*stride+stride-1]=xx[thid];
    }
  
    //backward substitution
    numThreads = restSystemSize;
    
    for (int j = 0; j <(iteration-restIteration); j++)
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
__global__ void crpcrKernelBranchFree(T *a_d, T *b_d, T *c_d, T *d_d, T *x_d)
{
    int thid = threadIdx.x;
    int blid = blockIdx.x;

    int stride = 1;

    int numThreads = blockDim.x;

    const unsigned int systemSize = blockDim.x * 2;
    int iteration = (int)log2(T(systemSize/2));
    int restSystemSize = systemSize/2;
    int restIteration = (int)log2(T(restSystemSize/2));

    __syncthreads();

    extern __shared__ char shared[];

    T* a = (T*)shared;
    T* b = (T*)&a[systemSize+1];
    T* c = (T*)&b[systemSize+1];
    T* d = (T*)&c[systemSize+1];
    T* x = (T*)&d[systemSize+1];

    a[thid] = a_d[thid + blid * systemSize];
    a[thid + blockDim.x] = a_d[thid + blockDim.x + blid * systemSize];

    b[thid] = b_d[thid + blid * systemSize];
    b[thid + blockDim.x] = b_d[thid + blockDim.x + blid * systemSize];

    c[thid] = c_d[thid + blid * systemSize];
    c[thid + blockDim.x] = c_d[thid + blockDim.x + blid * systemSize];

    d[thid] = d_d[thid + blid * systemSize];
    d[thid + blockDim.x] = d_d[thid + blockDim.x + blid * systemSize];


    __syncthreads();

    for (int j = 0; j <(iteration-restIteration); j++)
    {
        stride *= 2;
        int delta = stride/2;
        if (thid < numThreads)
        { 
            int i = stride * thid + stride - 1;
            int iRight = i+delta;
            iRight = iRight%systemSize;

            T tmp1 = a[i] / b[i-delta];
            T tmp2 = c[i] / b[iRight];

            b[i] = b[i] - c[i-delta] * tmp1 - a[iRight] * tmp2;
            d[i] = d[i] - d[i-delta] * tmp1 - d[iRight] * tmp2;

            a[i] = -a[i-delta] * tmp1;
            c[i] = -c[iRight]  * tmp2;
        }

        numThreads /= 2;
        __syncthreads();    
    }

    T* aa = (T*)&x[systemSize+1];
    T* bb = (T*)&aa[restSystemSize];
    T* cc = (T*)&bb[restSystemSize];
    T* dd = (T*)&cc[restSystemSize];
    T* xx = (T*)&dd[restSystemSize];
  
    if(thid<restSystemSize)
    {
        aa[thid] = a[thid*stride+stride-1];
        bb[thid] = b[thid*stride+stride-1];
        cc[thid] = c[thid*stride+stride-1];
        dd[thid] = d[thid*stride+stride-1];

        T aNew, bNew, cNew, dNew;
        int delta = 1;

        __syncthreads();

        //parallel cyclic reduction
        for (int j = 0; j <restIteration; j++)
        {
            int i = thid;
            if(i < delta)
            {
                T tmp2 = cc[i] / bb[i+delta];
                bNew = bb[i] - aa[i+delta] * tmp2;
                dNew = dd[i] - dd[i+delta] * tmp2;
                aNew = 0;
                cNew = -cc[i+delta] * tmp2;
            }
            else
            {
            if((restSystemSize-i-1) < delta)
            {
                T tmp = aa[i] / bb[i-delta];
                bNew = bb[i] - cc[i-delta] * tmp;
                dNew = dd[i] - dd[i-delta] * tmp;
                aNew = -aa[i-delta] * tmp;
                cNew = 0;
            }
            else
            {
                T tmp1 = aa[i] / bb[i-delta];
                T tmp2 = cc[i] / bb[i+delta];
                bNew = bb[i] - cc[i-delta] * tmp1 - aa[i+delta] * tmp2;
                dNew = dd[i] - dd[i-delta] * tmp1 - dd[i+delta] * tmp2;
                aNew = -aa[i-delta] * tmp1;
                cNew = -cc[i+delta] * tmp2;
            }
            }
            __syncthreads();

            bb[i] = bNew;
            dd[i] = dNew;
            aa[i] = aNew;
            cc[i] = cNew;

            delta *=2;
            __syncthreads();
        }

        if (thid < delta)
        {
            int addr1 = thid;
            int addr2 = thid+delta;
            T tmp3 = bb[addr2]*bb[addr1]-cc[addr1]*aa[addr2];
            xx[addr1] = (bb[addr2]*dd[addr1]-cc[addr1]*dd[addr2])/tmp3;
            xx[addr2] = (dd[addr2]*bb[addr1]-dd[addr1]*aa[addr2])/tmp3;
        }
        __syncthreads(); 
        x[thid*stride+stride-1]=xx[thid];
    }

    //backward substitution
    numThreads = restSystemSize;
    
    for (int j = 0; j <(iteration-restIteration); j++)
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
void crpcr(T *a, T *b, T *c, T *d, T *x, int systemSize, int numSystems)
{
    const unsigned int num_threads_block = systemSize/2;
    int restSystemSize = systemSize/2;
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

    crpcrKernel<<< grid, threads,(systemSize+1)*5*sizeof(T)+restSystemSize*(5+0)*sizeof(float)>>>(d_a, d_b, d_c, d_d, d_x);
    //crpcrKernelBranchFree<<< grid, threads,(systemSize+1)*5*sizeof(float)+restSystemSize*(5+0)*sizeof(float)>>>(d_a, d_b, d_c, d_d, d_x);

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

