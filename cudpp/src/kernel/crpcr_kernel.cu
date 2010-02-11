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
 * crpcr_kernel.cu
 *
 * @brief CUDPP kernel-level CR-PCR tridiagonal solver
 */

/** \addtogroup cudpp_kernel
  * @{
  */
/** @name Hybrid CR-PCR solver (CRPCR)
 * @{
 */

/**
 * @brief Hybrid CR-PCR solver (CRPCR)
 *
 * This kernel solves a tridiagonal linear system using a hybrid CR-PCR algorithm.
 * The solver first reduces the system size using
 * cyclic reduction, then solves the intermediate system using parallel cyclic 
 * reduction to reduce shared memory bank conflicts and algorithmic steps, and 
 * finally switch back to cyclic reduction to solve all unknowns.
 *
 * @param[out] d_x Solution vector
 * @param[in] d_a Lower diagonal
 * @param[in] d_b Main diagonal
 * @param[in] d_c Upper diagonal
 * @param[in] d_d Right hand side
 */

template <class T>
__global__ void crpcrKernel(T *d_a, T *d_b, T *d_c, T *d_d, T *d_x)
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

    a[thid] = d_a[thid + blid * systemSize];
    a[thid + blockDim.x] = d_a[thid + blockDim.x + blid * systemSize];

    b[thid] = d_b[thid + blid * systemSize];
    b[thid + blockDim.x] = d_b[thid + blockDim.x + blid * systemSize];

    c[thid] = d_c[thid + blid * systemSize];
    c[thid + blockDim.x] = d_c[thid + blockDim.x + blid * systemSize];

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

    d_x[thid + blid * systemSize] = x[thid];
    d_x[thid + blockDim.x + blid * systemSize] = x[thid + blockDim.x];
}

template <class T>
__global__ void crpcrKernelBranchFree(T *d_a, T *d_b, T *d_c, T *d_d, T *d_x)
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

    a[thid] = d_a[thid + blid * systemSize];
    a[thid + blockDim.x] = d_a[thid + blockDim.x + blid * systemSize];

    b[thid] = d_b[thid + blid * systemSize];
    b[thid + blockDim.x] = d_b[thid + blockDim.x + blid * systemSize];

    c[thid] = d_c[thid + blid * systemSize];
    c[thid + blockDim.x] = d_c[thid + blockDim.x + blid * systemSize];

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

    d_x[thid + blid * systemSize] = x[thid];
    d_x[thid + blockDim.x + blid * systemSize] = x[thid + blockDim.x];
}

/** @} */ // end crpcr functions
/** @} */ // end cudpp_kernel

