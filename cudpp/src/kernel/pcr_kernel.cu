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
 * @brief CUDPP kernel-level PCR tridiagonal solver
 */

/** \addtogroup cudpp_kernel
  * @{
  */
/** @name  Parallel cyclic reduction solver (PCR)
 * @{
 */

/**
 * @brief Parallel cyclic reduction solver (PCR)
 *
 * This kernel solves a tridiagonal linear system using the PCR algorithm.
 *
 * @param[out] d_x Solution vector
 * @param[in] d_a Lower diagonal
 * @param[in] d_b Main diagonal
 * @param[in] d_c Upper diagonal
 * @param[in] d_d Right hand side
 */

template <class T>
__global__ void pcrKernel(T *d_a, T *d_b, T *d_c, T *d_d, T *d_x)
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

    a[thid] = d_a[thid + blid * systemSize];
    b[thid] = d_b[thid + blid * systemSize];
    c[thid] = d_c[thid + blid * systemSize];
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
    d_x[thid + blid * systemSize] = x[thid];
}

template <class T>
__global__ void pcrKernelBranchFree(T *d_a, T *d_b, T *d_c, T *d_d, T *d_x)
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

    a[thid] = d_a[thid + blid * systemSize];
    b[thid] = d_b[thid + blid * systemSize];
    c[thid] = d_c[thid + blid * systemSize];
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

    d_x[thid + blid * systemSize] = x[thid];
}

/** @} */ // end tridiagonal functions
/** @} */ // end cudpp_kernel

