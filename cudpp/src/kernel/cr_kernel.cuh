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
 * @brief CUDPP kernel-level CR tridiagonal solver
 */

/** \addtogroup cudpp_kernel
  * @{
  */
/** @name Cyclic reduction solver (CR)
 * @{
 */

/**
 * @brief Cyclic reduction solver (CR)
 *
 * This kernel solves a tridiagonal linear system using the CR algorithm.
 *
 * @param[out] d_x Solution vector
 * @param[in] d_a Lower diagonal
 * @param[in] d_b Main diagonal
 * @param[in] d_c Upper diagonal
 * @param[in] d_d Right hand side
 */

template <class T>
__global__ void crKernel(T *d_a, T *d_b, T *d_c, T *d_d, T *d_x)
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

    a[thid] = d_a[thid + blid * systemSize];
    a[thid + blockDim.x] = d_a[thid + blockDim.x + blid * systemSize];

    b[thid] = d_b[thid + blid * systemSize];
    b[thid + blockDim.x] = d_b[thid + blockDim.x + blid * systemSize];

    c[thid] = d_c[thid + blid * systemSize];
    c[thid + blockDim.x] = d_c[thid + blockDim.x + blid * systemSize];

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

    d_x[thid + blid * systemSize] = x[thid];
    d_x[thid + blockDim.x + blid * systemSize] = x[thid + blockDim.x];
}

/** @} */ // end tridiagonal functions
/** @} */ // end cudpp_kernel

