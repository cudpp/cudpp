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
__global__ void pcr_small_systems_kernel(T *a_d, T *b_d, T *c_d, T *d_d, T *x_d)
{
    int thid = threadIdx.x;
    int blid = blockIdx.x;
    int delta = 1;
    const unsigned int system_size = blockDim.x;
    int Iteration = (int)log2(T(system_size/2));

    __syncthreads();

    extern __shared__ char shared[];

    T* a = (T*)shared;
    T* b = (T*)&a[system_size];//
    T* c = (T*)&b[system_size];
    T* d = (T*)&c[system_size];
    T* x = (T*)&d[system_size];

    a[thid] = a_d[thid + blid * system_size];
    b[thid] = b_d[thid + blid * system_size];
    c[thid] = c_d[thid + blid * system_size];
    d[thid] = d_d[thid + blid * system_size];
  
    T aNew, bNew, cNew, dNew;
  
    __syncthreads();

    //parallel cyclic reduction
    for (int j = 0; j <Iteration; j++)
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
            if((system_size-i-1) < delta)
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
    x_d[thid + blid * system_size] = x[thid];
}

template <class T>
__global__ void pcr_small_systems_kernel_branch_free(T *a_d, T *b_d, T *c_d, T *d_d, T *x_d)
{
    int thid = threadIdx.x;
    int blid = blockIdx.x;
    int delta = 1;
    const unsigned int system_size = blockDim.x;
    int Iteration = (int)log2(T(system_size/2));

    __syncthreads();

    extern __shared__ char shared[];

    T* a = (T*)shared;
    T* b = (T*)&a[system_size+1];
    T* c = (T*)&b[system_size+1];
    T* d = (T*)&c[system_size+1];
    T* x = (T*)&d[system_size+1];

    a[thid] = a_d[thid + blid * system_size];
    b[thid] = b_d[thid + blid * system_size];
    c[thid] = c_d[thid + blid * system_size];
    d[thid] = d_d[thid + blid * system_size];
  
    T aNew, bNew, cNew, dNew;
  
    __syncthreads();

    //parallel cyclic reduction
    for (int j = 0; j <Iteration; j++)
    {
        int i = thid;

        int iRight = i+delta;
        iRight = iRight%system_size;

        int iLeft = i-delta;
        iLeft = iLeft%system_size;

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

    x_d[thid + blid * system_size] = x[thid];
}

template <class T>
void pcr_small_systems(T *a, T *b, T *c, T *d, T *x, int system_size, int num_systems)
{
    const unsigned int num_threads_block = system_size;
    const unsigned int mem_size = sizeof(T)*num_systems*system_size;

    // allocate device memory input and output arrays
    T* device_a;
    T* device_b;
    T* device_c;
    T* device_d;
    T* device_x;

    CUDA_SAFE_CALL( cudaMalloc( (void**) &device_a,mem_size));
    CUDA_SAFE_CALL( cudaMalloc( (void**) &device_b,mem_size));
    CUDA_SAFE_CALL( cudaMalloc( (void**) &device_c,mem_size));
    CUDA_SAFE_CALL( cudaMalloc( (void**) &device_d,mem_size));
    CUDA_SAFE_CALL( cudaMalloc( (void**) &device_x,mem_size));

    // copy host memory to device input array
    CUDA_SAFE_CALL( cudaMemcpy( device_a, a,mem_size, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL( cudaMemcpy( device_b, b,mem_size, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL( cudaMemcpy( device_c, c,mem_size, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL( cudaMemcpy( device_d, d,mem_size, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL( cudaMemcpy( device_x, x,mem_size, cudaMemcpyHostToDevice));

    // setup execution parameters
    dim3  grid(num_systems, 1, 1);
    dim3  threads(num_threads_block, 1, 1);

    //pcr_small_systems_kernel<<< grid, threads,(system_size+1)*5*sizeof(T)>>>(device_a, device_b, device_c, device_d, device_x);
    pcr_small_systems_kernel_branch_free<<< grid, threads,(system_size+1)*5*sizeof(T)>>>(device_a, device_b, device_c, device_d, device_x);

    // copy result from device to host
    CUDA_SAFE_CALL( cudaMemcpy(x, device_x,mem_size, cudaMemcpyDeviceToHost));

    // cleanup memory
    CUDA_SAFE_CALL(cudaFree(device_a));
    CUDA_SAFE_CALL(cudaFree(device_b));
    CUDA_SAFE_CALL(cudaFree(device_c));
    CUDA_SAFE_CALL(cudaFree(device_d));
    CUDA_SAFE_CALL(cudaFree(device_x));
}

/** @} */ // end tridiagonal functions
/** @} */ // end cudpp_kernel

