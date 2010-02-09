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
__global__ void crpcr_small_systems_kernel(T *a_d, T *b_d, T *c_d, T *d_d, T *x_d)
{
    int thid = threadIdx.x;
    int blid = blockIdx.x;
    int stride = 1;
    int thid_num = blockDim.x;
    const unsigned int system_size = blockDim.x * 2;
    int Iteration = (int)log2(T(system_size/2));
    int rest_system_size = system_size/2;
    int restIteration = (int)log2(T(rest_system_size/2));

    __syncthreads();

    extern __shared__ char shared[];
    T* a = (T*)shared;
    T* b = (T*)&a[system_size+1];
    T* c = (T*)&b[system_size+1];
    T* d = (T*)&c[system_size+1];
    T* x = (T*)&d[system_size+1];

    a[thid] = a_d[thid + blid * system_size];
    a[thid + blockDim.x] = a_d[thid + blockDim.x + blid * system_size];

    b[thid] = b_d[thid + blid * system_size];
    b[thid + blockDim.x] = b_d[thid + blockDim.x + blid * system_size];

    c[thid] = c_d[thid + blid * system_size];
    c[thid + blockDim.x] = c_d[thid + blockDim.x + blid * system_size];

    d[thid] = d_d[thid + blid * system_size];
    d[thid + blockDim.x] = d_d[thid + blockDim.x + blid * system_size];

    __syncthreads();

    for (int j = 0; j <(Iteration-restIteration); j++)
    {
        stride *= 2;
        int delta = stride/2;
        if (thid < thid_num)
        {
            int i = stride * thid + stride - 1;
            if(i == system_size - 1)
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
        thid_num /= 2;
        __syncthreads();    
    }
    
    T* aa = (T*)&x[system_size+1];
    T* bb = (T*)&aa[rest_system_size];
    T* cc = (T*)&bb[rest_system_size];
    T* dd = (T*)&cc[rest_system_size];
    T* xx = (T*)&dd[rest_system_size];

    if(thid<rest_system_size)
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
            else if((rest_system_size-i-1) < delta)
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
    thid_num = rest_system_size;
    
    for (int j = 0; j <(Iteration-restIteration); j++)
    {
        int delta = stride/2;
        __syncthreads();
        if (thid < thid_num)
        {
            int i = stride * thid + stride/2 - 1;
            if(i == delta - 1)
            x[i] = (d[i] - c[i]*x[i+delta])/b[i];
            else
            x[i] = (d[i] - a[i]*x[i-delta] - c[i]*x[i+delta])/b[i];
        }
        stride /= 2;
        thid_num *= 2;
    }

    __syncthreads();    

    x_d[thid + blid * system_size] = x[thid];
    x_d[thid + blockDim.x + blid * system_size] = x[thid + blockDim.x];
}

template <class T>
__global__ void crpcr_small_systems_kernel_branch_free(T *a_d, T *b_d, T *c_d, T *d_d, T *x_d)
{
    int thid = threadIdx.x;
    int blid = blockIdx.x;

    int stride = 1;

    int thid_num = blockDim.x;

    const unsigned int system_size = blockDim.x * 2;
    int Iteration = (int)log2(T(system_size/2));
    int rest_system_size = system_size/2;
    int restIteration = (int)log2(T(rest_system_size/2));

    __syncthreads();

    extern __shared__ char shared[];

    T* a = (T*)shared;
    T* b = (T*)&a[system_size+1];
    T* c = (T*)&b[system_size+1];
    T* d = (T*)&c[system_size+1];
    T* x = (T*)&d[system_size+1];

    a[thid] = a_d[thid + blid * system_size];
    a[thid + blockDim.x] = a_d[thid + blockDim.x + blid * system_size];

    b[thid] = b_d[thid + blid * system_size];
    b[thid + blockDim.x] = b_d[thid + blockDim.x + blid * system_size];

    c[thid] = c_d[thid + blid * system_size];
    c[thid + blockDim.x] = c_d[thid + blockDim.x + blid * system_size];

    d[thid] = d_d[thid + blid * system_size];
    d[thid + blockDim.x] = d_d[thid + blockDim.x + blid * system_size];


    __syncthreads();

    for (int j = 0; j <(Iteration-restIteration); j++)
    {
        stride *= 2;
        int delta = stride/2;
        if (thid < thid_num)
        { 
            int i = stride * thid + stride - 1;
            int iRight = i+delta;
            iRight = iRight%system_size;

            T tmp1 = a[i] / b[i-delta];
            T tmp2 = c[i] / b[iRight];

            b[i] = b[i] - c[i-delta] * tmp1 - a[iRight] * tmp2;
            d[i] = d[i] - d[i-delta] * tmp1 - d[iRight] * tmp2;

            a[i] = -a[i-delta] * tmp1;
            c[i] = -c[iRight]  * tmp2;
        }

        thid_num /= 2;
        __syncthreads();    
    }

    T* aa = (T*)&x[system_size+1];
    T* bb = (T*)&aa[rest_system_size];
    T* cc = (T*)&bb[rest_system_size];
    T* dd = (T*)&cc[rest_system_size];
    T* xx = (T*)&dd[rest_system_size];
  
    if(thid<rest_system_size)
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
            if((rest_system_size-i-1) < delta)
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
    thid_num = rest_system_size;
    
    for (int j = 0; j <(Iteration-restIteration); j++)
    {
        int delta = stride/2;
        __syncthreads();
        if (thid < thid_num)
        {
            int i = stride * thid + stride/2 - 1;
            if(i == delta - 1)
            x[i] = (d[i] - c[i]*x[i+delta])/b[i];
            else
            x[i] = (d[i] - a[i]*x[i-delta] - c[i]*x[i+delta])/b[i];
        }
        stride /= 2;
        thid_num *= 2;
    }

    __syncthreads();

    x_d[thid + blid * system_size] = x[thid];
    x_d[thid + blockDim.x + blid * system_size] = x[thid + blockDim.x];
}

template <class T>
void crpcr_small_systems(T *a, T *b, T *c, T *d, T *x, int system_size, int num_systems)
{
    const unsigned int num_threads_block = system_size/2;
    int rest_system_size = system_size/2;
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

    crpcr_small_systems_kernel<<< grid, threads,(system_size+1)*5*sizeof(T)+rest_system_size*(5+0)*sizeof(float)>>>(device_a, device_b, device_c, device_d, device_x);
    //crpcr_small_systems_kernel_branch_free<<< grid, threads,(system_size+1)*5*sizeof(float)+rest_system_size*(5+0)*sizeof(float)>>>(device_a, device_b, device_c, device_d, device_x);

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

