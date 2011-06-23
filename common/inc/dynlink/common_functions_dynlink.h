/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA) 
 * associated with this source code for terms and conditions that govern 
 * your use of this NVIDIA software.
 * 
 */

#if !defined(__COMMON_FUNCTIONS_H__)
#define __COMMON_FUNCTIONS_H__

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#if defined(__cplusplus) && defined(__CUDACC__)

#include "host_defines.h"

#include <time.h>
#include <string.h>

extern "C"
{

/*DEVICE_BUILTIN*/
extern __host__ __device__ clock_t clock(void) __THROW;

/*DEVICE_BUILTIN*/
extern __host__ __device__ void *memset(void *s, int c, size_t n) __THROW;

/*DEVICE_BUILTIN*/
extern __host__ __device__ void *memcpy(void *d, const void *s, size_t n) __THROW;

}

#elif !defined(__CUDACC__)

#include "crt/func_macro.h"

__device_func__(clock_t __cuda_clock(void))
{
  return clock();
}

__device_func__(void *__cuda_memset(void *s, int c, size_t n))
{
  return memset(s, c, n);
}

__device_func__(void *__cuda_memcpy(void *d, const void *s, size_t n))
{
  return memcpy(d, s, n);
}

#endif /* __cplusplus && __CUDACC__ */

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#include "math_functions_dynlink.h"

#endif /* !__COMMON_FUNCTIONS_H__ */

