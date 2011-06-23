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

#if !defined(__DEVICE_FUNCTIONS_H__)
#define __DEVICE_FUNCTIONS_H__

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#if defined(__cplusplus) && defined(__CUDACC__)

#include "device_types.h"
#include "host_defines.h"

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

extern "C"
{

/*DEVICE_BUILTIN*/
extern __device__ int                    __mulhi(int, int);
/*DEVICE_BUILTIN*/
extern __device__ unsigned int           __umulhi(unsigned int, unsigned int);

/*DEVICE_BUILTIN*/
extern __device__ long long int          __mul64hi(long long int, long long int);
/*DEVICE_BUILTIN*/
extern __device__ unsigned long long int __umul64hi(unsigned long long int, unsigned long long int);

/*DEVICE_BUILTIN*/
extern __device__ float                  __int_as_float(int);
/*DEVICE_BUILTIN*/
extern __device__ int                    __float_as_int(float);

/*DEVICE_BUILTIN*/
extern __device__ void                   __syncthreads(void);
/*DEVICE_BUILTIN*/
extern __device__ void                   __trap(void);
/*DEVICE_BUILTIN*/
extern __device__ void                   __brkpt(int);

/*DEVICE_BUILTIN*/
extern __device__ float                  __saturatef(float);

/*DEVICE_BUILTIN*/
extern __device__ unsigned int           __sad(int, int, unsigned int);
/*DEVICE_BUILTIN*/
extern __device__ unsigned int           __usad(unsigned int, unsigned int, unsigned int);

/*DEVICE_BUILTIN*/
extern __device__ int                    __mul24(int, int);
/*DEVICE_BUILTIN*/
extern __device__ unsigned int           __umul24(unsigned int, unsigned int);

/*DEVICE_BUILTIN*/
extern __device__ float                  fdividef(float, float);
/*DEVICE_BUILTIN*/
extern __device__ float                  __fdividef(float, float);

/*DEVICE_BUILTIN*/
extern __device__ double                 fdivide(double, double);

/*DEVICE_BUILTIN*/
extern __device__ float                  __sinf(float) __THROW;

/*DEVICE_BUILTIN*/
extern __device__ float                  __cosf(float) __THROW;

/*DEVICE_BUILTIN*/
extern __device__ float                  __tanf(float) __THROW;

/*DEVICE_BUILTIN*/
extern __device__ void                   __sincosf(float, float*, float*) __THROW;

/*DEVICE_BUILTIN*/
extern __device__ float                  __expf(float) __THROW;

/*DEVICE_BUILTIN*/
extern __device__ float                  __exp10f(float) __THROW;

/*DEVICE_BUILTIN*/
extern __device__ float                  __log2f(float) __THROW;

/*DEVICE_BUILTIN*/
extern __device__ float                  __log10f(float) __THROW;

/*DEVICE_BUILTIN*/
extern __device__ float                  __logf(float) __THROW;

/*DEVICE_BUILTIN*/
extern __device__ float                  __powf(float, float) __THROW;

/*DEVICE_BUILTIN*/
extern __device__ int                    __float2int_rz(float);
/*DEVICE_BUILTIN*/
extern __device__ int                    __float2int_ru(float);
/*DEVICE_BUILTIN*/
extern __device__ int                    __float2int_rd(float);
/*DEVICE_BUILTIN*/
extern __device__ int                    __float2int_rn(float);

/*DEVICE_BUILTIN*/
extern __device__ unsigned int           __float2uint_rz(float);
/*DEVICE_BUILTIN*/
extern __device__ unsigned int           __float2uint_ru(float);
/*DEVICE_BUILTIN*/
extern __device__ unsigned int           __float2uint_rd(float);
/*DEVICE_BUILTIN*/
extern __device__ unsigned int           __float2uint_rn(float);

/*DEVICE_BUILTIN*/
extern __device__ float                  __int2float_rz(int);
/*DEVICE_BUILTIN*/
extern __device__ float                  __int2float_ru(int);
/*DEVICE_BUILTIN*/
extern __device__ float                  __int2float_rd(int);
/*DEVICE_BUILTIN*/
extern __device__ float                  __int2float_rn(int);

/*DEVICE_BUILTIN*/
extern __device__ float                  __uint2float_rz(unsigned int);
/*DEVICE_BUILTIN*/
extern __device__ float                  __uint2float_ru(unsigned int);
/*DEVICE_BUILTIN*/
extern __device__ float                  __uint2float_rd(unsigned int);
/*DEVICE_BUILTIN*/
extern __device__ float                  __uint2float_rn(unsigned int);

/*DEVICE_BUILTIN*/
extern __device__ long long int          __float2ll_rz(float);
/*DEVICE_BUILTIN*/
extern __device__ long long int          __float2ll_rn(float);

/*DEVICE_BUILTIN*/
extern __device__ unsigned long long int __float2ull_rz(float);

/*DEVICE_BUILTIN*/
extern __device__ float                  __ll2float_rn(long long int);

/*DEVICE_BUILTIN*/
extern __device__ float                  __ull2float_rn(unsigned long long int);

/*DEVICE_BUILTIN*/
extern __device__ float                  __fadd_rz(float, float);
/*DEVICE_BUILTIN*/
extern __device__ float                  __fmul_rz(float, float);
/*DEVICE_BUILTIN*/
extern __device__ float                  __fadd_rn(float, float);
/*DEVICE_BUILTIN*/
extern __device__ float                  __fmul_rn(float, float);

/*DEVICE_BUILTIN*/
extern __device__ int                    __clz(int);
/*DEVICE_BUILTIN*/
extern __device__ int                    __ffs(int);
/*DEVICE_BUILTIN*/
extern __device__ int                    __popc(unsigned int);


/*DEVICE_BUILTIN*/
extern __device__ int                    __clzll(long long int);
/*DEVICE_BUILTIN*/
extern __device__ int                    __ffsll(long long int);
/*DEVICE_BUILTIN*/
extern __device__ int                    __popcll(unsigned long long int);

#if !defined(CUDA_NO_SM_13_DOUBLE_INTRINSICS)

/*DEVICE_BUILTIN*/
extern __device__ int                    __double2int_rz(double);

/*DEVICE_BUILTIN*/
extern __device__ unsigned int           __double2uint_rz(double);

/*DEVICE_BUILTIN*/
extern __device__ long long int          __double2ll_rz(double);

/*DEVICE_BUILTIN*/
extern __device__ unsigned long long int __double2ull_rz(double);

#endif /* ! CUDA_NO_SM_13_DOUBLE_INTRINSICS */

}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

static __inline__ __device__ int mulhi(int a, int b)
{
  return __mulhi(a, b);
}

static __inline__ __device__ unsigned int mulhi(unsigned int a, unsigned int b)
{
  return __umulhi(a, b);
}

static __inline__ __device__ unsigned int mulhi(int a, unsigned int b)
{
  return __umulhi((unsigned int)a, b);
}

static __inline__ __device__ unsigned int mulhi(unsigned int a, int b)
{
  return __umulhi(a, (unsigned int)b);
}

static __inline__ __device__ long long int mul64hi(long long int a, long long int b)
{
  return __mul64hi(a, b);
}

static __inline__ __device__ unsigned long long int mul64hi(unsigned long long int a, unsigned long long int b)
{
  return __umul64hi(a, b);
}

static __inline__ __device__ unsigned long long int mul64hi(long long int a, unsigned long long int b)
{
  return __umul64hi((unsigned long long int)a, b);
}

static __inline__ __device__ unsigned long long int mul64hi(unsigned long long int a, long long int b)
{
  return __umul64hi(a, (unsigned long long int)b);
}

static __inline__ __device__ int float_as_int(float a)
{
  return __float_as_int(a);
}

static __inline__ __device__ float int_as_float(int a)
{
  return __int_as_float(a);
}

static __inline__ __device__ float saturate(float a)
{
  return __saturatef(a);
}

static __inline__ __device__ int mul24(int a, int b)
{
  return __mul24(a, b);
}

static __inline__ __device__ unsigned int umul24(unsigned int a, unsigned int b)
{
  return __umul24(a, b);
}

static __inline__ __device__ void trap(void)
{
  __trap();
}

static __inline__ __device__ void brkpt(int c)
{
  __brkpt(c);
}

static __inline__ __device__ void syncthreads(void)
{
  __syncthreads();
}

static __inline__ __device__ int float2int(float a, enum cudaRoundMode mode = cudaRoundZero)
{
  return mode == cudaRoundNearest ? __float2int_rn(a) :
         mode == cudaRoundPosInf  ? __float2int_ru(a) :
         mode == cudaRoundMinInf  ? __float2int_rd(a) :
                                    __float2int_rz(a);
}

static __inline__ __device__ unsigned int float2uint(float a, enum cudaRoundMode mode = cudaRoundZero)
{
  return mode == cudaRoundNearest ? __float2uint_rn(a) :
         mode == cudaRoundPosInf  ? __float2uint_ru(a) :
         mode == cudaRoundMinInf  ? __float2uint_rd(a) :
                                    __float2uint_rz(a);
}

static __inline__ __device__ float int2float(int a, enum cudaRoundMode mode = cudaRoundNearest)
{
  return mode == cudaRoundZero   ? __int2float_rz(a) :
         mode == cudaRoundPosInf ? __int2float_ru(a) :
         mode == cudaRoundMinInf ? __int2float_rd(a) :
                                   __int2float_rn(a);
}

static __inline__ __device__ float uint2float(unsigned int a, enum cudaRoundMode mode = cudaRoundNearest)
{
  return mode == cudaRoundZero   ? __uint2float_rz(a) :
         mode == cudaRoundPosInf ? __uint2float_ru(a) :
         mode == cudaRoundMinInf ? __uint2float_rd(a) :
                                   __uint2float_rn(a);
}

#elif !defined(__CUDACC__)

#include "crt/func_macro.h"

#include "host_defines.h"
#include "math_constants.h"

#if !defined(__CUDABE__)

__device_func__(int                    __cuda___isnan(double a));
__device_func__(int                    __cuda___isnanf(float a));
__device_func__(int                    __double2int_rz(double));
__device_func__(unsigned int           __double2uint_rz(double));
__device_func__(long long int          __double2ll_rz(double));
__device_func__(unsigned long long int __double2ull_rz(double));

#define __internal_clamp(val, max, min, nan)                                         \
       if (sizeof(val) == sizeof(double) && __cuda___isnan((double)val)) return nan; \
       if (sizeof(val) == sizeof(float) && __cuda___isnanf((float)val)) return nan;  \
       if (val >= max) return max;                                                   \
       if (val <= min) return min

/*******************************************************************************
*                                                                              *
* HOST IMPLEMENTATIONS FOR FUNCTIONS WITH BUILTIN NVOPENCC OPREATIONS          *
*                                                                              *
*******************************************************************************/

__device_func__(int __mulhi(int a, int b))
{
  long long int c = (long long int)a * (long long int)b;

  return (int)(c >> 32);
}

__device_func__(unsigned int __umulhi(unsigned int a, unsigned int b))
{
  unsigned long long int c = (unsigned long long int)a * (unsigned long long int)b;

  return (unsigned int)(c >> 32);
}

__device_func__(unsigned long long int __umul64hi(unsigned long long int a, unsigned long long int b))
{
  unsigned int           a_lo = (unsigned int)a;
  unsigned long long int a_hi = a >> 32;
  unsigned int           b_lo = (unsigned int)b;
  unsigned long long int b_hi = b >> 32;
  unsigned long long int m1 = a_lo * b_hi;
  unsigned long long int m2 = a_hi * b_lo;
  unsigned int           carry;

  carry = (0ULL + __umulhi(a_lo, b_lo) + (unsigned int)m1 + (unsigned int)m2) >> 32;

  return a_hi * b_hi + (m1 >> 32) + (m2 >> 32) + carry;
}

__device_func__(long long int __mul64hi(long long int a, long long int b))
{
  long long int res;
  res = __umul64hi(a, b);
  if (a < 0LL) res = res - b;
  if (b < 0LL) res = res - a;
  return res;
}

__device_func__(float __saturatef(float a))
{
  return a >= 1.0f ? 1.0f : a <= 0.0f ? 0.0f : a;
}

__device_func__(unsigned int __sad(int a, int b, unsigned int c))
{
  long long int diff = (long long int)a - (long long int)b;

  return (unsigned int)(__cuda_llabs(diff) + (long long int)c);
}

__device_func__(unsigned int __usad(unsigned int a, unsigned int b, unsigned int c))
{
  long long int diff = (long long int)a - (long long int)b;

  return (unsigned int)(__cuda_llabs(diff) + (long long int)c);
}

__device_func__(int __mul24(int a, int b))
{
#if !defined(__MULTI_CORE__)
  a &= 0xffffff;
  a = (a & 0x800000) != 0 ? a | ~0xffffff : a;
  b &= 0xffffff;
  b = (b & 0x800000) != 0 ? b | ~0xffffff : b;
#endif /* !__MULTI_CORE__ */

  return a * b;
}

__device_func__(unsigned int __umul24(unsigned int a, unsigned int b))
{
#if !defined(__MULTI_CORE__)
  a &= 0xffffff;
  b &= 0xffffff;
#endif /* !__MULTI_CORE__ */

  return a * b;
}

__device_func__(float __int_as_float(int a))
{
  volatile union {int a; float b;} u;

  u.a = a;

  return u.b;
}

__device_func__(int __float_as_int(float a))
{
  volatile union {float a; int b;} u;

  u.a = a;

  return u.b;
}

__device_func__(long long int __internal_float2ll_kernel(float a, long long int max, long long int min, long long int nan, enum cudaRoundMode rndMode))
{
  unsigned long long int res, t = 0ULL;
  int shift;
  unsigned int ia;

  __internal_clamp(a, max, min, nan);
  ia = __float_as_int(a);
  shift = 189 - ((ia >> 23) & 0xff);
  res = (unsigned long long int)(((ia << 8) | 0x80000000) >> 1) << 32;
  if (shift >= 64) {
    t = res;
    res = 0;
  } else if (shift) {
    t = res << (64 - shift);
    res = res >> shift;
  }
  if (rndMode == cudaRoundNearest && (long long int)t < 0LL) {
    res += t == 0x8000000000000000ULL ? res & 1ULL : 1ULL;
  }
  else if (rndMode == cudaRoundMinInf && t != 0ULL && ia > 0x80000000) {
    res++;
  }
  else if (rndMode == cudaRoundPosInf && t != 0ULL && (int)ia > 0) {
    res++;
  }
  if ((int)ia < 0) res = (unsigned long long int)-(long long int)res;
  return (long long int)res;
}

__device_func__(int __internal_float2int(float a, enum cudaRoundMode rndMode))
{
  return (int)__internal_float2ll_kernel(a, 2147483647LL, -2147483648LL, 0LL, rndMode);
}

__device_func__(int __float2int_rz(float a))
{
#if defined(__MULTI_CORE__)
  return (int)a;
#else /* __MULTI_CORE__ */
  return __internal_float2int(a, cudaRoundZero);
#endif /* __MULTI_CORE__ */
}

__device_func__(int __float2int_ru(float a))
{
  return __internal_float2int(a, cudaRoundPosInf);
}

__device_func__(int __float2int_rd(float a))
{
  return __internal_float2int(a, cudaRoundMinInf);
}

__device_func__(int __float2int_rn(float a))
{
  return __internal_float2int(a, cudaRoundNearest);
}

__device_func__(long long int __internal_float2ll(float a, enum cudaRoundMode rndMode))
{
  return __internal_float2ll_kernel(a, 9223372036854775807LL, -9223372036854775807LL -1LL, -9223372036854775807LL -1LL, rndMode);
}

__device_func__(long long int __float2ll_rz(float a))
{
#if defined(__MULTI_CORE__)
  return (long long int)a;
#else /* __MULTI_CORE__ */
  return __internal_float2ll(a, cudaRoundZero);
#endif /* __MULTI_CORE__ */
}

__device_func__(long long int __float2ll_ru(float a))
{
  return __internal_float2ll(a, cudaRoundPosInf);
}

__device_func__(long long int __float2ll_rd(float a))
{
  return __internal_float2ll(a, cudaRoundMinInf);
}

__device_func__(long long int __float2ll_rn(float a))
{
  return __internal_float2ll(a, cudaRoundNearest);
}

__device_func__(unsigned long long int __internal_float2ull_kernel(float a, unsigned long long int max, unsigned long long int nan, enum cudaRoundMode rndMode))
{
  unsigned long long int res, t = 0ULL;
  int shift;
  unsigned int ia;

  __internal_clamp(a, max, 0LL, nan);
  ia = __float_as_int(a);
  shift = 190 - ((ia >> 23) & 0xff);
  res = (unsigned long long int)((ia << 8) | 0x80000000) << 32;
  if (shift >= 64) {
    t = res >> (int)(shift > 64);
    res = 0;
  } else if (shift) {
    t = res << (64 - shift);
    res = res >> shift;
  }
  if (rndMode == cudaRoundNearest && (long long int)t < 0LL) {
    res += t == 0x8000000000000000ULL ? res & 1ULL : 1ULL;
  }
  else if (rndMode == cudaRoundPosInf && t != 0ULL) {
    res++;
  }
  return res;
}

__device_func__(unsigned int __internal_float2uint(float a, enum cudaRoundMode rndMode))
{
  return (unsigned int)__internal_float2ull_kernel(a, 4294967295U, 0U, rndMode);
}

__device_func__(unsigned int __float2uint_rz(float a))
{
#if defined(__MULTI_CORE__)
  return (unsigned int)a;
#else /* __MULTI_CORE__ */
  return __internal_float2uint(a, cudaRoundZero);
#endif /* __MULTI_CORE__ */
}

__device_func__(unsigned int __float2uint_ru(float a))
{
  return __internal_float2uint(a, cudaRoundPosInf);
}

__device_func__(unsigned int __float2uint_rd(float a))
{
  return __internal_float2uint(a, cudaRoundMinInf);
}

__device_func__(unsigned int __float2uint_rn(float a))
{
  return __internal_float2uint(a, cudaRoundNearest);
}

__device_func__(unsigned long long int __internal_float2ull(float a, enum cudaRoundMode rndMode))
{
  return __internal_float2ull_kernel(a, 18446744073709551615ULL, 9223372036854775808ULL, rndMode);
}

__device_func__(unsigned long long int __float2ull_rz(float a))
{
#if defined(__MULTI_CORE__)
  return (unsigned long long int)a;
#else /* __MULTI_CORE__ */
  return __internal_float2ull(a, cudaRoundZero);
#endif /* __MULTI_CORE__ */
}

__device_func__(unsigned long long int __float2ull_ru(float a))
{
  return __internal_float2ull(a, cudaRoundPosInf);
}

__device_func__(unsigned long long int __float2ull_rd(float a))
{
  return __internal_float2ull(a, cudaRoundMinInf);
}

__device_func__(unsigned long long int __float2ull_rn(float a))
{
  return __internal_float2ull(a, cudaRoundNearest);
}

__device_func__(int __internal_normalize64(unsigned long long int *a))
{
  int lz = 0;

  if ((*a & 0xffffffff00000000ULL) == 0ULL) {
    *a <<= 32;
    lz += 32;
  }
  if ((*a & 0xffff000000000000ULL) == 0ULL) {
    *a <<= 16;
    lz += 16;
  }
  if ((*a & 0xff00000000000000ULL) == 0ULL) {
    *a <<= 8;
    lz += 8;
  }
  if ((*a & 0xf000000000000000ULL) == 0ULL) {
    *a <<= 4;
    lz += 4;
  }
  if ((*a & 0xC000000000000000ULL) == 0ULL) {
    *a <<= 2;
    lz += 2;
  }
  if ((*a & 0x8000000000000000ULL) == 0ULL) {
    *a <<= 1;
    lz += 1;
  }
  return lz;
}

__device_func__(int __internal_normalize(unsigned int *a))
{
  unsigned long long int t = (unsigned long long int)*a;
  int lz = __internal_normalize64(&t);
  
  *a = (unsigned int)(t >> 32);

  return lz - 32;
}

__device_func__(float __internal_int2float_kernel(int a, enum cudaRoundMode rndMode))
{
  volatile union {
    float f;
    unsigned int i;
  } res;
  int shift;
  unsigned int t;
  res.i = a;
  if (a == 0) return res.f;
  if (a < 0) res.i = (unsigned int)-a;
  shift = __internal_normalize((unsigned int*)&res.i);
  t = res.i << 24;
  res.i = (res.i >> 8);
  res.i += (127 + 30 - shift) << 23;
  if (a < 0) res.i |= 0x80000000;
  if ((rndMode == cudaRoundNearest) && (t >= 0x80000000)) {
    res.i += (t == 0x80000000) ? (res.i & 1) : (t >> 31);
  }
  else if ((rndMode == cudaRoundMinInf) && t && (a < 0)) {
    res.i++;
  }
  else if ((rndMode == cudaRoundPosInf) && t && (a > 0)) {
    res.i++;
  }
  return res.f;
}

__device_func__(float __int2float_rz(int a))
{
  return __internal_int2float_kernel(a, cudaRoundZero);
}

__device_func__(float __int2float_ru(int a))
{
  return __internal_int2float_kernel(a, cudaRoundPosInf);
}

__device_func__(float __int2float_rd(int a))
{
  return __internal_int2float_kernel(a, cudaRoundMinInf);
}

__device_func__(float __int2float_rn(int a))
{
#if defined(__MULTI_CORE__)
  return (float)a;
#else /* __MULTI_CORE__ */
  return __internal_int2float_kernel(a, cudaRoundNearest);
#endif /* __MULTI_CORE__ */
}

__device_func__(float __internal_uint2float_kernel(unsigned int a, enum cudaRoundMode rndMode))
{
  volatile union {
    float f;
    unsigned int i;
  } res;
  int shift;
  unsigned int t;
  res.i = a;
  if (a == 0) return res.f;
  shift = __internal_normalize((unsigned int*)&res.i);
  t = res.i << 24;
  res.i = (res.i >> 8);
  res.i += (127 + 30 - shift) << 23;
  if ((rndMode == cudaRoundNearest) && (t >= 0x80000000)) {
    res.i += (t == 0x80000000) ? (res.i & 1) : (t >> 31);
  }
  else if ((rndMode == cudaRoundPosInf) && t) {
    res.i++;
  }
  return res.f;
}

__device_func__(float __uint2float_rz(unsigned int a))
{
  return __internal_uint2float_kernel(a, cudaRoundZero);
}

__device_func__(float __uint2float_ru(unsigned int a))
{
  return __internal_uint2float_kernel(a, cudaRoundPosInf);
}

__device_func__(float __uint2float_rd(unsigned int a))
{
  return __internal_uint2float_kernel(a, cudaRoundMinInf);
}

__device_func__(float __uint2float_rn(unsigned int a))
{
#if defined(__MULTI_CORE__)
  return (float)a;
#else /* __MULTI_CORE__ */
  return __internal_uint2float_kernel(a, cudaRoundNearest);
#endif /* __MULTI_CORE__ */
}

__device_func__(float __ll2float_rn(long long int a))
{
  return (float)a;
}      

__device_func__(float __ull2float_rn(unsigned long long int a))
{
#if defined(__MULTI_CORE__)
  return (float)a;
#else /* __MULTI_CORE__ */
  unsigned long long int temp;
  unsigned int res, t;
  int shift;
  if (a == 0ULL) return 0.0f;
  temp = a;
  shift = __internal_normalize64(&temp);
  temp = (temp >> 8) | ((temp & 0xffULL) ? 1ULL : 0ULL);
  res = (unsigned int)(temp >> 32);
  t = (unsigned int)temp;
  res += (127 + 62 - shift) << 23; /* add in exponent */
  res += t == 0x80000000 ? res & 1 : t >> 31;
  return __int_as_float(res);
#endif /* __MULTI_CORE__ */
}      

__device_func__(float __internal_fmul_kernel(float a, float b, int rndNearest))
{
  unsigned long long product;
  volatile union {
    float f;
    unsigned int i;
  } xx, yy;
  unsigned expo_x, expo_y;
    
  xx.f = a;
  yy.f = b;

  expo_y = 0xFF;
  expo_x = expo_y & (xx.i >> 23);
  expo_x = expo_x - 1;
  expo_y = expo_y & (yy.i >> 23);
  expo_y = expo_y - 1;
    
  if ((expo_x <= 0xFD) && 
      (expo_y <= 0xFD)) {
multiply:
    expo_x = expo_x + expo_y;
    expo_y = xx.i ^ yy.i;
    xx.i = xx.i & 0x00ffffff;
    yy.i = yy.i << 8;
    xx.i = xx.i | 0x00800000;
    yy.i = yy.i | 0x80000000;
    /* compute product */
    product = ((unsigned long long)xx.i) * yy.i;
    expo_x = expo_x - 127 + 2;
    expo_y = expo_y & 0x80000000;
    xx.i = (unsigned int)(product >> 32);
    yy.i = (unsigned int)(product & 0xffffffff);
    /* normalize mantissa */
    if (xx.i < 0x00800000) {
      xx.i = (xx.i << 1) | (yy.i >> 31);
      yy.i = (yy.i << 1);
      expo_x--;
    }
    if (expo_x <= 0xFD) {
      xx.i = xx.i | expo_y;          /* OR in sign bit */
      xx.i = xx.i + (expo_x << 23);  /* add in exponent */
      /* round result to nearest or even */
      if (yy.i < 0x80000000) return xx.f;
      xx.i += (((yy.i == 0x80000000) ? (xx.i & 1) : (yy.i >> 31)) 
               && rndNearest);
      return xx.f;
    } else if ((int)expo_x >= 254) {
      /* overflow: return infinity */
      xx.i = (expo_y | 0x7F800000) - (!rndNearest);
      return xx.f;
    } else {
      /* zero, denormal, or smallest normal */
      expo_x = ((unsigned int)-((int)expo_x));
      if (expo_x > 25) {
        /* massive underflow: return 0 */
        xx.i = expo_y;
        return xx.f;
      } else {
        yy.i = (xx.i << (32 - expo_x)) | ((yy.i) ? 1 : 0);
        xx.i = expo_y + (xx.i >> expo_x);
        xx.i += (((yy.i == 0x80000000) ? (xx.i & 1) : (yy.i >> 31)) 
                 && rndNearest);
        return xx.f;
      }
    }
  } else {
    product = xx.i ^ yy.i;
    product = product & 0x80000000;
    if (!(xx.i & 0x7fffffff)) {
      if (expo_y != 254) {
        xx.i = (unsigned int)product;
        return xx.f;
      }
      expo_y = yy.i << 1;
      if (expo_y == 0xFF000000) {
        xx.i = expo_y | 0x00C00000;
      } else {
        xx.i = yy.i | 0x00400000;
      }
      return xx.f;
    }
    if (!(yy.i & 0x7fffffff)) {
      if (expo_x != 254) {
        xx.i = (unsigned int)product;
        return xx.f;
      }
      expo_x = xx.i << 1;
      if (expo_x == 0xFF000000) {
        xx.i = expo_x | 0x00C00000;
      } else {
        xx.i = xx.i | 0x00400000;
      }
      return xx.f;
    }
    if ((expo_y != 254) && (expo_x != 254)) {
      expo_y++;
      expo_x++;
      if (expo_x == 0) {
        expo_y |= xx.i & 0x80000000;
        /*
         * If both operands are denormals, we only need to normalize 
         * one of them as the result will be either a denormal or zero.
         */
        xx.i = xx.i << 8;
        while (!(xx.i & 0x80000000)) {
          xx.i <<= 1;
          expo_x--;
        }
        xx.i = (xx.i >> 8) | (expo_y & 0x80000000);
        expo_y &= ~0x80000000;
        expo_y--;
        goto multiply;
      }
      if (expo_y == 0) {
        expo_x |= yy.i & 0x80000000;
        yy.i = yy.i << 8;
        while (!(yy.i & 0x80000000)) {
          yy.i <<= 1;
          expo_y--;
        }
        yy.i = (yy.i >> 8) | (expo_x & 0x80000000);
        expo_x &= ~0x80000000;
        expo_x--;
        goto multiply;
      }
    }
    expo_x = xx.i << 1;
    expo_y = yy.i << 1;
    /* if x is NaN, return x */
    if (expo_x > 0xFF000000) {
      /* cvt any SNaNs to QNaNs */
      xx.i = xx.i | 0x00400000;
      return xx.f;
    }
    /* if y is NaN, return y */
    if (expo_y > 0xFF000000) {
      /* cvt any SNaNs to QNaNs */
      xx.i = yy.i | 0x00400000;
      return xx.f;
    } 
    xx.i = (unsigned int)product | 0x7f800000;
    return xx.f;
  }
}

__device_func__(float __internal_fadd_kernel(float a, float b, int rndNearest))
{
  volatile union {
    float f;
    unsigned int i;
  } xx, yy;
  unsigned int expo_x;
  unsigned int expo_y;
  unsigned int temp;

  xx.f = a;
  yy.f = b;

  /* make bigger operand the augend */
  expo_y = yy.i << 1;
  if (expo_y > (xx.i << 1)) {
    expo_y = xx.i;
    xx.i   = yy.i;
    yy.i   = expo_y;
  }
    
  temp = 0xff;
  expo_x = temp & (xx.i >> 23);
  expo_x = expo_x - 1;
  expo_y = temp & (yy.i >> 23);
  expo_y = expo_y - 1;
    
  if ((expo_x <= 0xFD) && 
      (expo_y <= 0xFD)) {
        
add:
    expo_y = expo_x - expo_y;
    if (expo_y > 25) {
      expo_y = 31;
    }
    temp = xx.i ^ yy.i;
    xx.i = xx.i & ~0x7f000000;
    xx.i = xx.i |  0x00800000;
    yy.i = yy.i & ~0xff000000;
    yy.i = yy.i |  0x00800000;
        
    if ((int)temp < 0) {
      /* signs differ, effective subtraction */
      temp = 32 - expo_y;
      temp = (expo_y) ? (yy.i << temp) : 0;
      temp = (unsigned int)(-((int)temp));
      xx.i = xx.i - (yy.i >> expo_y) - (temp ? 1 : 0);
      if (xx.i & 0x00800000) {
        if (expo_x <= 0xFD) {
          xx.i = xx.i & ~0x00800000; /* lop off integer bit */
          xx.i = (xx.i + (expo_x << 23)) + 0x00800000;
          if (temp < 0x80000000) return xx.f;
          xx.i += (((temp == 0x80000000) ? (xx.i & 1) : (temp >> 31))
                   && rndNearest);
          return xx.f;
        }
      } else {
        if ((temp | (xx.i << 1)) == 0) {
          /* operands cancelled, resulting in a clean zero */
          xx.i = 0;
          return xx.f;
        }
        /* normalize result */
        yy.i = xx.i & 0x80000000;
        do {
          xx.i = (xx.i << 1) | (temp >> 31);
          temp <<= 1;
          expo_x--;
        } while (!(xx.i & 0x00800000));
        xx.i = xx.i | yy.i;
      }
    } else {
      /* signs are the same, effective addition */
      temp = 32 - expo_y;
      temp = (expo_y) ? (yy.i << temp) : 0;
      xx.i = xx.i + (yy.i >> expo_y);
      if (!(xx.i & 0x01000000)) {
        if (expo_x <= 0xFD) {
          expo_y = xx.i & 1;
          xx.i = xx.i + (expo_x << 23);
          if (temp < 0x80000000) return xx.f;
          xx.i += (((temp == 0x80000000) ? expo_y : (temp >> 31)) 
                   && rndNearest);
          return xx.f;
        }
      } else {
        /* normalize result */
        temp = (xx.i << 31) | (temp >> 1);
        /* not ANSI compliant: xx.i = (((int)xx.i)>>1) & ~0x40000000 */
        xx.i = ((xx.i & 0x80000000) | (xx.i >> 1)) & ~0x40000000;
        expo_x++;
      }
    }
    if (expo_x <= 0xFD) {
      expo_y = xx.i & 1;
      xx.i += (((temp == 0x80000000) ? expo_y : (temp >> 31)) 
               && rndNearest);
      xx.i = xx.i + (expo_x << 23);
      return xx.f;
    }
    if ((int)expo_x >= 254) {
      /* overflow: return infinity */
        xx.i = ((xx.i & 0x80000000) | 0x7f800000) - (!rndNearest);
        return xx.f;
    }
    /* underflow: denormal, or smallest normal */
    expo_y = expo_x + 32;
    yy.i = xx.i &  0x80000000;
    xx.i = xx.i & ~0xff000000;
        
    expo_x = (unsigned int)(-((int)expo_x));
    temp = xx.i << expo_y | ((temp) ? 1 : 0);
    xx.i = yy.i | (xx.i >> expo_x);
    xx.i += (((temp == 0x80000000) ? (xx.i & 1) : (temp >> 31)) 
             && rndNearest);
    return xx.f;
  } else {
    /* handle special cases separately */
    if (!(yy.i << 1)) {
      if (xx.i == 0x80000000) {
          xx.i = yy.i;
      }
      if ((xx.i << 1) > 0xff000000) {
          xx.i |= 0x00400000;
      }
      return xx.f;
    }
    if ((expo_y != 254) && (expo_x != 254)) {
      /* remove sign bits */
      if (expo_x == (unsigned int) -1) {
        temp = xx.i & 0x80000000;
        xx.i = xx.i << 8;
        while (!(xx.i & 0x80000000)) {
          xx.i <<= 1;
          expo_x--;
        }
        expo_x++;
        xx.i = (xx.i >> 8) | temp;
      }
      if (expo_y == (unsigned int) -1) {
        temp = yy.i & 0x80000000;
        yy.i = yy.i << 8;
        while (!(yy.i & 0x80000000)) {
          yy.i <<= 1;
          expo_y--;
        }
        expo_y++;
        yy.i = (yy.i >> 8) | temp;
      }
      goto add;
    }
    expo_x = xx.i << 1;
    expo_y = yy.i << 1;
    /* if x is NaN, return x */
    if (expo_x > 0xff000000) {
      /* cvt any SNaNs to QNaNs */
      xx.i = xx.i | 0x00400000;
      return xx.f;
    }
    /* if y is NaN, return y */
    if (expo_y > 0xff000000) {
      /* cvt any SNaNs to QNaNs */
      xx.i = yy.i | 0x00400000;
      return xx.f;
    }
    if ((expo_x == 0xff000000) && (expo_y == 0xff000000)) {
      /*
       * subtraction of infinities with the same sign, and addition of
       * infinities of unlike sign is undefined: return NaN INDEFINITE
       */
      expo_x = xx.i ^ yy.i;
      xx.i = xx.i | ((expo_x) ? 0xffc00000 : 0);
      return xx.f;
    }
    /* handle infinities */
    if (expo_y == 0xff000000) {
      xx.i = yy.i;
    }
    return xx.f;
  }
}

__device_func__(float __fadd_rz(float a, float b))
{
  return __internal_fadd_kernel(a, b, 0);
}

__device_func__(float __fmul_rz(float a, float b))
{
  return __internal_fmul_kernel(a, b, 0);
}

__device_func__(float __fadd_rn(float a, float b))
{
  return __internal_fadd_kernel(a, b, 1);
}

__device_func__(float __fmul_rn(float a, float b))
{
  return __internal_fmul_kernel(a, b, 1);
}

__device_func__(void __brkpt(int c))
{
  /* TODO */
}

#if defined(__MULTI_CORE__)

#define __syncthreads() \
        __builtin___syncthreads()

#else /* __MULTI_CORE__ */

extern int CUDARTAPI __cudaSynchronizeThreads(void**, void*);

#if defined(__GNUC__)

__device_func__(inline __attribute__((always_inline)) void __syncthreads(void))
{
  volatile int _ = 0;
  L: if (__cudaSynchronizeThreads((void**)&&L, (void*)&_)) goto L;
}

#elif defined(_WIN32)

#define __syncthreads() \
        (void)__cudaSynchronizeThreads((void**)0, (void*)0)

#endif /* __GNUC__ */

#endif /* __MULTI_CORE__ */

#if defined(__GNUC__)

__device_func__(void __trap(void))
{
  __builtin_trap();
}

#elif defined(_WIN32)

__device_func__(void __trap(void))
{
  __debugbreak();
}

#endif /* __GNUC__ */

#endif /* !__CUDABE__ */

/*******************************************************************************
*                                                                              *
* DEVICE IMPLEMENTATIONS FOR FUNCTIONS WITH BUILTIN NVOPENCC OPERATIONS        *
*                                                                              *
*******************************************************************************/

__device_func__(float __fdividef(float a, float b))
{
#if defined(__MULTI_CORE__)
  return a / b;
#elif defined(__CUDABE__)
  return a / b;
#else /* __MULTI_CORE__ */
  /* match range restrictions of the device function */
  if (__cuda_fabsf(b) > CUDART_TWO_TO_126_F) {
    if (__cuda_fabsf(a) <= CUDART_NORM_HUGE_F) {
      return ((a / b) / CUDART_NORM_HUGE_F) / CUDART_NORM_HUGE_F;
    } else {
      return CUDART_NAN_F;
    }
  } else {
    return a / b;
  }
#endif /* __MULTI_CORE__ */
}

__device_func__(float __sinf(float a))
{
  return sinf(a);
}

__device_func__(float __cosf(float a))
{
  return cosf(a);
}

__device_func__(float __log2f(float a))
{
  return log2f(a);
}

/*******************************************************************************
*                                                                              *
* SHARED HOST AND DEVICE IMPLEMENTATIONS                                       *
*                                                                              *
*******************************************************************************/

__device_func__(float __internal_accurate_fdividef(float a, float b))
{
  if (__cuda_fabsf(b) > CUDART_TWO_TO_126_F) {
    a *= .25f;
    b *= .25f;
  }
  return __fdividef(a, b);
}

__device_func__(float __tanf(float a))
{
#if defined(__MULTI_CORE__)
  return tanf(a);
#else /* __MULTI_CORE__ */
  return __sinf(a) / __cosf(a);
#endif /* __MULTI_CORE__ */
}

__device_func__(void __sincosf(float a, float *sptr, float *cptr))
{
#if defined(__MULTI_CORE__)
  sincosf(a, sptr, cptr);
#else /* __MULTI_CORE__ */
  *sptr = __sinf(a);
  *cptr = __cosf(a);
#endif /* __MULTI_CORE__ */
}

__device_func__(float __expf(float a))
{
#if defined(__MULTI_CORE__)
  return expf(a);
#else /* __MULTI_CORE__ */
  return __cuda_exp2f(a * CUDART_L2E_F);
#endif /* __MULTI_CORE__ */
}

__device_func__(float __exp10f(float a))
{
#if defined(__MULTI_CORE__)
  return exp10f(a);
#else /* __MULTI_CORE__ */
  return __cuda_exp2f(a * CUDART_L2T_F);
#endif /* __MULTI_CORE__ */
}

__device_func__(float __log10f(float a))
{
#if defined(__MULTI_CORE__)
  return log10f(a);
#else /* __MULTI_CORE__ */
  return CUDART_LG2_F * __log2f(a);
#endif /* __MULTI_CORE__ */
}

__device_func__(float __logf(float a))
{
#if defined(__MULTI_CORE__)
  return logf(a);
#else /* __MULTI_CORE__ */
  return CUDART_LN2_F * __log2f(a);
#endif /* __MULTI_CORE__ */
}

__device_func__(float __powf(float a, float b))
{
#if defined(__MULTI_CORE__)
  return powf(a, b);
#else /* __MULTI_CORE__ */
  return __cuda_exp2f(b * __log2f(a));
#endif /* __MULTI_CORE__ */
}

__device_func__(float fdividef(float a, float b))
{
#if defined(__MULTI_CORE__)
  return a / b;
#elif defined(__USE_FAST_MATH__)
  return __fdividef(a, b);
#else /* __MULTI_CORE__ */
  return __internal_accurate_fdividef(a, b);
#endif /* __MULTI_CORE__ */
}

__device_func__(int __clz(int a))
{
  return (a)?(158-(__float_as_int(__uint2float_rz((unsigned int)a))>>23)):32;
}

__device_func__(int __ffs(int a))
{
  return 32 - __clz (a & -a);
}

__device_func__(int __popc(unsigned int a))
{
  a = a - ((a >> 1) & 0x55555555);
  a = (a & 0x33333333) + ((a >> 2) & 0x33333333);
  a = (a + (a >> 4)) & 0x0f0f0f0f;
  a = ((__umul24(a, 0x808080) << 1) + a) >> 24;
  return a;
}

__device_func__(int __clzll(long long int a))
{
  int ahi = ((int)((unsigned long long)a >> 32));
  int alo = ((int)((unsigned long long)a & 0xffffffffULL));
  int res;
  if (ahi) {
      res = 0;
  } else {
      res = 32;
      ahi = alo;
  }
  res = res + __clz(ahi);
  return res;
}

__device_func__(int __ffsll(long long int a))
{
  return 64 - __clzll (a & -a);
}

__device_func__(int __popcll(unsigned long long int a))
{
  unsigned int ahi = ((unsigned int)(a >> 32));
  unsigned int alo = ((unsigned int)(a & 0xffffffffULL));
  alo = alo - ((alo >> 1) & 0x55555555);
  alo = (alo & 0x33333333) + ((alo >> 2) & 0x33333333);
  ahi = ahi - ((ahi >> 1) & 0x55555555);
  ahi = (ahi & 0x33333333) + ((ahi >> 2) & 0x33333333);
  alo = alo + ahi;
  alo = (alo & 0x0f0f0f0f) + ((alo >> 4) & 0x0f0f0f0f);
  alo = ((__umul24(alo, 0x808080) << 1) + alo) >> 24;
  return alo;
}

#if defined(CUDA_DOUBLE_MATH_FUNCTIONS) && defined(CUDA_FLOAT_MATH_FUNCTIONS)

#error -- conflicting mode for double math routines

#endif /* CUDA_DOUBLE_MATH_FUNCTIONS && CUDA_FLOAT_MATH_FUNCTIONS */

#if defined(CUDA_FLOAT_MATH_FUNCTIONS)

__device_func__(double fdivide(double a, double b))
{
  return (double)fdividef((float)a, (float)b);
}

#if !defined(__CUDABE__)

__device_func__(int __double2int_rz(double a))
{
  return __float2int_rz((float)a);
}

__device_func__(unsigned int __double2uint_rz(double a))
{
  return __float2uint_rz((float)a);
}

__device_func__(long long int __double2ll_rz(double a))
{
  return __float2ll_rz((float)a);
}

__device_func__(unsigned long long int __double2ull_rz(double a))
{
  return __float2ull_rz((float)a);
}

#endif /* !__CUDABE__ */

#endif /* CUDA_FLOAT_MATH_FUNCTIONS */

#if defined(CUDA_DOUBLE_MATH_FUNCTIONS)

__device_func__(double fdivide(double a, double b))
{
  return a / b;
}

#if !defined(__CUDABE__)

__device_func__(int __internal_double2int(double a, enum cudaRoundMode rndMode));
__device_func__(unsigned int __internal_double2uint(double a, enum cudaRoundMode rndMode));
__device_func__(long long int __internal_double2ll(double a, enum cudaRoundMode rndMode));
__device_func__(unsigned long long int __internal_double2ull(double a, enum cudaRoundMode rndMode));

__device_func__(int __double2int_rz(double a))
{
  return __internal_double2int(a, cudaRoundZero);
}

__device_func__(unsigned int __double2uint_rz(double a))
{
  return __internal_double2uint(a, cudaRoundZero);
}

__device_func__(long long int __double2ll_rz(double a))
{
  return __internal_double2ll(a, cudaRoundZero);
}

__device_func__(unsigned long long int __double2ull_rz(double a))
{
  return __internal_double2ull(a, cudaRoundZero);
}

#endif /* !__CUDABE__ */

#endif /* CUDA_DOUBLE_MATH_FUNCTIONS */

#endif /* __cplusplus && __CUDACC__ */

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#include "sm_11_atomic_functions.h"
#include "sm_12_atomic_functions.h"
#include "sm_13_double_functions.h"
#include "texture_fetch_functions_dynlink.h"

#endif /* !__DEVICE_FUNCTIONS_H__ */
