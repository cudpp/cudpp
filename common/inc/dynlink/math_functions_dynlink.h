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

#if !defined(__MATH_FUNCTIONS_H__)
#define __MATH_FUNCTIONS_H__

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#if defined(__cplusplus) && defined(__CUDACC__)

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#include "host_defines.h"

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

extern "C"
{

/*DEVICE_BUILTIN*/
extern __host__ __device__ int           abs(int) __THROW;
/*DEVICE_BUILTIN*/
extern __host__ __device__ long int      labs(long int) __THROW;
/*DEVICE_BUILTIN*/
extern __host__ __device__ long long int llabs(long long int) __THROW;
/*DEVICE_BUILTIN*/
extern __host__ __device__ double        fabs(double) __THROW;
/*DEVICE_BUILTIN*/
extern __host__ __device__ float         fabsf(float) __THROW;

/*DEVICE_BUILTIN*/
extern __host__ __device__ int           min(int, int);
/*DEVICE_BUILTIN*/
extern __host__ __device__ unsigned int  umin(unsigned int, unsigned int);
/*DEVICE_BUILTIN*/
extern __host__ __device__ float         fminf(float, float) __THROW;
/*DEVICE_BUILTIN*/
extern __host__ __device__ double        fmin(double, double) __THROW;

/*DEVICE_BUILTIN*/
extern __host__ __device__ int           max(int, int);
/*DEVICE_BUILTIN*/
extern __host__ __device__ unsigned int  umax(unsigned int, unsigned int);
/*DEVICE_BUILTIN*/
extern __host__ __device__ float         fmaxf(float, float) __THROW;
/*DEVICE_BUILTIN*/
extern __host__ __device__ double        fmax(double, double) __THROW;

/*DEVICE_BUILTIN*/
extern __host__ __device__ double        sin(double) __THROW;
/*DEVICE_BUILTIN*/
extern __host__ __device__ float         sinf(float) __THROW;

/*DEVICE_BUILTIN*/
extern __host__ __device__ double        cos(double) __THROW;
/*DEVICE_BUILTIN*/
extern __host__ __device__ float         cosf(float) __THROW;

/*DEVICE_BUILTIN*/
extern __host__ __device__ void          sincos(double, double*, double*) __THROW;
/*DEVICE_BUILTIN*/
extern __host__ __device__ void          sincosf(float, float*, float*) __THROW;

/*DEVICE_BUILTIN*/
extern __host__ __device__ double        tan(double) __THROW;
/*DEVICE_BUILTIN*/
extern __host__ __device__ float         tanf(float) __THROW;

/*DEVICE_BUILTIN*/
extern __host__ __device__ double        sqrt(double) __THROW;
/*DEVICE_BUILTIN*/
extern __host__ __device__ float         sqrtf(float) __THROW;

/*DEVICE_BUILTIN*/
extern __host__ __device__ double        rsqrt(double);
/*DEVICE_BUILTIN*/
extern __host__ __device__ float         rsqrtf(float);

/*DEVICE_BUILTIN*/
extern __host__ __device__ double        exp2(double) __THROW;
/*DEVICE_BUILTIN*/
extern __host__ __device__ float         exp2f(float) __THROW;

/*DEVICE_BUILTIN*/
extern __host__ __device__ double        exp10(double) __THROW;
/*DEVICE_BUILTIN*/
extern __host__ __device__ float         exp10f(float) __THROW;

/*DEVICE_BUILTIN*/
extern __host__ __device__ double        expm1(double) __THROW;
/*DEVICE_BUILTIN*/
extern __host__ __device__ float         expm1f(float) __THROW;

/*DEVICE_BUILTIN*/
extern __host__ __device__ double        log2(double) __THROW;
/*DEVICE_BUILTIN*/
extern __host__ __device__ float         log2f(float) __THROW;

/*DEVICE_BUILTIN*/
extern __host__ __device__ double        log10(double) __THROW;
/*DEVICE_BUILTIN*/
extern __host__ __device__ float         log10f(float) __THROW;

/*DEVICE_BUILTIN*/
extern __host__ __device__ double        log(double) __THROW;
/*DEVICE_BUILTIN*/
extern __host__ __device__ float         logf(float) __THROW;

/*DEVICE_BUILTIN*/
extern __host__ __device__ double        log1p(double) __THROW;
/*DEVICE_BUILTIN*/
extern __host__ __device__ float         log1pf(float) __THROW;

/*DEVICE_BUILTIN*/
extern __host__ __device__ double        floor(double) __THROW;
/*DEVICE_BUILTIN*/
extern __host__ __device__ float         floorf(float) __THROW;

/*DEVICE_BUILTIN*/
extern __host__ __device__ double        exp(double) __THROW;
/*DEVICE_BUILTIN*/
extern __host__ __device__ float         expf(float) __THROW;

/*DEVICE_BUILTIN*/
extern __host__ __device__ double        cosh(double) __THROW;
/*DEVICE_BUILTIN*/
extern __host__ __device__ float         coshf(float) __THROW;

/*DEVICE_BUILTIN*/
extern __host__ __device__ double        sinh(double) __THROW;
/*DEVICE_BUILTIN*/
extern __host__ __device__ float         sinhf(float) __THROW;

/*DEVICE_BUILTIN*/
extern __host__ __device__ double        tanh(double) __THROW;
/*DEVICE_BUILTIN*/
extern __host__ __device__ float         tanhf(float) __THROW;

/*DEVICE_BUILTIN*/
extern __host__ __device__ double        acosh(double) __THROW;
/*DEVICE_BUILTIN*/
extern __host__ __device__ float         acoshf(float) __THROW;

/*DEVICE_BUILTIN*/
extern __host__ __device__ double        asinh(double) __THROW;
/*DEVICE_BUILTIN*/
extern __host__ __device__ float         asinhf(float) __THROW;

/*DEVICE_BUILTIN*/
extern __host__ __device__ double        atanh(double) __THROW;
/*DEVICE_BUILTIN*/
extern __host__ __device__ float         atanhf(float) __THROW;

/*DEVICE_BUILTIN*/
extern __host__ __device__ double        ldexp(double, int) __THROW;
/*DEVICE_BUILTIN*/
extern __host__ __device__ float         ldexpf(float, int) __THROW;

/*DEVICE_BUILTIN*/
extern __host__ __device__ double        logb(double) __THROW;
/*DEVICE_BUILTIN*/
extern __host__ __device__ float         logbf(float) __THROW;

/*DEVICE_BUILTIN*/
extern __host__ __device__ int           ilogb(double) __THROW;
/*DEVICE_BUILTIN*/
extern __host__ __device__ int           ilogbf(float) __THROW;

/*DEVICE_BUILTIN*/
extern __host__ __device__ double        scalbn(double, int) __THROW;
/*DEVICE_BUILTIN*/
extern __host__ __device__ float         scalbnf(float, int) __THROW;

/*DEVICE_BUILTIN*/
extern __host__ __device__ double        scalbln(double, long int) __THROW;
/*DEVICE_BUILTIN*/
extern __host__ __device__ float         scalblnf(float, long int) __THROW;

/*DEVICE_BUILTIN*/
extern __host__ __device__ double        frexp(double, int*) __THROW;
/*DEVICE_BUILTIN*/
extern __host__ __device__ float         frexpf(float, int*) __THROW;

/*DEVICE_BUILTIN*/
extern __host__ __device__ double        round(double) __THROW;
/*DEVICE_BUILTIN*/
extern __host__ __device__ float         roundf(float) __THROW;

/*DEVICE_BUILTIN*/
extern __host__ __device__ long int      lround(double) __THROW;
/*DEVICE_BUILTIN*/
extern __host__ __device__ long int      lroundf(float) __THROW;

/*DEVICE_BUILTIN*/
extern __host__ __device__ long long int llround(double) __THROW;
/*DEVICE_BUILTIN*/
extern __host__ __device__ long long int llroundf(float) __THROW;

/*DEVICE_BUILTIN*/
extern __host__ __device__ double        rint(double) __THROW;
/*DEVICE_BUILTIN*/
extern __host__ __device__ float         rintf(float) __THROW;

/*DEVICE_BUILTIN*/
extern __host__ __device__ long int      lrint(double) __THROW;
/*DEVICE_BUILTIN*/
extern __host__ __device__ long int      lrintf(float) __THROW;

/*DEVICE_BUILTIN*/
extern __host__ __device__ long long int llrint(double) __THROW;
/*DEVICE_BUILTIN*/
extern __host__ __device__ long long int llrintf(float) __THROW;

/*DEVICE_BUILTIN*/
extern __host__ __device__ double        nearbyint(double) __THROW;
/*DEVICE_BUILTIN*/
extern __host__ __device__ float         nearbyintf(float) __THROW;

/*DEVICE_BUILTIN*/
extern __host__ __device__ double        ceil(double) __THROW;
/*DEVICE_BUILTIN*/
extern __host__ __device__ float         ceilf(float) __THROW;

/*DEVICE_BUILTIN*/
extern __host__ __device__ double        trunc(double) __THROW;
/*DEVICE_BUILTIN*/
extern __host__ __device__ float         truncf(float) __THROW;

/*DEVICE_BUILTIN*/
extern __host__ __device__ double        fdim(double, double) __THROW;
/*DEVICE_BUILTIN*/
extern __host__ __device__ float         fdimf(float, float) __THROW;

/*DEVICE_BUILTIN*/
extern __host__ __device__ double        atan2(double, double) __THROW;
/*DEVICE_BUILTIN*/
extern __host__ __device__ float         atan2f(float, float) __THROW;

/*DEVICE_BUILTIN*/
extern __host__ __device__ double        atan(double) __THROW;
/*DEVICE_BUILTIN*/
extern __host__ __device__ float         atanf(float) __THROW;

/*DEVICE_BUILTIN*/
extern __host__ __device__ double        asin(double) __THROW;
/*DEVICE_BUILTIN*/
extern __host__ __device__ float         asinf(float) __THROW;

/*DEVICE_BUILTIN*/
extern __host__ __device__ double        acos(double) __THROW;
/*DEVICE_BUILTIN*/
extern __host__ __device__ float         acosf(float) __THROW;

/*DEVICE_BUILTIN*/
extern __host__ __device__ double        hypot(double, double) __THROW;
/*DEVICE_BUILTIN*/
extern __host__ __device__ float         hypotf(float, float) __THROW;

/*DEVICE_BUILTIN*/
extern __host__ __device__ double        cbrt(double) __THROW;
/*DEVICE_BUILTIN*/
extern __host__ __device__ float         cbrtf(float) __THROW;

/*DEVICE_BUILTIN*/
extern __host__ __device__ double        pow(double, double) __THROW;
/*DEVICE_BUILTIN*/
extern __host__ __device__ float         powf(float, float) __THROW;

/*DEVICE_BUILTIN*/
extern __host__ __device__ double        modf(double, double*) __THROW;
/*DEVICE_BUILTIN*/
extern __host__ __device__ float         modff(float, float*) __THROW;

/*DEVICE_BUILTIN*/
extern __host__ __device__ double        fmod(double, double) __THROW;
/*DEVICE_BUILTIN*/
extern __host__ __device__ float         fmodf(float, float) __THROW;

/*DEVICE_BUILTIN*/
extern __host__ __device__ double        remainder(double, double) __THROW;
/*DEVICE_BUILTIN*/
extern __host__ __device__ float         remainderf(float, float) __THROW;

/*DEVICE_BUILTIN*/
extern __host__ __device__ double        remquo(double, double, int*) __THROW;
/*DEVICE_BUILTIN*/
extern __host__ __device__ float         remquof(float, float, int*) __THROW;

/*DEVICE_BUILTIN*/
extern __host__ __device__ double        erf(double) __THROW;
/*DEVICE_BUILTIN*/
extern __host__ __device__ float         erff(float) __THROW;

/*DEVICE_BUILTIN*/
extern __host__ __device__ double        erfc(double) __THROW;
/*DEVICE_BUILTIN*/
extern __host__ __device__ float         erfcf(float) __THROW;

/*DEVICE_BUILTIN*/
extern __host__ __device__ double        lgamma(double) __THROW;
/*DEVICE_BUILTIN*/
extern __host__ __device__ float         lgammaf(float) __THROW;

/*DEVICE_BUILTIN*/
extern __host__ __device__ double        tgamma(double) __THROW;
/*DEVICE_BUILTIN*/
extern __host__ __device__ float         tgammaf(float) __THROW;

/*DEVICE_BUILTIN*/
extern __host__ __device__ double        copysign(double, double) __THROW;
/*DEVICE_BUILTIN*/
extern __host__ __device__ float         copysignf(float, float) __THROW;

/*DEVICE_BUILTIN*/
extern __host__ __device__ double        nextafter(double, double) __THROW;
/*DEVICE_BUILTIN*/
extern __host__ __device__ float         nextafterf(float, float) __THROW;

/*DEVICE_BUILTIN*/
extern __host__ __device__ double        nan(const char*) __THROW;
/*DEVICE_BUILTIN*/
extern __host__ __device__ float         nanf(const char*) __THROW;

/*DEVICE_BUILTIN*/
extern __host__ __device__ int           __isinf(double) __THROW;
/*DEVICE_BUILTIN*/
extern __host__ __device__ int           __isinff(float) __THROW;

/*DEVICE_BUILTIN*/
extern __host__ __device__ int           __isnan(double) __THROW;
/*DEVICE_BUILTIN*/
extern __host__ __device__ int           __isnanf(float) __THROW;

#ifdef __APPLE__

/*DEVICE_BUILTIN*/
extern __host__ __device__ int           __isfinited(double) __THROW;
/*DEVICE_BUILTIN*/
extern __host__ __device__ int           __isfinitef(float) __THROW;
/*DEVICE_BUILTIN*/
extern __host__ __device__ int           __signbitd(double) __THROW;

#else /* __APPLE__ */

/*DEVICE_BUILTIN*/
extern __host__ __device__ int           __finite(double) __THROW;
/*DEVICE_BUILTIN*/
extern __host__ __device__ int           __finitef(float) __THROW;
/*DEVICE_BUILTIN*/
extern __host__ __device__ int           __signbit(double) __THROW;

#endif /* __APPLE__ */

/*DEVICE_BUILTIN*/
extern __host__ __device__ int           __signbitf(float) __THROW;

/*DEVICE_BUILTIN*/
extern __host__ __device__ double        fma(double, double, double) __THROW;
/*DEVICE_BUILTIN*/
extern __host__ __device__ float         fmaf(float, float, float) __THROW;

}

#include <math.h>
#include <stdlib.h>
#include <cmath>
#include <cstdlib>

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#if defined(__GNUC__)

/* these are here to avoid warnings on the call graph.
   long double is not supported on the device */
/*DEVICE_BUILTIN*/
extern __host__ __device__ int __signbitl(long double) __THROW;
/*DEVICE_BUILTIN*/
extern __host__ __device__ int __isinfl(long double) __THROW;
/*DEVICE_BUILTIN*/
extern __host__ __device__ int __isnanl(long double) __THROW;

#if defined(__APPLE__)

/*DEVICE_BUILTIN*/
extern __host__ __device__ int __isfinite(long double) __THROW;

#else /* __APPLE__ */

/*DEVICE_BUILTIN*/
extern __host__ __device__ int __finitel(long double) __THROW;

#endif /* __APPLE__ */

#if defined(__APPLE__)

#define signbit(x) \
        (sizeof(x) == sizeof(float) ? __signbitf(x) : sizeof(x) == sizeof(double) ? __signbitd(x) : __signbitl(x))
#define isfinite(x) \
        (sizeof(x) == sizeof(float) ? __isfinitef(x) : sizeof(x) == sizeof(double) ? __isfinited(x) : __isfinite(x))

#else /* __APPLE__ */

#define signbit(x) \
        (sizeof(x) == sizeof(float) ? __signbitf(x) : sizeof(x) == sizeof(double) ? __signbit(x) : __signbitl(x))
#define isfinite(x) \
        (sizeof(x) == sizeof(float) ? __finitef(x) : sizeof(x) == sizeof(double) ? __finite(x) : __finitel(x))

#endif /* __APPLE__ */

#define isnan(x) \
        (sizeof(x) == sizeof(float) ? __isnanf(x) : sizeof(x) == sizeof(double) ? __isnan(x) : __isnanl(x))
#define isinf(x) \
        (sizeof(x) == sizeof(float) ? __isinff(x) : sizeof(x) == sizeof(double) ? __isinf(x) : __isinfl(x))

namespace __gnu_cxx
{
  extern __host__ __device__ long long int abs(long long int);
}

namespace std
{
  template<typename T> extern __host__ __device__ T __pow_helper(T, int);
  template<typename T> extern __host__ __device__ T __cmath_power(T, unsigned int);
}

using std::abs;
using std::fabs;
using std::ceil;
using std::floor;
using std::sqrt;
using std::pow;
using std::log;
using std::log10;
using std::fmod;
using std::modf;
using std::exp;
using std::frexp;
using std::ldexp;
using std::asin;
using std::sin;
using std::sinh;
using std::acos;
using std::cos;
using std::cosh;
using std::atan;
using std::atan2;
using std::tan;
using std::tanh;

#elif defined(_WIN32)

static __inline__ __host__ __device__ long long int abs(long long int a)
{
  return llabs(a);
}

static __inline__ __host__ __device__ int signbit(double a)
{
  return __signbit(a);
}

static __inline__ __host__ __device__ int signbit(float a)
{
  return __signbitf(a);
}

static __inline__ __host__ __device__ int isinf(double a)
{
  return __isinf(a);
}

static __inline__ __host__ __device__ int isinf(float a)
{
  return __isinff(a);
}

static __inline__ __host__ __device__ int isnan(double a)
{
  return __isnan(a);
}

static __inline__ __host__ __device__ int isnan(float a)
{
  return __isnanf(a);
}

static __inline__ __host__ __device__ int isfinite(double a)
{
  return __finite(a);
}

static __inline__ __host__ __device__ int isfinite(float a)
{
  return __finitef(a);
}

template<class T> extern __host__ __device__ T _Pow_int(T, int);

#endif /* !_WIN32 */

#if defined(__GNUC__)
namespace std {
#endif /* __GNUC__ */

extern __host__ __device__ long int abs(long int);
extern __host__ __device__ float    abs(float);
extern __host__ __device__ double   abs(double);
extern __host__ __device__ float    fabs(float);
extern __host__ __device__ float    ceil(float);
extern __host__ __device__ float    floor(float);
extern __host__ __device__ float    sqrt(float);
extern __host__ __device__ float    pow(float, float);
extern __host__ __device__ float    pow(float, int);
extern __host__ __device__ double   pow(double, int);
extern __host__ __device__ float    log(float);
extern __host__ __device__ float    log10(float);
extern __host__ __device__ float    fmod(float, float);
extern __host__ __device__ float    modf(float, float*);
extern __host__ __device__ float    exp(float);
extern __host__ __device__ float    frexp(float, int*);
extern __host__ __device__ float    ldexp(float, int);
extern __host__ __device__ float    asin(float);
extern __host__ __device__ float    sin(float);
extern __host__ __device__ float    sinh(float);
extern __host__ __device__ float    acos(float);
extern __host__ __device__ float    cos(float);
extern __host__ __device__ float    cosh(float);
extern __host__ __device__ float    atan(float);
extern __host__ __device__ float    atan2(float, float);
extern __host__ __device__ float    tan(float);
extern __host__ __device__ float    tanh(float);

#if defined(__GNUC__)
}
#endif /* __GNUC__ */

static __inline__ __host__ __device__ float logb(float a)
{
  return logbf(a);
}

static __inline__ __host__ __device__ int ilogb(float a)
{
  return ilogbf(a);
}

static __inline__ __host__ __device__ float scalbn(float a, int b)
{
  return scalbnf(a, b);
}

static __inline__ __host__ __device__ float scalbln(float a, long int b)
{
  return scalblnf(a, b);
}

static __inline__ __host__ __device__ float exp2(float a)
{
  return exp2f(a);
}

static __inline__ __host__ __device__ float exp10(float a)
{
  return exp10f(a);
}

static __inline__ __host__ __device__ float expm1(float a)
{
  return expm1f(a);
}

static __inline__ __host__ __device__ float log2(float a)
{
  return log2f(a);
}

static __inline__ __host__ __device__ float log1p(float a)
{
  return log1pf(a);
}

static __inline__ __host__ __device__ float rsqrt(float a)
{
  return rsqrtf(a);
}

static __inline__ __host__ __device__ float acosh(float a)
{
  return acoshf(a);
}

static __inline__ __host__ __device__ float asinh(float a)
{
  return asinhf(a);
}

static __inline__ __host__ __device__ float atanh(float a)
{
  return atanhf(a);
}

static __inline__ __host__ __device__ float hypot(float a, float b)
{
  return hypotf(a, b);
}

static __inline__ __host__ __device__ float cbrt(float a)
{
  return cbrtf(a);
}

static __inline__ __host__ __device__ void sincos(float a, float *sptr, float *cptr)
{
  sincosf(a, sptr, cptr);
}

static __inline__ __host__ __device__ float erf(float a)
{
  return erff(a);
}

static __inline__ __host__ __device__ float erfc(float a)
{
  return erfcf(a);
}

static __inline__ __host__ __device__ float lgamma(float a)
{
  return lgammaf(a);
}

static __inline__ __host__ __device__ float tgamma(float a)
{
  return tgammaf(a);
}

static __inline__ __host__ __device__ float copysign(float a, float b)
{
  return copysignf(a, b);
}

static __inline__ __host__ __device__ double copysign(double a, float b)
{
  return copysign(a, (double)b);
}

static __inline__ __host__ __device__ float copysign(float a, double b)
{
  return copysignf(a, (float)b);
}

static __inline__ __host__ __device__ float nextafter(float a, float b)
{
  return nextafterf(a, b);
}

static __inline__ __host__ __device__ float remainder(float a, float b)
{
  return remainderf(a, b);
}

static __inline__ __host__ __device__ float remquo(float a, float b, int *quo)
{
  return remquof(a, b, quo);
}

static __inline__ __host__ __device__ float round(float a)
{
  return roundf(a);
}

static __inline__ __host__ __device__ long int lround(float a)
{
  return lroundf(a);
}

static __inline__ __host__ __device__ long long int llround(float a)
{
  return llroundf(a);
}

static __inline__ __host__ __device__ float trunc(float a)
{
  return truncf(a);
}

static __inline__ __host__ __device__ float rint(float a)
{
  return rintf(a);
}

static __inline__ __host__ __device__ long int lrint(float a)
{
  return lrintf(a);
}

static __inline__ __host__ __device__ long long int llrint(float a)
{
  return llrintf(a);
}

static __inline__ __host__ __device__ float nearbyint(float a)
{
  return nearbyintf(a);
}

static __inline__ __host__ __device__ float fdim(float a, float b)
{
  return fdimf(a, b);
}

static __inline__ __host__ __device__ float fma(float a, float b, float c)
{
  return fmaf(a, b, c);
}

static __inline__ __host__ __device__ unsigned int min(unsigned int a, unsigned int b)
{
  return umin(a, b);
}

static __inline__ __host__ __device__ unsigned int min(int a, unsigned int b)
{
  return umin((unsigned int)a, b);
}

static __inline__ __host__ __device__ unsigned int min(unsigned int a, int b)
{
  return umin(a, (unsigned int)b);
}

static __inline__ __host__ __device__ float min(float a, float b)
{
  return fminf(a, b);
}

static __inline__ __host__ __device__ double min(double a, double b)
{
  return fmin(a, b);
}

static __inline__ __host__ __device__ double min(float a, double b)
{
  return fmin((double)a, b);
}

static __inline__ __host__ __device__ double min(double a, float b)
{
  return fmin(a, (double)b);
}

static __inline__ __host__ __device__ unsigned int max(unsigned int a, unsigned int b)
{
  return umax(a, b);
}

static __inline__ __host__ __device__ unsigned int max(int a, unsigned int b)
{
  return umax((unsigned int)a, b);
}

static __inline__ __host__ __device__ unsigned int max(unsigned int a, int b)
{
  return umax(a, (unsigned int)b);
}

static __inline__ __host__ __device__ float max(float a, float b)
{
  return fmaxf(a, b);
}

static __inline__ __host__ __device__ double max(double a, double b)
{
  return fmax(a, b);
}

static __inline__ __host__ __device__ double max(float a, double b)
{
  return fmax((double)a, b);
}

static __inline__ __host__ __device__ double max(double a, float b)
{
  return fmax(a, (double)b);
}

#elif !defined(__CUDACC__)

#include "crt/func_macro.h"

#define INT_MAX \
        ((int)((unsigned int)-1 >> 1))

#if defined(__GNUC__)

__func__(int __cuda_error_not_implememted(void));

#define __cuda___signbitl(a) \
        __cuda_error_not_implememted()
#define __cuda___isinfl(a) \
        __cuda_error_not_implememted()
#define __cuda___isnanl(a) \
        __cuda_error_not_implememted()

#if defined(__APPLE__)

#define __cuda___isfinite(a) \
        __cuda_error_not_implememted()

#else /* __APPLE__ */

#define __cuda___finitel(a) \
        __cuda_error_not_implememted()

#endif /* __APPLE__ */

#endif /* __GNUC__ */

#if !defined(__CUDABE__) && !defined(__MULTI_CORE__)

#if defined(_WIN32)

__func__(double log2(double a))
{
  return log(a) / log(2.0);
}

__func__(float log2f(float a))
{
  return (float)log2((double)a);
}

__func__(double exp2(double a))
{
  return pow(2.0, a);
}

__func__(float exp2f(float a))
{
  return (float)exp2((double)a);
}

__func__(long long int llabs(long long int a))
{
  return a < 0ll ? -a : a;
}

#endif /* _WIN32 */

#endif /* !__CUDABE__ && !__MULTI_CORE__ */

__device_func__(int __cuda_abs(int a))
{
  return abs(a);
}

__device_func__(float __cuda_fabsf(float a))
{
  return fabsf(a);
}

__device_func__(long long int __cuda_llabs(long long int a))
{
#if defined (__cplusplus)
  return ::llabs(a);
#else /* __cplusplus */
  return llabs(a);
#endif /* __cplusplus */
}

__device_func__(float __cuda_exp2f(float a))
{
  return exp2f(a);
}

#include "device_functions_dynlink.h"
#include "math_constants.h"

__device_func__(int __cuda___signbitf(float a))
{
  return (int)((unsigned int)__float_as_int(a) >> 31);
}

/* The copysign() function returns a with its sign changed to 
 * match the sign of b.
 */
__device_func__(float __cuda_copysignf(float a, float b))
{
  return __int_as_float((__float_as_int(b) &  0x80000000) | 
                        (__float_as_int(a) & ~0x80000000));
}

#if !defined(__CUDABE__) && !defined(__MULTI_CORE__)

/*******************************************************************************
*                                                                              *
* HOST UTILITY IMPLEMENTATIONS                                                 *
*                                                                              *
*******************************************************************************/

__func__(int min(int a, int b))
{
  return a < b ? a : b;
}

__func__(unsigned int umin(unsigned int a, unsigned int b))
{
  return a < b ? a : b;
}

__func__(int max(int a, int b))
{
  return a > b ? a : b;
}

__func__(unsigned int umax(unsigned int a, unsigned int b))
{
  return a > b ? a : b;
}

#if defined(_WIN32)

__func__(double fmax(double a, double b))
{
  return a > b ? a : b;
}

__func__(double fmin(double a, double b))
{
  return a < b ? a : b;
}

__func__(float fmaxf(float a, float b))
{
  return (float)fmax((double)a, (double)b);
}

__func__(float fminf(float a, float b))
{
  return (float)fmin((double)a, (double)b);
}

__func__(int __signbit(double a))
{
  volatile union {
    double               d;
    signed long long int l;
  } cvt;

  cvt.d = a;

  return cvt.l < 0ll;
}

__func__(double copysign(double a, double b))
{
  volatile union {
    double                 d;
    unsigned long long int l;
  } cvta, cvtb;

  cvta.d = a;
  cvtb.d = b;
  cvta.l = (cvta.l & 0x7fffffffffffffffULL) | (cvtb.l & 0x8000000000000000ULL);

  return cvta.d;
}

__func__(int __signbitf(float a))
{
  return __cuda___signbitf(a);
}

__func__(float copysignf(float a, float b))
{
  return __cuda_copysignf(a, b);
}

#endif /* _WIN32 */

/*******************************************************************************
*                                                                              *
* HOST IMPLEMENTATIONS FOR FUNCTIONS WITH BUILTIN NVOPENCC OPREATIONS          *
*                                                                              *
*******************************************************************************/

__device_func__(float __internal_nearbyintf(float a))
{
  float fa = fabsf(a);

  if (fa >= CUDART_TWO_TO_23_F) {
    return a;
  } else {
    volatile float u = CUDART_TWO_TO_23_F + fa;

    u = u - CUDART_TWO_TO_23_F;
    return copysignf(u, a);
  }
}

__device_func__(float __internal_fminf(float a, float b))
{
  volatile union {
    float        f;
    unsigned int i;
  } cvta, cvtb;

  cvta.f = a;
  cvtb.f = b;
  if ((cvta.i << 1) > 0xff000000) return b;
  if ((cvtb.i << 1) > 0xff000000) return a;
  if ((cvta.i | cvtb.i) == 0x80000000) {
    return CUDART_NEG_ZERO_F;
  }
  return a < b ? a : b;
}

__device_func__(float __internal_fmaxf(float a, float b))
{
  volatile union {
    float        f;
    unsigned int i;
  } cvta, cvtb;

  cvta.f = a;
  cvtb.f = b;
  if ((cvta.i << 1) > 0xff000000) return b;
  if ((cvtb.i << 1) > 0xff000000) return a;
  if ((cvta.f == 0.0f) && (cvtb.f == 0.0f)) {
    cvta.i &= cvtb.i;
    return cvta.f;
  }
  return a > b ? a : b;
}

#if defined(_WIN32)

__func__(double trunc(double a))
{
  return a < 0.0 ? ceil(a) : floor(a);
}

__func__(double nearbyint(double a))
{
  double fa = fabs(a);
  if (fa >= CUDART_TWO_TO_52) {
    return a;
  } else {
    double u = CUDART_TWO_TO_52 + fa;
    u = u - CUDART_TWO_TO_52;
    return copysign(u, a);
  }
}

__func__(float truncf(float a))
{
  return (float)trunc((double)a);
}

__func__(float nearbyintf(float a))
{
  return __internal_nearbyintf(a);
}

#endif /* _WIN32 */

#endif /* !__CUDABE__ && !__MULTI_CORE__ */

/*******************************************************************************
*                                                                              *
* DEVICE IMPLEMENTATIONS FOR FUNCTIONS WITH BUILTIN NVOPENCC OPERATIONS        *
*                                                                              *
*******************************************************************************/

__device_func__(long int __cuda_labs(long int a))
{
  return labs(a);
}

__device_func__(float __cuda_ceilf(float a))
{
  return ceilf(a);
}

__device_func__(float __cuda_floorf(float a))
{
  return floorf(a);
}

__device_func__(float __cuda_sqrtf(float a))
{
   return sqrtf(a);
}

__device_func__(float __cuda_rsqrtf(float a))
{
   return 1.0f / sqrtf(a);
}

__device_func__(float __cuda_truncf(float a))
{
  return truncf(a);
}

__device_func__(int __cuda_max(int a, int b))
{
  return max(a, b);
}

__device_func__(int __cuda_min(int a, int b))
{
  return min(a, b);
}

__device_func__(unsigned int __cuda_umax(unsigned int a, unsigned int b))
{
  return umax(a, b);
}

__device_func__(unsigned int __cuda_umin(unsigned int a, unsigned int b))
{
  return umin(a, b);
}

__device_func__(long long int __cuda_llrintf(float a))
{
#if defined(__MULTI_CORE__)
  return llrintf(a);
#else /* __MULTI_CORE__ */
  return __float2ll_rn(a);
#endif /* __MULTI_CORE__ */
}

__device_func__(long int __cuda_lrintf(float a))
{
#if defined(__MULTI_CORE__)
  return lrintf(a);
#else /* __MULTI_CORE__ */
#if defined(__LP64__)
  return (long int)__cuda_llrintf(a);
#else /* __LP64__ */
  return (long int)__float2int_rn(a);
#endif /* __LP64__ */
#endif /* __MULTI_CORE__ */
}

__device_func__(float __cuda_nearbyintf(float a))
{
#if defined(__MULTI_CORE__)
  return nearbyintf(a);
#elif defined(__CUDABE__)
  return roundf(a);
#else /* __CUDABE__ */
  return __internal_nearbyintf(a);
#endif /* __CUDABE__ */
}

__device_func__(float __cuda_fmaxf(float a, float b))
{
#if defined(__MULTI_CORE__)
  return fmaxf(a, b);
#elif defined(__CUDABE__)
  return fmaxf(a, b);
#else /* __MULTI_CORE__ */
  return __internal_fmaxf(a, b);
#endif /* __MULTI_CORE__ */
}

__device_func__(float __cuda_fminf(float a, float b))
{
#if defined(__MULTI_CORE__)
  return fminf(a, b);
#elif defined(__CUDABE__)
  return fminf(a, b);
#else /* __MULTI_CORE__ */
  return __internal_fminf(a, b);
#endif /* __MULTI_CORE__ */
}

/*******************************************************************************
*                                                                              *
* DEVICE IMPLEMENTATIONS FOR FUNCTIONS WITHOUT BUILTIN NVOPENCC OPERATIONS     *
*                                                                              *
*******************************************************************************/
/*!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 *  NOTE: Any division operators in this section are mapped to fast division,
 *       i.e. a reciprocal followed by a multiply. This produces incorrect
 *       results if the divisor is greater than 2^126, in which case zero is
 *       returned due to underflow-with-flush-to-zero in the reciprocal. If
 *       full range division is required, it is required to call the device 
 *       function __internal_accurate_fdividef(). Examples are implementations
 *       of the functions __cuda_atan2f() and __cuda_hypotf() .
 *!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 */

__device_func__(int __cuda___finitef(float a))
{
  return __cuda_fabsf(a) < CUDART_INF_F;
}

#if defined(__APPLE__)

__device_func__(int __cuda___isfinitef(float a))
{
  return __cuda___finitef(a);
}

#endif /* __APPLE__ */

__device_func__(int __cuda___isinff(float a))
{
  return __cuda_fabsf(a) == CUDART_INF_F;
}

__device_func__(int __cuda___isnanf(float a))
{
  return !(__cuda_fabsf(a) <= CUDART_INF_F);
}

__device_func__(float __cuda_nextafterf(float a, float b))
{
  unsigned int ia;
  unsigned int ib;
  ia = __float_as_int(a);
  ib = __float_as_int(b);
#if defined(__CUDABE__)
  if ((ia << 1) < 0x01000000) ia &= 0x80000000; /* DAZ */
  if ((ib << 1) < 0x01000000) ib &= 0x80000000; /* DAZ */
#endif /* __CUDABE__ */
  if (__cuda___isnanf(a) || __cuda___isnanf(b)) return a + b; /*NaN*/
  if (__int_as_float (ia | ib) == 0.0f) return __int_as_float(ib);
#if defined(__CUDABE__)
  if (__int_as_float(ia) == 0.0f) {
    return __cuda_copysignf(CUDART_TWO_TO_M126_F, b); /* FTZ */
  }
#else /* __CUDABE__ */
  if (__int_as_float(ia) == 0.0f) {
    return __cuda_copysignf(CUDART_MIN_DENORM_F, b); /*crossover*/
  }
#endif /* __CUDABE__ */
  if ((a < b) && (a < 0.0f)) ia--;
  if ((a < b) && (a > 0.0f)) ia++;
  if ((a > b) && (a < 0.0f)) ia++;
  if ((a > b) && (a > 0.0f)) ia--;
  a = __int_as_float(ia);
#if defined(__CUDABE__)
  if (__cuda_fabsf(a) < CUDART_TWO_TO_M126_F) {
    a = __int_as_float(ia & 0x80000000); /* FTZ */
  }
#endif /* __CUDABE__ */
  return a;
}

__device_func__(float __cuda_nanf(const char *tagp))
{
  /* the GPU only has one canonical QNaN, so return that */
  return CUDART_NAN_F;
}

/* approximate 2*atanh(a/2) for |a| < 0.245 */
__device_func__(float __internal_atanhf_kernel(float a_1, float a_2))
{
  float a, a2, t;

  a = a_1 + a_2;
  a2 = a * a;    
  t =          1.566305595598990E-001f/64.0f;
  t = t * a2 + 1.995081856004762E-001f/16.0f;
  t = t * a2 + 3.333382699617026E-001f/4.0f;
  t = t * a2;
  t = t * a + a_2;
  t = t + a_1;
  return t;
}  

/* compute atan(r) in first octant, i.e. 0 <= r <= 1
 * eps ~= 2.16e-7
 */
__device_func__(float __internal_atanf_kernel(float a))
{
  float t4, t0, t1;

  t4 = a * a;
  t0 =         - 5.674867153f;
  t0 = t4 *    - 0.823362947f + t0;
  t0 = t0 * t4 - 6.565555096f;
  t0 = t0 * t4;
  t0 = t0 * a;
  t1 = t4      + 11.33538818f;
  t1 = t1 * t4 + 28.84246826f;
  t1 = t1 * t4 + 19.69667053f;
  t1 = 1.0f / t1;
  a = t0 * t1 + a;
  return a;
}

/* approximate tangent on -pi/4...+pi/4 */
__device_func__(float __internal_tan_kernel(float a))
{
  float a2, s, t;

  a2 = a * a;
  t  = 4.114678393115178E-003f * a2 - 8.231194034909670E-001f;
  s  = a2 - 2.469348886157666E+000f;
  s  = 1.0f / s;
  t  = t * s;
  t  = t * a2;
  t  = t * a + a;
  return t;
}

__device_func__(float __internal_accurate_logf(float a))
{
  float t;
  float z;
  float m;
  int ia, e;
  ia = __float_as_int(a);
  /* handle special cases */
  if ((ia < 0x00800000) || (ia > 0x7f7fffff)) {
    return __logf(a);
  }
  /* log(a) = 2 * atanh((a-1)/(a+1)) */
  m = __int_as_float((ia & 0x807fffff) | 0x3f800000);
  e = ((unsigned)ia >> 23) - 127;
  if (m > CUDART_SQRT_TWO_F) {
    m = m * 0.5f;
    e = e + 1;
  }      
  t = m - 1.0f;
  z = m + 1.0f;
  z = t / z;
  z = -t * z;
  z = __internal_atanhf_kernel(t, z);
  z = (float)e * CUDART_LN2_F + z;
  return z;
}  

__device_func__(float2 __internal_log_ep(float a))
{
  float2 res;
  int expo;
  float m;
  float log_hi, log_lo;
  float t_hi, t_lo;
  float f, g, u, v, q;
#if !defined(__CUDABE__) && defined(__linux__) && !defined(__LP64__)
  volatile float r, s, e;
#else
  float r, s, e;
#endif
  expo = (__float_as_int(a) >> 23) & 0xff;
#if !defined(__CUDABE__)
  /* convert denormals to normals for computation of log(a) */
  if (expo == 0) {
    a *= CUDART_TWO_TO_24_F;
    expo = (__float_as_int(a) >> 23) & 0xff;
    expo -= 24;
  }  
#endif
  expo -= 127;
  m = __int_as_float((__float_as_int(a) & 0x807fffff) | 0x3f800000);
  if (m > CUDART_SQRT_TWO_F) {
    m = m * 0.5f;
    expo = expo + 1;
  }

  /* compute log(m) with extended precision using an algorithm from P.T.P.
   * Tang, "Table Driven Implementation of the Logarithm Function", TOMS, 
   * Vol. 16., No. 4, December 1990, pp. 378-400. A modified polynomial 
   * approximation to atanh(x) on the interval [-0.1716, 0.1716] is utilized.
   */
  f = m - 1.0f;
  g = m + 1.0f;
  g = 1.0f / g;
  u = 2.0f * f * g;
  v = u * u;
  q =         1.49356810919559350E-001f/64.0f;
  q = q * v + 1.99887797540072460E-001f/16.0f;
  q = q * v + 3.33333880955515580E-001f/4.0f;
  q = q * v;
  q = q * u;
  log_hi = __int_as_float(__float_as_int(u) & 0xfffff000);
  v = __int_as_float(__float_as_int(f) & 0xfffff000);
  u = 2.0f * (f - log_hi);
  f = f - v;
  u = u - log_hi * v;
  u = u - log_hi * f;
  u = g * u;
  /* compute log(m) = log_hi + u + q in double-single format*/

  /* log += u; |log| > |u| */
  r = log_hi + u;
  s = u - (r - log_hi);
  log_hi = r;
  log_lo = s;
  /* log += q; |log| > |q| */
  r = log_hi + q;
  s = ((log_hi - r) + q) + log_lo;
  log_hi = e = r + s;
  log_lo = (r - e) + s;

  /* log(2)*expo in double-single format */
  t_hi = expo * 0.6931457519f;    /* multiplication is exact */
  t_lo = expo * 1.4286067653e-6f;

  /* log(a) = log(m) + log(2)*expo;  if expo != 0, |log(2)*expo| > |log(m)| */
  r = t_hi + log_hi;
  s = (((t_hi - r) + log_hi) + log_lo) + t_lo;
  res.y = e = r + s;
  res.x = (r - e) + s;
  return res;
}

__device_func__(float __internal_accurate_log2f(float a))
{
  return CUDART_L2E_F * __internal_accurate_logf(a);
}

/* Based on: Guillaume Da Graça, David Defour. Implementation of Float-Float 
 * Operators on Graphics Hardware. RNC'7, pp. 23-32, 2006.
 */
__device_func__(float2 __internal_dsmul (float2 x, float2 y))
{
    float2 z;
#if !defined(__CUDABE__)
    volatile float up, vp, u1, u2, v1, v2, mh, ml;
#else
    float up, vp, u1, u2, v1, v2, mh, ml;
#endif /* defined(__CUDABE__) */
    up  = x.y * 4097.0f;
    u1  = (x.y - up) + up;
    u2  = x.y - u1;
    vp  = y.y * 4097.0f;
    v1  = (y.y - vp) + vp;
    v2  = y.y - v1;
    mh  = __fmul_rn(x.y,y.y);
    ml  = (((u1 * v1 - mh) + u1 * v2) + u2 * v1) + u2 * v2;
    ml  = (__fmul_rn(x.y,y.x) + __fmul_rn(x.x,y.y)) + ml;
    z.y = up = mh + ml;
    z.x = (mh - up) + ml;
    return z;
}

/* 160 bits of 2/PI for Payne-Hanek style argument reduction. */
static __constant__ unsigned int __cudart_i2opi_f [] = {
  0x3c439041,
  0xdb629599,
  0xf534ddc0,
  0xfc2757d1,
  0x4e441529,
  0xa2f9836e,
};

/* reduce argument to trig function to -pi/4...+pi/4 */
__device_func__(float __internal_trig_reduction_kernel(float a, int *quadrant))
{
  float j;
  int q;
  if (__cuda_fabsf(a) > CUDART_TRIG_PLOSS_F) {
    /* Payne-Hanek style argument reduction. */
    unsigned int ia = __float_as_int(a);
    unsigned int s = ia & 0x80000000;
    unsigned int result[7];
    unsigned int phi, plo;
    unsigned int hi, lo;
    unsigned int e;
    int idx;
    e = ((ia >> 23) & 0xff) - 128;
    ia = (ia << 8) | 0x80000000;
    /* compute x * 2/pi */
    idx = 4 - (e >> 5);
    hi = 0;
#if defined(__CUDABE__)
#pragma unroll 1
#endif /* __CUDABE__ */
    for (q = 0; q < 6; q++) {
      plo = __cudart_i2opi_f[q] * ia;
      phi = __umulhi (__cudart_i2opi_f[q], ia);
      lo = hi + plo;
      hi = phi + (lo < plo);
      result[q] = lo;
    }
    result[q] = hi;
    e = e & 31;
    /* shift result such that hi:lo<63:62> are the least significant
       integer bits, and hi:lo<61:0> are the fractional bits of the result
     */
    hi = result[idx+2];
    lo = result[idx+1];
    if (e) {
      q = 32 - e;
      hi = (hi << e) | (lo >> q);
      lo = (lo << e) | (result[idx] >> q);
    }
    q = hi >> 30;
    /* fraction */
    hi = (hi << 2) | (lo >> 30);
    lo = (lo << 2);
    e = (hi + (lo > 0)) > 0x80000000; /* fraction >= 0.5 */
    q += e;
    if (s) q = -q;
    if (e) {
      unsigned int t;
      hi = ~hi;
      lo = -(int)lo;
      t = (lo == 0);
      hi += t;
      s = s ^ 0x80000000;
    }
    *quadrant = q;
    /* normalize fraction */
    e = 0;
    while ((int)hi > 0) {
      hi = (hi << 1) | (lo >> 31);
      lo = (lo << 1);
      e--;
    }
    lo = hi * 0xc90fdaa2;
    hi = __umulhi(hi, 0xc90fdaa2);
    if ((int)hi > 0) {
      hi = (hi << 1) | (lo >> 31);
      lo = (lo << 1);
      e--;
    }
    hi = hi + (lo > 0);
    ia = s | (((e + 126) << 23) + (hi >> 8) + ((hi << 24) >= 0x80000000));
    return __int_as_float(ia);
  }
  q = __float2int_rn(a * CUDART_2_OVER_PI_F);
  j = (float)q;
  a = a - j * 1.5703125000000000e+000f;
  a = a - j * 4.8351287841796875e-004f;
  a = a - j * 3.1385570764541626e-007f;
  a = a - j * 6.0771005065061922e-011f;
  *quadrant = q;
  return a;
}

/* High quality implementation of expf(). A naive implementation, expf(x) =
 * exp2f (x * log2(e)), loses significant accuracy for large arguments, and
 * may return results with only 15 to 16 good bits (out of 24). The present
 * implementation limits the error to about 2 ulps across the entire argument
 * range. It does so by employing an extended precision representation for
 * ln(2) which is composited from ln2_hi = 0.6931457519f which provides the
 * most significant 16-bit of ln(2), and ln2_lo = 1.4286067653e-6f, which
 * provides the least significant 24 bits.
 */
__device_func__(float __internal_expf_kernel(float a, float scale))
{
  float j, z;

  j = __cuda_truncf(a * CUDART_L2E_F);
  z = a - j * 0.6931457519f;
  z = z - j * 1.4286067653e-6f;
  z = z * CUDART_L2E_F;
  z = __cuda_exp2f(z) * __cuda_exp2f(j + scale);
  return z;
}

__device_func__(float __internal_accurate_expf(float a))
{
  float z;
  z = __internal_expf_kernel(a, 0.0f);
  if (a < -105.0f) z = 0.0f;
  if (a >  105.0f) z = CUDART_INF_F;
  return z;
}

__device_func__(float __internal_accurate_exp10f(float a))
{
  float j, z;
  j = __cuda_truncf(a * CUDART_L2T_F);
  z = a - j * 3.0102920532226563e-001f;
  z = z - j * 7.9034171557301747e-007f;
  z = z * CUDART_L2T_F;
  z = __cuda_exp2f(z) * __cuda_exp2f(j);
  if (a < -46.0f) z = 0.0f;
  if (a >  46.0f) z = CUDART_INF_F;
  return z;
}

__device_func__(float __internal_lgammaf_pos(float a))
{
  float sum;
  float s, t;

  if (a == CUDART_INF_F) {
    return a;
  }
  if (a >= 3.0f) {
    if (a >= 7.8f) {
      /* Stirling approximation for a >= 8; coefficients from Hart et al, 
       * "Computer Approximations", Wiley 1968. Approximation 5401
       */
      s = 1.0f / a;
      t = s * s;
      sum =           0.77783067e-3f;
      sum = sum * t - 0.2777655457e-2f;
      sum = sum * t + 0.83333273853e-1f;
      sum = sum * s + 0.918938533204672f;
      s = 0.5f * __internal_accurate_logf(a);
      t = a - 0.5f;
      s = s * t;
      t = s - a;
      s = __fadd_rn(s, sum); /* prevent FMAD merging */
      t = t + s;
      return t;
    } else {
      a = a - 3.0f;
      s =       - 7.488903254816711E+002f;
      s = s * a - 1.234974215949363E+004f;
      s = s * a - 4.106137688064877E+004f;
      s = s * a - 4.831066242492429E+004f;
      s = s * a - 1.430333998207429E+005f;
      t =     a - 2.592509840117874E+002f;
      t = t * a - 1.077717972228532E+004f;
      t = t * a - 9.268505031444956E+004f;
      t = t * a - 2.063535768623558E+005f;
      t = s / t;
      t = t + a;
      return t;
    }
  } else if (a >= 1.5f) {
    a = a - 2.0f;
    t =       + 4.959849168282574E-005f;
    t = t * a - 2.208948403848352E-004f;
    t = t * a + 5.413142447864599E-004f;
    t = t * a - 1.204516976842832E-003f;
    t = t * a + 2.884251838546602E-003f;
    t = t * a - 7.382757963931180E-003f;
    t = t * a + 2.058131963026755E-002f;
    t = t * a - 6.735248600734503E-002f;
    t = t * a + 3.224670187176319E-001f;
    t = t * a + 4.227843368636472E-001f;
    t = t * a;
    return t;
  } else if (a >= 0.7f) {
    a = 1.0f - a;
    t =       + 4.588266515364258E-002f;
    t = t * a + 1.037396712740616E-001f;
    t = t * a + 1.228036339653591E-001f;
    t = t * a + 1.275242157462838E-001f;
    t = t * a + 1.432166835245778E-001f;
    t = t * a + 1.693435824224152E-001f;
    t = t * a + 2.074079329483975E-001f;
    t = t * a + 2.705875136435339E-001f;
    t = t * a + 4.006854436743395E-001f;
    t = t * a + 8.224669796332661E-001f;
    t = t * a + 5.772156651487230E-001f;
    t = t * a;
    return t;
  } else {
    t =       + 3.587515669447039E-003f;
    t = t * a - 5.471285428060787E-003f;
    t = t * a - 4.462712795343244E-002f;
    t = t * a + 1.673177015593242E-001f;
    t = t * a - 4.213597883575600E-002f;
    t = t * a - 6.558672843439567E-001f;
    t = t * a + 5.772153712885004E-001f;
    t = t * a;
    t = t * a + a;
    return -__internal_accurate_logf(t);
  }
}

/* approximate sine on -pi/4...+pi/4 */
__device_func__(float __internal_sin_kernel(float x))
{
  float x2, z;

  x2 = x * x;
  z  =        - 1.95152959e-4f;
  z  = z * x2 + 8.33216087e-3f;
  z  = z * x2 - 1.66666546e-1f;
  z  = z * x2;
  z  = z * x + x;

  return z;
}

/* approximate cosine on -pi/4...+pi/4 */
__device_func__(float __internal_cos_kernel(float x))
{
  float x2, z;

  x2 = x * x;
  z  =          2.44331571e-5f;
  z  = z * x2 - 1.38873163e-3f;
  z  = z * x2 + 4.16666457e-2f;
  z  = z * x2 - 5.00000000e-1f;
  z  = z * x2 + 1.00000000e+0f;
  return z;
}

__device_func__(float __internal_accurate_sinf(float a))
{
  float z;
  int   i;

  if ((__cuda___isinff(a)) || (a == CUDART_ZERO_F)) {
    return __fmul_rn (a, CUDART_ZERO_F);
  }
  z = __internal_trig_reduction_kernel(a, &i);
  /* here, abs(z) <= pi/4, and i has the quadrant */
  if (i & 1) {
    z = __internal_cos_kernel(z);
  } else {
    z = __internal_sin_kernel(z);
  }
  if (i & 2) {
    z = -z;
  }
  return z;
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__device_func__(float __cuda_rintf(float a))
{
#if defined(__MULTI_CORE__)
  return rintf(a);
#else /* __MULTI_CORE__ */
  return __cuda_nearbyintf(a);
#endif /* __MULTI_CORE__ */
}

__device_func__(float __cuda_sinf(float a))
{
#if defined(__MULTI_CORE__)
  return sinf(a);
#elif defined(__USE_FAST_MATH__)
  return __sinf(a);
#else /* __MULTI_CORE__ */
  return __internal_accurate_sinf(a);
#endif /* __MULTI_CORE__ */
}

__device_func__(float __cuda_cosf(float a))
{
#if defined(__MULTI_CORE__)
  return cosf(a);
#elif defined(__USE_FAST_MATH__)
  return __cosf(a);
#else /* __MULTI_CORE__ */
  float z;
  int i;

  if (__cuda___isinff(a)) {
    return CUDART_NAN_F;
  }
  z = __internal_trig_reduction_kernel(a, &i);
  /* here, abs(z) <= pi/4, and i has the quadrant */
  i++;
  if (i & 1) {
    z = __internal_cos_kernel(z);
  } else {
    z = __internal_sin_kernel(z);
  }
  if (i & 2) {
    z = -z;
  }
  return z;
#endif /* __MULTI_CORE__ */
}

__device_func__(float __cuda_tanf(float a))
{
#if defined(__MULTI_CORE__)
  return tanf(a);
#elif defined(__USE_FAST_MATH__)
  return __tanf(a);
#else /* __MULTI_CORE__ */
  float z;
  int   i;

  if (__cuda___isinff(a)) {
    return CUDART_NAN_F;
  }
  z = __internal_trig_reduction_kernel(a, &i);
  /* here, abs(z) <= pi/4, and i has the quadrant */
  z = __internal_tan_kernel(z);
  if (i & 1) {
    z = -1.0f / z;
  }
  return z;
#endif /* __MULTI_CORE__ */
}

__device_func__(float __cuda_log2f(float a))
{
#if defined(__MULTI_CORE__)
  return log2f(a);
#elif defined(__USE_FAST_MATH__)
  return __log2f(a);
#else /* __MULTI_CORE__ */
  return __internal_accurate_log2f(a);
#endif /* __MULTI_CORE__ */
}

__device_func__(float __cuda_expf(float a))
{
#if defined(__MULTI_CORE__)
  return expf(a);
#elif defined(__USE_FAST_MATH__)
  return __expf(a);
#else /* __MULTI_CORE__ */
  return __internal_accurate_expf(a);
#endif /* __MULTI_CORE__*/
}

__device_func__(float __cuda_exp10f(float a))
{
#if defined(__MULTI_CORE__)
  return exp10f(a);
#elif defined(__USE_FAST_MATH__)
  return __exp10f(a);
#else /* __MULTI_CORE__ */
  return __internal_accurate_exp10f(a);
#endif /* __MULTI_CORE__ */
}

__device_func__(float __cuda_coshf(float a))
{
  float z;

  a = __cuda_fabsf(a);
  z = __internal_expf_kernel(a, -2.0f);
  z = 2.0f * z + 0.125f / z;
  if (a >= 90.0f) {
    z = CUDART_INF_F;     /* overflow -> infinity */
  }
  return z;
}

__device_func__(float __cuda_sinhf(float a))
{
  float s, z;

  s = a;
  a = __cuda_fabsf(a);
  if (a < 1.0f) {         /* danger of catastrophic cancellation */
    float a2 = a * a;
    /* approximate sinh(x) on [0,1] with a polynomial */
    z =          2.816951222e-6f;
    z = z * a2 + 1.983615978e-4f;
    z = z * a2 + 8.333350058e-3f;
    z = z * a2 + 1.666666650e-1f;
    z = z * a2;
    z = z * a + a;
  } else {
    z = __internal_expf_kernel(a, -2.0f);
    z = 2.0f * z - 0.125f / z;
    if (a >= 90.0f) {
      z = CUDART_INF_F;     /* overflow -> infinity */
    }
  }
  return __cuda_copysignf(z, s);
}

__device_func__(float __cuda_tanhf(float a))
{
  float s, t;

  t = __cuda_fabsf(a);
  if (t < 0.55f) {
    float z, z2;
    z = t;
    z2 = z * z;
    t =          1.643758066599993e-2f;
    t = t * z2 - 5.267181327760551e-2f;
    t = t * z2 + 1.332072505223051e-1f;
    t = t * z2 - 3.333294663641083e-1f;
    t = t * z2;
    s = t * z + z;
  } else {
    s = 1.0f - 2.0f / (__internal_expf_kernel(2.0f * t, 0.0f) + 1.0f);
    if (t >= 88.0f) {
      s = 1.0f;
    }
  }
  return __cuda_copysignf(s, a);
}

__device_func__(float __cuda_atan2f(float a, float b))
{
#if defined(__MULTI_CORE__)
  return atan2f(a, b);
#else /* __MULTI_CORE__ */
  float t0, t1, t3, fa, fb;

  /* reduce arguments to first octant */
  /* r = (|x| < |y|) ? (|x| / |y|) : (|y| / |x|) */
  fb = __cuda_fabsf(b);
  fa = __cuda_fabsf(a);

  if (fa == 0.0f && fb == 0.0f) {
    t3 = __cuda___signbitf(b) ? CUDART_PI_F : 0;
  } else if ((fa == CUDART_INF_F) && (fb == CUDART_INF_F)) {
    t3 = __cuda___signbitf(b) ? CUDART_3PIO4_F : CUDART_PIO4_F;
  } else {
    /* can't use min, max because they do not propagate NaNs */
    if (fb < fa) {
      t0 = fa;
      t1 = fb;
    } else {
      t0 = fb;
      t1 = fa;
    }
    t3 = __internal_accurate_fdividef(t1, t0);
    t3 = __internal_atanf_kernel(t3);
    /* Map result according to octant. */
    if (fa > fb)  t3 = CUDART_PIO2_F - t3;
    if (b < 0.0f) t3 = CUDART_PI_F - t3;
  }
  t3 = __cuda_copysignf(t3, a);

  return t3;
#endif /* __MULTI_CORE__ */
}

__device_func__(float __cuda_atanf(float a))
{
  float t0, t1;

  /* reduce argument to first octant */
  t0 = __cuda_fabsf(a);
  t1 = t0;
  if (t0 > 1.0f) {
    t1 = 1.0f / t1;
  }
  /* approximate atan(r) in first octant */
  t1 = __internal_atanf_kernel(t1);
  /* map result according to octant. */
  if (t0 > 1.0f) {
    t1 = CUDART_PIO2_F - t1;
  }
  return __cuda_copysignf(t1, a);
}

/* approximate asin(a) on [0, 0.575] */
__device_func__(float __internal_asinf_kernel(float a))
{
  float t2, t3, t4;

  t2 = a * a;
  t3 =         - 0.501162291f;
  t3 = t3 * t2 + 0.915201485f;
  t3 = t3 * t2;
  t3 = t3 * a;
  t4 = t2      - 5.478654385f;
  t4 = t4 * t2 + 5.491230488f;
  t4 = 1.0f / t4;
  a = t3 * t4 + a;
  return a;
}

__device_func__(float __cuda_asinf(float a))
{
  float t0, t1, t2;

  t0 = __cuda_fabsf(a);
  t2 = 1.0f - t0;
  t2 = 0.5f * t2;
  t2 = __cuda_sqrtf(t2);
  t1 = t0 > 0.575f ? t2 : t0;
  t1 = __internal_asinf_kernel(t1);
  t2 = -2.0f * t1 + CUDART_PIO2_F;
  if (t0 > 0.575f) {
    t1 = t2;
  }
  return __cuda_copysignf(t1, a);
}

__device_func__(float __cuda_acosf(float a))
{
  float t0, t1, t2;

  t0 = __cuda_fabsf(a);
  t2 = 1.0f - t0;
  t2 = 0.5f * t2;
  t2 = __cuda_sqrtf(t2);
  t1 = t0 > 0.575f ? t2 : t0;
  t1 = __internal_asinf_kernel(t1);
  t1 = t0 > 0.575f ? 2.0f * t1 : CUDART_PIO2_F - t1;
  if (__cuda___signbitf(a)) {
    t1 = CUDART_PI_F - t1;
  }
  return t1;
}

__device_func__(float __cuda_logf(float a))
{
#if defined(__MULTI_CORE__)
  return logf(a);
#elif defined(__USE_FAST_MATH__)
  return __logf(a);
#else /* __MULTI_CORE__ */
  return __internal_accurate_logf(a);
#endif /* __MULTI_CORE__ */
}

__device_func__(float __cuda_log10f(float a))
{
#if defined(__MULTI_CORE__)
  return log10f(a);
#elif defined(__USE_FAST_MATH__)
  return __log10f(a);
#else /* __MULTI_CORE__ */
  return CUDART_LGE_F * __internal_accurate_logf(a);
#endif /* __MULTI_CORE__ */
}

__device_func__(float __cuda_log1pf(float a))
{
#if defined(__MULTI_CORE__)
  return log1pf(a);
#else /* __MULTI_CORE__ */
  float t;
#if !defined(__CUDABE__) && defined(_WIN32)
  /* MSVC doesn't handle negative zero correctly, so handle it separately */
  if (a == 0.0f) return a;
#endif /* !__CUDABE__ && _WIN32 */
  if (a >= -0.394f && a <= 0.65f) {
    /* log(a+1) = 2*atanh(a/(a+2)) */
    t = a + 2.0f;
    t = a / t;
    t = -a * t;
    t = __internal_atanhf_kernel (a, t);
  } else {
    t = __internal_accurate_logf (CUDART_ONE_F + a);
  }
  return t;
#endif /* __MULTI_CORE__ */
}

__device_func__(float __cuda_acoshf(float a))
{
#if defined(__MULTI_CORE__)
  return acoshf(a);
#else /* __MULTI_CORE__ */
  float t;

  t = a - 1.0f;
  if (__cuda_fabsf(t) > CUDART_TWO_TO_23_F) {
    /* for large a, acosh = log(2*a) */
    return CUDART_LN2_F + __internal_accurate_logf(a);
  } else {
    t = t + __cuda_sqrtf(a * t + t);
    return __cuda_log1pf(t);
  }
#endif /* __MULTI_CORE__ */
}

__device_func__(float __cuda_asinhf(float a))
{
#if defined(__MULTI_CORE__)
  return asinhf(a);
#else /* __MULTI_CORE__ */
  float fa, oofa, t;

  fa = __cuda_fabsf(a);
  if (fa > CUDART_TWO_TO_126_F) {   /* prevent intermediate underflow */
    t = CUDART_LN2_F + __logf(fa);  /* fast version is safe here */
  } else {
    oofa = 1.0f / fa;
    t = fa + fa / (oofa + __cuda_sqrtf(1.0f + oofa * oofa));
    t = __cuda_log1pf(t);
  }
  return __cuda_copysignf(t, a);
#endif /* __MULTI_CORE__ */
}

__device_func__(float __cuda_atanhf(float a))
{
#if defined(__MULTI_CORE__)
  return atanhf(a);
#else /* __MULTI_CORE__ */
  float fa, t;

  fa = __cuda_fabsf(a);
  t = (2.0f * fa) / (1.0f - fa);
  t = 0.5f * __cuda_log1pf(t);
  return __cuda_copysignf(t, a);
#endif /* __MULTI_CORE__ */
}

__device_func__(float __cuda_expm1f(float a))
{
  float t, z, j, u;
  /* expm1(a) = 2^t*(expm1(z)+1)-1 */
  t = __cuda_rintf (a * CUDART_L2E_F);
  z = a - t * 0.6931457519f;
  z = z - t * 1.4286067653e-6f;
  /* prevent loss of accuracy for args a tad outside [-0.5*log(2),0.5*log(2)]*/
  if (__cuda_fabsf(a) < 0.41f) {
    z = a;
    t = 0.0f;
  }
  /* prevent intermediate overflow */
  j = t;
  if (t == 128.0f) j = j - 1.0f; 
  /* expm1(z) on [log(2/3), log(3/2)] */
  u =         1.38795078474044430E-003f;
  u = u * z + 8.38241261853264930E-003f;
  u = u * z + 4.16678317762833940E-002f;
  u = u * z + 1.66663978874356580E-001f;
  u = u * z + 4.99999940395997040E-001f;
  u = u * z;
  u = u * z + z;
  if (a == 0.0f) u = a;            // preserve input of -0
  /* 2^j*[exmp1(z)+1]-1 = 2^j*expm1(z)+2^j-1 */
  z = __cuda_exp2f (j);
  a = z - 1.0f;
  if (a != 0.0f)   u = u * z + a;  // preserve -0 generated by FTZ
  if (t == 128.0f) u = u + u;      // work around intermediate overflow 
  /* handle massive overflow and underflow */
  if (j >  128.0f) u = CUDART_INF_F;
  if (j <  -25.0f) u = -1.0f;
  return u;
}

__device_func__(float __cuda_hypotf(float a, float b))
{
#if defined(__MULTI_CORE__)
  return hypotf(a, b);
#else /* __MULTI_CORE__ */
  float v, w, t;

  a = __cuda_fabsf(a);
  b = __cuda_fabsf(b);
  /* can't use min, max because they do not propagate NaNs */
  if (a > b) {
    v = a;
    w = b; 
  } else {
    v = b;
    w = a;
  }
  t = __internal_accurate_fdividef(w, v);
  t = 1.0f + t * t;
  t = v * __cuda_sqrtf(t);
  if (v == 0.0f) {
    t = v + w;
  }
  if ((v == CUDART_INF_F) || (w == CUDART_INF_F)) {
    t = CUDART_INF_F;
  }
  return t;
#endif /* __MULTI_CORE__ */
}

__device_func__(float __cuda_cbrtf(float a))
{
#if defined(__MULTI_CORE__)
  return cbrtf(a);
#else /* __MULTI_CORE__ */
  float s, t;

  s = __cuda_fabsf(a);
  if ((a == 0.0f) || (s == CUDART_INF_F)) {
    return a;
  } 
  t = __cuda_exp2f(CUDART_THIRD_F * __log2f(s)); /* initial approximation */
  t = t - (t - (s / (t * t))) * CUDART_THIRD_F;  /* refine approximation */
  if (__cuda___signbitf(a)) {
     t = -t;
  }
  return t;
#endif /* __MULTI_CORE__ */
}

__device_func__(float __cuda_erff(float a))
{
  float t, r, q;

  t = __cuda_fabsf(a);
  if (t < 1.0f) {
    t = t * t;
    r =        -5.58510127926029810E-004f;
    r = r * t + 4.90688891415893070E-003f;
    r = r * t - 2.67027980930150640E-002f;
    r = r * t + 1.12799056505903940E-001f;
    r = r * t - 3.76122956138427440E-001f;
    r = r * t + 1.12837911712623450E+000f;
    a = a * r;
  } else if (t <= CUDART_INF_F) { 
    /* coefficients from Hastings, "Approximations for Digital Computers",
     * Princeton University Press 1955. Sheet 45.
     */
    q = 0.3275911f * t + 1.0f;
    q = 1.0f / q;
    r =         1.061405429f;
    r = r * q - 1.453152027f;
    r = r * q + 1.421413741f;
    r = r * q - 0.284496736f;
    r = r * q + 0.254829592f;
    r = r * q;
    q = __internal_expf_kernel(-a * a, 0.0f);
    r = 1.0f - q * r;
    if (t >= 5.5f) {
      r = 1.0f;
    }
    a = __int_as_float (__float_as_int(r) | (__float_as_int(a) & 0x80000000));
  }
  return a;
}

__device_func__(float __cuda_erfcf(float a))
{
  if (a <= 0.55f) {
    return 1.0f - __cuda_erff(a);
  } else if (a > 10.0f) {
    return 0.0f;
  } else {
    float p;
    float q;
    float h;
    float l;
    /* This rational approximation has a slight accuracy issue since all the
     * coefficients have same sign so error accumulates when this is computed
     * in single precision. Also the division at the end isn't IEEE compliant.
     */
    p =       + 4.014893410762552E-006f;
    p = p * a + 5.640401259462436E-001f;
    p = p * a + 2.626649872281140E+000f;
    p = p * a + 5.486372652389673E+000f;
    p = p * a + 5.250714831459401E+000f;
    q =     a + 4.651376250488319E+000f;
    q = q * a + 1.026302828878470E+001f;
    q = q * a + 1.140762166021288E+001f;
    q = q * a + 5.251211619089947E+000f;
    /* Use reciprocal plus NR refinement for division */
    h = 1.0f / q;
    q = 2.0f * h - q * h * h;
    p = p * q;
    /* compute exp(-a*a) with extended precision to avoid error magnification*/
    h = __int_as_float(__float_as_int(a) & 0xfffff000);  /* upper 12 bits */
    l = __fadd_rn (a, -h);  /* lower 12 bits */
    q = __fmul_rn (-h, h);  /* this product is error free */
    q = __internal_expf_kernel(q, 0.0f);
    a = a + h;
    l = l * a;
    h = __internal_expf_kernel(-l, 0.0f);
    q = q * h;
    p = p * q;
    return p;
  }
}

__device_func__(float __cuda_lgammaf(float a))
{
  float t;
  float i;
  int quot;
  t = __internal_lgammaf_pos(__cuda_fabsf(a));
  if (a >= 0.0f) return t;
  a = __cuda_fabsf(a);
  i = __cuda_floorf(a);                   
  if (a == i) return CUDART_INF_F; /* a is an integer: return infinity */
  if (a < 1e-19f) return -__internal_accurate_logf(a);
  i = __cuda_rintf (2.0f * a);
  quot = (int)i;
  i = a - 0.5f * i;
  i = i * CUDART_PI_F;
  if (quot & 1) {
    i = __internal_cos_kernel(i);
  } else {
    i = __internal_sin_kernel(i);
  }
  i = __cuda_fabsf(i);
  t = CUDART_LNPI_F - __internal_accurate_logf(i * a) - t;
  return t;
}

__device_func__(float __cuda_ldexpf(float a, int b))
{
#if defined(__MULTI_CORE__)
  return ldexpf(a, b);
#else /* __MULTI_CORE__ */
  float fa = __cuda_fabsf(a);

  if ((fa == CUDART_ZERO_F) || (fa == CUDART_INF_F) || (b == 0)) {
    return a;
  }
  else if (__cuda_abs(b) < 126) {
    return a * __cuda_exp2f((float)b);
  }
  else if (__cuda_abs(b) < 252) {
    int bhalf = b / 2;
    return a * __cuda_exp2f((float)bhalf) * __cuda_exp2f((float)(b - bhalf));
  } 
  else {
    int bquarter = b / 4;
    float t = __cuda_exp2f((float)bquarter);
    return a * t * t * t * __cuda_exp2f((float)(b - 3 * bquarter));
  }
#endif /* __MULTI_CORE__ */
}

__device_func__(float __cuda_scalbnf(float a, int b))
{
#if defined(__MULTI_CORE__)
  return scalbnf(a, b);
#else /* __MULTI_CORE__ */
  /* On binary systems, ldexp(x,exp) is equivalent to scalbn(x,exp) */
  return __cuda_ldexpf(a, b);
#endif /* __MULTI_CORE__ */
}

__device_func__(float __cuda_scalblnf(float a, long int b))
{
#if defined(__MULTI_CORE__)
  return scalblnf(a, b);
#else /* __MULTI_CORE__ */
  int t;
  if (b > 2147483647L) {
    t = 2147483647;
  } else if (b < (-2147483647 - 1)) {
    t = (-2147483647 - 1);
  } else {
    t = (int)b;
  }
  return __cuda_scalbnf(a, t);
#endif /* __MULTI_CORE__ */
}

__device_func__(float __cuda_frexpf(float a, int *b))
{
  float fa = __cuda_fabsf(a);
  unsigned int expo;
  unsigned int denorm;

  if (fa < CUDART_TWO_TO_M126_F) {
    a *= CUDART_TWO_TO_24_F;
    denorm = 24;
  } else {
    denorm = 0;
  }
  expo = ((__float_as_int(a) >> 23) & 0xff);
  if ((fa == 0.0f) || (expo == 0xff)) {
    expo = 0;
    a = a + a;
  } else {  
    expo = expo - denorm - 126;
    a = __int_as_float(((__float_as_int(a) & 0x807fffff) | 0x3f000000));
  }
  *b = expo;
  return a;
}

__device_func__(float __cuda_modff(float a, float *b))
{
#if defined(__MULTI_CORE__)
  return modff(a, b);
#else /* __MULTI_CORE__ */
  float t;
  if (__cuda___finitef(a)) {
    t = __cuda_truncf(a);
    *b = t;
    t = a - t;
    return __cuda_copysignf(t, a);
  } else if (__cuda___isinff(a)) {
    t = 0.0f;
    *b = a;
    return __cuda_copysignf(t, a);
  } else {
    *b = a; 
    return a;
  }
#endif /* __MULTI_CORE__ */
}

__device_func__(float __cuda_fmodf(float a, float b))
{
#if defined(__MULTI_CORE__)
  return fmodf(a, b);
#else /* __MULTI_CORE__ */
  float orig_a = a;
  float orig_b = b;
  a = __cuda_fabsf(a);
  b = __cuda_fabsf(b);
  if (!((a <= CUDART_INF_F) && (b <= CUDART_INF_F))) {
    return orig_a + orig_b;
  }
  if ((a == CUDART_INF_F) || (b == CUDART_ZERO_F)) {
    return CUDART_NAN_F;
  } else if (a >= b) {
#if !defined(__CUDABE__)
    /* Need to be able to handle denormals correctly */
    int expoa = (a < CUDART_TWO_TO_M126_F) ? 
        ((int)__log2f(a)) : (((__float_as_int(a) >> 23) & 0xff) - 127);
    int expob = (b < CUDART_TWO_TO_M126_F) ? 
        ((int)__log2f(b)) : (((__float_as_int(b) >> 23) & 0xff) - 127);
    int scale = expoa - expob;
    float scaled_b = __cuda_ldexpf(b, scale);
    if (scaled_b <= 0.5f * a) {
      scaled_b *= 2.0f;
    }
#else /* !__CUDABE__ */
    float scaled_b = __int_as_float ((__float_as_int(b) & 0x007fffff) | 
                                     (__float_as_int(a) & 0x7f800000));
    if (scaled_b > a) {
      scaled_b *= 0.5f;
    }
#endif /* !__CUDABE__ */
    while (scaled_b >= b) {
      if (a >= scaled_b) {
        a -= scaled_b;
      }
      scaled_b *= 0.5f;
    }
    return __cuda_copysignf(a, orig_a);
  } else {
    return orig_a;
  }
#endif /* __MULTI_CORE__ */
}

__device_func__(float __cuda_remainderf(float a, float b))
{

  float twoa = 0.0f;
  unsigned int quot0 = 0;  /* quotient bit 0 */
  float orig_a = a;
  float orig_b = b;

  a = __cuda_fabsf(a);
  b = __cuda_fabsf(b);
  if (!((a <= CUDART_INF_F) && (b <= CUDART_INF_F))) {
    return orig_a + orig_b;
  }
  if ((a == CUDART_INF_F) || (b == CUDART_ZERO_F)) {
    return CUDART_NAN_F;
  } else if (a >= b) {
#if !defined(__CUDABE__)
    int expoa = (a < CUDART_TWO_TO_M126_F) ? 
        ((int)__log2f(a)) : (((__float_as_int(a) >> 23) & 0xff) - 127);
    int expob = (b < CUDART_TWO_TO_M126_F) ? 
        ((int)__log2f(b)) : (((__float_as_int(b) >> 23) & 0xff) - 127);
    int scale = expoa - expob;
    float scaled_b = __cuda_ldexpf(b, scale);
    if (scaled_b <= 0.5f * a) {
      scaled_b *= 2.0f;
    }
#else
    float scaled_b = __int_as_float ((__float_as_int(b) & 0x007fffff) | 
                                     (__float_as_int(a) & 0x7f800000));
    if (scaled_b > a) {
      scaled_b *= 0.5f;
    }
    /* check wether divisor is a power of two */
    if (a == scaled_b) {
      return __int_as_float(__float_as_int(orig_a) & 0x80000000);
    }    
#endif /* !__CUDABE__ */
    while (scaled_b >= b) {
      quot0 = 0;
      if (a >= scaled_b) {
        twoa = (2.0f * a - scaled_b) - scaled_b;
        a -= scaled_b;
        quot0 = 1;
      }
      scaled_b *= 0.5f;
    }
  }
  /* round quotient to nearest even */
#if !defined(__CUDABE__)
  twoa = 2.0f * a;
  if ((twoa > b) || ((twoa == b) && quot0)) {
    a -= b;
    a = __cuda_copysignf (a, -1.0f);
  }
#else /* !__CUDABE__ */
  if (a >= CUDART_TWO_TO_M126_F) {
    twoa = 2.0f * a;
    if ((twoa > b) || ((twoa == b) && quot0)) {
      a -= b;
      a = __cuda_copysignf (a, -1.0f);
    }
  } else {
    /* a already got flushed to zero, so use twoa instead */
    if ((twoa > b) || ((twoa == b) && quot0)) {
      a = 0.5f * (twoa - 2.0f * b);
      a = __cuda_copysignf (a, -1.0f);
    }
  }
#endif /* !__CUDABE__ */
  a = __int_as_float((__float_as_int(orig_a) & 0x80000000)^
                     __float_as_int(a));
  return a;
}

__device_func__(float __cuda_remquof(float a, float b, int* quo))
{
  float twoa = 0.0f;
  unsigned int quot = 0;  /* trailing quotient bits */
  unsigned int sign;
  float orig_a = a;
  float orig_b = b;

  /* quo has a value whose sign is the sign of x/y */
  sign = 0 - (__cuda___signbitf(a) != __cuda___signbitf(b));
  a = __cuda_fabsf(a);
  b = __cuda_fabsf(b);
  if (!((a <= CUDART_INF_F) && (b <= CUDART_INF_F))) {
    *quo = quot;
    return orig_a + orig_b;
  }
  if ((a == CUDART_INF_F) || (b == CUDART_ZERO_F)) {
    *quo = quot;
    return CUDART_NAN_F;
  } else if (a >= b) {
#if !defined(__CUDABE__)
    /* Need to be able to handle denormals correctly */
    int expoa = (a < CUDART_TWO_TO_M126_F) ? 
        ((int)__log2f(a)) : (((__float_as_int(a) >> 23) & 0xff) - 127);
    int expob = (b < CUDART_TWO_TO_M126_F) ? 
        ((int)__log2f(b)) : (((__float_as_int(b) >> 23) & 0xff) - 127);
    int scale = expoa - expob;
    float scaled_b = __cuda_ldexpf(b, scale);
    if (scaled_b <= 0.5f * a) {
      scaled_b *= 2.0f;
    }
#else
    float scaled_b = __int_as_float ((__float_as_int(b) & 0x007fffff) | 
                                     (__float_as_int(a) & 0x7f800000));
    if (scaled_b > a) {
      scaled_b *= 0.5f;
    }
    /* check wether divisor is a power of two */
    if (a == scaled_b) {
      a = __internal_accurate_fdividef(a,b) + 0.5f;
      quot = (a < 8.0f) ? (int)a : 0;
      quot = quot & CUDART_REMQUO_MASK_F;
      quot = quot ^ sign;
      quot = quot - sign;
      *quo = quot;
      return __int_as_float(__float_as_int(orig_a) & 0x80000000);
    }    
#endif /* !__CUDABE__ */
    while (scaled_b >= b) {
      quot <<= 1;
      if (a >= scaled_b) {
        twoa = (2.0f * a - scaled_b) - scaled_b;
        a -= scaled_b;
        quot += 1;
      }
      scaled_b *= 0.5f;
    }
  }
  /* round quotient to nearest even */
#if !defined(__CUDABE__)
  twoa = 2.0f * a;
  if ((twoa > b) || ((twoa == b) && (quot & 1))) {
    quot++;
    a -= b;
    a = __cuda_copysignf (a, -1.0f);
  }
#else /* !__CUDABE__ */
  if (a >= CUDART_TWO_TO_M126_F) {
    twoa = 2.0f * a;
    if ((twoa > b) || ((twoa == b) && (quot & 1))) {
      quot++;
      a -= b;
      a = __cuda_copysignf (a, -1.0f);
    }
  } else {
    /* a already got flushed to zero, so use twoa instead */
    if ((twoa > b) || ((twoa == b) && (quot & 1))) {
      quot++;
      a = 0.5f * (twoa - 2.0f * b);
      a = __cuda_copysignf (a, -1.0f);
    }
  }
#endif /* !__CUDABE__ */
  a = __int_as_float((__float_as_int(orig_a) & 0x80000000)^
                     __float_as_int(a));
  quot = quot & CUDART_REMQUO_MASK_F;
  quot = quot ^ sign;
  quot = quot - sign;
  *quo = quot;
  return a;
}

__device_func__(float __cuda_fmaf(float a, float b, float c))
{
  unsigned int xx, yy, zz, ww;
  unsigned int temp, s, u;
  unsigned int expo_x, expo_y, expo_z;

  xx = __float_as_int(a);
  yy = __float_as_int(b);
  zz = __float_as_int(c);

#if defined(__CUDABE__)
  /* Match 'denormals are zero' behavior of the GPU */
  if ((xx << 1) < 0x01000000) xx &= 0x80000000;
  if ((yy << 1) < 0x01000000) yy &= 0x80000000;
  if ((zz << 1) < 0x01000000) zz &= 0x80000000;
#endif /* __CUDABE__ */

  temp = 0xff;
  expo_x = temp & (xx >> 23);
  expo_x = expo_x - 1;
  expo_y = temp & (yy >> 23);
  expo_y = expo_y - 1;
  expo_z = temp & (zz >> 23);
  expo_z = expo_z - 1;

  if (!((expo_x <= 0xFD) && 
        (expo_y <= 0xFD) &&
        (expo_z <= 0xFD))) {
    /* fma (nan, y, z) --> nan
       fma (x, nan, z) --> nan
       fma (x, y, nan) --> nan 
    */
    if ((yy << 1) > 0xff000000) {
      return CUDART_NAN_F;
    }
    if ((zz << 1) > 0xff000000) {
      return CUDART_NAN_F;
    }
    if ((xx << 1) > 0xff000000) {
      return CUDART_NAN_F;
    }
    /* fma (0, inf, z) --> NaN
       fma (inf, 0, z) --> NaN
       fma (-inf,+y,+inf) --> NaN
       fma (+x,-inf,+inf) --> NaN
       fma (+inf,-y,+inf) --> NaN
       fma (-x,+inf,+inf) --> NaN
       fma (-inf,-y,-inf) --> NaN
       fma (-x,-inf,-inf) --> NaN
       fma (+inf,+y,-inf) --> NaN
       fma (+x,+inf,-inf) --> NaN
    */
    if ((((xx << 1) == 0) && ((yy << 1) == 0xff000000)) ||
        (((yy << 1) == 0) && ((xx << 1) == 0xff000000))) {
      return CUDART_NAN_F;
    }
    if ((zz << 1) == 0xff000000) {
      if (((yy << 1) == 0xff000000) || ((xx << 1) == 0xff000000)) {
        if ((int)(xx ^ yy ^ zz) < 0) {
          return CUDART_NAN_F;
        }
      }
    }
    /* fma (inf, y, z) --> inf
       fma (x, inf, z) --> inf
       fma (x, y, inf) --> inf
    */
    if ((xx << 1) == 0xff000000) {
      xx = xx ^ (yy & 0x80000000);
      return __int_as_float(xx);
    }
    if ((yy << 1) == 0xff000000) {
      yy = yy ^ (xx & 0x80000000);
      return __int_as_float(yy);
    }
    if ((zz << 1) == 0xff000000) {
      return __int_as_float(zz);
    }
    /* fma (+0, -y, -0) --> -0
       fma (-0, +y, -0) --> -0
       fma (+x, -0, -0) --> -0
       fma (-x, +0, -0) --> -0
    */
    if (zz == 0x80000000) {
      if (((xx << 1) == 0) || ((yy << 1) == 0)) {
        if ((int)(xx ^ yy) < 0) {
          return __int_as_float(zz);
        }
      }
    }
    /* fma (0, y, 0) --> +0
       fma (x, 0, 0) --> +0
    */
    if (((zz << 1) == 0) && 
        (((xx << 1) == 0) || ((yy << 1) == 0))) {
      zz &= 0x7fffffff;
      return __int_as_float(zz);
    }
    /* fma (0, y, z) --> z
       fma (x, 0, z) --> z
    */
    if (((xx << 1) == 0) || ((yy << 1) == 0)) {
      return __int_as_float(zz);
    }
    /* normalize x, if denormal */
    if (expo_x == (unsigned int)-1) {
      temp = xx & 0x80000000;
      xx = xx << 8;
      while (!(xx & 0x80000000)) {
        xx <<= 1;
        expo_x--;
      }
      expo_x++;
      xx = (xx >> 8) | temp;
    }
    /* normalize y, if denormal */
    if (expo_y == (unsigned int)-1) {
      temp = yy & 0x80000000;
      yy = yy << 8;
      while (!(yy & 0x80000000)) {
        yy <<= 1;
        expo_y--;
      }
      expo_y++;
      yy = (yy >> 8) | temp;
    }
    /* normalize z, if denormal */
    if ((expo_z == (unsigned int)-1) && ((zz << 1) != 0)) {
      temp = zz & 0x80000000;
      zz = zz << 8;
      while (!(zz & 0x80000000)) {
        zz <<= 1;
        expo_z--;
      }
      expo_z++;
      zz = (zz >> 8) | temp;
    }
  }
    
  expo_x = expo_x + expo_y;
  expo_y = xx ^ yy;
  xx = xx & 0x00ffffff;
  yy = yy << 8;
  xx = xx | 0x00800000;
  yy = yy | 0x80000000;
  
  s = __umulhi(xx, yy);
  yy = xx * yy;
  xx = s;
  expo_x = expo_x - 127 + 2;
  expo_y = expo_y & 0x80000000;

  /* normalize mantissa */
  if (xx < 0x00800000) {
      xx = (xx << 1) | (yy >> 31);
      yy = (yy << 1);
      expo_x--;
  }
  temp = 0;
  if ((zz << 1) != 0) { /* z is not zero */
    s = zz & 0x80000000;
    zz &= 0x00ffffff;
    zz |= 0x00800000;
    ww = 0;
    /* compare and swap. put augend into xx:yy */
    if ((int)expo_z > (int)expo_x) {
      temp = expo_z;
      expo_z = expo_x;
      expo_x = temp;
      temp = zz;
      zz = xx;
      xx = temp;
      temp = ww;
      ww = yy;
      yy = temp;
      temp = expo_y;
      expo_y = s;
      s = temp;
    }
    /* augend_sign = expo_y, augend_mant = xx:yy, augend_expo = expo_x */
    /* addend_sign = s, addend_mant = zz:ww, addend_expo = expo_z */
    expo_z = expo_x - expo_z;
    u = expo_y ^ s;
    if (expo_z <= 49) {
      /* denormalize addend */
      temp = 0;
      while (expo_z >= 32) {
        temp = ww | (temp != 0);
        ww = zz;
        zz = 0;
        expo_z -= 32;
      }
      if (expo_z) {
        temp = ((temp >> expo_z) | (ww << (32 - expo_z)) | 
                ((temp << (32 - expo_z)) != 0));
        ww = (ww >> expo_z) | (zz << (32 - expo_z));
        zz = (zz >> expo_z);
      }
    } else {
      temp = 1;
      ww = 0;
      zz = 0;
    }            
    if ((int)u < 0) {
      /* signs differ, effective subtraction */
      temp = (unsigned)(-(int)temp);
      s = (temp != 0);
      u = yy - s;
      s = u > yy;
      yy = u - ww;
      s += yy > u;
      xx = (xx - zz) - s;
      if (!(xx | yy | temp)) {
        /* complete cancelation, return 0 */
        return __int_as_float(xx);
      }
      if ((int)xx < 0) {
        /* Oops, augend had smaller mantissa. Negate mantissa and flip
           sign of result
         */
        temp = ~temp;
        yy = ~yy;
        xx = ~xx;
        if (++temp == 0) {
          if (++yy == 0) {
            ++xx;
          }
        }
        expo_y ^= 0x80000000;
      }
      /* normalize mantissa, if necessary */
      while (!(xx & 0x00800000)) {
        xx = (xx << 1) | (yy >> 31);
        yy = (yy << 1);
        expo_x--;
      }
    } else {
      /* signs are the same, effective addition */
      yy = yy + ww;
      s =  yy < ww;
      xx = xx + zz + s;
      if (xx & 0x01000000) {
        temp = temp | (yy << 31);
        yy = (yy >> 1) | (xx << 31);
        xx = ((xx & 0x80000000) | (xx >> 1)) & ~0x40000000;
        expo_x++;
      }
    }
  }
  temp = yy | (temp != 0);
  if (expo_x <= 0xFD) {
    /* normal */
    xx |= expo_y; /* or in sign bit */
    s = xx & 1; /* mantissa lsb */
    xx += (temp == 0x80000000) ? s : (temp >> 31);
    xx = xx + (expo_x << 23); /* add in exponent */
    return __int_as_float(xx);
  } else if ((int)expo_x >= 126) {
    /* overflow */
    xx = expo_y | 0x7f800000;
    return __int_as_float(xx);
  }
  /* subnormal */
  expo_x = (unsigned int)(-(int)expo_x);
  if (expo_x > 25) {
    /* massive underflow: return 0 */
    return __int_as_float(expo_y);
  }
  yy = (xx << (32 - expo_x)) | ((yy) ? 1 : 0);
  xx = expo_y + (xx >> expo_x);
  xx = xx + ((yy==0x80000000) ? (xx & 1) : (yy >> 31));
  xx |= expo_y; /* or in sign bit */
#if defined(__CUDABE__)
  /* Match 'flush to zero' response of the GPU */
  if ((xx << 1) < 0x01000000) xx = expo_y;
#endif /* __CUDABE__ */
  return __int_as_float(xx);
}

__device_func__(float __internal_accurate_powf(float a, float b))
{
  float2 loga, prod;
#if !defined(__CUDABE__) && defined(_MSC_VER) && !defined(_WIN64)
  volatile float t;
#else
  float t;
#endif

  /* compute log(a) in double-single format*/
  loga = __internal_log_ep(a);

  /* prevent overflow during extended precision multiply */
  if (__cuda_fabsf(b) > 1.0e34f) b *= 1.220703125e-4f;
  prod.y = b;
  prod.x = 0.0f;
  prod = __internal_dsmul (prod, loga);

  /* prevent intermediate overflow in exponentiation */
  if (__float_as_int(prod.y) == 0x42b17218) {
    prod.y = __int_as_float(__float_as_int(prod.y) - 1);
    prod.x = prod.x + __int_as_float(0x37000000);
  }

  /* compute pow(a,b) = exp(b*log(a)) */
  t = __cuda_expf(prod.y);
  /* prevent -INF + INF = NaN */
  if (t != CUDART_INF_F) {
    /* if prod.x is much smaller than prod.y, then exp(prod.y+prod.x) ~= 
     * exp(prod.y) + prod.x * exp(prod.y) 
     */
    t = t * prod.x + t;
  }
  return t;
}

__device_func__(float __cuda_powif(float a, int b))
{
  unsigned int e = __cuda_abs(b);
  float        r = 1.0f;

  while (1) {
    if ((e & 1) != 0) {
      r = r * a;
    }
    e = e >> 1;
    if (e == 0) {
      return b < 0 ? 1.0f/r : r;
    }
    a = a * a;
  }
}

__device_func__(double __cuda_powi(double a, int b))
{
  unsigned int e = __cuda_abs(b);
  double       r = 1.0;

  while (1) {
    if ((e & 1) != 0) {
      r = r * a;
    }
    e = e >> 1;
    if (e == 0) {
      return b < 0 ? 1.0/r : r;
    }
    a = a * a;
  }
}

__device_func__(float __cuda_powf(float a, float b))
{
#if defined(__MULTI_CORE__)
  return powf(a, b);
#elif defined(__USE_FAST_MATH__)
  return __powf(a, b);
#else /* __MULTI_CORE__ */
  int bIsOddInteger;
  float t;
  if (a == 1.0f || b == 0.0f) {
    return 1.0f;
  } 
  if (__cuda___isnanf(a) || __cuda___isnanf(b)) {
    return a + b;
  }
  if (a == CUDART_INF_F) {
    return __cuda___signbitf(b) ? CUDART_ZERO_F : CUDART_INF_F;
  }
  if (__cuda___isinff(b)) {
    if (a == -1.0f) {
      return 1.0f;
    }
    t = (__cuda_fabsf(a) > 1.0f) ? CUDART_INF_F : CUDART_ZERO_F;
    if (b < CUDART_ZERO_F) {
      t = 1.0f / t;
    }
    return t;
  }
  bIsOddInteger = (b - (2.0f * floorf(0.5f * b))) == 1.0f;
  if (a == CUDART_ZERO_F) {
    t = bIsOddInteger ? a : CUDART_ZERO_F;
    if (b < CUDART_ZERO_F) {
      t = 1.0f / t;
    }
    return t;
  } 
  if (a == -CUDART_INF_F) {
    t = (b < CUDART_ZERO_F) ? -1.0f/a : -a;
    if (bIsOddInteger) {
      t = __int_as_float(__float_as_int(t) ^ 0x80000000);
    }
    return t;
  } 
  if ((a < CUDART_ZERO_F) && (b != __cuda_truncf(b))) {
    return CUDART_NAN_F;
  }
  t = __cuda_fabsf(a);
  t = __internal_accurate_powf(t, b);
  if ((a < CUDART_ZERO_F) && bIsOddInteger) {
    t = __int_as_float(__float_as_int(t) ^ 0x80000000);
  }
  return t;
#endif /* __MULTI_CORE__ */
}

/* approximate 1.0/(x*gamma(x)) on [-0.5,0.5] */
__device_func__(float __internal_tgammaf_kernel(float a))
{
  float t;
  t =       - 1.05767296987211380E-003f;
  t = t * a + 7.09279059435508670E-003f;
  t = t * a - 9.65347121958557050E-003f;
  t = t * a - 4.21736613253687960E-002f;
  t = t * a + 1.66542401247154280E-001f;
  t = t * a - 4.20043267827838460E-002f;
  t = t * a - 6.55878234051332940E-001f;
  t = t * a + 5.77215696929794240E-001f;
  t = t * a + 1.00000000000000000E+000f;
  return t;
}

/* Based on: Kraemer, W.: "Berechnung der Gammafunktion G(x) fuer reelle Punkt-
   und Intervallargumente". Zeitschrift fuer angewandte Mathematik und 
   Mechanik, Vol. 70 (1990), No. 6, pp. 581-584
*/
__device_func__(float __cuda_tgammaf(float a))
{
  float s, xx, x=a;
  if (x >= 0.0f) {
    if (x > 36.0f) x = 36.0f; /* clamp */
    s = 1.0f;
    xx = x;
    if (x > 34.03f) { /* prevent premature overflow */
      xx -= 1.0f;
    }
    while (xx > 1.5f) {
      xx = xx - 1.0f;
      s = s * xx;
    }
    if (x >= 0.5f) {
      xx = xx - 1.0f;
    }
    xx = __internal_tgammaf_kernel(xx);
    if (x < 0.5f) {
      xx = xx * x;
    }
    s = s / xx;
    if (x > 34.03f) {
      /* Cannot use s = s * x - s due to intermediate overflow! */
      xx = x - 1.0f;
      s = s * xx;
    }
    return s;
  } else {
    if (x == __cuda_floorf(x)) {  /* x is negative integer */
      x = CUDART_NAN_F;  /* NaN, propagates through on device */
#if !defined(__CUDABE__)
      return x;
#endif /* !__CUDABE__ */
    } 
    if (x < -41.1f) x = -41.1f; /* clamp */
    xx = x;
    if (x < -34.03f) {   /* prevent overflow in intermediate result */        
      xx += 6.0f;
    } 
    s = xx;
    while (xx < -0.5f) {
      xx = xx + 1.0f;
      s = s * xx;
    }
    xx = __internal_tgammaf_kernel(xx);
    s = s * xx;
    s = 1.0f / s;
    if (x < -34.03f) {
      xx = x;
      xx *= (x + 1.0f);
      xx *= (x + 2.0f);
      xx *= (x + 3.0f);
      xx *= (x + 4.0f);
      xx *= (x + 5.0f);
      xx = 1.0f / xx;
      s = s * xx;
      if ((a < -42.0f) && !(((int)a)&1)) {
        s = CUDART_NEG_ZERO_F;
      }
    }    
    return s;
  }
}

__device_func__(float __cuda_roundf(float a))
{
#if defined(__MULTI_CORE__)
  return roundf(a);
#else /* __MULTI_CORE__ */
  float fa = __cuda_fabsf(a);
  if (fa > CUDART_TWO_TO_23_F) {
    return a;
  } else {
    float u = __cuda_floorf(fa + 0.5f);
    if (fa < 0.5f) u = 0.0f;
    return __cuda_copysignf(u, a);
  }
#endif /* __MULTI_CORE__ */
}

__device_func__(long long int __internal_llroundf_kernel(float a))
{
  unsigned long long int res, t = 0LL;
  int shift;
  unsigned int ia = __float_as_int(a);
  if ((ia << 1) > 0xff000000) return 0LL;
  if ((int)ia >= 0x5f000000) return 0x7fffffffffffffffLL;
  if (ia >= 0xdf000000) return 0x8000000000000000LL;
  shift = 189 - ((ia >> 23) & 0xff);
  res = ((long long int)(((ia << 8) | 0x80000000) >> 1)) << 32;
  if (shift >= 64) {
    t = res;
    res = 0;
  } else if (shift) {
    t = res << (64 - shift);
    res = res >> shift;
  }
  if (t >= 0x8000000000000000LL) {
      res++;
  }
  if ((int)ia < 0) res = (unsigned long long int)(-(long long int)res);
  return (long long int)res;
}

__device_func__(long long int __cuda_llroundf(float a))
{
#if defined(__MULTI_CORE__)
  return llroundf(a);
#else /* __MULTI_CORE__ */
  return __internal_llroundf_kernel(a);
#endif /* __MULTI_CORE__ */
}

__device_func__(long int __cuda_lroundf(float a))
{
#if defined(__MULTI_CORE__)
  return lroundf(a);
#else /* __MULTI_CORE__ */
#if defined(__LP64__)
  return (long int)__cuda_llroundf(a);
#else /* __LP64__ */
#if !defined(__CUDABE__)
  if (__cuda___isnanf(a)) return 0L;
  if (a >=  CUDART_TWO_TO_31_F) return 2147483647L;
  if (a <= -CUDART_TWO_TO_31_F) return (-2147483647L - 1L);
#endif /* __CUDABE__ */
  return (long int)(__cuda_roundf(a));
#endif /* __LP64__ */
#endif /* __MULTI_CORE__ */
}

__device_func__(float __cuda_fdimf(float a, float b))
{
  float t;
  t = a - b;    /* default also handles NaNs */
  if (a <= b) {
    t = 0.0f;
  }
  return t;
}

__device_func__(int __cuda_ilogbf(float a))
{
  unsigned int i;
  int expo;
  a = __cuda_fabsf(a);
  if (a <= CUDART_TWO_TO_M126_F) {
    /* handle zero and denormals */
    if (a == 0.0f) {
      expo = -INT_MAX-1;
    } else {
      expo = -126;
      i = __float_as_int(a);
      i = i << 8;
      while ((int)i >= 0) {
        expo--;
        i = i + i;
      }
    }
  } else {
    i = __float_as_int(a);
    expo = ((int)((i >> 23) & 0xff)) - 127;
    if ((i == 0x7f800000)) {
      expo = INT_MAX;
    }
    if ((i > 0x7f800000)) {
      expo = -INT_MAX-1;
    }
  } 
  return expo;
}

__device_func__(float __cuda_logbf(float a))
{
#if defined(__MULTI_CORE__)
  return logbf(a);
#else /* __MULTI_CORE__ */
  unsigned int i;
  int expo;
  float res;
#if !defined(__CUDABE__)
  if (__cuda___isnanf(a)) return a + a;
#endif /* !__CUDABE__ */
  a = __cuda_fabsf(a);
  if (a <= CUDART_TWO_TO_M126_F) {
    /* handle zero and denormals */
    if (a == 0.0f) {
      res = -CUDART_INF_F;
    } else {
      expo = -126;
      i = __float_as_int(a);
      i = i << 8;
      while ((int)i >= 0) {
        expo--;
        i = i + i;
      }
      res = (float)expo;
    }
  } else {
    i = __float_as_int(a);
    expo = ((int)((i >> 23) & 0xff)) - 127;
    res = (float)expo;
    if ((i >= 0x7f800000)) {  
      /* return +INF or canonical NaN */
      res = a + a;
    }
  } 
  return res;
#endif /* __MULTI_CORE__ */
}

__device_func__(void __cuda_sincosf(float a, float *sptr, float *cptr))
{
#if defined(__MULTI_CORE__)
  sincosf(a, sptr, cptr);
#elif defined(__USE_FAST_MATH__)
  __sincosf(a, sptr, cptr);
#else /* __MULTI_CORE__ */
  float t, u, s, c;
  int quadrant;
  if (__cuda___isinff(a)) {
    *sptr = CUDART_NAN_F;
    *cptr = CUDART_NAN_F;
    return;
  }
  if (a == CUDART_ZERO_F) {
    *sptr = a;
    *cptr = 1.0f;
    return;
  } 
  t = __internal_trig_reduction_kernel(a, &quadrant);
  u = __internal_cos_kernel(t);
  t = __internal_sin_kernel(t);
  if (quadrant & 1) {
    s = u;
    c = t;
  } else {
    s = t;
    c = u;
  }
  if (quadrant & 2) {
    s = -s;
  }
  quadrant++;
  if (quadrant & 2) {
    c = -c;
  }
  *sptr = s;
  *cptr = c;
#endif /* __MULTI_CORE__ */
}

#if !defined(__CUDABE__) && !defined(__MULTI_CORE__)

/*******************************************************************************
*                                                                              *
* ONLY FOR HOST CODE! NOT FOR EMULATION AND HW EXECUTION                       *
*                                                                              *
*******************************************************************************/

__func__(double rsqrt(double a))
{
  return 1.0 / sqrt(a);
}

__func__(float rsqrtf(float a))
{
  return (float)rsqrt((double)a);
}

/*******************************************************************************
*                                                                              *
*  HOST IMPLEMENTATION FOR DOUBLE ROUTINES                                     *
*                                                                              *
*******************************************************************************/

#if defined(_WIN32) || defined(__APPLE__)

__func__(double exp10(double a))
{
  return pow(10.0, a);
}

#endif /* _WIN32 || __APPLE__ */

#if defined (_WIN32)
__func__(int __finite(double a))
{
  volatile union {
    double                 d;
    unsigned long long int l;
  } cvt;

  cvt.d = a;

  return cvt.l << 1 < 0xffe0000000000000ull;
}

__func__(int __isnan(double a))
{
  volatile union {
    double                 d;
    unsigned long long int l;
  } cvt;

  cvt.d = a;

  return cvt.l << 1 > 0xffe0000000000000ull;
}

__func__(int __isinf(double a))
{
  volatile union {
    double                 d;
    unsigned long long int l;
  } cvt;

  cvt.d = a;

  return cvt.l << 1 == 0xffe0000000000000ull;
}

__func__(double round(double a))
{
  double fa = fabs(a);

  if (fa > CUDART_TWO_TO_52) {
    return a;
  } else {
    double u = floor(fa + 0.5);

    if (__signbit(a)) {
      u = -u;
    }
    return u;
  }
}

__func__(long int lround(double a))
{
  return (long int)round(a);
}

__func__(long long int llround(double a))
{
  return (long long int)round(a);
}

__func__(double rint(double a))
{
  double fa = fabs(a);
  double u = CUDART_TWO_TO_52 + fa;
  if (fa >= CUDART_TWO_TO_52) {
    u = a;
  } else {
    u = u - CUDART_TWO_TO_52;
    if (__signbit(a)) {
      u = -u;
    }
  }
  return u;  
}

__func__(long int lrint(double a))
{
  return (long int)rint(a);
}

__func__(long long int llrint(double a))
{
  return (long long int)rint(a);
}

__func__(double fdim(double a, double b))
{
  if (a > b) {
    return (a - b);
  } else if (a <= b) {
    return 0.0;
  } else if (__isnan(a)) {
    return a;
  } else {
    return b;
  }
}

__func__(double scalbn(double a, int b))
{
  return ldexp(a, b);
}

__func__(double scalbln(double a, long int b))
{
  int t;
  if (b > 2147483647L) {
    t = 2147483647;
  } else if (b < (-2147483647 - 1)) {
    t = (-2147483647 - 1);
  } else {
    t = (int)b;
  }
  return scalbn(a, t);
}

/*  
 * The following is based on: David Goldberg, "What every computer scientist 
 * should know about floating-point arithmetic", ACM Computing Surveys, Volume 
 * 23, Issue 1, March 1991.
 */
__func__(double log1p(double a))
{
  volatile double u, m;

  u = 1.0 + a;
  if (u == 1.0) {
    /* a very close to zero */
    u = a;
  } else {
    m = u - 1.0;
    u = log(u);
    if (a < 1.0) {
      /* a somewhat close to zero */
      u = a * u;
      u = u / m;
    }
  }
  return u;
}

/*
 * This code based on: http://www.cs.berkeley.edu/~wkahan/Math128/Sumnfp.pdf
 */
__func__(double expm1(double a))
{
  volatile double u, m;

  u = exp(a);
  m = u - 1.0;
  if (m == 0.0) {
    /* a very close zero */
    m = a;
  } 
  else if (fabs(a) < 1.0) {
    /* a somewhat close zero */
    u = log(u);
    m = m * a;
    m = m / u;
  }
  return m;
}

__func__(double cbrt(double a))
{
  double s, t;
  if (a == 0.0 || __isinf(a)) {
    return a;
  } 
  s = fabs(a);
  t = exp2(CUDART_THIRD * log2(s));           /* initial approximation */
  t = t - (t - (s / (t * t))) * CUDART_THIRD; /* refine approximation */
  if (__signbit(a)) {
    t = -t;
  }
  return t;
}

__func__(double acosh(double a))
{
  double s, t;

  t = a - 1.0;
  if (t == a) {
    return log(2.0) + log(a);
  } else {
    s = a + 1.0;
    t = t + sqrt(s * t);
    return log1p(t);
  }
}

__func__(double asinh(double a))
{
  double fa, oofa, t;

  fa = fabs(a);
  oofa = 1.0 / fa;
  t = fa + fa / (oofa + sqrt(1.0 + oofa * oofa));
  t = log1p(t);
  if (__signbit(a)) {
    t = -t;
  }
  return t;
}

__func__(double atanh(double a))
{
  double fa, t;

  fa = fabs(a);
  t = (2.0 * fa) / (1.0 - fa);
  t = 0.5 * log1p(t);
  if (__signbit(a)) {
    t = -t;
  }
  return t;
}

__func__(int ilogb(double a))
{
  volatile union {
    double                 d;
    unsigned long long int l;
  } x;
  unsigned long long int i;
  int expo = -1022;

  if (__isnan(a)) return -INT_MAX-1;
  if (__isinf(a)) return INT_MAX;
  x.d = a;
  i = x.l & 0x7fffffffffffffffull;
  if (i == 0) return -INT_MAX-1;
  if (i >= 0x0010000000000000ull) {
    return (int)(((i >> 52) & 0x7ff) - 1023);
  }
  while (i < 0x0010000000000000ull) {
    expo--;
    i <<= 1;
  }
  return expo;
}

__func__(double logb(double a))
{
  volatile union {
    double                 d;
    unsigned long long int l;
  } x;
  unsigned long long int i;
  int expo = -1022;

  if (__isnan(a)) return a;
  if (__isinf(a)) return fabs(a);
  x.d = a;
  i = x.l & 0x7fffffffffffffffull;
  if (i == 0) return -1.0/a;
  if (i >= 0x0010000000000000ull) {
    return (double)((int)((i >> 52) & 0x7ff) - 1023);
  }
  while (i < 0x0010000000000000ull) {
    expo--;
    i <<= 1;
  }
  return (double)expo;
}

__func__(double fma(double a, double b, double c))
{
  return __fma_rn(a, b, c);
}

__func__(void sincos(double a, double *sptr, double *cptr))
{
  *sptr = sin(a);
  *cptr = cos(a);
}

/*******************************************************************************
*                                                                              *
*  HOST IMPLEMENTATION FOR FLOAT. BASED ON DOUBLE                              *
*                                                                              *
*******************************************************************************/

__func__(float roundf(float a))
{
  return (float)round((double)a);
}

__func__(long int lroundf(float a))
{
  return lround((double)a);
}

__func__(long long int llroundf(float a))
{
  return llround((double)a);
}

__func__(float rintf(float a))
{
  return (float)rint((double)a);
}

__func__(long int lrintf(float a))
{
  return lrint((double)a);
}

__func__(long long int llrintf(float a))
{
  return llrint((double)a);
}

__func__(float logbf(float a))
{
  return (float)logb((double)a);
}

__func__(float scalblnf(float a, long int b))
{
  return (float)scalbln((double)a, b);
}

__func__(float acoshf(float a))
{
  return (float)acosh((double)a);
}

__func__(float asinhf(float a))
{
  return (float)asinh((double)a);
}

__func__(float atanhf(float a))
{
  return (float)atanh((double)a);
}

__func__(float cbrtf(float a))
{
  return (float)cbrt((double)a);
}

__func__(float expm1f(float a))
{
  return (float)expm1((double)a);
}

__func__(float exp10f(float a))
{
  return (float)exp10((double)a);
}

__func__(float fdimf(float a, float b))
{
  return (float)fdim((double)a, (double)b);
}

__func__(float hypotf(float a, float b))
{
  return (float)hypot((double)a, (double)b);
}

__func__(float log1pf(float a))
{
  return (float)log1p((double)a);
}

__func__(float scalbnf(float a, int b))
{
  return (float)scalbn((double)a, b);
}

__func__(float fmaf(float a, float b, float c))
{
  return (float)fma((double)a, (double)b, (double)c);
}

__func__(void sincosf(float a, float *sptr, float *cptr))
{
  double s, c;

  sincos((double)a, &s, &c);
  *sptr = (float)s;
  *cptr = (float)c;
}

__func__(int ilogbf(float a))
{
  return ilogb((double)a);
}

__func__(float erff(float a))
{
  return (float)erf((double)a);
}

__func__(float erfcf(float a))
{
  return (float)erfc((double)a);
}

__func__(float lgammaf(float a))
{
  return (float)lgamma((double)a);
}

__func__(float tgammaf(float a))
{
  return (float)tgamma((double)a);
}

__func__(float remquof(float a, float b, int *quo))
{
  return (float)remquo((double)a, (double)b, quo);
}

__func__(float remainderf(float a, float b))
{
  return (float)remainder((double)a, (double)b);
}

__func__(float nextafterf(float a, float b))
{
  return (float)nextafter((double)a, (double)b);
}

/*******************************************************************************
*                                                                              *
*  HOST IMPLEMENTATION FOR FLOAT. USE DEVICE IMPLEMENTATION IF POSSIBLE        *
*                                                                              *
*******************************************************************************/

__func__(int __finitef(float a))
{
  return __cuda___finitef(a);
}

__func__(int __isinff(float a))
{
  return __cuda___isinff(a);
}

__func__(int __isnanf(float a))
{
  return __cuda___isnanf(a);
}

/*******************************************************************************
*                                                                              *
*  HOST IMPLEMENTATION FOR DOUBLE. USE FLOAT DEVICE IMPLEMENTATION IF POSSIBLE *
*                                                                              *
*******************************************************************************/

__func__(double lgamma(double a))
{
  return (double)__cuda_lgammaf((float)a);
}

__func__(double tgamma(double a))
{
  return (double)__cuda_tgammaf((float)a);
}

__func__(double erf(double a))
{
  return (double)__cuda_erff((float)a);
}

__func__(double erfc(double a))
{
  return (double)__cuda_erfcf((float)a);
}

__func__(double remquo(double a, double b, int *quo))
{
  return (double)__cuda_remquof((float)a, (float)b, quo);
}

__func__(double remainder(double a, double b))
{
  return (double)__cuda_remainderf((float)a, (float)b);
}

__func__(double nextafter(double a, double b))
{
  return (double)__cuda_nextafterf((float)a, (float)b);
}

#endif /* _WIN32 */

#endif /* !__CUDABE__ && !__MULTI_CORE__ */

#endif /* __cplusplus && __CUDACC__ */

#if defined(CUDA_DOUBLE_MATH_FUNCTIONS) && defined(CUDA_FLOAT_MATH_FUNCTIONS)

#error -- conflicting mode for double math routines

#endif /* CUDA_DOUBLE_MATH_FUNCTIONS && CUDA_FLOAT_MATH_FUNCTIONS */

#if defined(CUDA_FLOAT_MATH_FUNCTIONS)

#include "math_functions_dbl_ptx1.h"

#endif /* CUDA_FLOAT_MATH_FUNCTIONS */

#if defined(CUDA_DOUBLE_MATH_FUNCTIONS)

#include "math_functions_dbl_ptx3_dynlink.h"

#endif /* CUDA_DOUBLE_MATH_FUNCTIONS */

#endif /* !__MATH_FUNCTIONS_H__ */
