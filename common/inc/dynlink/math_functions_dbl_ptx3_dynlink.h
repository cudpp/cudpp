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

#if !defined(__MATH_FUNCTIONS_DBL_PTX3_H__)
#define __MATH_FUNCTIONS_DBL_PTX3_H__

/* True double precision implementations, since native double support */

#if defined(__cplusplus) && defined(__CUDACC__)

#elif !defined(__CUDACC__)

#include "crt/func_macro.h"

#define INT_MAX \
        ((int)((unsigned int)-1 >> 1))

#include "device_functions_dynlink.h"
#include "math_constants.h"

/*******************************************************************************
*                                                                              *
* DEVICE IMPLEMENTATIONS FOR FUNCTIONS WITH BUILTIN NVOPENCC OPREATIONS        *
*                                                                              *
*******************************************************************************/

__device_func__(double __cuda_fabs(double a))
{
  return fabs(a);
}

__device_func__(double __cuda_fmax(double a, double b))
{
#if !defined(__CUDABE__)
  volatile union {
    double                 d;
    unsigned long long int l;
  } cvta, cvtb;
  cvta.d = a;
  cvtb.d = b;
  if ((cvtb.l << 1) > 0xffe0000000000000ULL) return a;
  if ((cvta.l << 1) > 0xffe0000000000000ULL) return b;
  if ((cvta.d == 0.0) && (cvtb.d == 0.0)) {
    cvta.l &= cvtb.l;
    return cvta.d;
  }
  return a > b ? a : b;
#else    
  return fmax(a, b);
#endif /* !defined(__CUDABE__) */
}

__device_func__(double __cuda_fmin(double a, double b))
{
#if !defined(__CUDABE__)
  volatile union {
    double                 d;
    unsigned long long int l;
  } cvta, cvtb;
  cvta.d = a;
  cvtb.d = b;
  if ((cvtb.l << 1) > 0xffe0000000000000ULL) return a;
  if ((cvta.l << 1) > 0xffe0000000000000ULL) return b;
  if ((cvta.l | cvtb.l) == 0x8000000000000000ULL) {
    return CUDART_NEG_ZERO  ;
  }
  return a < b ? a : b;
#else    
  return fmin(a, b);
#endif /* !defined(__CUDABE__) */
}

__device_func__(double __cuda_ceil(double a))
{
  return ceil(a);
}

__device_func__(double __cuda_floor(double a))
{
  return floor(a);
}

__device_func__(double __cuda_trunc(double a))
{
  return trunc(a);
}

__device_func__(double __cuda_nearbyint(double a))
{
#if defined(__CUDABE__)
  return round(a);
#else /* __CUDABE__ */
  double res = nearbyint(a);
#if defined(__APPLE__)
  if ((a != 0.0) && (__cuda_fabs(a) <= 0.5)) {
    res = fabs(res) * ((a < 0.0) ? -3e-324 : 3e-324);
  }
#endif /* __APPLE__ */
  return res;
#endif /* __CUDABE__ */
}

/*******************************************************************************
*                                                                              *
* DEVICE IMPLEMENTATIONS FOR FUNCTIONS WITHOUT BUILTIN NVOPENCC OPREATIONS     *
*                                                                              *
*******************************************************************************/

__device_func__(double __cuda_rint(double a))
{
  return __cuda_nearbyint(a);
}

__device_func__(long int __cuda_lrint(double a))
{
#if defined(__LP64__)
  return (long int)__double2ll_rn(a);
#else /* __LP64__ */
  return (long int)__double2int_rn(a);
#endif /* __LP64__ */
}

__device_func__(long long int __cuda_llrint(double a))
{
  return __double2ll_rn(a);
}

__device_func__(int __cuda___signbit(double a))
{
  return (int)((unsigned int)__double2hiint(a) >> 31);
}

__device_func__(int __cuda___finite(double a))
{
  return __cuda_fabs(a) < CUDART_INF;
}

__device_func__(int __cuda___isinf(double a))
{
  return __cuda_fabs(a) == CUDART_INF;
}

__device_func__(int __cuda___isnan(double a))
{
  return !(__cuda_fabs(a) <= CUDART_INF);
}

__device_func__(double __cuda_copysign(double a, double b))
{
  int alo, ahi, bhi;

  bhi = __double2hiint(b);
  alo = __double2loint(a);
  ahi = __double2hiint(a);
  ahi = (bhi & 0x80000000) | (ahi & ~0x80000000);
  return __hiloint2double(ahi, alo);
}

/* 1152 bits of 2/PI for Payne-Hanek style argument reduction. */
static __constant__ unsigned long long int __cudart_i2opi_d [] = {
  0x6bfb5fb11f8d5d08ULL,
  0x3d0739f78a5292eaULL,
  0x7527bac7ebe5f17bULL,
  0x4f463f669e5fea2dULL,
  0x6d367ecf27cb09b7ULL,
  0xef2f118b5a0a6d1fULL,
  0x1ff897ffde05980fULL,
  0x9c845f8bbdf9283bULL,
  0x3991d639835339f4ULL,
  0xe99c7026b45f7e41ULL,
  0xe88235f52ebb4484ULL,
  0xfe1deb1cb129a73eULL,
  0x06492eea09d1921cULL,
  0xb7246e3a424dd2e0ULL,
  0xfe5163abdebbc561ULL,
  0xdb6295993c439041ULL,
  0xfc2757d1f534ddc0ULL,
  0xa2f9836e4e441529ULL,
};

__device_func__(double __internal_trig_reduction_kerneld(double a, int *quadrant))
{
  double j;
  int q;
  if (__cuda_fabs(a) > CUDART_TRIG_PLOSS) {
    /* Payne-Hanek style argument reduction. */
    unsigned long long int ia;
    unsigned long long int s;
    unsigned long long int result[5];
    unsigned long long int phi, plo;
    unsigned long long int hi, lo;
    unsigned int e;
    int idx;

    ia = __double_as_longlong(a);
    s = ia & 0x8000000000000000ULL;
    e = (unsigned int)(((ia >> 52) & 0x7ff) - 1024);
    ia = (ia << 11) | 0x8000000000000000ULL;
    /* compute x * 2/pi */
    idx = 16 - (e >> 6);
    hi = 0;
#if defined(__CUDABE__)
#pragma unroll 1
#endif /* __CUDABE__ */
    for (q = (idx-1); q < min(18,idx+3); q++) {
      plo = __cudart_i2opi_d[q] * ia;
      phi = __umul64hi (__cudart_i2opi_d[q], ia);
      lo = hi + plo;
      hi = phi + (lo < plo);
      result[q-(idx-1)] = lo;
    }
    result[q-(idx-1)] = hi;
    e = e & 63;
    /* shift result such that hi:lo<127:126> are the least significant
       integer bits, and hi:lo<125:0> are the fractional bits of the result
     */
    hi = result[3];
    lo = result[2];
    if (e) {
      q = 64 - e;
      hi = (hi << e) | (lo >> q);
      lo = (lo << e) | (result[1] >> q);
    }
    q = (int)(hi >> 62);
    /* fraction */
    hi = (hi << 2) | (lo >> 62);
    lo = (lo << 2);
    e = (hi | (lo > 0)) > 0x8000000000000000ULL; /* fraction >= 0.5 */
    q += e;
    if (s) q = -q;
    if (e) {
      unsigned long long int t;
      hi = ~hi;
      lo = -(long long int)lo;
      t = (lo == 0ULL);
      hi += t;
      s = s ^ 0x8000000000000000ULL;
    }
    *quadrant = q;
    /* normalize fraction */
    e = 0;
    while ((long long int)hi > 0) {
      hi = (hi << 1) | (lo >> 63);
      lo = (lo << 1);
      e--;
    }
    lo = hi * 0xC90FDAA22168C235ULL;
    hi = __umul64hi (hi, 0xC90FDAA22168C235ULL);
    if ((long long int)hi > 0) {
      hi = (hi << 1) | (lo >> 63);
      lo = (lo << 1);
      e--;
    }
    hi = hi + (lo > 0);
    ia = s | ((((unsigned long long int)(e + 1022)) << 52) + (hi >> 11) +
              ((hi << 53) >= 0x8000000000000000ULL));
    return __longlong_as_double(ia);
  } 
  /* NOTE: for an input of -0, this returns -0 */
  q = __double2int_rn (a * CUDART_2_OVER_PI);
  j = (double)q;
  /* Constants from S. Boldo, M. Daumas, R.-C. Li: "Formally Verified Argument
   * Reduction with a Fused-Multiply-Add", retrieved from the internet at
   * http://arxiv.org/PS_cache/arxiv/pdf/0708/0708.3722v1.pdf
   */
  a = __fma_rn (-j, 1.5707963267948966e+000, a);
  a = __fma_rn (-j, 6.1232339957367574e-017, a);
  a = __fma_rn (-j, 8.4784276603688985e-032, a);
  *quadrant = q;
  return a;
}

/* approximate sine on -pi/4...+pi/4 */
__device_func__(double __internal_sin_kerneld(double x))
{
  double x2, z;
  x2 = x * x;
  z =                   1.5896230157221844E-010;
  z = __fma_rn (z, x2, -2.5050747762850355E-008);
  z = __fma_rn (z, x2,  2.7557313621385676E-006);
  z = __fma_rn (z, x2, -1.9841269829589539E-004);
  z = __fma_rn (z, x2,  8.3333333333221182E-003);
  z = __fma_rn (z, x2, -1.6666666666666630E-001);
  z  = z * x2;
  z  = __fma_rn (z, x, x);
  return z;
}

/* approximate cosine on -pi/4...+pi/4 */
__device_func__(double __internal_cos_kerneld(double x))
{
  double x2, z;
  x2 = x * x;
  z  =                  -1.136788825395985E-011;   
  z  = __fma_rn (z, x2,  2.087588480545065E-009);
  z  = __fma_rn (z, x2, -2.755731555403950E-007);
  z  = __fma_rn (z, x2,  2.480158729365970E-005);
  z  = __fma_rn (z, x2, -1.388888888888074E-003);
  z  = __fma_rn (z, x2,  4.166666666666664E-002);
  z  = __fma_rn (z, x2, -5.000000000000000E-001);
  z  = __fma_rn (z, x2,  1.000000000000000E+000);
  return z;
}

/* approximate tangent on -pi/4...+pi/4 */
__device_func__(double __internal_tan_kerneld(double x, int i))
{
  double x2, z, q;
  x2 = x * x;
  z =                   9.8006287203286300E-006;
  z = __fma_rn (z, x2, -2.4279526494179897E-005);
  z = __fma_rn (z, x2,  4.8644173130937162E-005);
  z = __fma_rn (z, x2, -2.5640012693782273E-005);
  z = __fma_rn (z, x2,  6.7223984330880073E-005);
  z = __fma_rn (z, x2,  8.3559287318211639E-005);
  z = __fma_rn (z, x2,  2.4375039850848564E-004);
  z = __fma_rn (z, x2,  5.8886487754856672E-004);
  z = __fma_rn (z, x2,  1.4560454844672040E-003);
  z = __fma_rn (z, x2,  3.5921008885857180E-003);
  z = __fma_rn (z, x2,  8.8632379218613715E-003);
  z = __fma_rn (z, x2,  2.1869488399337889E-002);
  z = __fma_rn (z, x2,  5.3968253972902704E-002);
  z = __fma_rn (z, x2,  1.3333333333325342E-001);
  z = __fma_rn (z, x2,  3.3333333333333381E-001);
  z = z * x2;
  q = __fma_rn (z, x, x);
  if (i) {
    double s = q - x; 
    double w = __fma_rn (z, x, -s); // tail of q
    z = 1.0 / q;
    z = -z;
    s = __fma_rn (q, z, 1.0);
    q = __fma_rn (z, __fma_rn (z, w, s), z);
  }           
  return q;
}

__device_func__(double __cuda_sqrt(double a))
{
  return sqrt(a);
}

__device_func__(double __cuda_rsqrt(double a))
{
#if !defined(__CUDABE__)
    return 1.0 / sqrt(a);
#else
    return rsqrt(a);
#endif
}

/* approximates exp(a)-1 on [-log(1.5),log(1.5)] accurate to 1 ulp */
__device_func__(double __internal_expm1_kernel (double a))
{
  double t;
  t =                 2.08842685477913050E-009;
  t = __fma_rn (t, a, 2.51366409033551950E-008);
  t = __fma_rn (t, a, 2.75574612072447230E-007);
  t = __fma_rn (t, a, 2.75571539284473460E-006);
  t = __fma_rn (t, a, 2.48015869443077950E-005);
  t = __fma_rn (t, a, 1.98412699878799470E-004);
  t = __fma_rn (t, a, 1.38888888892029890E-003);
  t = __fma_rn (t, a, 8.33333333327662860E-003);
  t = __fma_rn (t, a, 4.16666666666656370E-002);
  t = __fma_rn (t, a, 1.66666666666667380E-001);
  t = __fma_rn (t, a, 5.00000000000000000E-001);
  t = t * a;
  t = __fma_rn (t, a, a);
  return t;
}

/* approximate 2*atanh(0.5*a) on [-0.25,0.25] */
__device_func__(double __internal_atanh_kernel (double a_1, double a_2))
{
  double a, a2, t;

  a = a_1 + a_2;
  a2 = a * a;
  t =                  7.597322383488143E-002/65536.0;
  t = __fma_rn (t, a2, 6.457518383364042E-002/16384.0);          
  t = __fma_rn (t, a2, 7.705685707267146E-002/4096.0);
  t = __fma_rn (t, a2, 9.090417561104036E-002/1024.0);
  t = __fma_rn (t, a2, 1.111112158368149E-001/256.0);
  t = __fma_rn (t, a2, 1.428571416261528E-001/64.0);
  t = __fma_rn (t, a2, 2.000000000069858E-001/16.0);
  t = __fma_rn (t, a2, 3.333333333333198E-001/4.0);
  t = t * a2;
  t = __fma_rn (t, a, a_2);
  t = t + a_1;
  return t;
}

__device_func__(double __internal_exp2i_kernel(int b))
{
  return __hiloint2double((b + 1023) << 20, 0);
}

__device_func__(double __internal_half(double a))
{
  unsigned int ihi, ilo;
  ilo = __double2loint(a);
  ihi = __double2hiint(a);
  return __hiloint2double(ihi - 0x00100000, ilo);
}

__device_func__(double __internal_twice(double a))
{
  unsigned int ihi, ilo;
  ilo = __double2loint(a);
  ihi = __double2hiint(a);
  return __hiloint2double(ihi + 0x00100000, ilo);
}

__device_func__(double __cuda_sin(double a))
{
  double z;
  int i;
  if (__cuda___isinf(a) || (a == CUDART_ZERO)) {
    return __dmul_rn(a, CUDART_ZERO);
  }
  z = __internal_trig_reduction_kerneld(a, &i);
  /* here, abs(z) <= pi/4, and i has the quadrant */
  if (i & 1) {
    z = __internal_cos_kerneld(z);
  } else {
    z = __internal_sin_kerneld(z);
  }
  if (i & 2) {
    z = -z;
  }
  return z;
}

__device_func__(double __cuda_cos(double a))
{
  double z;
  int i;
  if (__cuda___isinf(a)) {
    return CUDART_NAN;
  }
  z = __internal_trig_reduction_kerneld(a, &i);
  /* here, abs(z) <= pi/4, and i has the quadrant */
  i++;
  if (i & 1) {
    z = __internal_cos_kerneld(z);
  } else {
    z = __internal_sin_kerneld(z);
  }
  if (i & 2) {
    z = -z;
  }
  return z;
}

__device_func__(void __cuda_sincos(double a, double *sptr, double *cptr))
{
  double t, u, s, c;
  int i;
  if (__cuda___isinf(a)) {
    *sptr = CUDART_NAN;
    *cptr = CUDART_NAN;
    return;
  }
  if (a == CUDART_ZERO) {
    *sptr = a;
    *cptr = 1.0;
    return;
  } 
  t = __internal_trig_reduction_kerneld(a, &i);
  u = __internal_cos_kerneld(t);
  t = __internal_sin_kerneld(t);
  if (i & 1) {
    s = u;
    c = t;
  } else {
    s = t;
    c = u;
  }
  if (i & 2) {
    s = -s;
  }
  i++;
  if (i & 2) {
    c = -c;
  }
  *sptr = s;
  *cptr = c;
}

__device_func__(double __cuda_tan(double a))
{
  double z;
  int i;
  if (__cuda___isinf(a)) {
    return CUDART_NAN;
  }
  z = __internal_trig_reduction_kerneld(a, &i);
  /* here, abs(z) <= pi/4, and i has the quadrant */
  z = __internal_tan_kerneld(z, i & 1);
  return z;
}

__device_func__(double __cuda_log(double a))
{
  double m, f, g, u, v, tmp, q, ulo, log_lo, log_hi;
  int ihi, ilo;
  int e = 0;

  ihi = __double2hiint(a);
  ilo = __double2loint(a);
  if (__cuda___isnan(a)) {
    return a + a;
  }
  /* log(x) is undefined for x < 0.0, return INDEFINITE */
  if (a < 0.0) {
    return CUDART_NAN;
  }
  /* log(0) = -INF */
  if (a == 0) {
    return -CUDART_INF;
  }
  /* log(INF) = INF */
  if (__cuda___isinf(a)) {
    return a;
  }
  /* normalize denormals */
  if (a < CUDART_TWO_TO_M1022) {
    a = a * CUDART_TWO_TO_54;
    e = -54;
    ihi = __double2hiint(a);
    ilo = __double2loint(a);
  }
  /* a = m * 2^e. m <= sqrt(2): log2(a) = log2(m) + e.
   * m > sqrt(2): log2(a) = log2(m/2) + (e+1)
   */
  e += ((ihi >> 20) & 0x7ff) - 1023;
  m = __hiloint2double ((ihi & 0x800fffff) | 0x3ff00000, ilo);
  if (m > CUDART_SQRT_TWO) {
    m = __internal_half(m);
    e = e + 1;
  }
  /* log((1+m)/(1-m)) = 2*atanh(m). log(m) = 2*atanh ((m-1)/(m+1)) */
  f = m - 1.0;
  g = m + 1.0;
  g = 1.0 / g;
  u = f * g;
  u = u + u;  
  /* u = 2.0 * (m - 1.0) / (m + 1.0) */
  v = u * u;
  q =                 6.7261411553826339E-2/65536.0;
  q = __fma_rn (q, v, 6.6133829643643394E-2/16384.0);
  q = __fma_rn (q, v, 7.6940931149150890E-2/4096.0);
  q = __fma_rn (q, v, 9.0908745692137444E-2/1024.0);
  q = __fma_rn (q, v, 1.1111111499059706E-1/256.0);
  q = __fma_rn (q, v, 1.4285714283305975E-1/64.0);
  q = __fma_rn (q, v, 2.0000000000007223E-1/16.0);
  q = __fma_rn (q, v, 3.3333333333333326E-1/4.0);
  tmp = __internal_twice (f - u);
  tmp = __fma_rn (-u, f, tmp); // tmp = remainder of division
  ulo = g * tmp;               // less significant quotient bits
  /* u + ulo = 2.0 * (m - 1.0) / (m + 1.0) to more than double precision */
  q = q * v;
  q = q * u;      
  /* log_hi + log_lo = log(m) to more than double precision */ 
  log_hi = u;
  log_lo = ulo + q;
  /* log_hi + log_lo = log(m)+e*log(2)=log(a) to more than double precision */
  q   = __fma_rn ( e, CUDART_LN2_HI, log_hi);
  tmp = __fma_rn (-e, CUDART_LN2_HI, q);
  tmp = tmp - log_hi;
  log_hi = q;
  log_lo = log_lo - tmp;
  log_lo = __fma_rn (e, CUDART_LN2_LO, log_lo);
  return log_hi + log_lo;
}

/* Requires |x.y| > |y.y|. 8 DP operations */
__device_func__(double2 __internal_ddadd_xgty (double2 x, double2 y))
{
    double2 z;
#if defined(__GNUC__) && !defined(__CUDABE__)
    volatile double r, s, e;
#else 
    double r, s, e;
#endif
    r = x.y + y.y;
    e = x.y - r;
    s = ((e + y.y) + y.x) + x.x;
    z.y = e = r + s;
    z.x = (r - e) + s;
    return z;
}

/* Take full advantage of FMA. Only 8 DP operations */
__device_func__(double2 __internal_ddmul (double2 x, double2 y))
{
#if defined(__GNUC__) && !defined(__CUDABE__)
    volatile double e;
#else
    double e;
#endif
    double2 t, z;
    t.y = x.y * y.y;
    t.x = __fma_rn (x.y, y.y, -t.y);
    t.x = __fma_rn (x.x, y.x, t.x);
    t.x = __fma_rn (x.y, y.x, t.x);
    t.x = __fma_rn (x.x, y.y, t.x);
    z.y = e = t.y + t.x;
    z.x = (t.y - e) + t.x;
    return z;
}

__device_func__(double2 __internal_log_ext_prec(double a))
{
  double2 res;
  double2 qq, cc, uu, tt;
  double f, g, u, v, q, ulo, tmp, m;
  int ilo, ihi, expo;

  ihi = __double2hiint(a);
  ilo = __double2loint(a);
  expo = (ihi >> 20) & 0x7ff;
  /* convert denormals to normals for computation of log(a) */
  if (expo == 0) {
    a *= CUDART_TWO_TO_54;
    ihi = __double2hiint(a);
    ilo = __double2loint(a);
    expo = (ihi >> 20) & 0x7ff;
    expo -= 54;
  }  
  expo -= 1023;
  /* log(a) = log(m*2^expo) = 
     log(m) + log(2)*expo, if m < sqrt(2), 
     log(m*0.5) + log(2)*(expo+1), if m >= sqrt(2)
  */
  m = __hiloint2double((ihi & 0x800fffff) | 0x3ff00000, ilo);
  if (m > CUDART_SQRT_TWO) {
    m = __internal_half(m);
    expo = expo + 1;
  }
  /* compute log(m) with extended precision using an algorithm derived from 
   * P.T.P. Tang, "Table Driven Implementation of the Logarithm Function", 
   * TOMS, Vol. 16., No. 4, December 1990, pp. 378-400. A modified polynomial 
   * approximation to atanh(x) on the interval [-0.1716, 0.1716] is utilized.
   */
  f = m - 1.0;
  g = m + 1.0;
  g = 1.0 / g;
  u = f * g;
  u = u + u;  
  /* u = 2.0 * (m - 1.0) / (m + 1.0) */
  v = u * u;
  q =                 6.6253631649203309E-2/65536.0;
  q = __fma_rn (q, v, 6.6250935587260612E-2/16384.0);
  q = __fma_rn (q, v, 7.6935437806732829E-2/4096.0);
  q = __fma_rn (q, v, 9.0908878711093280E-2/1024.0);
  q = __fma_rn (q, v, 1.1111111322892790E-1/256.0);
  q = __fma_rn (q, v, 1.4285714284546502E-1/64.0);
  q = __fma_rn (q, v, 2.0000000000003113E-1/16.0);
  q = q * v;
  /* u + ulo = 2.0 * (m - 1.0) / (m + 1.0) to more than double precision */
  tmp = __internal_twice (f - u);
  tmp = __fma_rn (-u, f, tmp); // tmp = remainder of division
  ulo = g * tmp;               // less significand quotient bits
  /* switch to double-double at this point */
  qq.y = q;
  qq.x = 0.0;
  uu.y = u;
  uu.x = ulo;
  cc.y =  3.3333333333333331E-1/4.0;
  cc.x = -9.8201492846582465E-18/4.0;
  qq = __internal_ddadd_xgty (cc, qq);
  /* compute log(m) in double-double format */
  qq = __internal_ddmul(qq, uu);
  qq = __internal_ddmul(qq, uu);
  qq = __internal_ddmul(qq, uu);
  uu = __internal_ddadd_xgty (uu, qq);
  u   = uu.y;
  ulo = uu.x;
  /* log(2)*expo in double-double format */
  tt.y = expo * 6.9314718055966296e-001;  /* multiplication is exact */
  tt.x = expo * 2.8235290563031577e-013;
  /* log(a) = log(m) + log(2)*expo;  if expo != 0, |log(2)*expo| > |log(m)| */
  res = __internal_ddadd_xgty (tt, uu);
  return res;
}

__device_func__(double __cuda_log2(double a))
{
  double t;
  t = __cuda_log(a);
  return __fma_rn (t, CUDART_L2E_HI, t * CUDART_L2E_LO);
}

__device_func__(double __cuda_log10(double a))
{
  double t;
  t = __cuda_log(a);
  return __fma_rn (t, CUDART_LGE_HI, t * CUDART_LGE_LO);
}

__device_func__(double __cuda_log1p(double a))
{
  double t;
#if !defined(__CUDABE__) 
  if (__cuda___isnan(a)) {
    return a + a;
  }
#endif
  if ((a < -0.4) || (a > CUDART_TWOTHIRD)) {
    return __cuda_log (a + 1.0);
  }
  /* Compute log2(a+1) = 2*atanh(a/(a+2)) */
  t = a + 2.0;
  t = a / t;
  t = -a * t;
  t = __internal_atanh_kernel(a, t);
  return t;
}

__device_func__(double __internal_exp_kernel(double a, int scale))
{ 
  double t, fac, z;
  int i;
  /* exp(a) = 2^(rint(a/log(2)) + z) = 2^(i + z) */
  t = __cuda_rint (a * CUDART_L2E);
  i = (int)t;
  z = __fma_rn (t, -CUDART_LN2_HI, a);
  z = __fma_rn (t, -CUDART_LN2_LO, z);
  fac = 2.0;
  if (i <= -1021) {
    i += 55;
    fac = CUDART_TWO_TO_M54;
  }
  t = __internal_expm1_kernel(z);
  /* exp(a) = 2^i * 2^z */
  z = __hiloint2double((1022 + i + scale) << 20, 0);
  t = __fma_rn (t, z, z);
  t = t * fac;
  return t;
}

__device_func__(double __cuda_exp(double a))
{
  if (a > CUDART_LN2_X_1024) {
    return CUDART_INF;
  }
  if (a <= -CUDART_LN2_X_1075) {
    return CUDART_ZERO;
  }
  a = __internal_exp_kernel(a, 0);
  return a;
}

__device_func__(double __cuda_exp2(double a))
{
  double z;
  double t;
  double fac;
  int i;

  if (a >= 1024.0) {
    return CUDART_INF;
  }
  if (a < -1075.0) {
    return CUDART_ZERO;
  }
  t = __cuda_rint (a);
  z = a - t;
  i = (int)t;
  fac = 2.0;
  if (i <= -1021) {
    i += 55;
    fac = CUDART_TWO_TO_M54;
  }
  /* 2^z = exp(log(2)*z) */
  z = __fma_rn (z, CUDART_LN2_HI, z * CUDART_LN2_LO);
  t = __internal_expm1_kernel(z);
  z = __hiloint2double((1022 + i) << 20, 0);
  t = __fma_rn (t, z, z);
  t = t * fac;
  return t;
}

__device_func__(double __cuda_exp10(double a))
{
  double z;
  double t;
  double fac;
  int i;

  if (a >= CUDART_LG2_X_1024) {
    return CUDART_INF;
  }
  if (a < -CUDART_LG2_X_1075) {
    return CUDART_ZERO;
  }
  t = __cuda_rint (a * CUDART_L2T);
  i = (int)t;
  z = __fma_rn (t, -CUDART_LG2_HI, a);
  z = __fma_rn (t, -CUDART_LG2_LO, z);
  fac = 2.0;
  if (i <= -1021) {
    i += 55;
    fac = CUDART_TWO_TO_M54;
  }
  /* 2^z = exp(log(10)*z) */
  z = __fma_rn (z, CUDART_LNT_HI, z * CUDART_LNT_LO);
  t = __internal_expm1_kernel(z);
  /* exp(a) = 2^i * 2^z */
  z = __hiloint2double((1022 + i) << 20, 0);
  t = __fma_rn (t, z, z);
  t = t * fac;
  return t;
}

__device_func__(double __cuda_expm1(double a))
{
  double t, z, u;
  int i, j;
  if (a > CUDART_LN2_X_1024) {
    return CUDART_INF;
  }
  if (a < -53.0) {
    return -1.0; 
  }
  t = __cuda_rint (a * CUDART_L2E);
  i = (int)t;
  z = __fma_rn (t, -CUDART_LN2_HI, a);
  z = __fma_rn (t, -CUDART_LN2_LO, z);
  if (__cuda_fabs(a) < 0.405465108) {
    z = a;
    i = 0;
  }
  j = (i == 1024) ? (i - 1) : i;
  t = __internal_expm1_kernel(z);
  u = __hiloint2double((1023 + j) << 20, 0);
  a = u - 1.0;
  t = __fma_rn (t, u, a);
  if (z == 0.0) t = z;              /* preserve -0 */
  if (i == 1024) t = t + t;
  return t;
}

__device_func__(double __cuda_cosh(double a))
{
  double z;
  if (__cuda___isnan(a)) {
    return a + a;
  }
  a = __cuda_fabs(a);
  z = __internal_exp_kernel(a, -2);
  z = __fma_rn(2.0, z, 0.125 / z);
  if (a >= CUDART_LN2_X_1025) {
    z = CUDART_INF_F;     /* overflow -> infinity */
  }
  return z;
}

__device_func__(double __cuda_sinh(double a))
{
  double s, z;
  s = a;
  a = __cuda_fabs(a);
  if (a < 1.0) {          /* danger of catastrophic cancellation */
    double a2 = a * a;
    /* approximate sinh(x) on [0,1] with a polynomial */
    z =                  1.632386098183803E-010;
    z = __fma_rn (z, a2, 2.504854501385687E-008);
    z = __fma_rn (z, a2, 2.755734274788706E-006);
    z = __fma_rn (z, a2, 1.984126976294102E-004);
    z = __fma_rn (z, a2, 8.333333333452911E-003);
    z = __fma_rn (z, a2, 1.666666666666606E-001);
    z = z * a2;
    z = __fma_rn (z, a, a);
  } else if (a < 2.0) {   /* work around accuracy issue in vicinity of 1.4 */
    z = __cuda_expm1(a);
    z = __internal_half (z + z / (z + 1.0));
  } else {
    z = __internal_exp_kernel(a, -1);
    z = z + (1.0 / (-4.0 * z));
    if (a >= CUDART_LN2_X_1025) {
      z = CUDART_INF;     /* overflow -> infinity */
    }
  }
  z = __cuda_copysign(z, s);
  return z;
}

__device_func__(double __cuda_tanh(double a))
{
  double t;
  t = __cuda_fabs(a);
  if (t >= 0.55) {
    double s;
    s = 1.0 - 2.0 / (__internal_exp_kernel(2.0 * t, 0) + 1.0);
    if (t > 350.0) {
      s = 1.0;       /* overflow -> 1.0 */
    }
    a = __cuda_copysign(s, a);
  } else {
    double a2;
    a2 = a * a;
    t =                   5.102147717274194E-005;
    t = __fma_rn (t, a2, -2.103023983278533E-004);
    t = __fma_rn (t, a2,  5.791370145050539E-004);
    t = __fma_rn (t, a2, -1.453216755611004E-003);
    t = __fma_rn (t, a2,  3.591719696944118E-003);
    t = __fma_rn (t, a2, -8.863194503940334E-003);
    t = __fma_rn (t, a2,  2.186948597477980E-002);
    t = __fma_rn (t, a2, -5.396825387607743E-002);
    t = __fma_rn (t, a2,  1.333333333316870E-001);
    t = __fma_rn (t, a2, -3.333333333333232E-001);
    t = t * a2;
    t = __fma_rn (t, a, a);
    a = __cuda_copysign(t, a);
  }
  return a;
}

__device_func__(double __internal_atan_kernel(double a))
{
  double t, a2;
  a2 = a * a;
  t =             -2.0258553044438358E-005 ;
  t = __fma_rn (t, a2,  2.2302240345758510E-004);
  t = __fma_rn (t, a2, -1.1640717779930576E-003);
  t = __fma_rn (t, a2,  3.8559749383629918E-003);
  t = __fma_rn (t, a2, -9.1845592187165485E-003);
  t = __fma_rn (t, a2,  1.6978035834597331E-002);
  t = __fma_rn (t, a2, -2.5826796814495994E-002);
  t = __fma_rn (t, a2,  3.4067811082715123E-002);
  t = __fma_rn (t, a2, -4.0926382420509971E-002);
  t = __fma_rn (t, a2,  4.6739496199157994E-002);
  t = __fma_rn (t, a2, -5.2392330054601317E-002);
  t = __fma_rn (t, a2,  5.8773077721790849E-002);
  t = __fma_rn (t, a2, -6.6658603633512573E-002);
  t = __fma_rn (t, a2,  7.6922129305867837E-002);
  t = __fma_rn (t, a2, -9.0909012354005225E-002);
  t = __fma_rn (t, a2,  1.1111110678749424E-001);
  t = __fma_rn (t, a2, -1.4285714271334815E-001);
  t = __fma_rn (t, a2,  1.9999999999755019E-001);
  t = __fma_rn (t, a2, -3.3333333333331860E-001);
  t = t * a2;
  t = __fma_rn (t, a, a);
  return t;
}

__device_func__(double __cuda_atan2(double a, double b))
{
  double t0, t1, t3;
  if (__cuda___isnan(a) || __cuda___isnan(b)) {
    return a + b;
  }
  /* reduce arguments to first octant */
  /* r = (|x| < |y|) ? (|x| / |y|) : (|y| / |x|) */
  t3 = __cuda_fabs(b);
  t1 = __cuda_fabs(a);
  if (t3 == 0.0 && t1 == 0.0) {
    t3 = __cuda___signbit(b) ? CUDART_PI : 0;
  } else if (__cuda___isinf(t3) && __cuda___isinf(t1)) {
    t3 = __cuda___signbit(b) ? CUDART_3PIO4 : CUDART_PIO4;
  } else {
    t0 = __cuda_fmax (t1, t3);
    t1 = __cuda_fmin (t1, t3);
    t3 = t1 / t0;
    t3 = __internal_atan_kernel(t3);
    /* Map result according to octant. */
    if (__cuda_fabs(a) > __cuda_fabs(b)) t3 = CUDART_PIO2 - t3;
    if (b < 0.0)                         t3 = CUDART_PI - t3;
  }
  t3 = __cuda_copysign(t3, a);
  return t3;
}

__device_func__(double __cuda_atan(double a))
{
  double t0, t1;
  /* reduce argument to first octant */
  t0 = __cuda_fabs(a);
  t1 = t0;
  if (t0 > 1.0) {
    t1 = 1.0 / t1;
  }
  /* approximate atan(r) in first octant */
  t1 = __internal_atan_kernel(t1);
  /* map result according to octant. */
  if (t0 > 1.0) {
    t1 = CUDART_PIO2 - t1;
  }
  return __cuda_copysign (t1, a);
}

/* b should be the square of a */
__device_func__(double __internal_asin_kernel(double a, double b))
{
  double r;
  r =                  6.259798167646803E-002;
  r = __fma_rn (r, b, -7.620591484676952E-002);
  r = __fma_rn (r, b,  6.686894879337643E-002);
  r = __fma_rn (r, b, -1.787828218369301E-002); 
  r = __fma_rn (r, b,  1.745227928732326E-002);
  r = __fma_rn (r, b,  1.000422754245580E-002);
  r = __fma_rn (r, b,  1.418108777515123E-002);
  r = __fma_rn (r, b,  1.733194598980628E-002);
  r = __fma_rn (r, b,  2.237350511593569E-002);
  r = __fma_rn (r, b,  3.038188875134962E-002);
  r = __fma_rn (r, b,  4.464285849810986E-002);
  r = __fma_rn (r, b,  7.499999998342270E-002);
  r = __fma_rn (r, b,  1.666666666667375E-001);
  r = r * b;
  return r;
}

__device_func__(double __cuda_asin(double a))
{
  double t0, t1;
  t0 = __cuda_fabs(a);
  if (t0 > 1.0) {
    return CUDART_NAN;
  }
  if (t0 > 0.575) {
    t1 = __fma_rn (-0.5, t0, 0.5);
    t0 = __cuda_sqrt (t1);
    t1 = __internal_asin_kernel (t0, t1);
    t0 = -2.0 * t0;
    t1 = __fma_rn (t0, t1, CUDART_PIO2_LO);
    t0 = t0 + CUDART_PIO4_HI;
    t1 = t0 + t1;
    t1 = t1 + CUDART_PIO4_HI;
  } else {
    t1 = t0 * t0;
    t1 = __internal_asin_kernel (t0, t1);
    t1 = __fma_rn (t1, t0, t0);
  }
  return __cuda_copysign(t1, a);
}

__device_func__(double __cuda_acos(double a))
{
  double t0, t1;
#if !defined(__CUDABE__) 
  if (__cuda___isnan(a)) {
    return a + a;
  }
#endif
  t0 = __cuda_fabs (a);
  if (t0 > 0.575) {
    t1 = __fma_rn (-0.5, t0, 0.5);
    t0 = __cuda_sqrt(t1);
    t1 = __internal_asin_kernel (t0, t1);
    t0 = __fma_rn (t1, t0, t0);
    t0 = 2.0 * t0;
    if (__cuda___signbit(a)) {    
      t0 = __fma_rn (1.0, t0, -CUDART_PI_LO);
      t0 = CUDART_PI_HI - t0;
    }
  } else {
    t1 = t0 * t0;
    t1 = __internal_asin_kernel (t0, t1);
    t0 = __fma_rn (t1, t0, t0);
    if (__cuda___signbit(a)) {
      t0 = __fma_rn (1.0, t0, +CUDART_PIO2_LO);
      t0 = CUDART_PIO2_HI + t0;
    } else {
      t0 = __fma_rn (1.0, t0, -CUDART_PIO2_LO);
      t0 = CUDART_PIO2_HI - t0;
    }
  } 
  return t0;
}

__device_func__(double __cuda_acosh(double a))
{
  double t;
#if !defined(__CUDABE__) 
  if (__cuda___isnan(a)) {
    return a + a;
  }
#endif
  t = a - 1.0;
  if (__cuda_fabs(t) > CUDART_TWO_TO_52) {
    /* for large a, acosh = log(2*a) */
    return CUDART_LN2 + __cuda_log(a);
  } else {
    t = t + __cuda_sqrt(__fma_rn(a, t, t));
    return __cuda_log1p(t);
  }  
}

__device_func__(double __cuda_asinh(double a))
{
#if SLIGHTLY_MORE_ACCURATE_BUT_SLOWER
  double fa, oofa, t;
  fa = __cuda_fabs(a);
  if (fa > 8.9884657373828596e+307) { /* prevent intermediate underflow */
    t = CUDART_LN2 + __cuda_log(fa);
  } else {
    oofa = 1.0 / fa;
    t = fa + fa / (oofa + __cuda_sqrt(__fma_rn(oofa, oofa, 1.0)));
    t = __cuda_log1p(t);
  }
#else
  double fa, t;
  fa = __cuda_fabs(a);
  if (fa > 1.0e153) {
    t = CUDART_LN2 + __cuda_log(fa);
  } else {
    t = fa * fa;
    t = __cuda_log1p (fa + t / (1.0 + __cuda_sqrt(1.0 + t)));
  }
#endif
  return __cuda_copysign(t, a);  
}

__device_func__(double __cuda_atanh(double a))
{
  double fa, t;
#if !defined(__CUDABE__) 
  if (__cuda___isnan(a)) {
    return a + a;
  }
#endif
  fa = __cuda_fabs(a);
  t = (2.0 * fa) / (1.0 - fa);
  t = 0.5 * __cuda_log1p(t);
#if !defined(__CUDABE__) 
  if (__cuda___isnan(t)) {
    return t;
  }
#endif
  if (__cuda___signbit(a)) {
    t = -t;
  }
  return t;
}

__device_func__(double __cuda_hypot(double a, double b))
{
  double v, w, t;
  if (__cuda___isinf(a) || __cuda___isinf(b)) {
    return CUDART_INF;
  }
  if (__cuda___isnan(a) || __cuda___isnan(b)) {
    return a + b;
  }
  a = __cuda_fabs(a);
  b = __cuda_fabs(b);
  v = __cuda_fmax(a, b);
  w = __cuda_fmin(a, b);
  t = w / v;
  t = __fma_rn (t, t, 1.0);
  t = v * __cuda_sqrt(t);
  if (v == 0.0) {
    t = v + w;
  }
  return t;
}

__device_func__(double __cuda_cbrt(double a))
{
  float s;
  double t, r;
  int ilo, ihi, expo, nexpo, denorm;
  if ((a == 0.0) || !(__cuda___finite(a))) {
    return a + a;
  } 
  t = __cuda_fabs(a);
  ilo = __double2loint(t);
  ihi = __double2hiint(t);
  expo = ((int)((unsigned int)ihi >> 20) & 0x7ff);
  denorm = 0;
  if (expo == 0) {
    /* denormal */
    t = t * CUDART_TWO_TO_54;
    denorm = 18;
    ilo = __double2loint(t);
    ihi = __double2hiint(t);
    expo = ((int)((unsigned int)ihi >> 20) & 0x7ff);
  }
  /* scale into float range */
  nexpo = __float2int_rn(CUDART_THIRD_F * (float)(expo - 1022));
  ihi -= (3 * nexpo) << 20;
  r = __hiloint2double(ihi, ilo);
  /* initial approximation */
  s = (float)r;
  t = __cuda_exp2f(CUDART_THIRD_F * __log2f(s));
  /* refine approximation */  
  t = t - (t - (r / (t * t))) * CUDART_THIRD;
  t = t - (t - (r / (t * t))) * CUDART_THIRD;
  /* scale result back into double range */
  ilo = __double2loint(t);
  ihi = __double2hiint(t);
  ihi += (nexpo - denorm) << 20;
  t = __hiloint2double(ihi, ilo);
  if (__cuda___signbit(a)) {
     t = -t;
  }
  return t;
}

__device_func__(double __internal_accurate_pow(double a, double b))
{
  double2 loga;
  double2 prod;
  double t_hi, t_lo;
  double tmp;
#if !defined(__CUDABE__) && defined(__linux__) && !defined(__LP64__)
  volatile double e;
#else
  double e;
#endif

  /* compute log(a) in double-double format*/
  loga = __internal_log_ext_prec(a);

  /* prevent overflow during extended precision multiply */
  if (__cuda_fabs(b) > 1e304) b *= 1.220703125e-4;
  /* compute b * log(a) in double-double format */
  t_hi = loga.y * b;
  t_lo = __fma_rn (loga.y, b, -t_hi);
  t_lo = __fma_rn (loga.x, b, t_lo);
  prod.y = e = t_hi + t_lo;
  prod.x = (t_hi - e) + t_lo;

  /* compute pow(a,b) = exp(b*log(a)) */
  tmp = __cuda_exp(prod.y);
  /* prevent -INF + INF = NaN */
  if (!__cuda___isinf(tmp)) {
    /* if prod.x is much smaller than prod.y, then exp(prod.y + prod.x) ~= 
     * exp(prod.y) + prod.x * exp(prod.y) 
     */
    tmp = __fma_rn (tmp, prod.x, tmp);
  }
  return tmp;
}

__device_func__(double __cuda_pow(double a, double b))
{
  int bIsOddInteger;
  double t;
  if (a == 1.0 || b == 0.0) {
    return 1.0;
  } 
  if (__cuda___isnan(a) || __cuda___isnan(b)) {
    return a + b;
  }
  if (a == CUDART_INF) {
    return __cuda___signbit(b) ?  CUDART_ZERO : CUDART_INF;
  }
  if (__cuda___isinf(b)) {
    if (a == -1.0) {
      return 1.0;
    }
    t = __cuda_fabs(a) > 1.0 ? CUDART_INF : CUDART_ZERO;
    if (b < CUDART_ZERO) {
      t = 1.0 / t;
    }
    return t;
  }
  bIsOddInteger = (b - (2.0 * __cuda_floor(0.5 * b))) == 1.0;
  if (a == CUDART_ZERO) {
    t = bIsOddInteger ? a : CUDART_ZERO;
    if (b < CUDART_ZERO) {
      t = 1.0 / t;
    }
    return t;
  } 
  if (a == -CUDART_INF) {
    t = (b < CUDART_ZERO) ? -1.0/a : -a;
    if (bIsOddInteger) {
      t = __longlong_as_double(__double_as_longlong(t)^0x8000000000000000ULL);
    }
    return t;
  } 
  if ((a < CUDART_ZERO) && (b != __cuda_trunc(b))) {
    return CUDART_NAN;
  } 
  t = __cuda_fabs(a);
  t = __internal_accurate_pow(t, b);
  if ((a < CUDART_ZERO) && bIsOddInteger) {
    t = __longlong_as_double(__double_as_longlong(t) ^ 0x8000000000000000ULL); 
  }
  return t;
}

__device_func__(double __cuda_erf(double a))
{
  double t, r, q;
#if !defined(__CUDABE__) 
  if (__cuda___isnan(a)) {
    return a + a;
  }
#endif
  t = __cuda_fabs(a);
  if (t >= 1.0) {
    r =                 -1.28836351230756500E-019;
    r = __fma_rn (r, t,  1.30597472161093370E-017);
    r = __fma_rn (r, t, -6.33924401259620500E-016);
    r = __fma_rn (r, t,  1.96231865908940140E-014);
    r = __fma_rn (r, t, -4.35272243559990750E-013);
    r = __fma_rn (r, t,  7.37083927929352150E-012);
    r = __fma_rn (r, t, -9.91402142550461630E-011);
    r = __fma_rn (r, t,  1.08817017167760820E-009);
    r = __fma_rn (r, t, -9.93918713097634620E-009);
    r = __fma_rn (r, t,  7.66739923255145500E-008);
    r = __fma_rn (r, t, -5.05440278302806720E-007);
    r = __fma_rn (r, t,  2.87474157099000620E-006);
    r = __fma_rn (r, t, -1.42246725399722510E-005);
    r = __fma_rn (r, t,  6.16994555079419460E-005);
    r = __fma_rn (r, t, -2.36305221938908790E-004);
    r = __fma_rn (r, t,  8.05032844055371070E-004);
    r = __fma_rn (r, t, -2.45833366629108140E-003);
    r = __fma_rn (r, t,  6.78340988296706120E-003);
    r = __fma_rn (r, t, -1.70509103597554640E-002);
    r = __fma_rn (r, t,  3.93322852515666300E-002);
    r = __fma_rn (r, t, -8.37271292613764040E-002);
    r = __fma_rn (r, t,  1.64870423707623280E-001);
    r = __fma_rn (r, t, -2.99729521787681470E-001);
    r = __fma_rn (r, t,  4.99394435612628580E-001);
    r = __fma_rn (r, t, -7.52014596480123030E-001);
    r = __fma_rn (r, t,  9.99933138314926250E-001);
    r = __fma_rn (r, t, -1.12836725321102670E+000);
    r = __fma_rn (r, t,  9.99998988715182450E-001);
    q = __internal_exp_kernel(-t * t, 0);
    r = __fma_rn (r, -q, 1.0);
    if (t >= 6.5) {
      r = 1.0;
    }    
    a = __cuda_copysign (r, a);
  } else {
    q = t * t;
    r =                 -7.77946848895991420E-010;
    r = __fma_rn (r, q,  1.37109803980285950E-008);
    r = __fma_rn (r, q, -1.62063137584932240E-007);
    r = __fma_rn (r, q,  1.64471315712790040E-006);
    r = __fma_rn (r, q, -1.49247123020098620E-005);
    r = __fma_rn (r, q,  1.20552935769006260E-004);
    r = __fma_rn (r, q, -8.54832592931448980E-004);
    r = __fma_rn (r, q,  5.22397760611847340E-003);
    r = __fma_rn (r, q, -2.68661706431114690E-002);
    r = __fma_rn (r, q,  1.12837916709441850E-001);
    r = __fma_rn (r, q, -3.76126389031835210E-001);
    r = __fma_rn (r, q,  1.12837916709551260E+000);
    a = r * a;
  }
  return a;
}

__device_func__(double __cuda_erfc(double a))
{
  double p, q, h, l;
  if (__cuda___isnan(a)) {
    return a + a;
  }
  if (a <= 0.55) {
    return 1.0 - __cuda_erf(a);
  } 
  if (a > 27.3) {
    return 0.0;
  }
  if (a <= 5.0) {
    p =                 5.64189549785304440E-001;
    p = __fma_rn (p, a, 8.17405083437083490E+000);
    p = __fma_rn (p, a, 5.68958722557864720E+001);
    p = __fma_rn (p, a, 2.42568747802647010E+002);
    p = __fma_rn (p, a, 6.80381374390412930E+002);
    p = __fma_rn (p, a, 1.25873132236024590E+003);
    p = __fma_rn (p, a, 1.43925353963809330E+003);
    p = __fma_rn (p, a, 8.15949420587659230E+002);
    q =              a+ 1.44881247113239940E+001;
    q = __fma_rn (q, a, 1.01345387970210510E+002);
    q = __fma_rn (q, a, 4.37184684964009650E+002);
    q = __fma_rn (q, a, 1.25588209225251330E+003);
    q = __fma_rn (q, a, 2.43864015012647630E+003);
    q = __fma_rn (q, a, 3.10570469008816280E+003);
    q = __fma_rn (q, a, 2.35995386578953550E+003);
    q = __fma_rn (q, a, 8.15949420692539320E+002);
  } else {
    p =                 5.64189583545675280E-001;
    p = __fma_rn (p, a, 2.04728556066513970E+000);
    p = __fma_rn (p, a, 6.75128409662943610E+000);
    p = __fma_rn (p, a, 1.10459345071747900E+001);
    p = __fma_rn (p, a, 1.22570382896313600E+001);
    p = __fma_rn (p, a, 6.01884641114116460E+000);
    q =              a+ 3.62871917534986780E+000;
    q = __fma_rn (q, a, 1.24663395327043550E+001);
    q = __fma_rn (q, a, 2.13927672803974790E+001);
    q = __fma_rn (q, a, 2.72082423532866070E+001);
    q = __fma_rn (q, a, 1.86422906830006700E+001);
    q = __fma_rn (q, a, 6.13809834548870550E+000);
  }
  p = p / q;
  h = -a * a;
  l = __fma_rn (-a, a, -h);
  q = __internal_exp_kernel(h, 0);
  q = __fma_rn (q, l, q);
  p = p * q;
  return p;
}

/* approximate 1.0/(a*gamma(a)) on [-0.5,0.5] */
__device_func__(double __internal_tgamma_kernel(double a))
{
  double t;
  t =                 -4.42689340712524750E-010;
  t = __fma_rn (t, a, -2.02665918466589540E-007);
  t = __fma_rn (t, a,  1.13812117211195270E-006);
  t = __fma_rn (t, a, -1.25077348166307480E-006);
  t = __fma_rn (t, a, -2.01365017404087710E-005);
  t = __fma_rn (t, a,  1.28050126073544860E-004);
  t = __fma_rn (t, a, -2.15241408115274180E-004);
  t = __fma_rn (t, a, -1.16516754597046040E-003);
  t = __fma_rn (t, a,  7.21894322484663810E-003);
  t = __fma_rn (t, a, -9.62197153268626320E-003);
  t = __fma_rn (t, a, -4.21977345547223940E-002);
  t = __fma_rn (t, a,  1.66538611382503560E-001);
  t = __fma_rn (t, a, -4.20026350341054440E-002);
  t = __fma_rn (t, a, -6.55878071520257120E-001);
  t = __fma_rn (t, a,  5.77215664901532870E-001);
  t = __fma_rn (t, a,  1.00000000000000000E+000);
  return t;
}

/* Stirling approximation for gamma(a), a > 20 */
__device_func__(double __internal_stirling_poly(double a))
{
  double x = 1.0 / a;
  double z = 0.0;
  z = __fma_rn (z, x,  8.3949872067208726e-004);
  z = __fma_rn (z, x, -5.1717909082605919e-005);
  z = __fma_rn (z, x, -5.9216643735369393e-004);
  z = __fma_rn (z, x,  6.9728137583658571e-005);
  z = __fma_rn (z, x,  7.8403922172006662e-004);
  z = __fma_rn (z, x, -2.2947209362139917e-004);
  z = __fma_rn (z, x, -2.6813271604938273e-003);
  z = __fma_rn (z, x,  3.4722222222222220e-003);
  z = __fma_rn (z, x,  8.3333333333333329e-002);
  z = __fma_rn (z, x,  1.0000000000000000e+000);
  return z;
}

__device_func__(double __internal_tgamma_stirling(double a))
{
  if (a < 1.7162437695630274e+002) {
#if defined(__GNUC__) && !defined(__CUDABE__)
    volatile double t_hi, t_lo, e;
#else     
    double t_hi, t_lo, e;
#endif
    double2 loga, prod;
    double z = __internal_stirling_poly (a);
    double b = a - 0.5;

    /* compute log(a) in double-double format*/
    loga = __internal_log_ext_prec(a);

    /* compute (a - 0.5) * log(a) in double-double format */
    t_hi = loga.y * b;
    t_lo = __fma_rn (loga.y, b, -t_hi);
    t_lo = __fma_rn (loga.x, b, t_lo);
    prod.y = e = t_hi + t_lo;
    prod.x = (t_hi - e) + t_lo;

    /* compute (a - 0.5) * log(a) - a in double-double format */
    loga.y = -a;
    loga.x = 0.0;
    prod = __internal_ddadd_xgty (prod, loga);

    /* compute pow(a,b) = exp(b*log(a)) */
    a = __cuda_exp(prod.y);
    /* prevent -INF + INF = NaN */
    if (!__cuda___isinf(a)) {
      /* if prod.x is much smaller than prod.y, then exp(prod.y + prod.x) ~= 
       * exp(prod.y) + prod.x * exp(prod.y) 
       */
      a = __fma_rn (a, prod.x, a);
    }
    a = __fma_rn (a, CUDART_SQRT_2PI_HI, a * CUDART_SQRT_2PI_LO);
    return a * z;
  } else {
    return CUDART_INF;
  }
}

__device_func__(double __cuda_tgamma(double a))
{
  double s, xx, x = a;
  if (__cuda___isnan(a)) {
    return a + a;
  }
  if (__cuda_fabs(x) < 15.0) {
     /* Based on: Kraemer, W.: "Berechnung der Gammafunktion G(x) fuer reelle 
      * Punkt- und Intervallargumente". Zeitschrift fuer angewandte Mathematik 
      * und Mechanik, Vol. 70 (1990), No. 6, pp. 581-584
      */
    if (x >= 0.0) {
      s = 1.0;
      xx = x;
      while (xx > 1.5) {
        s = __fma_rn(s, xx, -s);
        xx = xx - 1.0;
      }
      if (x >= 0.5) {
        xx = xx - 1.0;
      }
      xx = __internal_tgamma_kernel (xx);
      if (x < 0.5) {
        xx = xx * x;
      }
      s = s / xx;
    } else {
      xx = x;
      s = xx;
      if (x == __cuda_floor(x)) {
        return CUDART_NAN;
      }
      while (xx < -0.5) {
        s = __fma_rn (s, xx, s);
        xx = xx + 1.0;
      }
      xx = __internal_tgamma_kernel (xx);
      s = s * xx;
      s = 1.0 / s;
    }
    return s;
  } else {
    if (x >= 0.0) {
      return __internal_tgamma_stirling (x);
    } else {
      double t;
      int quot;
      if (x == __cuda_floor(x)) {
        return CUDART_NAN;
      }
      if (x < -185.0) {
        int negative;
        x = __cuda_floor(x);
        negative = ((x - (2.0 * __cuda_floor(0.5 * x))) == 1.0);
        return negative ? CUDART_NEG_ZERO : CUDART_ZERO;
      }
      /* compute sin(pi*x) accurately */
      xx = __cuda_rint (__internal_twice(x));
      quot = (int)xx;
      xx = __fma_rn (-0.5, xx, x);
      xx = xx * CUDART_PI;
      if (quot & 1) {
        xx = __internal_cos_kerneld (xx);
      } else {
        xx = __internal_sin_kerneld (xx);
      }
      if (quot & 2) {
        xx = -xx;
      }
      x = __cuda_fabs (x);
      s = __cuda_exp (-x);
      t = x - 0.5;
      if (x > 140.0) t = __internal_half(t);
      t = __cuda_pow (x, t);
      if (x > 140.0) s = s * t;
      s = s * __internal_stirling_poly (x);
      s = s * x;
      s = s * xx;
      s = 1.0 / s;
      s = __fma_rn (s, CUDART_SQRT_PIO2_HI, CUDART_SQRT_PIO2_LO * s);
      s = s / t;
      return s;
    }
  }
}

__device_func__(double __internal_lgamma_pos(double a))
{
  double sum;
  double s, t;

  if (a == CUDART_INF) {
    return a;
  }
  if (a >= 3.0) {
    if (a >= 8.0) {
      /* Stirling approximation; coefficients from Hart et al, "Computer 
       * Approximations", Wiley 1968. Approximation 5404. 
       */
      s = 1.0 / a;
      t = s * s;
      sum =                   -0.1633436431e-2;
      sum = __fma_rn (sum, t,  0.83645878922e-3);
      sum = __fma_rn (sum, t, -0.5951896861197e-3);
      sum = __fma_rn (sum, t,  0.793650576493454e-3);
      sum = __fma_rn (sum, t, -0.277777777735865004e-2);
      sum = __fma_rn (sum, t,  0.833333333333331018375e-1);
      sum = __fma_rn (sum, s,  0.918938533204672);
      s = __internal_half(__cuda_log (a));
      t = a - 0.5;
      s = s * t;
      t = s - a;
      s = s + sum;
      t = t + s;
      return t;
    } else {
      a = a - 3.0;
      s =                 -4.02412642744125560E+003;
      s = __fma_rn (s, a, -2.97693796998962000E+005);
      s = __fma_rn (s, a, -6.38367087682528790E+006);
      s = __fma_rn (s, a, -5.57807214576539320E+007);
      s = __fma_rn (s, a, -2.24585140671479230E+008);
      s = __fma_rn (s, a, -4.70690608529125090E+008);
      s = __fma_rn (s, a, -7.62587065363263010E+008);
      s = __fma_rn (s, a, -9.71405112477113250E+008);
      t =              a  -1.02277248359873170E+003;
      t = __fma_rn (t, a, -1.34815350617954480E+005);
      t = __fma_rn (t, a, -4.64321188814343610E+006);
      t = __fma_rn (t, a, -6.48011106025542540E+007);
      t = __fma_rn (t, a, -4.19763847787431360E+008);
      t = __fma_rn (t, a, -1.25629926018000720E+009);
      t = __fma_rn (t, a, -1.40144133846491690E+009);
      t = s / t;
      t = t + a;
      return t;
    }
  } else if (a >= 1.5) {
    a = a - 2.0;
    t =                  9.84839283076310610E-009;
    t = __fma_rn (t, a, -6.69743850483466500E-008);
    t = __fma_rn (t, a,  2.16565148880011450E-007);
    t = __fma_rn (t, a, -4.86170275781575260E-007);
    t = __fma_rn (t, a,  9.77962097401114400E-007);
    t = __fma_rn (t, a, -2.03041287574791810E-006);
    t = __fma_rn (t, a,  4.36119725805364580E-006);
    t = __fma_rn (t, a, -9.43829310866446590E-006);
    t = __fma_rn (t, a,  2.05106878496644220E-005);
    t = __fma_rn (t, a, -4.49271383742108440E-005);
    t = __fma_rn (t, a,  9.94570466342226000E-005);
    t = __fma_rn (t, a, -2.23154589559238440E-004);
    t = __fma_rn (t, a,  5.09669559149637430E-004);
    t = __fma_rn (t, a, -1.19275392649162300E-003);
    t = __fma_rn (t, a,  2.89051032936815490E-003);
    t = __fma_rn (t, a, -7.38555102806811700E-003);
    t = __fma_rn (t, a,  2.05808084278121250E-002);
    t = __fma_rn (t, a, -6.73523010532073720E-002);
    t = __fma_rn (t, a,  3.22467033424113040E-001);
    t = __fma_rn (t, a,  4.22784335098467190E-001);
    t = t * a;
    return t;
  } else if (a >= 0.7) {
    a = 1.0 - a;
    t =                 1.17786911519331130E-002;  
    t = __fma_rn (t, a, 3.89046747413522300E-002);
    t = __fma_rn (t, a, 5.90045711362049900E-002);
    t = __fma_rn (t, a, 6.02143305254344420E-002);
    t = __fma_rn (t, a, 5.61652708964839180E-002);
    t = __fma_rn (t, a, 5.75052755193461370E-002);
    t = __fma_rn (t, a, 6.21061973447320710E-002);
    t = __fma_rn (t, a, 6.67614724532521880E-002);
    t = __fma_rn (t, a, 7.14856037245421020E-002);
    t = __fma_rn (t, a, 7.69311251313347100E-002);
    t = __fma_rn (t, a, 8.33503129714946310E-002);
    t = __fma_rn (t, a, 9.09538288991182800E-002);
    t = __fma_rn (t, a, 1.00099591546322310E-001);
    t = __fma_rn (t, a, 1.11334278141734510E-001);
    t = __fma_rn (t, a, 1.25509666613462880E-001);
    t = __fma_rn (t, a, 1.44049896457704160E-001);
    t = __fma_rn (t, a, 1.69557177031481600E-001);
    t = __fma_rn (t, a, 2.07385551032182120E-001);
    t = __fma_rn (t, a, 2.70580808427600350E-001);
    t = __fma_rn (t, a, 4.00685634386517050E-001);
    t = __fma_rn (t, a, 8.22467033424113540E-001);
    t = __fma_rn (t, a, 5.77215664901532870E-001);
    t = t * a;
    return t;
  } else {
    t=                  -9.04051686831357990E-008;
    t = __fma_rn (t, a,  7.06814224969349250E-007);
    t = __fma_rn (t, a, -3.80702154637902830E-007);
    t = __fma_rn (t, a, -2.12880892189316100E-005);
    t = __fma_rn (t, a,  1.29108470307156190E-004);
    t = __fma_rn (t, a, -2.15932815215386580E-004);
    t = __fma_rn (t, a, -1.16484324388538480E-003);
    t = __fma_rn (t, a,  7.21883433044470670E-003);
    t = __fma_rn (t, a, -9.62194579514229560E-003);
    t = __fma_rn (t, a, -4.21977386992884450E-002);
    t = __fma_rn (t, a,  1.66538611813682460E-001);
    t = __fma_rn (t, a, -4.20026350606819980E-002);
    t = __fma_rn (t, a, -6.55878071519427450E-001);
    t = __fma_rn (t, a,  5.77215664901523870E-001);
    t = t * a;
    t = __fma_rn (t, a, a);
    return -__cuda_log (t);
  }
}

__device_func__(double __cuda_lgamma(double a))
{
  double t;
  double i;
  long long int quot;
  if (__cuda___isnan(a)) {
    return a + a;
  }
  t = __internal_lgamma_pos(__cuda_fabs(a));
  if (a >= 0.0) return t;
  a = __cuda_fabs(a);
  i = __cuda_floor(a);       
  if (a == i) return CUDART_INF; /* a is an integer: return infinity */
  if (a < 1e-19) return -__cuda_log(a);
  i = __cuda_rint (2.0 * a);
  quot = (long long int)i;
  i = __fma_rn (-0.5, i, a);
  i = i * CUDART_PI;
  if (quot & 1) {
    i = __internal_cos_kerneld(i);
  } else {
    i = __internal_sin_kerneld(i);
  }
  i = __cuda_fabs(i);
  t = __cuda_log(CUDART_PI / (i * a)) - t;
  return t;
}

__device_func__(double __cuda_ldexp(double a, int b))
{
  double fa = __cuda_fabs (a);
  if ((fa == CUDART_ZERO) || (fa == CUDART_INF) || (!(fa <= CUDART_INF))) {
    return a + a;
  }
  if (b == 0) {
    return a;
  }
  if (b >  2200) b =  2200;
  if (b < -2200) b = -2200;
  if (__cuda_abs (b) < 1022) {
    return a * __internal_exp2i_kernel(b);
  }
  if (__cuda_abs (b) < 2044) {
    int bhalf = b / 2;
    return a * __internal_exp2i_kernel (bhalf) * 
           __internal_exp2i_kernel (b - bhalf);
  } else {
    int bquarter = b / 4;
    double t = __internal_exp2i_kernel(bquarter);
    return a * t * t * t *__internal_exp2i_kernel (b - 3 * bquarter);
  }
}

__device_func__(double __cuda_scalbn(double a, int b))
{
  /* On binary systems, ldexp(x,exp) is equivalent to scalbn(x,exp) */
  return __cuda_ldexp(a, b);
}

__device_func__(double __cuda_scalbln(double a, long int b))
{
#if defined(__LP64__)
  /* clamp to integer range prior to conversion */
  if (b < -2147483648L) b = -2147483648L;
  if (b >  2147483647L) b =  2147483647L;
#endif
  return __cuda_scalbn(a, (int)b);
}

__device_func__(double __cuda_frexp(double a, int *b))
{
  double fa = __cuda_fabs(a);
  unsigned int expo;
  unsigned int denorm;

  if (fa < CUDART_TWO_TO_M1022) {
    a *= CUDART_TWO_TO_54;
    denorm = 54;
  } else {
    denorm = 0;
  }
  expo = (__double2hiint(a) >> 20) & 0x7ff;
  if ((fa == 0.0) || (expo == 0x7ff)) {
    expo = 0;
    a = a + a;
  } else {
    expo = expo - denorm - 1022;
    a = __longlong_as_double((__double_as_longlong(a) & 0x800fffffffffffffULL)|
                              0x3fe0000000000000ULL);
  }
  *b = expo;
  return a;  
}

__device_func__(double __cuda_modf(double a, double *b))
{
  double t;
  if (__cuda___finite(a)) {
    t = __cuda_trunc(a);
    *b = t;
    t = a - t;
    return __cuda_copysign(t, a);
  } else if (__cuda___isinf(a)) {
    t = 0.0;
    *b = a;
    return __cuda_copysign(t, a);
  } else {
    *b = a + a; 
    return a + a;
  }  
}

__device_func__(double __cuda_fmod(double a, double b))
{
  double orig_a = a;
  double orig_b = b;
  a = __cuda_fabs(a);
  b = __cuda_fabs(b);
  if (!((a <= CUDART_INF) && (b <= CUDART_INF))) {
      return orig_a + orig_b;
  }
  if (a == CUDART_INF || b == 0.0) {
    return CUDART_NAN;
  } else if (a >= b) {
    int bhi = __double2hiint(b);
    int blo = __double2loint(b);
    int ahi = __double2hiint(a);
    double scaled_b = 0.0;
    if (b < CUDART_TWO_TO_M1022) {
      double t = b;
      while ((t < a) && (t < CUDART_TWO_TO_M1022)) {
        t = t + t;
      }
      bhi = __double2hiint(t);
      blo = __double2loint(t);
      scaled_b = t;
    }
    if (a >= CUDART_TWO_TO_M1022) {
      scaled_b = __hiloint2double ((bhi & 0x000fffff)|(ahi & 0x7ff00000), blo);
    }
    if (scaled_b > a) {
      scaled_b *= 0.5;
    }
    while (scaled_b >= b) {
      if (a >= scaled_b) {
        a -= scaled_b;
      }
      scaled_b *= 0.5;
    }
    return __cuda_copysign (a, orig_a);
  } else {
    return orig_a;
  }
}

__device_func__(double __cuda_remainder(double a, double b))
{
  double orig_a;
  double twoa = 0.0;
  unsigned int quot0 = 0;  /* quotient bit 0 */
  int bhi;
  int blo;
  int ahi;
  if (__cuda___isnan(a) || __cuda___isnan(b)) {
    return a + b;
  }
  orig_a = a;
  a = __cuda_fabs(a);
  b = __cuda_fabs(b);
  if (a == CUDART_INF || b == 0.0) {
    return CUDART_NAN;
  } else if (a >= b) {
    double scaled_b = 0.0;
    bhi = __double2hiint(b);
    blo = __double2loint(b);
    ahi = __double2hiint(a);
    if (b < CUDART_TWO_TO_M1022) {
      double t = b;
      while ((t < a) && (t < CUDART_TWO_TO_M1022)) {
        t = t + t;
      }
      bhi = __double2hiint(t);
      blo = __double2loint(t);
      scaled_b = t;
    }
    if (a >= CUDART_TWO_TO_M1022) {
      scaled_b = __hiloint2double ((bhi & 0x000fffff)|(ahi & 0x7ff00000), blo);
    }
    if (scaled_b > a) {
      scaled_b *= 0.5;
    }
    while (scaled_b >= b) {
      quot0 = 0;
      if (a >= scaled_b) {
        a -= scaled_b;
        quot0 = 1;
      }
      scaled_b *= 0.5;
    }
  }
  /* round quotient to nearest even */
  twoa = a + a;
  if ((twoa > b) || ((twoa == b) && quot0)) {
    a -= b;
  }
  bhi = __double2hiint(a);
  blo = __double2loint(a);
  ahi = __double2hiint(orig_a);
  a = __hiloint2double((ahi & 0x80000000) ^ bhi, blo);
  return a;
}

__device_func__(double __cuda_remquo(double a, double b, int *c))
{
  double orig_a;
  double twoa = 0.0;
  unsigned int quot = 0;  /* trailing quotient bits  */
  unsigned int sign;
  int bhi;
  int blo;
  int ahi;
  if (__cuda___isnan(a) || __cuda___isnan(b)) {
    *c = quot;
    return a + b;
  }
  orig_a = a;
  sign = 0 - (__cuda___signbit(a) != __cuda___signbit(b));
  a = __cuda_fabs(a);
  b = __cuda_fabs(b);
  if (a == CUDART_INF || b == 0.0) {
    *c = quot;
    return CUDART_NAN;
  } else if (a >= b) {
    double scaled_b = 0.0;
    bhi = __double2hiint(b);
    blo = __double2loint(b);
    ahi = __double2hiint(a);
    if (b < CUDART_TWO_TO_M1022) {
      double t = b;
      while ((t < a) && (t < CUDART_TWO_TO_M1022)) {
        t = t + t;
      }
      bhi = __double2hiint(t);
      blo = __double2loint(t);
      scaled_b = t;
    }
    if (a >= CUDART_TWO_TO_M1022) {
      scaled_b = __hiloint2double ((bhi & 0x000fffff)|(ahi & 0x7ff00000), blo);
    }
    if (scaled_b > a) {
      scaled_b *= 0.5;
    }
    while (scaled_b >= b) {
      quot <<= 1;
      if (a >= scaled_b) {
        a -= scaled_b;
        quot += 1;
      }
      scaled_b *= 0.5;
    }
  }
  /* round quotient to nearest even */
  twoa = a + a;
  if ((twoa > b) || ((twoa == b) && (quot & 1))) {
    quot++;
    a -= b;
  }
  bhi = __double2hiint(a);
  blo = __double2loint(a);
  ahi = __double2hiint(orig_a);
  a = __hiloint2double((ahi & 0x80000000) ^ bhi, blo);
  quot = quot & CUDART_REMQUO_MASK_F;
  quot = quot ^ sign;
  quot = quot - sign;
  *c = quot;
  return a;
}

__device_func__(double __cuda_nextafter(double a, double b))
{
  unsigned long long int ia;
  unsigned long long int ib;
  ia = __double_as_longlong(a);
  ib = __double_as_longlong(b);
  if (__cuda___isnan(a) || __cuda___isnan(b)) return a + b; /* NaN */
  if (((ia | ib) << 1) == 0ULL) return b;
  if ((ia + ia) == 0ULL) {
    return __cuda_copysign (CUDART_MIN_DENORM, b); /* crossover */
  }
  if ((a < b) && (a < 0.0)) ia--;
  if ((a < b) && (a > 0.0)) ia++;
  if ((a > b) && (a < 0.0)) ia++;
  if ((a > b) && (a > 0.0)) ia--;
  a = __longlong_as_double(ia);
  return a;
}

__device_func__(double __cuda_nan(const char *s))
{
  unsigned long long i = 0;
  int c;
  int ovfl = 0;
  int invld = 0;
  if (*s == '0') {
    s++;
    if ((*s == 'x') || (*s == 'X')) {
      s++; 
      while (*s == '0') s++;
      while (*s) {
        if (i > 0x0fffffffffffffffULL) {
          ovfl = 1;
        }
        c = (((*s) >= 'A') && ((*s) <= 'F')) ? (*s + 'a' - 'A') : (*s);
        if ((c >= 'a') && (c <= 'f')) { 
          c = c - 'a' + 10;
          i = i * 16 + c;
        } else if ((c >= '0') && (c <= '9')) { 
          c = c - '0';
          i = i * 16 + c;
        } else {
          invld = 1;
        }
        s++;
      }
    } else {
      while (*s == '0') s++;
      while (*s) {
        if (i > 0x1fffffffffffffffULL) {
          ovfl = 1;
        }
        c = *s;
        if ((c >= '0') && (c <= '7')) { 
          c = c - '0';
          i = i * 8 + c;
        } else {
          invld = 1; 
        }
        s++;
      }
    }
  } else {
    while (*s) {
      c = *s;
      if ((i > 1844674407370955161ULL) || 
          ((i == 1844674407370955161ULL) && (c > '5'))) {
        ovfl = 1;
      }
      if ((c >= '0') && (c <= '9')) { 
        c = c - '0';
        i = i * 10 + c;
      } else {
        invld = 1;
      }
      s++;
    }
  }
  if (ovfl) {
    i = ~0ULL;
  }
  if (invld) {
    i = 0ULL;
  }
  i = (i & 0x000fffffffffffffULL) | 0x7ff8000000000000ULL;
  return __longlong_as_double(i);
}

__device_func__(double __cuda_round(double a))
{
  double fa = __cuda_fabs(a);
  if (fa > CUDART_TWO_TO_52) {
    return a;
  } else {
    double u = __cuda_floor(fa + 0.5);
    if (fa < 0.5) u = 0;
    return __cuda_copysign(u, a);
  }
}

__device_func__(long long int __cuda_llround(double a))
{
#if !defined(__CUDABE__)
  if (a >=  9223372036854775807.0) return 0x7fffffffffffffffLL;
  if (a <= -9223372036854775808.0) return 0x8000000000000000LL;
#endif /* !__CUDABE__ */
  return (long long int)(__cuda_round(a));
}

__device_func__(long int __cuda_lround(double a))
{
#if defined(__LP64__)
  return (long int)(__cuda_llround(a));
#else /* __LP64__ */
#if !defined(__CUDABE__)
  if (__cuda___isnan(a))  return 0x80000000L;
  if (a >=  2147483647.0) return 0x7fffffffL;
  if (a <= -2147483648.0) return 0x80000000L;
#endif /* !__CUDABE__ */
  return (long int)(__cuda_round(a));
#endif /* __LP64__ */
}

__device_func__(double __cuda_fdim(double a, double b))
{
  double t;
  t = a - b;    /* default also takes care of NaNs */
  if (a <= b) {
    t = 0.0;
  }
  return t;
}

__device_func__(int __cuda_ilogb(double a))
{
  unsigned long long int i;
  unsigned int ihi;
  unsigned int ilo;
  if (__cuda___isnan(a)) return -INT_MAX-1;
  if (__cuda___isinf(a)) return INT_MAX;
  if (a == 0.0) return -INT_MAX-1;
  a = __cuda_fabs(a);
  ilo = __double2loint(a);
  ihi = __double2hiint(a);
  i = ((unsigned long long int)ihi) << 32 | (unsigned long long int)ilo;
  if (a >= CUDART_TWO_TO_M1022) {
    return ((int)((ihi >> 20) & 0x7ff)) - 1023;
  } else {
    int expo = -1022;
    while (i < 0x0010000000000000ULL) {
      expo--;
      i = i + i;
    }
    return expo;
  }
}

__device_func__(double __cuda_logb(double a))
{
  unsigned long long int i;
  unsigned int ihi;
  unsigned int ilo;
  if (__cuda___isnan(a)) return a + a;
  a = __cuda_fabs(a);
  if (a == CUDART_INF) return a;
  if (a == 0.0) return -CUDART_INF;
  ilo = __double2loint(a);
  ihi = __double2hiint(a);
  i = ((unsigned long long int)ihi) << 32 | (unsigned long long int)ilo;
  if (a >= CUDART_TWO_TO_M1022) {
    return (double)((int)((ihi >> 20) & 0x7ff)) - 1023;
  } else {
    int expo = -1022;
    while (i < 0x0010000000000000ULL) {
      expo--;
      i = i + i;
    }
    return (double)expo;
  }
}

__device_func__(double __cuda_fma(double a, double b, double c))
{
  return __fma_rn(a, b, c);
}

#if __APPLE__
__device_func__(int __cuda___isfinited(double a))
{
  return __cuda___finite(a);
}

__device_func__(int __cuda___signbitd(double a))
{
  return __cuda___signbit(a);
}
#endif

#endif /* __cplusplus && __CUDACC__ */

#endif /* __MATH_FUNCTIONS_DBL_PTX3_H__ */
