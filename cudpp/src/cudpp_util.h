// -------------------------------------------------------------
// cuDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision: 5636 $
// $Date: 2009-07-02 13:39:38 +1000 (Thu, 02 Jul 2009) $
// ------------------------------------------------------------- 
// This source code is distributed under the terms of license.txt in
// the root directory of this source distribution.
// ------------------------------------------------------------- 

/**
 * @file
 * cudpp_util.h
 *
 * @brief C++ utility functions and classes used internally to cuDPP
 */

#ifndef __CUDPP_UTIL_H__
#define __CUDPP_UTIL_H__

#ifdef WIN32
#include <windows.h>
#endif

#include <cuda.h>
#include <cudpp.h>
#include <limits.h>
#include <float.h>

#if (CUDA_VERSION >= 3000)
#define LAUNCH_BOUNDS(x) __launch_bounds__((x))
#define LAUNCH_BOUNDS_MINBLOCKs(x, y) __launch_bounds__((x),(y))
#else
#define LAUNCH_BOUNDS(x)
#define LAUNCH_BOUNDS_MINBLOCKS(x, y)
#endif


/** @brief Determine if \a n is a power of two.
  * @param n Value to be checked to see if it is a power of two
  * @returns True if \a n is a power of two, false otherwise
  */
inline bool 
isPowerOfTwo(int n)
{
    return ((n&(n-1))==0) ;
}

/** @brief Determine if an integer \a n is a multiple of an integer \a f.
  * @param n Multiple
  * @param f Factor
  * @returns True if \a n is a multiple of \a f, false otherwise
  */
inline bool
isMultiple(int n, int f)
{
    if (isPowerOfTwo(f))
        return ((n&(f-1))==0);
    else
        return (n%f==0);
}

/** @brief Compute the smallest power of two larger than \a x.
  * @param x Input value
  * @returns The smallest power f two larger than \a x
  */
inline unsigned int 
ceilPow2( unsigned int x ) 
{
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

/** @brief Compute the largest power of two smaller than or equal to \a x.
  * @param x Input value
  * @returns The largest power of two smaller than or equal to \a x.
  */
inline unsigned int 
floorPow2(unsigned int x)
{
    return ceilPow2(x) >> 1;
}

/** @brief Returns the maximum value for type \a T.
  * 
  * Implemented using template specialization on \a T.
  */
template <class T> 
__host__ __device__ inline T getMax() { return 0; }
/** @brief Returns the minimum value for type \a T.
* 
* Implemented using template specialization on \a T.
*/
template <class T> 
__host__ __device__ inline T getMin() { return 0; }
// type specializations for the above
// getMax
template <> __host__ __device__ inline int getMax() { return INT_MAX; }
template <> __host__ __device__ inline unsigned int getMax() { return INT_MAX; }
template <> __host__ __device__ inline float getMax() { return FLT_MAX; }
template <> __host__ __device__ inline char getMax() { return (char)INT_MAX; }
template <> __host__ __device__ inline unsigned char getMax() { return (unsigned char)INT_MAX; }
// getMin
template <> __host__ __device__ inline int getMin() { return INT_MIN; }
template <> __host__ __device__ inline unsigned int getMin() { return 0; }
template <> __host__ __device__ inline float getMin() { return -FLT_MAX; }
template <> __host__ __device__ inline char getMin() { return (char)INT_MIN; }
template <> __host__ __device__ inline unsigned char getMin() { return (unsigned char)0; }

/** @brief Returns the maximum of three values. 
  * @param a First value. 
  * @param b Second value. 
  * @param c Third value. 
  * @returns The maximum of \a a, \a b and \a c.
  */
template<class T>
inline int max3(T a, T b, T c)
{       
    return (a > b) ? ((a > c)? a : c) : ((b > c) ? b : c);
}

/** @brief Utility template struct for generating small vector types from scalar types
  *
  * Given a base scalar type (\c int, \c float, etc.) and a vector length (1 through 4) as 
  * template parameters, this struct defines a vector type (\c float3, \c int4, etc.) of the 
  * specified length and base type.  For example:
  * \code
  * template <class T>
  * __device__ void myKernel(T *data)
  * {
  *     typeToVector<T,4>::Result myVec4;             // create a vec4 of type T
  *     myVec4 = (typeToVector<T,4>::Result*)data[0]; // load first element of data as a vec4
  * }
  * \endcode
  *
  * This functionality is implemented using template specialization.  Currently specializations
  * for int, float, and unsigned int vectors of lengths 2-4 are defined.  Note that this results 
  * in types being generated at compile time -- there is no runtime cost.  typeToVector is used by 
  * the optimized scan \c __device__ functions in scan_cta.cu.
  */
template <typename T, int N>
struct typeToVector
{
    typedef T Result;
};

template<>
struct typeToVector<int, 4>
{
    typedef int4 Result;
};
template<>
struct typeToVector<unsigned int, 4>
{
    typedef uint4 Result;
};
template<>
struct typeToVector<float, 4>
{
    typedef float4 Result;
};
template<>
struct typeToVector<int, 3>
{
    typedef int3 Result;
};
template<>
struct typeToVector<unsigned int, 3>
{
    typedef uint3 Result;
};
template<>
struct typeToVector<float, 3>
{
    typedef float3 Result;
};
template<>
struct typeToVector<int, 2>
{
    typedef int2 Result;
};
template<>
struct typeToVector<unsigned int, 2>
{
    typedef uint2 Result;
};
template<>
struct typeToVector<float, 2>
{
    typedef float2 Result;
};

template <typename T>
class OperatorAdd
{
public:
    __device__ T operator()(const T a, const T b) { return a + b; }
    __device__ T identity() { return (T)0; }
};

template <typename T>
class OperatorMultiply
{
public:
    __device__ T operator()(const T a, const T b) { return a * b; }
    __device__ T identity() { return (T)1; }
};

template <typename T>
class OperatorMax
{
public:
    __device__ T operator() (const T a, const T b) const { return max(a, b); }
    __device__ T identity() const { return (T)0; }
};

template <>
__device__ int OperatorMax<int>::identity() const { return INT_MIN; }
template <>
__device__ unsigned int OperatorMax<unsigned int>::identity() const { return 0; }
template <>
__device__ float OperatorMax<float>::identity() const { return -FLT_MAX; }
template <>
__device__ double OperatorMax<double>::identity() const { return -DBL_MAX; }

template <typename T>
class OperatorMin
{
public:
    __device__ T operator() (const T a, const T b) const { return min(a, b); }
    __device__ T identity() const { return (T)0; }
};

template <>
__device__ int OperatorMin<int>::identity() const { return INT_MAX; }
template <>
__device__ unsigned int OperatorMin<unsigned int>::identity() const { return UINT_MAX; }
template <>
__device__ float OperatorMin<float>::identity() const { return FLT_MAX; }
template <>
__device__ double OperatorMin<double>::identity() const { return DBL_MAX; }

#endif // __CUDPP_UTIL_H__

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
