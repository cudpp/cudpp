// -------------------------------------------------------------
// cuDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision: $
// $Date: $
// ------------------------------------------------------------- 
// This source code is distributed under the terms of license.txt in
// the root directory of this source distribution.
// ------------------------------------------------------------- 

/**
 * @file
 * cudpp_testrig_utils.h
 *
 */

#ifndef __CUDPP_TESTRIG_UTILS_H__
#define __CUDPP_TESTRIG_UTILS_H__

#include <math.h>
#include <cstdio>
#include <climits>
#include <float.h>
#include <algorithm>


// template specializations defined below after class definitions

template <typename T>
class VectorSupport
{
public:
    static void fillVector(T *a, size_t numElements, unsigned int keybits, T range);
    static int  verifySort(T *keysSorted, unsigned int *valuesSorted, T *keysUnsorted, size_t len);
};

template <typename T>
class OperatorAdd
{
public:
    T operator()(const T& a, const T& b) { return a + b; }
    T identity() { return (T)0; }
};

template <typename T>
class OperatorMultiply
{
public:
    T operator()(const T& a, const T& b) { return a * b; }
    T identity() { return (T)1; }
};

template <typename T>
class OperatorMax
{
public:
    T operator() (const T& a, const T& b) const { return std::max(a, b); }
    T identity() const { return (T)0; }
};

template <typename T>
class OperatorMin
{
public:
    T operator() (const T& a, const T& b) const { return std::min(a, b); }
    T identity() const { return (T)0; }
};

// specializations
template <> int OperatorMax<int>::identity() const;
template <> unsigned int OperatorMax<unsigned int>::identity() const;
template <> float OperatorMax<float>::identity() const;
template <> double OperatorMax<double>::identity() const;

template <> int OperatorMin<int>::identity() const;
template <> unsigned int OperatorMin<unsigned int>::identity() const;
template <> float OperatorMin<float>::identity() const;
template <> double OperatorMin<double>::identity() const;


template<>
void VectorSupport<unsigned int>::fillVector(unsigned int *a, 
                                             size_t numElements, 
                                             unsigned int keybits, 
                                             unsigned int range);

template<>
void VectorSupport<int>::fillVector(int *a, 
                                    size_t numElements, 
                                    unsigned int keybits, 
                                    int range);

template<> void VectorSupport<float>::fillVector(float *a, 
                                                 size_t numElements, 
                                                 unsigned int keybits, 
                                                 float range);

template<> void VectorSupport<double>::fillVector(double *a, 
                                                  size_t numElements, 
                                                  unsigned int keybits, 
                                                  double range);

// assumes the values were initially indices into the array, for simplicity of 
// checking correct order of values
template<> int VectorSupport<unsigned int>::verifySort(unsigned int *keysSorted, 
                                                       unsigned int *valuesSorted, 
                                                       unsigned int *keysUnsorted, 
                                                       size_t len);

template<> int VectorSupport<float>::verifySort(float *keysSorted, 
                                                unsigned int *valuesSorted, 
                                                float *keysUnsorted, 
                                                size_t len);

#endif // __CUDPP_TESTRIG_UTILS_H__
