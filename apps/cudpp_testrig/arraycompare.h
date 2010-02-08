// -------------------------------------------------------------
// cuDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
//  $Revision: 3632 $
//  $Date: 2007-08-26 06:15:39 +0100 (Sun, 26 Aug 2007) $
// ------------------------------------------------------------- 
// This source code is distributed under the terms of license.txt in
// the root directory of this source distribution.
// ------------------------------------------------------------- 

/**
 * @file
 * arraycompare.h
 * 
 * @brief Templatization of array comparisons for cutil
 * 
 * To completely templatize tests (in cudpp_testrig) with cutil, we
 * need to use template specialization to wrap up CUTIL's array
 * comparison and file writing functions for different types.
 */

#ifndef _ARRAY_COMPARE_
#define _ARRAY_COMPARE_

// Here's the generic wrapper for cutCompare*
template<class T>
class ArrayComparator
{
public:
    CUTBoolean compare( const T* reference, T* data, unsigned int len)
    {
        fprintf(stderr, "Error: no comparison function implemented for this type\n");
        return CUTFalse;
    }
};

// Here's the specialization for ints:
template<>
class ArrayComparator<int>
{
public:
    CUTBoolean compare( const int* reference, int* data, unsigned int len)
    {
        return cutComparei(reference, data, len);
    }
};

// Here's the specialization for floats:
template<>
class ArrayComparator<float>
{
public:
    CUTBoolean compare( const float* reference, float* data, unsigned int len)
    {
        return cutComparef(reference, data, len);
    }
};

// Here's the generic wrapper for cutWriteFile*
template<class T>
class ArrayFileWriter
{
public:
    CUTBoolean write(const char* filename, T* data, unsigned int len, float epsilon)
    {
        fprintf(stderr, "Error: no file write function implemented for this type\n");
        return CUTFalse;
    }
};

// Here's the specialization for ints:
template<>
class ArrayFileWriter<int>
{
public:
    CUTBoolean write(const char* filename, int* data, unsigned int len, float epsilon)
    {
        return cutWriteFilei(filename, data, len);
    }
};

// Here's the specialization for floats:
template<>
class ArrayFileWriter<float>
{
public:
    CUTBoolean write(const char* filename, float* data, unsigned int len, float epsilon)
    {
        return cutWriteFilef(filename, data, len, epsilon);
    }
};

#endif //_ARRAY_COMPARE_

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
