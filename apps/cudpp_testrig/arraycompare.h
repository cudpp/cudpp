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

#define MIN_EPSILON_ERROR 1e-3f // from cutil

// Here's the generic wrapper for cutCompare*
template<class T>
class ArrayComparator
{
public:
    CUTBoolean compare( const T* reference, T* data, unsigned int len)
    {
        // get rid of compiler warnings
        reference = reference;
        data = data;
        len = len;
        fprintf(stderr,
                "Error: no comparison function implemented for this type\n");
        return CUTFalse;
    }
    CUTBoolean compare_e( const T* reference, T* data, unsigned int len,
                          float epsilon)
    {
        // get rid of compiler warnings
        reference = reference;
        data = data;
        len = len;
        epsilon = epsilon; 
        fprintf(stderr,
                "Error: no comparison function implemented for this type\n");
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
    CUTBoolean compare_e( const int* reference, int* data, unsigned int len,
                          float epsilon)
    {
        epsilon = epsilon;
        return cutComparei(reference, data, len);
    }
};

// Here's the specialization for uints:
template<>
class ArrayComparator<unsigned int>
{
public:
    CUTBoolean compare( const unsigned int* reference, unsigned int* data, 
                        unsigned int len)
    {
        return cutComparei((int *) reference, (int *) data, len);
    }
    CUTBoolean compare_e( const unsigned int* reference, unsigned int* data, 
                          unsigned int len, float epsilon)
    {
        epsilon = epsilon;
        return cutComparei((int *) reference, (int *) data, len);
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
    CUTBoolean compare_e( const float* reference, float* data, unsigned int len,
                          float epsilon)
    {
        return cutComparefe(reference, data, len, epsilon);
    }
};

// Here's the specialization for doubles:
template<>
class ArrayComparator<double>
{
public:
    CUTBoolean compare( const double* reference, double* data, unsigned int len)
    {
        // get rid of compiler warnings
        reference = reference;
        data = data;
        len = len;
        fprintf(stderr,
                "Error: no comparison function implemented for this type\n");
        return CUTFalse;
    }
    CUTBoolean compare_e( const double* reference, double* data, 
                          unsigned int len, float epsilon)
    {
        // If we set epsilon to be 0, let's set a minimum threshold
        double max_error = std::max( (double)epsilon, 
                                     (double)MIN_EPSILON_ERROR );
        int error_count = 0;
        bool result = true;

        for( unsigned int i = 0; i < len; ++i) {
            double diff = fabs(reference[i] - data[i]);
            bool comp = (diff < max_error);
            result &= comp;
            if( ! comp) 
            {
                error_count++;
                // printf("ERROR(epsilon=%4.3f), i=%d, (ref)0x%02x / (data)0x%02x / (diff)%d\n", max_error, i, reference[i], data[i], (unsigned int)diff);
            }
        }

        if (error_count) {
            printf("total # of errors = %d\n", error_count);
        }
        return (error_count == 0) ? CUTTrue : CUTFalse;

    }
};

// Here's the specialization for longlong:
template<>
class ArrayComparator<long long>
{
public:
    CUTBoolean compare( const long long* reference, long long* data, 
                        unsigned int len)
    {
        int error_count = 0;
        for( unsigned int i = 0; i < len; ++i) {
            bool comp = (reference[i] == data[i]);
            if( ! comp) 
            {
                error_count++;
                // printf("ERROR(epsilon=%4.3f), i=%d, (ref)0x%02x / (data)0x%02x / (diff)%d\n", max_error, i, reference[i], data[i], (unsigned int)diff);
            }
        }
        if (error_count) {
            printf("total # of errors = %d\n", error_count);
        }
        return (error_count == 0) ? CUTTrue : CUTFalse;
    }
    CUTBoolean compare_e( const long long* reference, long long* data, 
                          unsigned int len, float epsilon)
    {
        epsilon = epsilon;
        return ArrayComparator<long long>::compare(reference, data, len);

    }
};

// Here's the specialization for ulonglong:
template<>
class ArrayComparator<unsigned long long>
{
public:
    CUTBoolean compare( const unsigned long long* reference, 
                        unsigned long long* data, 
                        unsigned int len)
    {
        ArrayComparator<long long> ll;
        return ll.compare((long long *) reference,
                          (long long *) data,
                          len);
    }
    CUTBoolean compare_e( const unsigned long long* reference, 
                          unsigned long long* data, 
                          unsigned int len, float epsilon)
    {
        ArrayComparator<long long> ll;
        return ll.compare_e((long long *) reference,
                            (long long *) data,
                            len, epsilon);
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
        epsilon = epsilon;
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
