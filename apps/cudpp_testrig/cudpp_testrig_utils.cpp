#include "cudpp_testrig_utils.h"


template <>
int OperatorMax<int>::identity() const { return INT_MIN; }
template <>
unsigned int OperatorMax<unsigned int>::identity() const { return 0; }
template <>
float OperatorMax<float>::identity() const { return -FLT_MAX; }
template <>
double OperatorMax<double>::identity() const { return -DBL_MAX; }

template <>
int OperatorMin<int>::identity() const { return INT_MAX; }
template <>
unsigned int OperatorMin<unsigned int>::identity() const { return UINT_MAX; }
template <>
float OperatorMin<float>::identity() const { return FLT_MAX; }
template <>
double OperatorMin<double>::identity() const { return DBL_MAX; }

template<>
void VectorSupport<unsigned int>::fillVector(unsigned int *a, size_t numElements, unsigned int keybits, unsigned int range)
{
    // Fill up with some random data
    int keyshiftmask = 0;
    if (keybits > 16) keyshiftmask = (1 << (keybits - 16)) - 1;
    int keymask = 0xffff;
    if (keybits < 16) keymask = (1 << keybits) - 1;            

    srand(95123);
    for(unsigned int i=0; i < numElements; ++i)   
    { 
        a[i] = ((rand() & keyshiftmask)<<16) | (rand() & keymask); 
    }
}

template<>
void VectorSupport<float>::fillVector(float *a, size_t numElements, unsigned int keybits, float range)
{
    srand(95123);
    for(unsigned int j = 0; j < numElements; j++)
    {
        //a[j] = pow(-1,(float)j)*(float)((rand()<<16) | rand());          
        a[j] = pow(-1, (float)j) * (range * (rand() / (float)RAND_MAX));
    }
}

template<>
void VectorSupport<double>::fillVector(double *a, size_t numElements, unsigned int keybits, double range)
{
    srand(95123);
    for(unsigned int j = 0; j < numElements; j++)
    {
        a[j] = pow(-1, (double)j) * (range * (rand() / (double)RAND_MAX));
    }
}

// assumes the values were initially indices into the array, for simplicity of 
// checking correct order of values
template<>
int VectorSupport<unsigned int>::verifySort(unsigned int *keysSorted, unsigned int *valuesSorted, 
                                            unsigned int *keysUnsorted, size_t len)
{
    int retval = 0;

    for(unsigned int i=0; i<len-1; ++i)
    {	   
        if( (keysSorted[i])>(keysSorted[i+1]) )
        {
            printf("Unordered key[%u]:%u > key[%u]:%u\n", i, keysSorted[i], i+1, keysSorted[i+1]);
            retval = 1;
            break;
        }		
    }

    if (valuesSorted)
    {
        for(unsigned int i=0; i<len; ++i)
        {
            if( keysUnsorted[(valuesSorted[i])] != keysSorted[i] )
            {
                printf("Incorrectly sorted value[%u] (%u) %u != %u\n", 
                    i, valuesSorted[i], keysUnsorted[valuesSorted[i]], keysSorted[i]);
                retval = 1;
                break;
            }
        }
    }

    return retval;
}

template<>
int VectorSupport<float>::verifySort(float *keysSorted, unsigned int *valuesSorted, 
                                     float *keysUnsorted, size_t len)
{
    int retval = 0;

    for(unsigned int i=0; i<len-1; ++i)
    {	   
        if( (keysSorted[i])>(keysSorted[i+1]) )
        {
            printf("Unordered key[%u]:%f > key[%u]:%f\n", i, keysSorted[i], i+1, keysSorted[i+1]);
            retval = 1;
            break;
        }		
    }

    if (valuesSorted)
    {
        for(unsigned int i=0; i<len; ++i)
        {
            if( keysUnsorted[(valuesSorted[i])] != keysSorted[i] )
            {
                printf("Incorrectly sorted value[%u] (%u) %f != %f\n", 
                    i, valuesSorted[i], keysUnsorted[valuesSorted[i]], keysSorted[i]);
                retval = 1;
                break;
            }
        }
    }

    return retval;
}