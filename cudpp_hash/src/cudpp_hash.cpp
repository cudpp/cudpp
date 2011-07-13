#include <cuda_runtime.h>
#include "cudpp_hash.h"
#include "cudpp_plan.h"

#include "hash_table.h"         // HashTable class
#include "hash_compacting.h"    // CompactingHashTable class
#include "hash_multivalue.h"    // MultivalueHashTable class

typedef CUDPPHashTableInternal<CudaHT::CuckooHashing::HashTable> hti_basic;
typedef CUDPPHashTableInternal<CudaHT::CuckooHashing::CompactingHashTable> hti_compacting;
typedef CUDPPHashTableInternal<CudaHT::CuckooHashing::MultivalueHashTable> hti_multivalue;
typedef CUDPPHashTableInternal<void> hti_void;

// cudppHashTable will create some sort of internal struct that you
// write. It will then cast the pointer to that struct to a
// CUDPPHandle (just like cudppPlan() does), and return that.
CUDPP_HASH_DLL
CUDPPResult cudppHashTable(CUDPPHandle theCudpp_, CUDPPHandle *hash_table, 
                           const CUDPPHashTableConfig *config)
{
    /* first check: is this device >= 2.0? if not, say so and exit. */
    int dev;
    if (cudaGetDevice(&dev) != cudaSuccess)
    {
        fprintf(stderr, "Can't get current device (cudppHashTable)\n");
        return CUDPP_ERROR_UNKNOWN;
    }

    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess)
    {
        fprintf(stderr, "Can't get current device properties "
                "(cudppHashTable)\n");
        return CUDPP_ERROR_UNKNOWN;
    }

    if (prop.major < 2)
    {
        fprintf(stderr, "Hash tables are only supported on devices with "
                "compute capability 2.0 or greater.\n");
        return CUDPP_ERROR_ILLEGAL_CONFIGURATION;
    }

    switch(config->type)
    {
    case CUDPP_BASIC_HASH_TABLE:
    {
        CudaHT::CuckooHashing::HashTable * basic_table = 
            new CudaHT::CuckooHashing::HashTable();
        basic_table->setTheCudpp(theCudpp_);
        basic_table->Initialize(config->kInputSize, config->space_usage);
        hti_basic * hti = new hti_basic(config, basic_table);
        if (!hti)
        {
            return CUDPP_ERROR_UNKNOWN;
        }
        else
        {
            *hash_table = hti->getHandle();
            return CUDPP_SUCCESS;
        }
        break;
    }
    case CUDPP_COMPACTING_HASH_TABLE:
    {
        CudaHT::CuckooHashing::CompactingHashTable * compacting_table = 
            new CudaHT::CuckooHashing::CompactingHashTable();
        compacting_table->setTheCudpp(theCudpp_);
        compacting_table->Initialize(config->kInputSize, config->space_usage);
        hti_compacting * hti = new hti_compacting(config, compacting_table);
        if (!hti)
        {
            return CUDPP_ERROR_UNKNOWN;
        }
        else
        {
            *hash_table = hti->getHandle();
            return CUDPP_SUCCESS;
        }
        break;
    }
    case CUDPP_MULTIVALUE_HASH_TABLE:
    {
        CudaHT::CuckooHashing::MultivalueHashTable * multivalue_table = 
            new CudaHT::CuckooHashing::MultivalueHashTable();
        multivalue_table->setTheCudpp(theCudpp_);
        multivalue_table->Initialize(config->kInputSize, config->space_usage);
        hti_multivalue * hti = new hti_multivalue(config, multivalue_table);
        if (!hti)
        {
            return CUDPP_ERROR_UNKNOWN;
        }
        else
        {
            *hash_table = hti->getHandle();
            return CUDPP_SUCCESS;
        }
        break;
    }
    case CUDPP_INVALID_HASH_TABLE:
        return CUDPP_ERROR_ILLEGAL_CONFIGURATION;
        break;
    }
    return CUDPP_ERROR_ILLEGAL_CONFIGURATION;
}

// Then cudppHashTableInsert/Retrieve, or any other functions that
// operate on it, take the CUDPPHandle as input, and call
// getPlanPtrFromHandle<T>(handle), where T is the type of the
// internal struct you define, to get back the pointer to the struct.
CUDPP_HASH_DLL
CUDPPResult cudppHashInsert(CUDPPHandle theCudpp_, CUDPPHandle plan, 
                            const void* d_keys, const void* d_vals, 
                            unsigned int num)
{
    (void) theCudpp_;           // suppress compiler warning

    // the other way to do this hacky thing is to have inherited classes
    // from CUDPPHashTableInternal maybe?
    hti_void * hti_init = (hti_void *) getPlanPtrFromHandle<hti_void>(plan);
    switch(hti_init->config.type)
    {
    case CUDPP_BASIC_HASH_TABLE:
    {
        hti_basic * hti = (hti_basic *) getPlanPtrFromHandle<hti_basic>(plan);
        bool s = hti->hash_table->Build(num, (const unsigned int *) d_keys, 
                                        (const unsigned int *) d_vals);
        return s ? CUDPP_SUCCESS : CUDPP_ERROR_UNKNOWN;
        break;
    }
    case CUDPP_COMPACTING_HASH_TABLE:
    {
        hti_compacting * hti =
            (hti_compacting *) getPlanPtrFromHandle<hti_compacting>(plan);
        bool s = hti->hash_table->Build(num, (const unsigned int *) d_keys, 
                                        (const unsigned int *) d_vals);
        return s ? CUDPP_SUCCESS : CUDPP_ERROR_UNKNOWN;
        break;
    } 
    case CUDPP_MULTIVALUE_HASH_TABLE:
    {
        hti_multivalue * hti =
            (hti_multivalue *) getPlanPtrFromHandle<hti_multivalue>(plan);
        bool s = hti->hash_table->Build(num, (const unsigned int *) d_keys, 
                                        (const unsigned int *) d_vals);
        return s ? CUDPP_SUCCESS : CUDPP_ERROR_UNKNOWN;
        break;
    } 
    case CUDPP_INVALID_HASH_TABLE:
        return CUDPP_ERROR_ILLEGAL_CONFIGURATION;
        break;
    }
    return CUDPP_ERROR_ILLEGAL_CONFIGURATION;
}

CUDPP_HASH_DLL
CUDPPResult cudppHashRetrieve(CUDPPHandle theCudpp_, CUDPPHandle plan, 
                              const void* d_keys, void* d_vals, size_t num)
{
    (void) theCudpp_;           // suppress compiler warning

    hti_void * hti_init = (hti_void *) getPlanPtrFromHandle<hti_void>(plan);
    switch(hti_init->config.type)
    {
    case CUDPP_BASIC_HASH_TABLE:
    {
        hti_basic * hti = (hti_basic *) getPlanPtrFromHandle<hti_basic>(plan);
        hti->hash_table->Retrieve(num, (const unsigned int *) d_keys, 
                                           (unsigned int *) d_vals);
        return CUDPP_SUCCESS;
        break;
    }
    case CUDPP_COMPACTING_HASH_TABLE:
    {
        hti_compacting * hti = 
            (hti_compacting *) getPlanPtrFromHandle<hti_compacting>(plan);
        hti->hash_table->Retrieve(num, (const unsigned int *) d_keys, 
                                  (unsigned int *) d_vals);
        return CUDPP_SUCCESS;
        break;
    }
    case CUDPP_MULTIVALUE_HASH_TABLE:
    {
        hti_multivalue * hti = 
            (hti_multivalue *) getPlanPtrFromHandle<hti_multivalue>(plan);
        hti->hash_table->Retrieve(num, (const unsigned int *) d_keys, 
                                  (unsigned int *) d_vals);
        return CUDPP_SUCCESS;
        break;
    }
    case CUDPP_INVALID_HASH_TABLE:
        return CUDPP_ERROR_ILLEGAL_CONFIGURATION;
        break;
    }
    return CUDPP_ERROR_ILLEGAL_CONFIGURATION;
}

CUDPP_HASH_DLL
CUDPPResult cudppDestroyHashTable(CUDPPHandle theCudpp_, CUDPPHandle plan)
{
    (void) theCudpp_;           // suppress compiler warning

    hti_void * hti_init = (hti_void *) getPlanPtrFromHandle<hti_void>(plan);
    switch(hti_init->config.type)
    {
    case CUDPP_BASIC_HASH_TABLE:
    {
        hti_basic * hti = (hti_basic *) getPlanPtrFromHandle<hti_basic>(plan);
        delete hti;
        return CUDPP_SUCCESS;
    }
    case CUDPP_COMPACTING_HASH_TABLE:
    {
        hti_compacting * hti = 
            (hti_compacting *) getPlanPtrFromHandle<hti_compacting>(plan);
        delete hti;
        return CUDPP_SUCCESS;
    }
    case CUDPP_MULTIVALUE_HASH_TABLE:
    {
        hti_multivalue * hti = 
            (hti_multivalue *) getPlanPtrFromHandle<hti_multivalue>(plan);
        delete hti;
        return CUDPP_SUCCESS;
    }
    case CUDPP_INVALID_HASH_TABLE:
        return CUDPP_ERROR_ILLEGAL_CONFIGURATION;
        break;
    }
    return CUDPP_ERROR_ILLEGAL_CONFIGURATION;
}

CUDPP_HASH_DLL
unsigned cudppHashGetNotFoundValue(CUDPPHandle theCudpp_)
{
    (void) theCudpp_;           // suppress compiler warning

    return CudaHT::CuckooHashing::kNotFound;
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
