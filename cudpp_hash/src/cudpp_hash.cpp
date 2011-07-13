#include "cudpp_hash.h"
#include "cudpp_plan.h"

#include "hash_table.h"         // HashTable class

typedef CUDPPHashTableInternal<CudaHT::CuckooHashing::HashTable> hti_basic;
typedef CUDPPHashTableInternal<void> hti_void;

// cudppHashTable will create some sort of internal struct that you
// write. It will then cast the pointer to that struct to a
// CUDPPHandle (just like cudppPlan() does), and return that.
CUDPP_HASH_DLL
CUDPPResult cudppHashTable(CUDPPHandle *hash_table, 
                           const CUDPPHashTableConfig *config)
{
    switch(config->type)
    {
    case CUDPP_BASIC_HASH_TABLE:
    {
        CudaHT::CuckooHashing::HashTable * basic_table = 
            new CudaHT::CuckooHashing::HashTable();
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
    case CUDPP_MULTIVALUE_HASH_TABLE:
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
CUDPPResult cudppHashInsert(CUDPPHandle plan, const void* d_keys, 
                            const void* d_vals, unsigned int num)
{
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
    case CUDPP_MULTIVALUE_HASH_TABLE:
    case CUDPP_INVALID_HASH_TABLE:
        return CUDPP_ERROR_ILLEGAL_CONFIGURATION;
        break;
    }
    return CUDPP_ERROR_ILLEGAL_CONFIGURATION;
}

CUDPP_HASH_DLL
CUDPPResult cudppHashRetrieve(CUDPPHandle plan, const void* d_keys, 
                              void* d_vals, size_t num)
{
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
    case CUDPP_MULTIVALUE_HASH_TABLE:
    case CUDPP_INVALID_HASH_TABLE:
        return CUDPP_ERROR_ILLEGAL_CONFIGURATION;
        break;
    }
    return CUDPP_ERROR_ILLEGAL_CONFIGURATION;
}

CUDPP_HASH_DLL
CUDPPResult cudppDestroyHashTable(CUDPPHandle plan)
{
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
    case CUDPP_MULTIVALUE_HASH_TABLE:
    case CUDPP_INVALID_HASH_TABLE:
        return CUDPP_ERROR_ILLEGAL_CONFIGURATION;
        break;
    }
    return CUDPP_ERROR_ILLEGAL_CONFIGURATION;
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
