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

/* @brief unsigned int indicating a not-found value in a hash table */
const unsigned int CUDPP_HASH_KEY_NOT_FOUND = CudaHT::CuckooHashing::kNotFound;

// cudppHashTable will create some sort of internal struct that you
// write. It will then cast the pointer to that struct to a
// CUDPPHandle (just like cudppPlan() does), and return that.

/**
 * @brief Creates a CUDPP hash table in GPU memory given an input hash
 * table configuration; returns the \a plan for that hash table.
 * 
 * Requires a CUDPPHandle for the CUDPP instance (to ensure thread
 * safety); call cudppCreate() to get this handle. 
 * 
 * The hash table implementation requires hardware capability 2.0 or
 * higher (64-bit atomic operations).
 * 
 * Hash table types and input parameters are discussed in
 * CUDPPHashTableType and CUDPPHashTableConfig.
 * 
 * After you are finished with the hash table, clean up with
 * cudppDestroyHashTable().
 * 
 * @param[in] cudppHandle Handle to CUDPP instance
 * @param[out] plan Handle to hash table instance
 * @param[in] config Configuration for hash table to be created
 * @returns CUDPPResult indicating if creation was successful
 * 
 * @see cudppCreate, cudppDestroyHashTable
 */
CUDPP_HASH_DLL
CUDPPResult
cudppHashTable(CUDPPHandle cudppHandle, CUDPPHandle *plan,
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
        basic_table->setTheCudpp(cudppHandle);
        basic_table->Initialize(config->kInputSize, config->space_usage);
        hti_basic * hti = new hti_basic(config, basic_table);
        if (!hti)
        {
            return CUDPP_ERROR_UNKNOWN;
        }
        else
        {
            *plan = hti->getHandle();
            return CUDPP_SUCCESS;
        }
        break;
    }
    case CUDPP_COMPACTING_HASH_TABLE:
    {
        CudaHT::CuckooHashing::CompactingHashTable * compacting_table = 
            new CudaHT::CuckooHashing::CompactingHashTable();
        compacting_table->setTheCudpp(cudppHandle);
        compacting_table->Initialize(config->kInputSize, config->space_usage);
        hti_compacting * hti = new hti_compacting(config, compacting_table);
        if (!hti)
        {
            return CUDPP_ERROR_UNKNOWN;
        }
        else
        {
            *plan = hti->getHandle();
            return CUDPP_SUCCESS;
        }
        break;
    }
    case CUDPP_MULTIVALUE_HASH_TABLE:
    {
        CudaHT::CuckooHashing::MultivalueHashTable * multivalue_table = 
            new CudaHT::CuckooHashing::MultivalueHashTable();
        multivalue_table->setTheCudpp(cudppHandle);
        multivalue_table->Initialize(config->kInputSize, config->space_usage);
        hti_multivalue * hti = new hti_multivalue(config, multivalue_table);
        if (!hti)
        {
            return CUDPP_ERROR_UNKNOWN;
        }
        else
        {
            *plan = hti->getHandle();
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
/**
 * @brief Inserts keys and values into a CUDPP hash table
 * 
 * Requires a CUDPPHandle for the hash table instance; call
 * cudppHashTable() to create the hash table and get this handle.
 *
 * \a d_keys and \a d_values should be in GPU memory. These should be
 * pointers to arrays of unsigned ints.
 *
 * Calls HashTable::Build internally.
 * 
 * @param[in] plan Handle to hash table instance
 * @param[in] d_keys GPU pointer to keys to be inserted
 * @param[in] d_vals GPU pointer to values to be inserted
 * @param[in] num Number of keys/values to be inserted
 * @returns CUDPPResult indicating if insertion was successful
 * 
 * @see cudppHashTable, cudppHashRetrieve, HashTable::Build,
 * CompactingHashTable::Build, MultivalueHashTable::Build
 */

CUDPP_HASH_DLL
CUDPPResult 
cudppHashInsert(CUDPPHandle plan, const void* d_keys, const void* d_vals,
                size_t num)
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

/**
 * @brief Retrieves values, given keys, from a CUDPP hash table
 * 
 * Requires a CUDPPHandle for the hash table instance; call
 * cudppHashTable() to create the hash table and get this handle.
 *
 * \a d_keys and \a d_values should be in GPU memory. These should be
 * pointers to arrays of unsigned ints.
 *
 * Calls HashTable::Retrieve internally.
 * 
 * @param[in] plan Handle to hash table instance
 * @param[in] d_keys GPU pointer to keys to be retrieved
 * @param[out] d_vals GPU pointer to values to be retrieved
 * @param[in] num Number of keys/values to be retrieved
 * @returns CUDPPResult indicating if retrieval was successful
 * 
 * @see cudppHashTable, cudppHashBuild, HashTable::Retrieve,
 * CompactingHashTable::Retrieve, MultivalueHashTable::Retrieve
 */
CUDPP_HASH_DLL
CUDPPResult
cudppHashRetrieve(CUDPPHandle plan, const void* d_keys, void* d_vals, 
                  size_t num)
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
                                  (uint2 *) d_vals);
        return CUDPP_SUCCESS;
        break;
    }
    case CUDPP_INVALID_HASH_TABLE:
        return CUDPP_ERROR_ILLEGAL_CONFIGURATION;
        break;
    }
    return CUDPP_ERROR_ILLEGAL_CONFIGURATION;
}

/**
 * @brief Destroys a hash table given its handle.
 * 
 * Requires a CUDPPHandle for the CUDPP instance (to ensure thread
 * safety); call cudppCreate() to get this handle. 
 * 
 * Requires a CUDPPHandle for the hash table instance; call
 * cudppHashTable() to get this handle.
 * 
 * @param[in] cudppHandle Handle to CUDPP instance
 * @param[in] plan Handle to hash table instance
 * @returns CUDPPResult indicating if destruction was successful
 * 
 * @see cudppHashTable
 */
CUDPP_HASH_DLL
CUDPPResult
cudppDestroyHashTable(CUDPPHandle /* cudppHandle */, CUDPPHandle plan)
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

/**
 * @brief Retrieves the size of the values array in a multivalue hash table
 * 
 * Only relevant for multivalue hash tables.
 * 
 * Requires a CUDPPHandle for the hash table instance; call
 * cudppHashTable() to get this handle.
 * 
 * @param[in] plan Handle to hash table instance
 * @param[out] size Pointer to size of multivalue hash table
 * @returns CUDPPResult indicating if operation was successful
 * 
 * @see cudppHashTable, cudppMultivalueHashGetAllValues
 */
CUDPP_HASH_DLL
CUDPPResult
cudppMultivalueHashGetValuesSize(CUDPPHandle plan, unsigned int * size)
{
    hti_void * hti_init = (hti_void *) getPlanPtrFromHandle<hti_void>(plan);
    if (hti_init->config.type != CUDPP_MULTIVALUE_HASH_TABLE)
    {
        // better be a MULTIVALUE
        return CUDPP_ERROR_ILLEGAL_CONFIGURATION;
    }
    hti_multivalue * hti = 
        (hti_multivalue *) getPlanPtrFromHandle<hti_multivalue>(plan);
    *size = hti->hash_table->get_values_size();
    return CUDPP_SUCCESS;
}

/**
 * @brief Retrieves a pointer to the values array in a multivalue hash table
 * 
 * Only relevant for multivalue hash tables.
 * 
 * Requires a CUDPPHandle for the hash table instance; call
 * cudppHashTable() to get this handle.
 * 
 * @param[in] plan Handle to hash table instance
 * @param[out] d_vals Pointer to pointer of values (in GPU memory)
 * @returns CUDPPResult indicating if operation was successful
 * 
 * @see cudppHashTable, cudppMultivalueHashGetValuesSize
 */
CUDPP_HASH_DLL
CUDPPResult
cudppMultivalueHashGetAllValues(CUDPPHandle plan, unsigned int ** d_vals)
{
    hti_void * hti_init = (hti_void *) getPlanPtrFromHandle<hti_void>(plan);
    if (hti_init->config.type != CUDPP_MULTIVALUE_HASH_TABLE)
    {
        // better be a MULTIVALUE
        return CUDPP_ERROR_ILLEGAL_CONFIGURATION;
    }
    hti_multivalue * hti = 
        (hti_multivalue *) getPlanPtrFromHandle<hti_multivalue>(plan);
    // @TODO fix up constness
    *d_vals = (unsigned*) (hti->hash_table->get_all_values());
    return CUDPP_SUCCESS;
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
