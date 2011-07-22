#include "cudpp.h"

#include "cudpp_config.h"

/**
 * @brief Supported types of hash tables
 *
 * @see CUDPPHashTableConfig
 */
enum CUDPPHashTableType
{
    CUDPP_BASIC_HASH_TABLE,     /**< Stores a single value per key.
                                 * Input is expected to be a set of
                                 * key-value pairs, where the keys are
                                 * all unique. */
    CUDPP_COMPACTING_HASH_TABLE,/**< Assigns each key a unique
                                 * identifier and allows O(1)
                                 * translation between the key and the
                                 * unique IDs. Input is a set of keys
                                 * that may, or may not, be
                                 * repeated. */
    CUDPP_MULTIVALUE_HASH_TABLE,/**< Can store multiple values for
                                 * each key. Multiple values for the
                                 * same key are represented by
                                 * different key-value pairs in the
                                 * input. */
    CUDPP_INVALID_HASH_TABLE,   /**< Invalid hash table; flags error
                                 * if used. */
};

/**
 * @brief Configuration struct for creating a hash table (CUDPPHashTable())
 * 
 * @see CUDPPHashTable, CUDDPHashTableType
 */
struct CUDPPHashTableConfig
{
    CUDPPHashTableType type;    /**< see CUDPPHashTableType */
    unsigned int kInputSize;    /**< number of elements to be stored
                                 * in hash table */
    float space_usage;          /**< space factor multiple for the
                                 * hash table; multiply space_usage by
                                 * kInputSize to get the actual space
                                 * allocation in GPU memory. 1.05 is
                                 * about the minimum possible to get a
                                 * working hash table. Larger values
                                 * use more space but take less time
                                 * to construct. */
};

extern CUDPP_DLL const unsigned int CUDPP_HASH_KEY_NOT_FOUND;

CUDPP_DLL CUDPPResult 
cudppHashTable(CUDPPHandle cudppHandle, 
               CUDPPHandle *plan,
               const CUDPPHashTableConfig *config);

CUDPP_DLL CUDPPResult
cudppDestroyHashTable(CUDPPHandle cudppHandle, 
                      CUDPPHandle plan);

CUDPP_DLL CUDPPResult
cudppHashInsert(CUDPPHandle plan, 
                const void* d_keys, 
                const void* d_vals,
                size_t num);

CUDPP_DLL CUDPPResult
cudppHashRetrieve(CUDPPHandle plan, 
                  const void* d_keys, 
                  void* d_vals, 
                  size_t num);

CUDPP_DLL CUDPPResult
cudppMultivalueHashGetValuesSize(CUDPPHandle plan, 
                                 unsigned int * size);

CUDPP_DLL CUDPPResult
cudppMultivalueHashGetAllValues(CUDPPHandle plan, 
                                unsigned int ** d_vals);

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:

