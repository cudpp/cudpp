#include "cudpp.h"

/* To use CUDPP_HASH as a static library, #define CUDPP_HASH_STATIC_LIB before 
 * including cudpp.h
 */
#ifndef CUDPP_HASH_DLL
    #ifdef _WIN32
        #ifdef CUDPP_HASH_STATIC_LIB
            #define CUDPP_HASH_DLL
        #else
        #ifdef BUILD_DLL
            #define CUDPP_HASH_DLL __declspec(dllexport)
        #else
            #define CUDPP_HASH_DLL __declspec(dllimport)
        #endif
        #endif
    #else
        #define CUDPP_HASH_DLL
    #endif
#endif

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

inline CUDPPHashTableType& operator++(CUDPPHashTableType& htt, int)
{
   const int i = static_cast<int>(htt);
   htt = static_cast<CUDPPHashTableType>(i + 1);
   return htt;
}

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

/* @brief Internal structure used to store CUDPP hash table */
template<class T>
class CUDPPHashTableInternal
{
public:
    CUDPPHashTableInternal(const CUDPPHashTableConfig * c, T * t) : 
        config(*c), hash_table(t) {}
    CUDPPHashTableConfig config;
    T * hash_table;
    // template<typename T> T getHashTablePtr()
    // {
        // return reinterpret_cast<T>(hash_table);
        // }
    //! @internal Convert this pointer to an opaque handle
    CUDPPHandle getHandle()
    {
        return reinterpret_cast<CUDPPHandle>(this);
    }
    ~CUDPPHashTableInternal() 
    {
        delete hash_table;
    }
};

extern const unsigned int CUDPP_HASH_KEY_NOT_FOUND;

CUDPPResult
cudppHashTable(CUDPPHandle cudppHandle, CUDPPHandle *plan,
               const CUDPPHashTableConfig *config);

CUDPPResult
cudppDestroyHashTable(CUDPPHandle cudppHandle, CUDPPHandle plan);

CUDPPResult
cudppHashInsert(CUDPPHandle plan, const void* d_keys, const void* d_vals,
                size_t num);

CUDPPResult
cudppHashRetrieve(CUDPPHandle plan, const void* d_keys, void* d_vals, 
                  size_t num);

CUDPPResult
cudppMultivalueHashGetValuesSize(CUDPPHandle plan, unsigned int * size);

CUDPPResult
cudppMultivalueHashGetAllValues(CUDPPHandle plan, unsigned int ** d_vals);

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:

