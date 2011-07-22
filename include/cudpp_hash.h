#include "cudpp.h"

enum CUDPPHashTableType
{
    CUDPP_BASIC_HASH_TABLE,
    CUDPP_COMPACTING_HASH_TABLE,
    CUDPP_MULTIVALUE_HASH_TABLE,
    CUDPP_INVALID_HASH_TABLE,
};

inline CUDPPHashTableType& operator++(CUDPPHashTableType& htt, int)
{
   const int i = static_cast<int>(htt);
   htt = static_cast<CUDPPHashTableType>(i + 1);
   return htt;
}

struct CUDPPHashTableConfig
{
    CUDPPHashTableType type;
    unsigned int kInputSize;
    float space_usage;
};

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

extern 
CUDPP_DLL
const unsigned int CUDPP_HASH_KEY_NOT_FOUND;

CUDPP_DLL 
CUDPPResult cudppHashTable(CUDPPHandle theCudpp_, 
                           CUDPPHandle *plan, 
                           const CUDPPHashTableConfig *config);

CUDPP_DLL 
CUDPPResult cudppHashInsert(CUDPPHandle theCudpp_, 
                            CUDPPHandle plan, 
                            const void* d_keys, 
                            const void* d_vals, 
                            unsigned int num);

CUDPP_DLL 
CUDPPResult cudppHashRetrieve(CUDPPHandle theCudpp_, 
                              CUDPPHandle plan, 
                              const void* d_keys, 
                              void* d_vals, 
                              size_t num);

CUDPP_DLL CUDPPResult
cudppDestroyHashTable(CUDPPHandle theCudpp_, 
                      CUDPPHandle plan);

CUDPP_DLL
CUDPPResult cudppMultivalueHashGetValuesSize(CUDPPHandle theCudpp_, 
                                             CUDPPHandle plan,
                                             unsigned int * size);

CUDPP_DLL
CUDPPResult cudppMultivalueHashGetAllValues(CUDPPHandle theCudpp_, 
                                            CUDPPHandle plan,
                                            unsigned int ** d_vals);
CUDPP_DLL
unsigned cudppHashGetNotFoundValue(CUDPPHandle theCudpp_);

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:

