// -------------------------------------------------------------
// cuDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision:$
// $Date:$
// ------------------------------------------------------------- 
// This source code is distributed under the terms of license.txt in
// the root directory of this source distribution.
// ------------------------------------------------------------- 

/**
 * @file hash_multivalue.h
 *
 * @brief Header for hash tables that store multiple values per key.
 */

#ifndef CUDAHT__CUCKOO__SRC__LIBRARY__HASH_MULTIVALUE__H
#define CUDAHT__CUCKOO__SRC__LIBRARY__HASH_MULTIVALUE__H

#include "hash_table.h"

/** \addtogroup cudpp_app 
  * @{
  */

/** \addtogroup cudpp_hash_data_structures
 * @{
 */

namespace CudaHT {
namespace CuckooHashing {

//! @class MultivalueHashTable
/*! @brief Hash table that stores multiple values per key. 
 * 
 *  A key with multiple values is represented by multiple key-value
 *  pairs in the input with the same key.
 *
 *  Querying the structure returns how many items the key has and its
 *  location in the array returned by \ref get_all_values().
 */
class MultivalueHashTable : public HashTable {
public:
    MultivalueHashTable();
    virtual ~MultivalueHashTable() {Release();}

    //! Build the multi-value hash table.
    /*! See \ref HashTable::Build() for an explanation of the parameters.
     *  Key-value pairs in the input with the same key are assumed to be
     *  values associated with the same key.
     *  @param[in] input_size   Number of key-value pairs being inserted.
     *  @param[in] d_keys       Device memory array containing all of the input keys.
     *  @param[in] d_vals       Device memory array containing the keys' values.
     *  @returns Whether the hash table was built successfully (true) or not (false).
     *  @see \ref HashTable::Build()
     */
    virtual bool Build(const unsigned  input_size,
                       const unsigned *d_keys,
                       const unsigned *d_vals);

    virtual void Release();

    //! Don't call this.
    /*! @todo Remove this function entirely somehow.
     */
    virtual void Retrieve(const unsigned   /* n_queries */,
                          const unsigned * /* d_keys */,
                          unsigned       * /* d_location_counts */)
    { 
        fprintf(stderr, "Wrong retrieve function.\n"); exit(1);
    }

    //! Retrieve from a multi-value hash table.
    /*! @param[in]   n_queries          Number of queries in the input.
     *  @param[in]   d_keys             Device mem: All of the query keys.
     *  @param[out]  d_location_counts  Contains the index of a query key's 
     *                                  first value and the number of values
     *                                  associated with the key.
     *                                  If a query fails, the number of values
     *                                  the key has will be marked as zero.
     */
    virtual void Retrieve(const unsigned  n_queries,
                          const unsigned *d_keys,
                          uint2    *d_location_counts);

    //! Returns the array of values, where each key's values are stored contiguously in memory.
    inline const unsigned* get_all_values() const {return d_sorted_values_;}

    //! Gets the total number of values between all of the keys.
    inline unsigned get_values_size() const {return sorted_values_size_;}

    //! Gets the location and number of values each key has.
    inline const uint2* get_index_counts() const {return d_index_counts_;}

    //! Initializes the multi-value hash table's memory.
    /*! See \ref HashTable::Initialize() for an explanation of the parameters.
     * 
     * @param[in] max_input_size  Largest expected number of items in the input.
     * @param[in] space_usage     Size of the hash table relative to the input.
     *                            Bigger tables are faster to build and 
     *                            retrieve from.
     * @param[in] num_functions   Number of hash functions to use. May be 2-5.
     *                            More hash functions make it easier to build
     *                            the table, but increase retrieval times.
     * @returns Whether the hash table was initialized successfully (true) or
     *                            not (false).
     * @see HashTable::Initialize()
     */
    virtual bool Initialize(const unsigned max_input_size,
                            const float    space_usage    = 1.2,
                            const unsigned num_functions  = 4);

private:
    // Multi-value hash data.
    unsigned *d_sorted_values_;
    unsigned  sorted_values_size_;
    uint2    *d_index_counts_;
    unsigned *d_unique_keys_;
    float     target_space_usage_;

    // Scratch memory.
    size_t    scanplan_;
    unsigned *d_scratch_is_unique_;
    unsigned *d_scratch_offsets_;
};

};  // namespace CuckooHashing
};  // namespace CudaHT

/** @} */ // end hash table data structures
/** @} */ // end cudpp_app

#endif

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
