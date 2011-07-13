/*! @file hash_multivalue.h
 *  @brief Include this file to create hash tables that store multiple values per key.
 */

#ifndef CUDAHT__CUCKOO__SRC__LIBRARY__HASH_MULTIVALUE__H
#define CUDAHT__CUCKOO__SRC__LIBRARY__HASH_MULTIVALUE__H

#include "hash_table.h"

namespace CudaHT {
namespace CuckooHashing {

//! @class MultivalueHashTable
/*! @brief Stores multiple values per key.
 *  @ingroup PublicInterface
 *  A key with multiple values is represented by multiple key-value pairs in the input
 *  with the same key.
 *
 *  Querying the structure returns how many items the key has and its location in
 *  the array returned by \ref get_all_values().
 */
class MultivalueHashTable : public HashTable {
public:
  MultivalueHashTable();
  virtual ~MultivalueHashTable() {Release();}

  //! Build the multi-value hash table.
  /*! See \ref HashTable::Build() for an explanation of the parameters.
   *  Key-value pairs in the input with the same key are assumed to be
   *  values associated with the same key.
   */
  virtual bool Build(const unsigned  input_size,
                     const unsigned *d_keys,
                     const unsigned *d_vals);

  virtual void Release();

  //! Don't call this.
  /*! @todo Remove this function entirely somehow.
   */
  virtual void Retrieve(const unsigned  n_queries,
                        const unsigned *d_keys,
                              unsigned *d_location_counts) { fprintf(stderr, "Wrong retrieve function.\n"); exit(1); }

  //! Retrieve from a multi-value hash table.
  /*! @param[in]   n_queries          Number of queries in the input.
   *  @param[in]   d_keys             Device mem: All of the query keys.
   *  @param[out]  d_location_counts  Contains the index of a query key's first value
   *                                  and the number of values assocatied with the key.
   *
   *  If a query fails, the number of values the key has will be marked as zero.
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
   */
  virtual bool Initialize(const unsigned max_input_size,
                          const float    space_usage    = 1.2,
                          const unsigned num_funcionts  = 4);

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

#endif
