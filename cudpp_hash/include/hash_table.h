/*! @file hash_table.h
 *  @brief Header for a basic hash table that stores one value per key.
 */

#ifndef CUDAHT__CUCKOO__SRC__LIBRARY__HASH_TABLE__H
#define CUDAHT__CUCKOO__SRC__LIBRARY__HASH_TABLE__H

#include <cudpp.h>
#include "definitions.h"

#include "hash_functions.h"
#include <cstdio>

/* --------------------------------------------------------------------------
   Doxygen definitions.
   -------------------------------------------------------------------------- */
/*! @namespace CudaHT
 *  @brief Encapsulates the hash table library.
 */

/*! @namespace CuckooHashing
 *  @brief Encapsulates the cuckoo hash table that uses stashes.
 */

/*! @defgroup PublicInterface Public Interface
 *  Code necessary for using the hash tables.
 */


/* -------------------------------------------------------------------------
   Hash table code.
   ------------------------------------------------------------------------- */
namespace CudaHT {
namespace CuckooHashing {

//! Compute how many thread blocks are required for the given number of threads.
dim3 ComputeGridDim(unsigned threads);

//! Compute how long an eviction chain is allowed to become for a given input size.
/*! \param[in] num_keys       Number of keys in the input.
 *  \param[in] table_size     Number of slots in the hash table.
 *  \param[in] num_functions  Number of hash functions being used.
 *  \returns The number of iterations that should be allowed.
 *
 *  The latter two parameters are only needed when using an empirical
 *  formula for computing the chain length.
 */
unsigned ComputeMaxIterations(const unsigned num_keys,
                              const unsigned table_size,
                              const unsigned num_functions);

//! Basic hash table that stores one value for each key.
/*! The input consists of two unsigned arrays of keys and values.
 *  None of the keys are expected to be repeated.
 *
 *  @todo Templatize the interface without forcing the header file to have CUDA calls.
 *  @ingroup PublicInterface
 */
class HashTable {
 public:
  HashTable() : table_size_(0),
                d_contents_(NULL),
                stash_count_(0),
                d_failures_(NULL) {}

  virtual ~HashTable() {Release();}

  //! Initialize the hash table's memory.  Must be called before \ref Build() and after the random number generator has been seeded.
  /*! @param[in] max_input_size   Largest expected number of items in the input.
   *  @param[in] space_usage      Size of the hash table relative to the input.  Bigger tables are faster to build and retrieve from.
   *  @param[in] num_functions    Number of hash functions to use.  May be 2-5.  More hash functions make it easier to build the table, but increase retrieval times.
   *  @returns Whether the initialization was successful.
   *
   *  The minimum space usage is dependent on the number of functions being used; for two through five functions, the
   *  minimum space usage is 2.1, 1.1, 1.03, and 1.02 respectively.
   */
  virtual bool Initialize(const unsigned max_input_size,
                          const float    space_usage    = 1.25,
                          const unsigned num_functions  = 4);

  //! Free all memory.
  virtual void Release();

  //! Build the hash table.
  /*! @param[in] input_size   Number of key-value pairs being inserted.
   *  @param[in] d_keys       Device memory array containing all of the input keys.
   *  @param[in] d_vals       Device memory array containing the keys' values.
   *  @returns Whether the hash table was built successfully or not.
   *
   *  Several attempts are allowed to build the hash table in case of failure.
   *  The input keys are expected to be completely unique.
   *  To reduce the chance of a failure, increase the space usage or number of functions.
   *  Keys are not allowed to be equal to \ref CudaHT::CuckooHashing::kKeyEmpty.
   */
  virtual bool Build(const unsigned  input_size,
                     const unsigned *d_keys,
                     const unsigned *d_vals);

  //! Query the hash table.
  /*! @param[in] n_queries        Number of keys in the query set.
   *  @param[in] d_query_keys     Device memory array containing all of the query keys.
   *  @param[in] d_query_results  Values for the query keys.
   *
   *  \ref kNotFound is returned for any query key that failed to be found in the table.
   */
  virtual void Retrieve(const unsigned  n_queries,
                        const unsigned *d_query_keys,
                              unsigned *d_query_results);

  //! @name Accessors
  /// @brief Mainly needed to use the __device__ CudaHT::retrieve() function directly.
  /// @{

  //! Returns how many slots the hash table has.
  inline unsigned     get_table_size()         const {return table_size_;}

  //! Returns how many items are stored in the stash.
  inline unsigned     get_stash_count()        const {return stash_count_;}

  //! Returns the constants used by the stash.
  inline uint2        get_stash_constants()    const {return stash_constants_;}

  //! Returns the hash table contents.
  inline const Entry* get_contents()           const {return d_contents_;}

  //! Returns the number of hash functions being used.
  inline unsigned     get_num_hash_functions() const {return num_hash_functions_;}

  //! When using two hash functions, returns the constants.
  inline Functions<2> get_constants_2()        const {return constants_2_;}

  //! When using three hash functions, returns the constants.
  inline Functions<3> get_constants_3()        const {return constants_3_;}

  //! When using four hash functions, returns the constants.
  inline Functions<4> get_constants_4()        const {return constants_4_;}

  //! When using five hash functions, returns the constants.
  inline Functions<5> get_constants_5()        const {return constants_5_;}

  /// @}

 protected:
  unsigned      table_size_;           //!< Size of the hash table.
  unsigned      num_hash_functions_;   //!< Number of hash functions being used.
  Entry        *d_contents_;           //!< Device memory: The hash table contents.  The stash is stored at the end.
  unsigned      stash_count_;          //!< Number of key-value pairs currently stored.
  uint2         stash_constants_;      //!< Hash function constants for the stash.

  Functions<2>  constants_2_;          //!< Constants for a set of two hash functions.
  Functions<3>  constants_3_;          //!< Constants for a set of three hash functions.
  Functions<4>  constants_4_;          //!< Constants for a set of four hash functions.
  Functions<5>  constants_5_;          //!< Constants for a set of five hash functions.

  unsigned     *d_failures_;           //!< Device memory: General use error flag.

  CUDPPHandle  theCudpp;               //!< CUDPP instance
};

};  // namespace CuckooHashing
};  // namespace CudaHT

#endif
