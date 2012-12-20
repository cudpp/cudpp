// -------------------------------------------------------------
// cuDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision: 5636 $
// $Date: 2009-07-02 13:39:38 +1000 (Thu, 02 Jul 2009) $
// ------------------------------------------------------------- 
// This source code is distributed under the terms of license.txt in
// the root directory of this source distribution.
// ------------------------------------------------------------- 

/**
 * @file
 * cudpp.cpp
 *
 * @brief Main library source file.  Implements wrappers for public
 * interface.  
 * 
 * Main library source file.  Implements wrappers for public
 * interface.  These wrappers call application-level operators.
 * As this grows we may decide to partition into multiple source
 * files.
 */

/**
 * \defgroup publicInterface CUDPP Public Interface
 * The CUDA public interface comprises the functions, structs, and enums
 * defined in cudpp.h.  Public interface functions call functions in the
 * \link cudpp_app Application-Level\endlink interface. The public 
 * interface functions include Plan Interface functions and Algorithm
 * Interface functions.  Plan Interface functions are used for creating
 * CUDPP Plan objects that contain configuration details, intermediate
 * storage space, and in the case of cudppSparseMatrix(), data.  The 
 * Algorithm Interface is the set of functions that do the real work 
 * of CUDPP, such as cudppScan() and cudppSparseMatrixVectorMultiply().
 *
 * @{
 */

/** @name Algorithm Interface
 * @{
 */

#include "cudpp.h"
#include "cudpp_manager.h"
#include "cudpp_scan.h"
#include "cudpp_segscan.h"
#include "cudpp_compact.h"
#include "cudpp_spmvmult.h"
#include "cudpp_mergesort.h"
#include "cudpp_radixsort.h"
#include "cudpp_rand.h"
#include "cudpp_reduce.h"
#include "cudpp_tridiagonal.h"

/**
 * @brief Performs a scan operation of numElements on its input in
 * GPU memory (d_in) and places the output in GPU memory
 * (d_out), with the scan parameters specified in the plan pointed to by
 * planHandle. 
 
 * The input to a scan operation is an input array, a binary associative 
 * operator (like + or max), and an identity element for that operator 
 * (+'s identity is 0). The output of scan is the same size as its input.
 * Informally, the output at each element is the result of operator
 * applied to each input that comes before it. For instance, the
 * output of sum-scan at each element is the sum of all the input
 * elements before that input.
 *
 * More formally, for associative operator
 * @htmlonly&oplus;@endhtmlonly@latexonly$\oplus$@endlatexonly,
 * <var>out<sub>i</sub></var> = <var>in<sub>0</sub></var>
 * @htmlonly&oplus;@endhtmlonly@latexonly$\oplus$@endlatexonly
 * <var>in<sub>1</sub></var>
 * @htmlonly&oplus;@endhtmlonly@latexonly$\oplus$@endlatexonly ...
 * @htmlonly&oplus;@endhtmlonly@latexonly$\oplus$@endlatexonly
 * <var>in<sub>i-1</sub></var>.
 * 
 * CUDPP supports "exclusive" and "inclusive" scans. For the ADD operator, 
 * an exclusive scan computes the sum of all input elements before the 
 * current element, while an inclusive scan computes the sum of all input 
 * elements up to and including the current element. 
 * 
 * Before calling scan, create an internal plan using cudppPlan().
 * 
 * After you are finished with the scan plan, clean up with cudppDestroyPlan(). 
 * 
 * @param[in] planHandle Handle to plan for this scan
 * @param[out] d_out output of scan, in GPU memory
 * @param[in] d_in input to scan, in GPU memory
 * @param[in] numElements number of elements to scan
 * @returns CUDPPResult indicating success or error condition 
 * 
 * @see cudppPlan, cudppDestroyPlan
 */
CUDPP_DLL
CUDPPResult cudppScan(const CUDPPHandle planHandle,
                      void              *d_out, 
                      const void        *d_in, 
                      size_t            numElements)
{
    CUDPPScanPlan *plan = 
        (CUDPPScanPlan*)getPlanPtrFromHandle<CUDPPScanPlan>(planHandle);

    if (plan != NULL)
    {
        if (plan->m_config.algorithm != CUDPP_SCAN)
            return CUDPP_ERROR_INVALID_PLAN;
            
        cudppScanDispatch(d_out, d_in, numElements, 1, plan);
        return CUDPP_SUCCESS;
    }
    else
        return CUDPP_ERROR_INVALID_HANDLE;
}

/**
 * @brief Performs a segmented scan operation of numElements on its input in
 * GPU memory (d_idata) and places the output in GPU memory
 * (d_out), with the scan parameters specified in the plan pointed to by
 * planHandle. 
 
 * The input to a segmented scan operation is an input array of data,
 * an input array of flags which demarcate segments, a binary associative 
 * operator (like + or max), and an identity element for that operator 
 * (+'s identity is 0). The array of flags is the same length as the input
 * with 1 marking the the first element of a segment and 0 otherwise. The 
 * output of segmented scan is the same size as its input. Informally, the 
 * output at each element is the result of operator applied to each input 
 * that comes before it in that segment. For instance, the output of 
 * segmented sum-scan at each element is the sum of all the input elements 
 * before that input in that segment.
 *
 * More formally, for associative operator
 * @htmlonly&oplus;@endhtmlonly@latexonly$\oplus$@endlatexonly,
 * <var>out<sub>i</sub></var> = <var>in<sub>k</sub></var>
 * @htmlonly&oplus;@endhtmlonly@latexonly$\oplus$@endlatexonly
 * <var>in<sub>k+1</sub></var>
 * @htmlonly&oplus;@endhtmlonly@latexonly$\oplus$@endlatexonly ...
 * @htmlonly&oplus;@endhtmlonly@latexonly$\oplus$@endlatexonly
 * <var>in<sub>i-1</sub></var>.
 * <i>k</i> is the index of the first element of the segment in which <i>i</i> lies
 * 
 * We support both "exclusive" and "inclusive" variants. For a segmented sum-scan, 
 * the exclusive variant computes the sum of all input elements before the 
 * current element in that segment, while the inclusive variant computes the 
 * sum of all input elements up to and including the current element, in 
 * that segment. 
 * 
 * Before calling segmented scan, create an internal plan using cudppPlan().
 * 
 * After you are finished with the scan plan, clean up with cudppDestroyPlan(). 
 * @param[in] planHandle Handle to plan for this scan
 * @param[out] d_out output of segmented scan, in GPU memory
 * @param[in] d_idata input data to segmented scan, in GPU memory
 * @param[in] d_iflags input flags to segmented scan, in GPU memory
 * @param[in] numElements number of elements to perform segmented scan on
 * @returns CUDPPResult indicating success or error condition 
 * 
 * @see cudppPlan, cudppDestroyPlan
 */
CUDPP_DLL
CUDPPResult cudppSegmentedScan(const CUDPPHandle  planHandle,
                               void               *d_out, 
                               const void         *d_idata,
                               const unsigned int *d_iflags,
                               size_t             numElements)
{
    CUDPPSegmentedScanPlan *plan = 
        (CUDPPSegmentedScanPlan*)getPlanPtrFromHandle<CUDPPSegmentedScanPlan>(planHandle);

    if (plan != NULL)
    {
        if (plan->m_config.algorithm != CUDPP_SEGMENTED_SCAN)
            return CUDPP_ERROR_INVALID_PLAN;
        
        cudppSegmentedScanDispatch(d_out, d_idata, d_iflags, numElements, plan);
        return CUDPP_SUCCESS;
    }
    else
        return CUDPP_ERROR_INVALID_HANDLE;
}

/**
 * @brief Performs numRows parallel scan operations of numElements
 * each on its input (d_in) and places the output in d_out,
 * with the scan parameters set by config. Exactly like cudppScan 
 * except that it runs on multiple rows in parallel.
 * 
 * Note that to achieve good performance with cudppMultiScan one should
 * allocate the device arrays passed to it so that all rows are aligned
 * to the correct boundaries for the architecture the app is running on.
 * The easy way to do this is to use cudaMallocPitch() to allocate a 
 * 2D array on the device.  Use the \a rowPitch parameter to cudppPlan() 
 * to specify this pitch. The easiest way is to pass the device pitch 
 * returned by cudaMallocPitch to cudppPlan() via \a rowPitch.
 * 
 * @param[in] planHandle handle to CUDPPScanPlan
 * @param[out] d_out output of scan, in GPU memory
 * @param[in] d_in input to scan, in GPU memory
 * @param[in] numElements number of elements (per row) to scan
 * @param[in] numRows number of rows to scan in parallel
 * @returns CUDPPResult indicating success or error condition 
 * 
 * @see cudppScan, cudppPlan
 */
CUDPP_DLL
CUDPPResult cudppMultiScan(const CUDPPHandle planHandle,
                           void              *d_out, 
                           const void        *d_in, 
                           size_t            numElements,
                           size_t            numRows)
{
    CUDPPScanPlan *plan = 
        (CUDPPScanPlan*)getPlanPtrFromHandle<CUDPPScanPlan>(planHandle);
    if (plan != NULL)
    {
        if (plan->m_config.algorithm != CUDPP_SCAN)
            return CUDPP_ERROR_INVALID_PLAN;
            
        cudppScanDispatch(d_out, d_in, numElements, numRows, plan);
        return CUDPP_SUCCESS;
    }
    else
        return CUDPP_ERROR_INVALID_HANDLE;
}


/**
 * @brief Given an array \a d_in and an array of 1/0 flags in \a 
 * deviceValid, returns a compacted array in \a d_out of corresponding
 * only the "valid" values from \a d_in.
 * 
 * Takes as input an array of elements in GPU memory
 * (\a d_in) and an equal-sized unsigned int array in GPU memory
 * (\a deviceValid) that indicate which of those input elements are
 * valid. The output is a packed array, in GPU memory, of only those
 * elements marked as valid.
 * 
 * Internally, uses cudppScan.
 *
 * Example:
 * \code
 * d_in    = [ a b c d e f ]
 * deviceValid = [ 1 0 1 1 0 1 ]
 * d_out   = [ a c d f ]
 * \endcode
 *
 * @todo [MJH] We need to evaluate whether cudppCompact should be a core member
 * of the public interface.  It's not clear to me that what the user always
 * wants is a final compacted array.  Often one just wants the array of indices
 * to which each input element should go in the output. The split() routine used
 * in radix sort might make more sense to expose.
 * 
 * @param[in] planHandle handle to CUDPPCompactPlan
 * @param[out] d_out compacted output
 * @param[out] d_numValidElements set during cudppCompact; is set with the
 * number of elements valid flags in the d_isValid input array
 * @param[in] d_in input to compact
 * @param[in] d_isValid which elements in d_in are valid
 * @param[in] numElements number of elements in d_in
 * @returns CUDPPResult indicating success or error condition 
 */
CUDPP_DLL
CUDPPResult cudppCompact(const CUDPPHandle  planHandle,
                         void               *d_out, 
                         size_t             *d_numValidElements,
                         const void         *d_in, 
                         const unsigned int *d_isValid,
                         size_t             numElements)
{
    CUDPPCompactPlan *plan = 
        (CUDPPCompactPlan*)getPlanPtrFromHandle<CUDPPCompactPlan>(planHandle);

    if (plan != NULL)
    {
        if (plan->m_config.algorithm != CUDPP_COMPACT)
            return CUDPP_ERROR_INVALID_PLAN;
        
        cudppCompactDispatch(d_out, d_numValidElements, d_in, d_isValid, 
            numElements, plan);
        return CUDPP_SUCCESS;
    }
    else
        return CUDPP_ERROR_INVALID_HANDLE;
}

/**
 * @brief Reduces an array to a single element using a binary associative operator
 * 
 * For example, if the operator is CUDPP_ADD, then:
 * \code
 * d_in    = [ 3 2 0 1 -4 5 0 -1 ]
 * d_out   = [ 6 ]
 * \endcode
 *
 * If the operator is CUDPP_MIN, then:
 * \code
 * d_in    = [ 3 2 0 1 -4 5 0 -1 ]
 * d_out   = [ -4 ]
 * \endcode
 *
 * Limits:
 * \a numElements must be at least 1, and is currently limited only by the addressable memory
 * in CUDA (and the output accuracy is limited by numerical precision).
 *
 * @param[in] planHandle handle to CUDPPReducePlan
 * @param[out] d_out Output of reduce (a single element) in GPU memory. 
 *                   Must be a pointer to an array of at least a single element.
 * @param[in] d_in Input array to reduce in GPU memory.  
 *                 Must be a pointer to an array of at least \a numElements elements.
 * @param[in] numElements the number of elements to reduce.  
 * @returns CUDPPResult indicating success or error condition 
 * 
 * @see cudppPlan
 */
CUDPP_DLL
CUDPPResult cudppReduce(const CUDPPHandle planHandle,
                        void              *d_out,
                        const void        *d_in,
                        size_t            numElements)
{
    CUDPPReducePlan *plan = 
        (CUDPPReducePlan*)getPlanPtrFromHandle<CUDPPReducePlan>(planHandle);

    if (plan != NULL)
    {
        if (plan->m_config.algorithm != CUDPP_REDUCE)
            return CUDPP_ERROR_INVALID_PLAN;
        
        cudppReduceDispatch(d_out, d_in, numElements, plan);
        return CUDPP_SUCCESS;
    }
    else
        return CUDPP_ERROR_INVALID_HANDLE;
}

/**
 * @brief Sorts key-value pairs or keys only
 * 
 * Takes as input an array of keys in GPU memory
 * (d_keys) and an optional array of corresponding values,
 * and outputs sorted arrays of keys and (optionally) values in place. 
 * Radix sort or Merge sort is selected through the configuration (.algorithm)
 * Key-value and key-only sort is selected through the configuration of 
 * the plan, using the options CUDPP_OPTION_KEYS_ONLY and 
 * CUDPP_OPTION_KEY_VALUE_PAIRS.
 *
 * Supported key types are CUDPP_FLOAT and CUDPP_UINT.  Values can be
 * any 32-bit type (internally, values are treated only as a payload
 * and cast to unsigned int).
 *
 * @todo Determine if we need to provide an "out of place" sort interface.
 * 
 * @param[in] planHandle handle to CUDPPSortPlan
 * @param[out] d_keys keys by which key-value pairs will be sorted
 * @param[in] d_values values to be sorted
 * @param[in] numElements number of elements in d_keys and d_values
 * @returns CUDPPResult indicating success or error condition 
 *
 * @see cudppPlan, CUDPPConfiguration, CUDPPAlgorithm
 */
CUDPP_DLL
CUDPPResult cudppRadixSort(const CUDPPHandle planHandle,
                      void              *d_keys,
                      void              *d_values,                      
                      size_t            numElements)
{
    
	
	
    CUDPPRadixSortPlan *plan = 
        (CUDPPRadixSortPlan*)getPlanPtrFromHandle<CUDPPRadixSortPlan>(planHandle);

    if (plan != NULL)
    {
        if (plan->m_config.algorithm != CUDPP_SORT_RADIX)
            return CUDPP_ERROR_INVALID_PLAN;
        
	if(plan->m_config.algorithm == CUDPP_SORT_RADIX)
            cudppRadixSortDispatch(d_keys, d_values, numElements, plan);
	
        return CUDPP_SUCCESS;
    }
    else
        return CUDPP_ERROR_INVALID_HANDLE;
}
/**
 * @brief Sorts key-value pairs or keys only
 * 
 * Takes as input an array of keys in GPU memory
 * (d_keys) and an optional array of corresponding values,
 * and outputs sorted arrays of keys and (optionally) values in place. 
 * Radix sort or Merge sort is selected through the configuration (.algorithm)
 * Key-value and key-only sort is selected through the configuration of 
 * the plan, using the options CUDPP_OPTION_KEYS_ONLY and 
 * CUDPP_OPTION_KEY_VALUE_PAIRS.
 *
 * Supported key types are CUDPP_FLOAT and CUDPP_UINT.  Values can be
 * any 32-bit type (internally, values are treated only as a payload
 * and cast to unsigned int).
 *
 * @todo Determine if we need to provide an "out of place" sort interface.
 * 
 * @param[in] planHandle handle to CUDPPSortPlan
 * @param[out] d_keys keys by which key-value pairs will be sorted
 * @param[in] d_values values to be sorted
 * @param[in] numElements number of elements in d_keys and d_values
 * @returns CUDPPResult indicating success or error condition 
 *
 * @see cudppPlan, CUDPPConfiguration, CUDPPAlgorithm
 */
CUDPP_DLL
CUDPPResult cudppMergeSort(const CUDPPHandle planHandle,
                      void              *d_keys,
                      void              *d_values,                      
                      size_t            numElements)
{    		
    CUDPPMergeSortPlan *plan = 
        (CUDPPMergeSortPlan*)getPlanPtrFromHandle<CUDPPMergeSortPlan>(planHandle);

    if (plan != NULL)
    {
        if (plan->m_config.algorithm != CUDPP_SORT_MERGE)
            return CUDPP_ERROR_INVALID_PLAN;   	
		cudppMergeSortDispatch(d_keys, d_values, numElements, plan);
	    return CUDPP_SUCCESS;
    }
    else
        return CUDPP_ERROR_INVALID_HANDLE;
}
/**
 * @brief Sorts strings. Keys are the first four characters of the string, 
 * and values are the addresses where the strings reside in memory (stringVals)
 * 
 * Takes as input an array of strings (broken up as first four chars (key), 
 * addresses (values), and the strings themselves (stringVals))
 *
 * 
 * @todo Determine if we need to provide an "out of place" sort interface.
 * 
 * @param[in] planHandle handle to CUDPPSortPlan
 * @param[in/out] d_keys keys (first four chars of string to be sorted)
 * @param[in/out] d_values addresses where the strings reside
 * @param[in] stringVals Original string input, series of characters each terminated by a null
 * @param[in] numElements number of elements in d_keys and d_values
 * @param[in] stringArrayLength Length in uint of the size of all strings
 * @returns CUDPPResult indicating success or error condition 
 *
 * @see cudppPlan, CUDPPConfiguration, CUDPPAlgorithm
 */
CUDPP_DLL
CUDPPResult cudppStringSort(const CUDPPHandle planHandle,
                      void              *d_keys,
                      void              *d_values,                      
		              void              *stringVals,
                      size_t            numElements,
		              size_t            stringArrayLength)
{    		
    CUDPPStringSortPlan *plan = 
        (CUDPPStringSortPlan*)getPlanPtrFromHandle<CUDPPStringSortPlan>(planHandle);

    if (plan != NULL)
    {
        if (plan->m_config.algorithm != CUDPP_SORT_STRING)
            return CUDPP_ERROR_INVALID_PLAN;   	
		cudppStringSortDispatch(d_keys, d_values, numElements, stringArrayLength, plan);
	    return CUDPP_SUCCESS;
    }
    else
        return CUDPP_ERROR_INVALID_HANDLE;
}


/** @brief Perform matrix-vector multiply y = A*x for arbitrary sparse matrix A and vector x
  *
  * Given a matrix object handle (which has been initialized using cudppSparseMatrix()),
  * This function multiplies the input vector \a d_x by the matrix referred to by
  * \a sparseMatrixHandle, returning the result in \a d_y.
  *
  * @param sparseMatrixHandle Handle to a sparse matrix object created with cudppSparseMatrix()
  * @param d_y The output vector, y
  * @param d_x The input vector, x
  * @returns CUDPPResult indicating success or error condition 
  * 
  * @see cudppSparseMatrix, cudppDestroySparseMatrix
  */
CUDPP_DLL
CUDPPResult cudppSparseMatrixVectorMultiply(const CUDPPHandle  sparseMatrixHandle,
                                            void               *d_y,
                                            const void         *d_x)
{
    CUDPPSparseMatrixVectorMultiplyPlan *plan = 
        (CUDPPSparseMatrixVectorMultiplyPlan*)
        getPlanPtrFromHandle<CUDPPSparseMatrixVectorMultiplyPlan>(sparseMatrixHandle);
    
    if (plan != NULL)
    {
        if (plan->m_config.algorithm != CUDPP_SPMVMULT)
            return CUDPP_ERROR_INVALID_PLAN;
        
        cudppSparseMatrixVectorMultiplyDispatch(d_y, d_x, plan);
        return CUDPP_SUCCESS;
    }
    else
        return CUDPP_ERROR_INVALID_HANDLE;
}

/**
 * @brief Rand puts \a numElements random 32-bit elements into \a d_out
 *
 
 * Outputs \a numElements random values to \a d_out. \a d_out must be of
 * type unsigned int, allocated in device memory.
 * 
 * The algorithm used for the random number generation is stored in \a planHandle.
 * Depending on the specification of the pseudo random number generator(PRNG),
 * the generator may have one or more seeds.  To set the seed, use cudppRandSeed().
 * 
 * @todo Currently only MD5 PRNG is supported.  We may provide more rand routines in 
 * the future.
 *
 * @param[in] planHandle Handle to plan for rand
 * @param[in] numElements number of elements in d_out.
 * @param[out] d_out output of rand, in GPU memory.  Should be an array of unsigned integers.
 * @returns CUDPPResult indicating success or error condition 
 *
 * @see cudppPlan, CUDPPConfiguration, CUDPPAlgorithm
 */
CUDPP_DLL
CUDPPResult cudppRand(const CUDPPHandle planHandle,
                      void *            d_out, 
                      size_t            numElements)
{
    CUDPPRandPlan * plan = 
        (CUDPPRandPlan *) getPlanPtrFromHandle<CUDPPRandPlan>(planHandle);

    if(plan != NULL)
    {
        if (plan->m_config.algorithm != CUDPP_RAND_MD5)
            return CUDPP_ERROR_INVALID_PLAN;
        
        //dispatch the rand algorithm here
        cudppRandDispatch(d_out, numElements, plan);
        return CUDPP_SUCCESS;
    }
    else
        return CUDPP_ERROR_INVALID_HANDLE;
}


/**@brief Sets the seed used for rand
 *
 * The seed is crucial to any random number generator as it allows a 
 * sequence of random numbers to be replicated.  Since there may be 
 * multiple different rand algorithms in CUDPP, cudppRandSeed 
 * uses \a planHandle to determine which seed to set.  Each rand 
 * algorithm has its own  unique set of seeds depending on what 
 * the algorithm needs.
 *
 * @param[in] planHandle the handle to the plan which specifies which rand seed to set
 * @param[in] seed the value which the internal cudpp seed will be set to
 * @returns CUDPPResult indicating success or error condition 
 */
CUDPP_DLL
CUDPPResult cudppRandSeed(const CUDPPHandle planHandle, 
                          unsigned int      seed)
{
    CUDPPRandPlan * plan = 
        (CUDPPRandPlan *) getPlanPtrFromHandle<CUDPPRandPlan>(planHandle);

    if (plan != NULL)
    {
        if (plan->m_config.algorithm != CUDPP_RAND_MD5)
            return CUDPP_ERROR_INVALID_PLAN;
        plan->m_seed = seed;
    }
    else
        return CUDPP_ERROR_INVALID_HANDLE;

    return CUDPP_SUCCESS;
}//end cudppRandSeed

/**
 * @brief Solves tridiagonal linear systems
 *
 * The solver uses a hybrid CR-PCR algorithm described in our papers "Fast
 * Fast Tridiagonal Solvers on the GPU" and "A Hybrid Method for Solving 
 * Tridiagonal Systems on the GPU". (See the \ref references bibliography).
 * Please refer to the papers for a complete description of the basic CR 
 * (Cyclic Reduction) and PCR (Parallel Cyclic Reduction) algorithms and their 
 * hybrid variants.
 *
 * - Both float and double data types are supported. 
 * - Both power-of-two and non-power-of-two system sizes are supported.
 * - The maximum system size could be limited by the maximum number of threads
 * of a CUDA block, the number of registers per multiprocessor, and the 
 * amount of shared memory available. For example, on the GTX 280 GPU, the 
 * maximum system size is 512 for the float datatype, and 256 for the double 
 * datatype, which is limited by the size of shared memory in this case. 
 * - The maximum number of systems is 65535, that is the maximum number of 
 * one-dimensional blocks that could be launched in a kernel call. Users could 
 * launch the kernel multiple times to solve more systems if required. 
 *
 * @param[out] d_x Solution vector
 * @param[in] planHandle Handle to plan for tridiagonal solver
 * @param[in] d_a Lower diagonal
 * @param[in] d_b Main diagonal
 * @param[in] d_c Upper diagonal
 * @param[in] d_d Right hand side
 * @param[in] systemSize The size of the linear system
 * @param[in] numSystems The number of systems to be solved
 * @returns CUDPPResult indicating success or error condition
 *
 * @see cudppPlan, CUDPPConfiguration, CUDPPAlgorithm
 */
CUDPP_DLL
CUDPPResult cudppTridiagonal(CUDPPHandle planHandle, 
                             void *d_a, 
                             void *d_b, 
                             void *d_c, 
                             void *d_d, 
                             void *d_x, 
                             int systemSize, 
                             int numSystems)
{   
    CUDPPTridiagonalPlan * plan = 
        (CUDPPTridiagonalPlan *) getPlanPtrFromHandle<CUDPPTridiagonalPlan>(planHandle);
    
    if(plan != NULL)
    {
        //dispatch the tridiagonal solver here
        return cudppTridiagonalDispatch(d_a, d_b, d_c, d_d, d_x, 
                                        systemSize, numSystems, plan);
    }
    else
        return CUDPP_ERROR_INVALID_HANDLE;
}

/** @} */ // end Algorithm Interface
/** @} */ // end of publicInterface group

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:

