// -------------------------------------------------------------
// CUDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision: 3572$
// $Date: 2007-11-19 13:58:06 +0000 (Mon, 19 Nov 2007) $
// ------------------------------------------------------------- 
// This source code is distributed under the terms of license.txt
// in the root directory of this source distribution.
// ------------------------------------------------------------- 
#ifndef __CUDPP_PLAN_H__
#define __CUDPP_PLAN_H__

typedef void* KernelPointer;
class CUDPPPlan;
class CUDPPManager;

#include "cudpp.h"

//! @internal Convert an opaque handle to a pointer to a plan
template <typename T>
T* getPlanPtrFromHandle(CUDPPHandle handle)
{
    return reinterpret_cast<T*>(handle);
}


/** @brief Base class for CUDPP Plan data structures
  *
  * CUDPPPlan and its subclasses provide the internal (i.e. not visible to the
  * library user) infrastructure for planning algorithm execution.  They 
  * own intermediate storage for CUDPP algorithms as well as, in some cases,
  * information about optimal execution configuration for the present hardware.
  * 
  */
class CUDPPPlan
{
public:
    CUDPPPlan(CUDPPManager *mgr, CUDPPConfiguration config, 
              size_t numElements, size_t numRows, size_t rowPitch);
    virtual ~CUDPPPlan() {}

    // Note anything passed to functions compiled by NVCC must be public
    CUDPPConfiguration m_config;        //!< @internal Options structure
    size_t             m_numElements;   //!< @internal Maximum number of input elements
    size_t             m_numRows;       //!< @internal Maximum number of input rows
    size_t             m_rowPitch;      //!< @internal Pitch of input rows in elements
    CUDPPManager      *m_planManager;  //!< @internal pointer to the manager of this plan
   
    //! @internal Convert this pointer to an opaque handle
    CUDPPHandle getHandle()
    {
        return reinterpret_cast<CUDPPHandle>(this);
    }
};

/** @brief Plan class for scan algorithm
  *
  */
class CUDPPScanPlan : public CUDPPPlan
{
public:
    CUDPPScanPlan(CUDPPManager *mgr, CUDPPConfiguration config, size_t numElements, size_t numRows, size_t rowPitch);
    virtual ~CUDPPScanPlan();

    void  **m_blockSums;          //!< @internal Intermediate block sums array
    size_t *m_rowPitches;         //!< @internal Pitch of each row in elements (for cudppMultiScan())
    size_t  m_numEltsAllocated;   //!< @internal Number of elements allocated (maximum scan size)
    size_t  m_numRowsAllocated;   //!< @internal Number of rows allocated (for cudppMultiScan())
    size_t  m_numLevelsAllocated; //!< @internal Number of levels allocaed (in _scanBlockSums)
};

/** @brief Plan class for segmented scan algorithm
*
*/
class CUDPPSegmentedScanPlan : public CUDPPPlan
{
public:
    CUDPPSegmentedScanPlan(CUDPPManager *mgr, CUDPPConfiguration config, size_t numElements);
    virtual ~CUDPPSegmentedScanPlan();

    void          **m_blockSums;          //!< @internal Intermediate block sums array
    unsigned int  **m_blockFlags;         //!< @internal Intermediate block flags array
    unsigned int  **m_blockIndices;       //!< @internal Intermediate block indices array
    size_t        m_numEltsAllocated;     //!< @internal Number of elements allocated (maximum scan size)
    size_t        m_numLevelsAllocated;   //!< @internal Number of levels allocaed (in _scanBlockSums)
};

/** @brief Plan class for compact algorithm
*
*/
class CUDPPCompactPlan : public CUDPPPlan
{
public:
    CUDPPCompactPlan(CUDPPManager *mgr, CUDPPConfiguration config, size_t numElements, size_t numRows, size_t rowPitch);
    virtual ~CUDPPCompactPlan();

    CUDPPScanPlan *m_scanPlan;         //!< @internal Compact performs a scan of type unsigned int using this plan
    unsigned int* m_d_outputIndices; //!< @internal Output address of compacted elements; this is the result of scan
    
};

/** @brief Plan class for reduce algorithm
*
*/
class CUDPPReducePlan : public CUDPPPlan
{
public:
    CUDPPReducePlan(CUDPPManager *mgr, CUDPPConfiguration config, size_t numElements);
    virtual ~CUDPPReducePlan();

    unsigned int m_threadsPerBlock;     //!< @internal number of threads to launch per block
    unsigned int m_maxBlocks;           //!< @internal maximum number of blocks to launch
    void         *m_blockSums;          //!< @internal Intermediate block sums array
};  

class CUDPPRadixSortPlan : public CUDPPPlan
{
public:
    CUDPPRadixSortPlan(CUDPPManager *mgr, CUDPPConfiguration config, size_t numElements);
    virtual ~CUDPPRadixSortPlan();
    bool           m_bKeysOnly;
    bool           m_bManualCoalesce;
    unsigned int   m_persistentCTAThreshold[2];
    unsigned int   m_persistentCTAThresholdFullBlocks[2];
    CUDPPScanPlan *m_scanPlan;        //!< @internal Sort performs a scan of type unsigned int using this plan
    unsigned int   m_keyBits;
    mutable void  *m_tempKeys;        //!< @internal Intermediate storage for keys
    mutable void  *m_tempValues;      //!< @internal Intermediate storage for values
    unsigned int  *m_counters;        //!< @internal Counter for each radix
    unsigned int  *m_countersSum;     //!< @internal Prefix sum of radix counters
    unsigned int  *m_blockOffsets;    //!< @internal Global offsets of each radix in each block

    enum RadixSortKernels
    {
        KERNEL_RSB_4_0_F_F_T,
        KERNEL_RSB_4_0_F_T_T,
        KERNEL_RSB_4_0_T_F_T,
        KERNEL_RSB_4_0_T_T_T,
        KERNEL_RSBKO_4_0_F_F_T,
        KERNEL_RSBKO_4_0_F_T_T,
        KERNEL_RSBKO_4_0_T_F_T,
        KERNEL_RSBKO_4_0_T_T_T,
        KERNEL_FRO_0_F_T,
        KERNEL_FRO_0_T_T,
        KERNEL_RD_0_F_F_F_T,
        KERNEL_RD_0_F_F_T_T,
        KERNEL_RD_0_F_T_F_T,
        KERNEL_RD_0_F_T_T_T,
        KERNEL_RD_0_T_F_F_T,
        KERNEL_RD_0_T_F_T_T,
        KERNEL_RD_0_T_T_F_T,
        KERNEL_RD_0_T_T_T_T,
        KERNEL_RDKO_0_F_F_F_T,
        KERNEL_RDKO_0_F_F_T_T,
        KERNEL_RDKO_0_F_T_F_T,
        KERNEL_RDKO_0_F_T_T_T,
        KERNEL_RDKO_0_T_F_F_T,
        KERNEL_RDKO_0_T_F_T_T,
        KERNEL_RDKO_0_T_T_F_T,
        KERNEL_RDKO_0_T_T_T_T,
        KERNEL_EK,
        NUM_KERNELS
    };
    unsigned int m_numCTAs[NUM_KERNELS];

};

/** @brief Plan class for sparse-matrix dense-vector multiply
*
*/
class CUDPPSparseMatrixVectorMultiplyPlan : public CUDPPPlan
{
public:
    CUDPPSparseMatrixVectorMultiplyPlan(CUDPPManager *mgr, 
                                        CUDPPConfiguration config, size_t numNZElts,
                                        const void         *A,
                                        const unsigned int *rowindx, 
                                        const unsigned int *indx, size_t numRows);
    virtual ~CUDPPSparseMatrixVectorMultiplyPlan();

    CUDPPSegmentedScanPlan *m_segmentedScanPlan; //!< @internal Performs a segmented scan of type T using this plan
    void             *m_d_prod;  //!< @internal Vector of products (of an element in A and its corresponding (thats is
                                 //!            belongs to the same row) element in x; this is the input and output of 
                                 //!            segmented scan
    unsigned int     *m_d_flags; //!< @internal Vector of flags where a flag is set if an element of A is the first element
                                 //!            of its row; this is the flags vector for segmented scan
    unsigned int     *m_d_rowFinalIndex; //!< @internal Vector of row end indices, which for each row specifies an index in A
                                         //!            which is the last element of that row. Resides in GPU memory. 
    unsigned int     *m_d_rowIndex; //!< @internal Vector of row end indices, which for each row specifies an index in A
                                    //!            which is the first element of that row. Resides in GPU memory. 
    unsigned int     *m_d_index;    //!<@internal Vector of column numbers one for each element in A 
    void             *m_d_A;        //!<@internal The A matrix 
    unsigned int     *m_rowFinalIndex; //!< @internal Vector of row end indices, which for each row specifies an index in A
                                       //!            which is the last element of that row. Resides in CPU memory.
    size_t           m_numRows; //!< Number of rows
    size_t           m_numNonZeroElements; //!<Number of non-zero elements
};

/** @brief Plan class for random number generator
*
*/
class CUDPPRandPlan : public CUDPPPlan
{
public:
    CUDPPRandPlan(CUDPPManager *mgr, CUDPPConfiguration config, size_t num_elements);

    unsigned int m_seed; //!< @internal the seed for the random number generator
};

/** @brief Plan class for tridiagonal solver
*
*/
class CUDPPTridiagonalPlan : public CUDPPPlan
{
public:
    CUDPPTridiagonalPlan(CUDPPManager *mgr, CUDPPConfiguration config);
};

#endif // __CUDPP_PLAN_H__
