// -------------------------------------------------------------
// cuDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision: 3572$
// $Date: 2007-11-19 13:58:06 +0000 (Mon, 19 Nov 2007) $
// ------------------------------------------------------------- 
// This source code is distributed under the terms of license.txt
// in the root directory of this source distribution.
// ------------------------------------------------------------- 

#include "cudpp.h"
#include "cudpp_plan_manager.h"
#include "cudpp_scan.h"
#include "cudpp_segscan.h"
#include "cudpp_compact.h"
#include "cudpp_spmvmult.h"
#include "cudpp_vgraph.h"
#include "cudpp_radixsort.h"

#include <assert.h>

CUDPPPlanManager* CUDPPPlanManager::m_instance = NULL;

CUDPPResult validateOptions(CUDPPConfiguration config, size_t /*numElements*/, size_t numRows, size_t /*rowPitch*/)
{
    CUDPPResult ret = CUDPP_SUCCESS;
    if ((config.options & CUDPP_OPTION_BACKWARD) && (config.options & CUDPP_OPTION_FORWARD))
        ret = CUDPP_ERROR_ILLEGAL_CONFIGURATION;
    if ((config.options & CUDPP_OPTION_EXCLUSIVE) && (config.options & CUDPP_OPTION_INCLUSIVE))
        ret = CUDPP_ERROR_ILLEGAL_CONFIGURATION;

    if (config.algorithm == CUDPP_COMPACT && numRows > 1)
        ret = CUDPP_ERROR_ILLEGAL_CONFIGURATION; //!< @todo: add support for multi-row cudppCompact

    return ret;
}

/** @addtogroup publicInterface
  * @{
  */

/** @name Plan Interface
 * @{
 */


/** @brief Create a CUDPP plan 
  * 
  * A plan is a data structure containing state and intermediate storage space
  * that CUDPP uses to execute algorithms on data.  A plan is created by 
  * passing to cudppPlan() a CUDPPConfiguration that specifies the algorithm,
  * operator, datatype, and options.  The size of the data must also be passed
  * to cudppPlan(), in the \a numElements, \a numRows, and \a rowPitch 
  * arguments.  These sizes are used to allocate internal storage space at the
  * time the plan is created.  The CUDPP planner may use the sizes, options,
  * and information about the present hardware to choose optimal settings.
  *
  * Note that \a numElements is the maximum size of the array to be processed
  * with this plan.  That means that a plan may be re-used to process (for 
  * example, to sort or scan) smaller arrays.  
  * 
  * @param[out] planHandle A pointer to an opaque handle to the internal plan
  * @param[in]  config The configuration struct specifying algorithm and options
  * @param[in]  numElements The maximum number of elements to be processed
  * @param[in]  numRows The number of rows (for 2D operations) to be processed
  * @param[in]  rowPitch The pitch of the rows of input data, in elements
  */
CUDPP_DLL
CUDPPResult cudppPlan(CUDPPHandle        *planHandle, 
                      CUDPPConfiguration config, 
                      size_t             numElements, 
                      size_t             numRows, 
                      size_t             rowPitch)
{
    CUDPPResult result = CUDPP_SUCCESS;

    CUDPPPlan *plan;

    result = validateOptions(config, numElements, numRows, rowPitch);
    if (result != CUDPP_SUCCESS)
    {
        *planHandle = CUDPP_INVALID_HANDLE;
        return result;
    }

    switch (config.algorithm)
    {
    case CUDPP_SCAN:
        {
            plan = new CUDPPScanPlan(config, numElements, numRows, rowPitch);
            break;
        }
    case CUDPP_COMPACT:
        {
            plan = new CUDPPCompactPlan(config, numElements, numRows, rowPitch);
            break;
        }
    case CUDPP_SORT_RADIX:
    //case CUDPP_SORT_RADIX_GLOBAL:
        {
            plan = new CUDPPRadixSortPlan(config, numElements);
            break;
        }
    case CUDPP_SEGMENTED_SCAN:
        {
            plan = new CUDPPSegmentedScanPlan(config, numElements);
            break;
        }
    //new rand plan
    case CUDPP_RAND_MD5:
        {
            plan = new CUDPPRandPlan(config, numElements);
            break;
        }
    case CUDPP_REDUCE:
    default:
        //! @todo: implement cudppReduce()
        return CUDPP_ERROR_ILLEGAL_CONFIGURATION; 
        break;
    }

    *planHandle = CUDPPPlanManager::AddPlan(plan);
    if (CUDPP_INVALID_HANDLE == *planHandle)
        return CUDPP_ERROR_UNKNOWN;
    else
        return CUDPP_SUCCESS;
}

/** @brief Destroy a CUDPP Plan
  *
  * Deletes the plan referred to by \a planHandle and all associated internal
  * storage.
  * 
  * @param[in] planHandle The CUDPPHandle to the plan to be destroyed
  */
CUDPP_DLL
CUDPPResult cudppDestroyPlan(CUDPPHandle planHandle)
{
    if (CUDPPPlanManager::RemovePlan(planHandle) == false)
        return CUDPP_ERROR_INVALID_HANDLE;
    else
        return CUDPP_SUCCESS;
}

/** @brief Create a CUDPP Sparse Matrix Object 
  *
  * The sparse matrix plan is a data structure containing state and intermediate storage space
  * that CUDPP uses to perform sparse matrix dense vector multiply.  This plan is created by 
  * passing to CUDPPSparseMatrixVectorMultiplyPlan() a CUDPPConfiguration that specifies the 
  * algorithm (sprarse matrix-dense vector multiply) and datatype, along with the sparse matrix
  * itself in CSR format.  The number of non-zero elements in the sparse matrix must also be passed
  * as \a numNonZeroElements. This is used to allocate internal storage space at the time the 
  * sparse matrix plan is created.
  *
  * @param[out] sparseMatrixHandle A pointer to an opaque handle to the sparse matrix object
  * @param[in]  config The configuration struct specifying algorithm and options
  * @param[in]  numNonZeroElements The number of non zero elements in the sparse matrix 
  * @param[in]  numRows This is the number of rows in y, x and A for y = A * x
  * @param[in]  A The matrix data
  * @param[in]  h_rowIndices An array containing the index of the start of each row in \a A
  * @param[in]  h_indices An array containing the index of each nonzero element in \a A
  */
CUDPP_DLL
CUDPPResult cudppSparseMatrix(CUDPPHandle        *sparseMatrixHandle, 
                              CUDPPConfiguration config, 
                              size_t             numNonZeroElements, 
                              size_t             numRows, 
                              const void         *A,
                              const unsigned int *h_rowIndices,
                              const unsigned int *h_indices)
{
    CUDPPResult result = CUDPP_SUCCESS;

    CUDPPPlan *sparseMatrix;

    if ((config.algorithm != CUDPP_SPMVMULT) || 
        (numNonZeroElements <= 0) || (numRows <= 0))
    {
        result = CUDPP_ERROR_ILLEGAL_CONFIGURATION;
    }

    if (result != CUDPP_SUCCESS)
    {
        *sparseMatrixHandle = CUDPP_INVALID_HANDLE;
        return result;
    }

    sparseMatrix = 
        new CUDPPSparseMatrixVectorMultiplyPlan(config, numNonZeroElements, A, 
                                                h_rowIndices, h_indices, numRows);

    *sparseMatrixHandle = CUDPPPlanManager::AddPlan(sparseMatrix);
    if (CUDPP_INVALID_HANDLE == *sparseMatrixHandle)
        return CUDPP_ERROR_UNKNOWN;
    else
        return CUDPP_SUCCESS;
}

/** @brief Destroy a CUDPP Sparse Matrix Object
  *
  * Deletes the sparse matrix data and plan referred to by \a sparseMatrixHandle 
  * and all associated internal storage.
  * 
  * @param[in] sparseMatrixHandle The CUDPPHandle to the matrix object to be destroyed
  */
CUDPP_DLL
CUDPPResult cudppDestroySparseMatrix(CUDPPHandle sparseMatrixHandle)
{
    return cudppDestroyPlan(sparseMatrixHandle);
}

/** @brief Initialize plan for v-graph data structure handle */
CUDPP_DLL
CUDPPResult cudppVGraph(CUDPPHandle        *vGraphHandle, 
                        CUDPPConfiguration config, 
                        size_t             num_nodes, 
                        size_t             num_edges, 
                        const unsigned int * segment_descriptor, // size: num_nodes
                        const unsigned int * cross_pointers,     // size: 2 * num_edges
                        const unsigned int * head_flags,         // size: 2 * num_edges
                        const float * weights)                   // size: 2 * num_edges
{
    CUDPPResult result = CUDPP_SUCCESS;

    CUDPPVGraphPlan *vGraph;

    if (result != CUDPP_SUCCESS)
    {
        *vGraphHandle = CUDPP_INVALID_HANDLE;
        return result;
    }

    vGraph = 
        new CUDPPVGraphPlan(config, num_nodes, num_edges, segment_descriptor, 
                            cross_pointers, head_flags, weights);

    *vGraphHandle = CUDPPPlanManager::AddPlan(vGraph);
    if (CUDPP_INVALID_HANDLE == *vGraphHandle)
        return CUDPP_ERROR_UNKNOWN;
    else
        return CUDPP_SUCCESS;
}

/** @brief Initialize plan for v-graph neighbor-reduce algorithm handle */
CUDPP_DLL
CUDPPResult cudppVGraphNRPlan(CUDPPHandle        *vGraphNRHandle, 
                              CUDPPConfiguration config, 
                              size_t             num_nodes, 
                              size_t             num_edges)
{
    CUDPPResult result = CUDPP_SUCCESS;

    CUDPPVGraphNRPlan *vGraphNR;

//     if ((config.algorithm != CUDPP_SPMVMULT) || 
//         (numNonZeroElements <= 0) || (numRows <= 0))
//     {
//         result = CUDPP_ERROR_ILLEGAL_CONFIGURATION;
//     }

    if (result != CUDPP_SUCCESS)
    {
        *vGraphNRHandle = CUDPP_INVALID_HANDLE;
        return result;
    }

    vGraphNR = new CUDPPVGraphNRPlan(config, num_nodes, num_edges);

    *vGraphNRHandle = CUDPPPlanManager::AddPlan(vGraphNR);
    if (CUDPP_INVALID_HANDLE == *vGraphNRHandle)
        return CUDPP_ERROR_UNKNOWN;
    else
        return CUDPP_SUCCESS;
}

/** @brief Initialize plan for v-graph distribute-excess algorithm handle */
CUDPP_DLL
CUDPPResult cudppVGraphDEPlan(CUDPPHandle        *vGraphDEHandle, 
                              CUDPPConfiguration config, 
                              size_t             num_nodes, 
                              size_t             num_edges)
{
    CUDPPResult result = CUDPP_SUCCESS;

    CUDPPVGraphDEPlan *vGraphDE;

    if (result != CUDPP_SUCCESS)
    {
        *vGraphDEHandle = CUDPP_INVALID_HANDLE;
        return result;
    }

    vGraphDE = new CUDPPVGraphDEPlan(config, num_nodes, num_edges);

    *vGraphDEHandle = CUDPPPlanManager::AddPlan(vGraphDE);
    if (CUDPP_INVALID_HANDLE == *vGraphDEHandle)
        return CUDPP_ERROR_UNKNOWN;
    else
        return CUDPP_SUCCESS;
}

/** @brief Initialize plan for v-graph minimum-spanning-tree algorithm handle */
CUDPP_DLL
CUDPPResult cudppVGraphMSTPlan(CUDPPHandle        *vGraphMSTHandle, 
                               CUDPPConfiguration config, 
                               size_t             num_nodes, 
                               size_t             num_edges)
{
    CUDPPResult result = CUDPP_SUCCESS;

    CUDPPVGraphMSTPlan *vGraphMST;

    if (result != CUDPP_SUCCESS)
    {
        *vGraphMSTHandle = CUDPP_INVALID_HANDLE;
        return result;
    }

    vGraphMST = new CUDPPVGraphMSTPlan(config, num_nodes, num_edges);

    *vGraphMSTHandle = CUDPPPlanManager::AddPlan(vGraphMST);
    if (CUDPP_INVALID_HANDLE == *vGraphMSTHandle)
        return CUDPP_ERROR_UNKNOWN;
    else
        return CUDPP_SUCCESS;
}

CUDPP_DLL
CUDPPResult cudppSetVGTemps(CUDPPHandle vGraphHandle,
                            unsigned int * temp,
                            unsigned int * temp2)
{
    CUDPPResult result = CUDPP_SUCCESS;

    CUDPPVGraphPlan *vGraph = 
        (CUDPPVGraphPlan*) CUDPPPlanManager::GetPlan(vGraphHandle);

    vGraph->m_d_temp = temp;
    vGraph->m_d_temp2 = temp2;
    
    return result;
}

CUDPP_DLL
CUDPPResult cudppSetVGMSTTemps(CUDPPHandle vGraphHandle,
                               unsigned int * temp,
                               unsigned int * temp2)
{
    CUDPPResult result = CUDPP_SUCCESS;

    CUDPPVGraphMSTPlan *vGraphMST = 
        (CUDPPVGraphMSTPlan*) CUDPPPlanManager::GetPlan(vGraphHandle);

    vGraphMST->m_d_temp = temp;
    vGraphMST->m_d_temp2 = temp2;
    
    return result;
}

CUDPP_DLL
CUDPPResult cudppDestroyVGraph(CUDPPHandle vGraphHandle)
{
    return cudppDestroyPlan(vGraphHandle);
}

CUDPP_DLL
CUDPPResult cudppDestroyVGraphNRPlan(CUDPPHandle vGraphNRHandle)
{
    return cudppDestroyPlan(vGraphNRHandle);
}

CUDPP_DLL
CUDPPResult cudppDestroyVGraphDEPlan(CUDPPHandle vGraphDEHandle)
{
    return cudppDestroyPlan(vGraphDEHandle);
}

CUDPP_DLL
CUDPPResult cudppDestroyVGraphMSTPlan(CUDPPHandle vGraphMSTHandle)
{
    return cudppDestroyPlan(vGraphMSTHandle);
}


/** @} */ // end Plan Interface
/** @} */ // end publicInterface


/** @brief Plan base class constructor
  * 
  * @param[in]  config The configuration struct specifying algorithm and options
  * @param[in]  numElements The maximum number of elements to be processed
  * @param[in]  numRows The number of rows (for 2D operations) to be processed
  * @param[in]  rowPitch The pitch of the rows of input data, in elements
  */
CUDPPPlan::CUDPPPlan(CUDPPConfiguration config, 
                     size_t numElements, 
                     size_t numRows, 
                     size_t rowPitch)
: m_config(config),
  m_numElements(numElements),
  m_numRows(numRows),
  m_rowPitch(rowPitch)
{
}

/** @brief Scan Plan constructor
* 
* @param[in]  config The configuration struct specifying algorithm and options
* @param[in]  numElements The maximum number of elements to be scanned
* @param[in]  numRows The maximum number of rows (for 2D operations) to be scanned
* @param[in]  rowPitch The pitch of the rows of input data, in elements
*/
CUDPPScanPlan::CUDPPScanPlan(CUDPPConfiguration config, 
                             size_t numElements, 
                             size_t numRows, 
                             size_t rowPitch)
: CUDPPPlan(config, numElements, numRows, rowPitch),
  m_blockSums(0),
  m_rowPitches(0),
  m_numEltsAllocated(0),
  m_numRowsAllocated(0),
  m_numLevelsAllocated(0)
{
    allocScanStorage(this);
}

/** @brief CUDPP scan plan destructor */
CUDPPScanPlan::~CUDPPScanPlan()
{
    freeScanStorage(this);
}

/** @brief SegmentedScan Plan constructor
* 
* @param[in]  config The configuration struct specifying options
* @param[in]  numElements The maximum number of elements to be scanned
*/
CUDPPSegmentedScanPlan::CUDPPSegmentedScanPlan(CUDPPConfiguration config, 
                                               size_t numElements)
: CUDPPPlan(config, numElements, 1, 0),
  m_blockSums(0),
  m_blockFlags(0),
  m_blockIndices(0),
  m_numEltsAllocated(0),
  m_numLevelsAllocated(0)
{
    allocSegmentedScanStorage(this);
}

/** @brief SegmentedScan plan destructor */
CUDPPSegmentedScanPlan::~CUDPPSegmentedScanPlan()
{
    freeSegmentedScanStorage(this);
}

/** @brief Compact Plan constructor
* 
* @param[in]  config The configuration struct specifying options
* @param[in]  numElements The maximum number of elements to be compacted
* @param[in]  numRows The number of rows (for 2D operations) to be compacted
* @param[in]  rowPitch The pitch of the rows of input data, in elements
*/
CUDPPCompactPlan::CUDPPCompactPlan(CUDPPConfiguration config, 
                                   size_t numElements, 
                                   size_t numRows, 
                                   size_t rowPitch)
: CUDPPPlan(config, numElements, numRows, rowPitch),
  m_d_outputIndices(0)
{
    assert(numRows == 1); //!< @todo Add support for multirow compaction

    CUDPPConfiguration scanConfig = 
    { 
      CUDPP_SCAN, 
      CUDPP_ADD, 
      CUDPP_UINT, 
      (config.options & CUDPP_OPTION_BACKWARD) ? 
        CUDPP_OPTION_BACKWARD | CUDPP_OPTION_EXCLUSIVE : 
        CUDPP_OPTION_FORWARD  | CUDPP_OPTION_EXCLUSIVE 
    };
    m_scanPlan = new CUDPPScanPlan(scanConfig, numElements, numRows, rowPitch);

    allocCompactStorage(this);
}

/** @brief Compact plan destructor */
CUDPPCompactPlan::~CUDPPCompactPlan()
{
    delete m_scanPlan;
    freeCompactStorage(this);
}

/** @brief Sort Plan constructor
* 
* @param[in]  config The configuration struct specifying algorithm and options
* @param[in]  numElements The maximum number of elements to be sorted
*/
/*CUDPPSortPlan::CUDPPSortPlan(CUDPPConfiguration config, size_t numElements)
: CUDPPPlan(config, numElements, 1, 0),
  m_scanPlan(0),
  m_d_temp(0),
  m_d_tempAddress(0)
{
    CUDPPConfiguration scanConfig = 
    { 
      CUDPP_SCAN, 
      CUDPP_ADD, 
      CUDPP_UINT, 
      CUDPP_OPTION_FORWARD | CUDPP_OPTION_EXCLUSIVE 
    };

    //if (config.algorithm == CUDPP_SORT_RADIX_GLOBAL)
    {
        m_scanPlan = new CUDPPScanPlan(scanConfig, numElements, 1, 0);
    }

    allocSortStorage(this);
}*/

/** @brief Sort plan destructor */
/*CUDPPSortPlan::~CUDPPSortPlan()
{
    delete m_scanPlan;
    freeSortStorage(this);
}*/

CUDPPRadixSortPlan::CUDPPRadixSortPlan(CUDPPConfiguration config, size_t numElements)
: CUDPPPlan(config, numElements, 1, 0),
  m_scanPlan(0),
  m_tempKeys(0),    
  m_tempValues(0),
  m_counters(0),
  m_countersSum(0),
  m_blockOffsets(0) 
{
    size_t numBlocks2 = ((numElements % (SORT_CTA_SIZE * 2)) == 0) ?
            (numElements / (SORT_CTA_SIZE * 2)) : (numElements / (SORT_CTA_SIZE * 2) + 1);

    CUDPPConfiguration scanConfig = 
    { 
      CUDPP_SCAN, 
      CUDPP_ADD, 
      CUDPP_UINT, 
      CUDPP_OPTION_FORWARD | CUDPP_OPTION_EXCLUSIVE 
    };    

    if(m_config.options == CUDPP_OPTION_KEYS_ONLY)
        m_bKeysOnly = true;
    else
        m_bKeysOnly = false;

    m_scanPlan = new CUDPPScanPlan(scanConfig, numBlocks2*16, 1, 0);    
        
    allocRadixSortStorage(this); 
}

CUDPPRadixSortPlan::~CUDPPRadixSortPlan()
{
    delete m_scanPlan;
    freeRadixSortStorage(this);
}

/** @brief SparseMatrixVectorMultiply Plan constructor
* 
* @param[in]  config The configuration struct specifying options
* @param[in]  numNonZeroElements The number of non-zero elements in sparse matrix
* @param[in]  A Array of non-zero matrix elements
* @param[in]  rowIndex Array of indices of the first element of each row 
*                     in the "flattened" version of the sparse matrix
* @param[in]  index Array of indices of non-zero elements in the matrix
* @param[in]  numRows The number of rows in the sparse matrix
*/
CUDPPSparseMatrixVectorMultiplyPlan::CUDPPSparseMatrixVectorMultiplyPlan(
                                                                         CUDPPConfiguration config,
                                                                         size_t             numNonZeroElements,
                                                                         const void         *A,
                                                                         const unsigned int *rowIndex,
                                                                         const unsigned int *index,
                                                                         size_t             numRows
                                                                         )
: CUDPPPlan(config, numNonZeroElements, 1, 0),
  m_segmentedScanPlan(0),
  m_d_prod(0),
  m_d_flags(0),
  m_d_rowFinalIndex(0),
  m_rowFinalIndex(0),
  m_numRows(numRows),
  m_numNonZeroElements(numNonZeroElements)  
{
    CUDPPConfiguration segScanConfig = 
    { 
      CUDPP_SEGMENTED_SCAN, 
      CUDPP_ADD, 
      config.datatype, 
      (CUDPP_OPTION_FORWARD | CUDPP_OPTION_INCLUSIVE) 
    };
    m_segmentedScanPlan = new CUDPPSegmentedScanPlan(segScanConfig, m_numNonZeroElements);

    // Generate an array of the indices of the last element of each row
    // in the "flattened" version of the sparse matrix
    m_rowFinalIndex = new unsigned int [m_numRows];
    for (unsigned int i=0; i < m_numRows; ++i)
    {
        if (i < m_numRows-1)
            m_rowFinalIndex[i] = rowIndex[i+1];
        else
            m_rowFinalIndex[i] = (unsigned int)numNonZeroElements;
    }

    allocSparseMatrixVectorMultiplyStorage(this, A, rowIndex, index);
}

/** @brief Sparse matrix-vector plan destructor */
CUDPPSparseMatrixVectorMultiplyPlan::~CUDPPSparseMatrixVectorMultiplyPlan()
{
    freeSparseMatrixVectorMultiplyStorage(this);
    delete m_segmentedScanPlan;
    delete [] m_rowFinalIndex;
}

/** @brief v-graph Plan constructor
* 
* @param[in]  config The configuration struct specifying options
* @param[in]  num_nodes The number of nodes (vertices) in the graph
* @param[in]  num_edges The number of edges the graph
* @param[in]  h_segment_descriptor host-side segment descriptor (number of edges per node/vertex)
* @param[in]  h_cross_pointers host-side pointers per edge pointing to the edge's other side
* @param[in]  h_head_flags host-side array per edge noting whether it's the first edge of the vertex (1) or not (0)
* @param[in]  h_weights host-side array per edge noting weight of that edge
*/

CUDPPVGraphPlan::
CUDPPVGraphPlan(CUDPPConfiguration config, 
                size_t             num_nodes, 
                size_t             num_edges, 
                const unsigned int * h_segment_descriptor, // size: num_nodes
                const unsigned int * h_cross_pointers,     // size: 2*num_edges
                const unsigned int * h_head_flags,         // size: 2*num_edges
                const float *        h_weights)            // size: 2*num_edges
    : CUDPPPlan(config, 2 * num_edges, 1, 0),
      m_num_nodes(num_nodes),
      m_num_edges(num_edges),
      m_d_segment_descriptor(NULL),
      m_d_cross_pointers(NULL),
      m_d_head_flags(NULL),
      m_d_weights(NULL),
      m_d_temp((void *) NULL),
      m_d_temp2((void *) NULL)
{
    initializeVGraphStorage(this, h_segment_descriptor, h_cross_pointers, 
                            h_head_flags, h_weights);
}

/* @brief Constructor for the neighbor-reduce plan data structure */
CUDPPVGraphNRPlan::
CUDPPVGraphNRPlan(CUDPPConfiguration config, 
                  size_t             num_nodes, 
                  size_t             num_edges)
    : CUDPPPlan(config, 2 * num_edges, 1, 0),
      m_num_nodes(num_nodes),
      m_num_edges(num_edges)
{
    CUDPPConfiguration scanConfig = 
        { 
            CUDPP_SCAN, 
            CUDPP_ADD, 
            CUDPP_INT,
            (CUDPP_OPTION_FORWARD | CUDPP_OPTION_INCLUSIVE) 
        };
    m_scanPlan = new CUDPPScanPlan(scanConfig, 2 * m_num_edges, 1, 0);

    CUDPPConfiguration segScanConfig = 
        { 
            CUDPP_SEGMENTED_SCAN, 
            CUDPP_ADD, 
            config.datatype, 
            (CUDPP_OPTION_FORWARD | CUDPP_OPTION_INCLUSIVE) 
        };
    m_segmentedScanPlan = 
        new CUDPPSegmentedScanPlan(segScanConfig, 2 * m_num_edges);
}

/* @brief Constructor for the v-graph distribute-excess plan data structure */
CUDPPVGraphDEPlan::
CUDPPVGraphDEPlan(CUDPPConfiguration config, 
                  size_t             num_nodes, 
                  size_t             num_edges)
    : CUDPPPlan(config, 2 * num_edges, 1, 0),
      m_num_nodes(num_nodes),
      m_num_edges(num_edges)
{
    CUDPPConfiguration scanConfig = 
        { 
            CUDPP_SCAN, 
            CUDPP_ADD, 
            CUDPP_INT,
            (CUDPP_OPTION_FORWARD | CUDPP_OPTION_INCLUSIVE) 
        };
    m_scanPlan = new CUDPPScanPlan(scanConfig, 2 * m_num_edges, 1, 0);

    CUDPPConfiguration segScanConfig = 
        { 
            CUDPP_SEGMENTED_SCAN, 
            CUDPP_ADD, 
            config.datatype, 
            (CUDPP_OPTION_FORWARD | CUDPP_OPTION_EXCLUSIVE) 
        };
    m_segmentedScanPlan = 
        new CUDPPSegmentedScanPlan(segScanConfig, 2 * m_num_edges);
}

/* @brief Constructor for the minimum-spanning-tree plan data structure */
CUDPPVGraphMSTPlan::
CUDPPVGraphMSTPlan(CUDPPConfiguration config, 
                   size_t             num_nodes, 
                   size_t             num_edges)
    : CUDPPPlan(config, 2 * num_edges, 1, 0),
      m_num_nodes(num_nodes),
      m_num_edges(num_edges)
{
    CUDPPConfiguration scanConfig = 
        { 
            CUDPP_SCAN, 
            CUDPP_ADD, 
            CUDPP_INT,
            (CUDPP_OPTION_FORWARD | CUDPP_OPTION_INCLUSIVE) 
        };
    m_scanPlan = new CUDPPScanPlan(scanConfig, 2 * m_num_edges, 1, 0);

    CUDPPConfiguration segScanConfig = 
        { 
            CUDPP_SEGMENTED_SCAN, 
            CUDPP_ADD, 
            config.datatype, 
            (CUDPP_OPTION_FORWARD | CUDPP_OPTION_INCLUSIVE) 
        };
    m_segmentedScanPlan = 
        new CUDPPSegmentedScanPlan(segScanConfig, 2 * m_num_edges);

    initializeVGraphMSTStorage(this);
}

/** @brief v-graph data structure plan destructor */
CUDPPVGraphPlan::~CUDPPVGraphPlan()
{
    freeVGraphStorage(this);
}


/** @brief v-graph neighbor-reduce algorithm plan destructor */
CUDPPVGraphNRPlan::~CUDPPVGraphNRPlan()
{
    freeVGraphNRStorage(this);
    delete m_scanPlan;
    delete m_segmentedScanPlan;
}

/** @brief v-graph distribute-excess algorithm plan destructor */
CUDPPVGraphDEPlan::~CUDPPVGraphDEPlan()
{
    freeVGraphDEStorage(this);
    delete m_scanPlan;
    delete m_segmentedScanPlan;
}

/** @brief v-graph minimum-spanning-tree algorithm plan destructor */
CUDPPVGraphMSTPlan::~CUDPPVGraphMSTPlan()
{
    freeVGraphMSTStorage(this);
    delete m_scanPlan;
    delete m_segmentedScanPlan;
}

/** @brief CUDPP Rand Plan Constructor
  * @param[in] config The configuration struct specifying options
  * @param[in] num_elements The number of elements to generate random bits for
  */
CUDPPRandPlan::CUDPPRandPlan(CUDPPConfiguration config, size_t num_elements) 
 : CUDPPPlan(config, num_elements, 1, 0),
   m_seed(0)
{
    
}

