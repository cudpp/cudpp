// ***************************************************************
//  cuDPP -- CUDA Data Parallel Primitives library
//  -------------------------------------------------------------
//  $Source$
//  $Revision: $
//  $Date: $
//  -------------------------------------------------------------
// This source code is distributed under the terms of license.txt in
// the root directory of this source distribution.
// ------------------------------------------------------------- 

/**
 * @file
 * vgraph_app.cu
 *
 * @brief CUDPP application-level vgraph routines
 */

/** \defgroup cudpp_app CUDPP Application-Level API
  * The CUDPP Application-Level API contains functions
  * that run on the host CPU and invoke GPU routines in 
  * the CUDPP \link cudpp_kernel Kernel-Level API\endlink. 
  * Application-Level API functions are used by
  * CUDPP \link publicInterface Public Interface\endlink
  * functions to implement CUDPP's core functionality.
  * @{
  */
#include "cudpp.h"
#include "cudpp_util.h"
#include "cudpp_plan.h"
#include "cudpp_globals.h"
#include "cudpp_scan.h"
#include "cudpp_segscan.h"
#include "cudpp_vgraph.h"
#include "kernel/vgraph_kernel.cu"

#include <cutil.h>
#include <cstdlib>
#include <cstdio>
#include <assert.h>

/** @name v-graph Functions
 * @{
 */
/* use soa not aos */

// debug only
template<class T>
void printDebugF(char * str, T * d, int len)
{
#ifdef __DEVICE_EMULATION__
    printf("%s:", str);
    for (int i = 0; i < len; i++)
    {
        printf(" %f", d[i]);
    }
    printf("\n");
    fflush(stdout);
#endif
}

template<class T>
void printDebugI(char * str, T * d, int len)
{
#ifdef __DEVICE_EMULATION__
    printf("%s:", str);
    for (int i = 0; i < len; i++)
    {
        printf(" %d", d[i]);
    }
    printf("\n");
    fflush(stdout);
#endif
}

void printDebug(char * str)
{
    printDebugI(str, (int *) NULL, 0);
}

void coredump()
{
    int * p = (int *) NULL;
    p[0] = 'x';
}

/** @brief At each vertex in a v-graph graph representation, reduce
 * the values of all vertices connected to that vertex.
 *
 * This function performs a reduction at each vertex of a graph. In
 * this graph, each vertex has a value. In this function, each vertex
 * reduces the values of all vertices to which it is connected in the
 * graph.
 *
 * As a small example, consider the graph 1-2-4-8. Using addition as
 * the reduction operation, this function would output the values
 * 2-5-10-4.
 * 
 * Computationally, this can be characterized as a sparse matrix (the
 * sparse 0-1 edge matrix) multiplied by a vertex value vector. This
 * function is computationally equivalent, though this function has
 * more packing and unpacking operations. 
 * 
 * We perform this computation in four steps:
 * 
 * -# Copy the index of each vertex across all the edges attached to
 * that vertex.
 *
 * -# For each edge, copy that vertex index to the vertex on the
 * other end of the edge, and fetch the vertex value at that edge.
 * 
 * -# Reduce the values at each vertex.
 *
 * -# Pack the resulting reduced values per vertex back into a vertex
 * vector.
 *
 * This does not work with non-fully-connected graphs. It is tested
 * with random graphs of fairly large sizes.
 * 
 * @param[in] plan Pointer to the CUDPPVGraphPlan object which stores
 * the v-graph data structure
 * @param[in] nrplan Pointer to the CUDPPVGraphNRPlan object which stores
 * the configuration needed by this algorithm
 * @param[out] d_out The per-vertex output array, where each vertex
 * V's output is the reduction of the values at all vertices incident
 * to V.
 * @param[in] d_idata The per-vertex input value array
 */
template <class T, CUDPPOperator op>
void vgNeighborReduce(const CUDPPVGraphPlan *plan,
                      const CUDPPVGraphNRPlan *nrplan,
                      T * d_out, const T * d_idata)
{
    /* idata, out are allocated per-node */
    /* temps (within plan) are allocated (2 *) per-edge */

    unsigned int numNodesBlocks = 
        max(1, (int)ceil((double)plan->m_num_nodes / 
                         ((double)VGNEIGHBORREDUCE_ELTS_PER_THREAD * 
                          CTA_SIZE)));

#if 0
    /* don't need this */
    bool nodesFillBlock = 
        (plan->m_num_nodes == 
         (numNodesBlocks * VGNEIGHBORREDUCE_ELTS_PER_THREAD * CTA_SIZE));  
#endif

    dim3  gridNodes(max(1, numNodesBlocks), 1, 1);

    unsigned int num2EdgeBlocks = 
        max(1,
            (int)ceil(2.0 * (double)plan->m_num_edges / 
                      ((double)VGNEIGHBORREDUCE_ELTS_PER_THREAD * CTA_SIZE)));

    dim3  grid2Edges(max(1, num2EdgeBlocks), 1, 1);

    dim3  threads(CTA_SIZE, 1, 1);

    /* Step 1: segmented copy of vertex indices to each edge incident
     * to that vertex. This is a segmented distribute operation, using
     * head_flags to indicate segment boundaries. */

    /* implemented with an inclusive scan of head flags rather than
     * {exclusive scan, inclusive segmented-scan}. The side effect is
     * that ***the resulting values are 1 larger than they should
     * be***. We clean this up in later steps. */

    cudppScanDispatch(plan->m_d_temp, plan->m_d_head_flags,
                      2 * plan->m_num_edges, 1, nrplan->m_scanPlan);
    /* in output of this call, temp is 1 too big */

    /* Step 2: First fetch the vertex value at each edge, then permute
     * that value to the opposite vertex at that edge (using
     * cross_pointers). Note that the indices in "temp" are one too
     * big; the kernel makes the proper correction. */

    gatherAndPermuteMinus1<<<grid2Edges, threads>>>
        ((T *) plan->m_d_temp2,
         (unsigned int *) plan->m_d_temp, 
         plan->m_d_cross_pointers,
         d_idata,
         2 * plan->m_num_edges);

    /* Step 3: use segmented scan (inclusive, forward) to compute
     * the segmented reduction of the values from step 2 at each vertex */

    cudppSegmentedScanDispatch(plan->m_d_temp2,
                               plan->m_d_temp2,
                               plan->m_d_head_flags,
                               2 * plan->m_num_edges,
                               nrplan->m_segmentedScanPlan);
    /* this scan really should be inclusive and backward to make the
     * next step easier */
    /* temp2 now has reduced value per segment at end of segment  */

    /* Step 4: pack result of segmented reduce back into per-vertex
     * array. Note that the indices in "temp" are (still) one too big;
     * the kernel makes the proper correction. */

    /* remember temp is still 1 too big */
    gatherFromEndOfSegment<<<grid2Edges, threads>>>
        (d_out, (T *) plan->m_d_temp2, (unsigned int *) plan->m_d_temp,
         plan->m_d_head_flags, 2 * plan->m_num_edges);
    
}

template <class T>
void vgDistributeExcess(const CUDPPVGraphPlan *plan,
                        const CUDPPVGraphDEPlan *deplan,
                        T * d_out, const T * d_capacity, const T * d_excess)
{
    /* capacity, out are allocated per-edge */
    /* excess is allocated per-node */
    /* temps (within deplan) are allocated per-edge */

    unsigned int numNodesBlocks = 
        max(1, (int)ceil((double)plan->m_num_nodes / 
                         ((double)VGDISTRIBUTEEXCESS_ELTS_PER_THREAD * 
                          CTA_SIZE)));

#if 0
    /* don't need this */
    bool nodesFillBlock = 
        (plan->m_num_nodes == 
         (numNodesBlocks * VGDISTRIBUTEEXCESS_ELTS_PER_THREAD * CTA_SIZE));  
#endif

    dim3  gridNodes(max(1, numNodesBlocks), 1, 1);

    unsigned int numEdgeBlocks = 
        max(1,
            (int)ceil((double)plan->m_num_edges / 
                      ((double)VGDISTRIBUTEEXCESS_ELTS_PER_THREAD * CTA_SIZE)));

    dim3  gridEdges(max(1, numEdgeBlocks), 1, 1);

    dim3  threads(CTA_SIZE, 1, 1);

    vGraphCopyDebug<<<gridEdges, threads>>>(d_out, d_capacity, plan->m_num_edges);

    // XXX I am still debugging - obviously this code doesn't work yet
    return;

    /* Step 1: segmented copy of vertex indices to each edge incident
     * to that vertex. This is a segmented distribute operation, using
     * head_flags to indicate segment boundaries. */

    /* implemented with an inclusive scan of head flags rather than
     * {exclusive scan, inclusive segmented-scan}. The side effect is
     * that ***the resulting values are 1 larger than they should
     * be***. We clean this up in later steps. */

    /* This results in d_temp containing 1+ the vertex ID in each
     * element. We use this result to distribute the excess into
     * d_temp. */

    cudppScanDispatch((unsigned int *) plan->m_d_temp, plan->m_d_head_flags,
                      plan->m_num_edges, 1, deplan->m_scanPlan);
    /* in output of this call, temp is 1 too big */

    /* the -1 in the template corrects for the 1-too-big in previous call */
    vGraphGatherAndZero<T, -1>
        <<<gridEdges, threads>>>((T *) plan->m_d_temp, d_excess, 
                                 (unsigned int *) plan->m_d_temp, 
                                 plan->m_d_head_flags, 
                                 plan->m_num_edges);

    /* step 2: distribute capacity into temp2 */
    cudppSegmentedScanDispatch((T *) plan->m_d_temp2,
                               d_capacity,
                               plan->m_d_head_flags,
                               plan->m_num_edges,
                               deplan->m_segmentedScanPlan);

    /* step 3: calculate excess distribution into d_out:
       min(capacity, max(temp-temp2,0) ) */
    vGraphDistributeExcessKernel<<<gridEdges, threads>>>(d_out, d_capacity, 
                                                         (T *) plan->m_d_temp,
                                                         (T *) plan->m_d_temp2,
                                                         plan->m_num_edges);

}

/** @brief Given a v-graph graph representation with weights at each
 * edge, return an output array with '1' marked in each edge that is
 * part of the minimum spanning tree.
 *
 * Algorithm:
 * 
 * Repeat until done (don't know how to do that yet, but "done" ==
 * "single tree"):
 *
 * -# Propagate vertex ID to each edge.
 *
 * -# For each (alive) vertex, decide if it's a child or a parent.
 *    Only continue if it's a child.
 *
 * -# Find the smallest edge weight per vertex. We do this with
 *    min-segmented-scan. Only continue if this smallest edge weight
 *    is to a parent. These edges are star edges.
 *
 * -# Star-merge all star edges into the parent vertex. Star-merge is
 *    the most complicated routine.
 *
 * @param[in] plan Pointer to the CUDPPVGraphPlan object which stores
 * the v-graph data structure
 * @param[in] mstplan Pointer to the CUDPPVGraphNRPlan object which stores
 * the configuration needed by this algorithm
 * @param[out] d_out The per-edge output array, where a '1' marks
 * edges that are part of the minimum spanning tree.
 */

template <class WeightsT>
void vgMinimumSpanningTree(const CUDPPVGraphPlan *plan,
                           const CUDPPVGraphMSTPlan *mstplan,
                           unsigned int * d_out)
{

    unsigned int numNodesBlocks = 
        max(1, (int)ceil((double)plan->m_num_nodes / 
                         ((double)VGMST_ELTS_PER_THREAD * 
                          CTA_SIZE)));

#if 0
    /* don't need this */
    bool nodesFillBlock = 
        (plan->m_num_nodes == 
         (numNodesBlocks * VGMST_ELTS_PER_THREAD * CTA_SIZE));  
#endif

    dim3  gridNodes(max(1, numNodesBlocks), 1, 1);

    unsigned int num2EdgeBlocks = 
        max(1,
            (int)ceil(2.0 * (double)plan->m_num_edges / 
                      ((double)VGMST_ELTS_PER_THREAD * CTA_SIZE)));

    dim3  grid2Edges(max(1, num2EdgeBlocks), 1, 1);

    dim3  threads(CTA_SIZE, 1, 1);

    int iter = 0;
    while (1)
    {

        /* preliminary step: clear starParent */
        clearStarParent<<<gridNodes, threads>>>(mstplan->m_d_starParent,
                                                plan->m_num_nodes);

        /* preliminary step: fill mstplan->{m_d_myHead,isEndOfSegment} */
        setHeadToOffset<<<grid2Edges, threads>>>(mstplan->m_d_myHead,
                                                 mstplan->m_d_isEndOfSegment,
                                                 plan->m_d_head_flags,
                                                 2 * plan->m_num_edges);
        printDebugI("offsetonly", mstplan->m_d_myHead, 2 * plan->m_num_edges);
        printDebugI("isEndOfSegment", mstplan->m_d_isEndOfSegment, 
                    2 * plan->m_num_edges);
        cudppSegmentedScanDispatch(mstplan->m_d_myHead,
                                   mstplan->m_d_myHead,
                                   plan->m_d_head_flags,
                                   2 * plan->m_num_edges,
                                   mstplan->m_segmentedScanPlan);
        printDebugI("myHead", mstplan->m_d_myHead, 2 * plan->m_num_edges);

        /* Step 1: segmented copy of vertex indices to each edge incident
         * to that vertex. This is a segmented distribute operation, using
         * head_flags to indicate segment boundaries. */
        
        /* implemented with an inclusive scan of head flags rather than
         * {exclusive scan, inclusive segmented-scan}. The side effect is
         * that ***the resulting values are 1 larger than they should
         * be***. We clean this up in later steps. */

        printDebugI("head flags", plan->m_d_head_flags, 2 * plan->m_num_edges);

        cudppScanDispatch(plan->m_d_temp, plan->m_d_head_flags,
                          2 * plan->m_num_edges, 1, mstplan->m_scanPlan);
        /* in output of this call, temp is 1 too big */
        printDebugI("vtx idx+1", (int *) plan->m_d_temp, 2 * plan->m_num_edges);
        vectorAdd<int, -1>
            <<<grid2Edges, threads>>>(mstplan->m_d_myVertexID,
                                      (int *) plan->m_d_temp,
                                      2 * plan->m_num_edges);
        printDebugI("myVtxID", mstplan->m_d_myVertexID, 2 * plan->m_num_edges);

        /* Step 2. Randomly set edge at each vertex to child (0) or
         * parent (1). Only sets segment heads (so really it's setting
         * per-vertex). */
        random0Or1<<<grid2Edges, threads>>>(mstplan->m_d_isParent, 
                                            plan->m_d_head_flags,
                                            iter, 2 * plan->m_num_edges);
        printDebugI("isParent", mstplan->m_d_isParent, 2 * plan->m_num_edges);
        /* then propagate the result */
        mstplan->m_segmentedScanPlan->m_config.op = CUDPP_ADD;
        cudppSegmentedScanDispatch(mstplan->m_d_isParent,
                                   mstplan->m_d_isParent,
                                   plan->m_d_head_flags,
                                   2 * plan->m_num_edges,
                                   mstplan->m_segmentedScanPlan);
        printDebugI("isParentProp", mstplan->m_d_isParent, 
                    2 * plan->m_num_edges);
        /* Now every edge knows if its vertex is a child or parent */

        /* Step 3. Find smallest edge per vertex. Warning: Will likely
         * not work correctly if two edges in same vertex have same
         * weight.*/
        /* 3a: segmented min-scan across vertex's edges */
        printDebugF("weights", (WeightsT *) plan->m_d_weights, 
                    2 * plan->m_num_edges);
        mstplan->m_segmentedScanPlan->m_config.op = CUDPP_MIN;
        cudppSegmentedScanDispatch(mstplan->m_d_temp,
                                   (WeightsT *) plan->m_d_weights,
                                   plan->m_d_head_flags,
                                   2 * plan->m_num_edges,
                                   mstplan->m_segmentedScanPlan);
        printDebugF("weights-min", (WeightsT *) mstplan->m_d_temp, 
                    2 * plan->m_num_edges);

        /* @todo rewrite 3b and 3c using myHead */
        /* 3b: gather minimum back to vertex, set non-head to 0 */
        /* what we want here is a backwards segmented inclusive scan */
        vGraphMSTGatherAndZero<WeightsT, -1>
            <<<grid2Edges, threads>>>((WeightsT *) mstplan->m_d_temp, 
                                      (WeightsT *) mstplan->m_d_temp,
                                      plan->m_d_segment_descriptor,
                                      (int *) plan->m_d_temp, 
                                      plan->m_d_head_flags, 
                                      2 * plan->m_num_edges);
        printDebugF("weights-gather", (WeightsT *) mstplan->m_d_temp, 
                    2 * plan->m_num_edges);
        /* 3c: propagate minimum across vertex's edges into temp */
        mstplan->m_segmentedScanPlan->m_config.op = CUDPP_ADD;
        cudppSegmentedScanDispatch(mstplan->m_d_temp,
                                   mstplan->m_d_temp,
                                   plan->m_d_head_flags,
                                   2 * plan->m_num_edges,
                                   mstplan->m_segmentedScanPlan);
        printDebugF("weights-prop", (WeightsT *) mstplan->m_d_temp, 
                    2 * plan->m_num_edges);
        /* now star edges are where weight == mstplan->m_d_tmp AND the min edge 
         * is a child AND the opposite edge is a parent */

        computeStarEdgesAndNeededSpace<<<grid2Edges, threads>>>
            (mstplan->m_d_neededSpace, 
             mstplan->m_d_isStarEdge,
             mstplan->m_d_starParent,
             mstplan->m_d_isParent,
             plan->m_d_cross_pointers,
             (WeightsT *) mstplan->m_d_temp, /* minweights */
             (WeightsT *) plan->m_d_weights,
             mstplan->m_d_myHead,
             mstplan->m_d_myVertexID,
             2 * plan->m_num_edges);
        
        printDebugI("neededSpace", mstplan->m_d_neededSpace, 
                    2 * plan->m_num_edges);
        printDebugI("isStarEdge", mstplan->m_d_isStarEdge, 
                    2 * plan->m_num_edges);
        printDebugI("starParent", mstplan->m_d_starParent, 
                    plan->m_num_nodes);

        /* edges that are going to get squished must be:
         * - edges in a child that has a star-edge AND
         *   - edges that are star-edges OR
         *   - edges that point into this vertex's star-parent
         */

        markSurvivingEdges<<<grid2Edges, threads>>>
            (mstplan->m_d_survivingEdges,
             mstplan->m_d_isStarEdge,
             mstplan->m_d_myVertexID,
             mstplan->m_d_isParent,
             mstplan->m_d_starParent,
             plan->m_d_cross_pointers,
             2 * plan->m_num_edges);
        printDebugI("survivingEdges", mstplan->m_d_survivingEdges, 
                    2 * plan->m_num_edges);

        return;

        vGraphCopyDebug
            <<<grid2Edges, threads>>>(d_out, 
                                      (unsigned int *) mstplan->m_d_temp,
                                      2 * plan->m_num_edges);



        iter++;
    }
}

/** @brief Initializes v-graph data structure (allocate and copy) from
 * host-side data
 *
 * @todo Reconcile need for neighbor-reduce's 2*num_edges vs.
 * distribute-excess's num_edges
 * 
 * @param[in] plan v-graph data structure plan
 * @param[in] h_segment_descriptor Unsigned int host array containing segment descriptor; should be an array of size num_nodes
 * @param[in] h_cross_pointers Unsigned int host array containing cross pointers; should be an array of size 2 * num_edges
 * @param[in] h_head_flags Unsigned int host array containing head flags; should be an array of size 2 * num_edges
 */

void initializeVGraphStorage(
    CUDPPVGraphPlan *plan,
    const unsigned int * h_segment_descriptor, // size: num_nodes
    const unsigned int * h_cross_pointers,     // size: 2 * num_edges
    const unsigned int * h_head_flags,         // size: 2 * num_edges
    const float * h_weights)                   // size: 2 * num_edges
{
    CUDA_SAFE_CALL(cudaMalloc((void **) &(plan->m_d_segment_descriptor),  
                              plan->m_num_nodes * sizeof(unsigned int)));
    CUDA_SAFE_CALL(cudaMalloc((void **) &(plan->m_d_cross_pointers),  
                              2 * plan->m_num_edges * sizeof(unsigned int)));
    CUDA_SAFE_CALL(cudaMalloc((void **) &(plan->m_d_head_flags),  
                              2 * plan->m_num_edges * sizeof(unsigned int)));
    CUDA_SAFE_CALL(cudaMalloc((void **) &(plan->m_d_weights),  
                              2 * plan->m_num_edges * sizeof(float)));

    CUDA_SAFE_CALL(cudaMemcpy(plan->m_d_segment_descriptor, 
                              h_segment_descriptor, 
                              plan->m_num_nodes * sizeof(unsigned int),
                              cudaMemcpyHostToDevice) );
    CUDA_SAFE_CALL(cudaMemcpy(plan->m_d_cross_pointers, 
                              h_cross_pointers, 
                              2 * plan->m_num_edges * sizeof(unsigned int),
                              cudaMemcpyHostToDevice) );
    CUDA_SAFE_CALL(cudaMemcpy(plan->m_d_head_flags, 
                              h_head_flags, 
                              2 * plan->m_num_edges * sizeof(unsigned int),
                              cudaMemcpyHostToDevice) );
    if (h_weights != NULL)
    {
        CUDA_SAFE_CALL(cudaMemcpy(plan->m_d_weights, 
                                  h_weights, 
                                  2 * plan->m_num_edges * sizeof(float),
                                  cudaMemcpyHostToDevice) );
    }

    CUT_CHECK_ERROR("initializeVGraphStorage");
}

#ifdef __cplusplus
extern "C" 
{
#endif

/** @brief Allocates space in v-graph "m_d_temp" array (size 2 * num_edges)
 *
 * @param[in] vgplan v-graph data structure plan
 */
CUDPPResult cudppVGraphAllocateTemp(CUDPPVGraphPlan * vgplan)
{
    if (vgplan != NULL)
    {
        CUDA_SAFE_CALL(cudaMalloc((void**) &(vgplan->m_d_temp), 
                                  2 * vgplan->m_num_edges * sizeof(int))); 
        return CUDPP_SUCCESS;
    }
    else
    {
        return CUDPP_ERROR_UNKNOWN;
    }
}

/** @brief Allocates space in v-graph "m_d_temp2" array (size 2 * num_edges)
 *
 * @param[in] vgplan v-graph data structure plan
 */
CUDPPResult cudppVGraphAllocateTemp2(CUDPPVGraphPlan * vgplan)
{
    if (vgplan != NULL)
    {
        CUDA_SAFE_CALL(cudaMalloc((void**) &(vgplan->m_d_temp2), 
                                  2 * vgplan->m_num_edges * sizeof(int))); 
        return CUDPP_SUCCESS;
    }
    else
    {
        return CUDPP_ERROR_UNKNOWN;
    }
}


/** @brief Frees and resets storage in v-graph
 *
 * @param[in] plan v-graph data structure plan
 */
void freeVGraphStorage(CUDPPVGraphPlan *plan)
{
    CUT_CHECK_ERROR("freeVGraphStorage");

    cudaFree(plan->m_d_segment_descriptor);
    cudaFree(plan->m_d_cross_pointers);
    cudaFree(plan->m_d_head_flags);
    cudaFree(plan->m_d_temp);
    cudaFree(plan->m_d_temp2);

    plan->m_num_nodes = 0;
    plan->m_num_edges = 0;
    plan->m_d_segment_descriptor = 0;
    plan->m_d_cross_pointers = 0;
    plan->m_d_head_flags = 0;
    plan->m_d_temp = 0;
    plan->m_d_temp2 = 0;
}

/** @brief Frees and resets storage in v-graph neighbor-reduce algorithm plan
 *
 * @param[in] plan v-graph neighbor-reduce algorithm plan
 */
void freeVGraphNRStorage(CUDPPVGraphNRPlan *plan)
{
    CUT_CHECK_ERROR("freeVGraphNRStorage");

    plan->m_num_nodes = 0;
    plan->m_num_edges = 0;
}

/** @brief Frees and resets storage in v-graph distribute-excess algorithm plan
 *
 * @param[in] plan v-graph distribute-excess algorithm plan
 */
void freeVGraphDEStorage(CUDPPVGraphDEPlan *plan)
{
    CUT_CHECK_ERROR("freeVGraphDEStorage");

    plan->m_num_nodes = 0;
    plan->m_num_edges = 0;
}

void initializeVGraphMSTStorage(CUDPPVGraphMSTPlan *plan)
{
    CUDA_SAFE_CALL(cudaMalloc((void **) &(plan->m_d_isParent),  
                              2 * plan->m_num_edges * sizeof(unsigned int)));
    CUDA_SAFE_CALL(cudaMalloc((void **) &(plan->m_d_myHead),  
                              2 * plan->m_num_edges * sizeof(unsigned int)));
    CUDA_SAFE_CALL(cudaMalloc((void **) &(plan->m_d_neededSpace),  
                              2 * plan->m_num_edges * sizeof(unsigned int)));
    CUDA_SAFE_CALL(cudaMalloc((void **) &(plan->m_d_isStarEdge),  
                              2 * plan->m_num_edges * sizeof(unsigned int)));
    CUDA_SAFE_CALL(cudaMalloc((void **) &(plan->m_d_isEndOfSegment),  
                              2 * plan->m_num_edges * sizeof(unsigned int)));
    CUDA_SAFE_CALL(cudaMalloc((void **) &(plan->m_d_myVertexID),  
                              2 * plan->m_num_edges * sizeof(unsigned int)));
    CUDA_SAFE_CALL(cudaMalloc((void **) &(plan->m_d_survivingEdges),  
                              2 * plan->m_num_edges * sizeof(unsigned int)));
    CUDA_SAFE_CALL(cudaMalloc((void **) &(plan->m_d_starParent),  
                              plan->m_num_nodes * sizeof(int)));
    CUT_CHECK_ERROR("initializeVGraphMSTStorage");
}

/** @brief Allocates space in v-graph MST "m_d_temp" array (size 2 * num_edges)
 *
 * @param[in] vgplan v-graph data structure plan
 */
CUDPPResult cudppVGraphMSTAllocateTemp(CUDPPVGraphMSTPlan * mstplan)
{
    if (mstplan != NULL)
    {
        CUDA_SAFE_CALL(cudaMalloc((void**) &(mstplan->m_d_temp), 
                                  2 * mstplan->m_num_edges * sizeof(float))); 
        return CUDPP_SUCCESS;
    }
    else
    {
        return CUDPP_ERROR_UNKNOWN;
    }
}

/** @brief Allocates space in v-graph MST "m_d_temp2" array (size 2 * num_edges)
 *
 * @param[in] vgplan v-graph data structure plan
 */
CUDPPResult cudppVGraphMSTAllocateTemp2(CUDPPVGraphMSTPlan * mstplan)
{
    if (mstplan != NULL)
    {
        CUDA_SAFE_CALL(cudaMalloc((void**) &(mstplan->m_d_temp2), 
                                  2 * mstplan->m_num_edges * sizeof(float))); 
        return CUDPP_SUCCESS;
    }
    else
    {
        return CUDPP_ERROR_UNKNOWN;
    }
}

/** @brief Frees and resets storage in v-graph minimum-spanning-tree algorithm plan
 *
 * @param[in] plan v-graph minimum-spanning-tree algorithm plan
 */
void freeVGraphMSTStorage(CUDPPVGraphMSTPlan *plan)
{
    CUT_CHECK_ERROR("freeVGraphMSTStorage");

    plan->m_num_nodes = 0;
    plan->m_num_edges = 0;

    cudaFree(plan->m_d_isParent);
    cudaFree(plan->m_d_myHead);
    cudaFree(plan->m_d_neededSpace);
    cudaFree(plan->m_d_isStarEdge);
    cudaFree(plan->m_d_isEndOfSegment);
    cudaFree(plan->m_d_temp);
    cudaFree(plan->m_d_temp2);
}

/** @brief Dispatch function to perform a neighbor reduction on a v-graph
 *
 * Currently only supports ADD-reduction on int data.
 * 
 * @param[out] d_out Per-vertex output vector: values from neighbor reduction
 * @param[in]  d_idata Per-vertex input vector: input to neighbor reduction
 * @param[in]  vgplan v-graph data structure
 * @param[in]  vgnrplan v-graph neighbor reduction algorithmic configuration
 */
void cudppVGNeighborReduceDispatch(const CUDPPVGraphPlan *vgplan,
                                   const CUDPPVGraphNRPlan *vgnrplan,
                                   void * d_out, const void * d_idata)
{    
    // needs to dispatch based on vgplan's datatype
    vgNeighborReduce<int,CUDPP_ADD>(vgplan, vgnrplan,
                                    (int *) d_out, (const int *) d_idata);
}

/** @brief Dispatch function to distribute excesses on a v-graph
 *
 * Currently only supports int data.
 * 
 * @param[out] d_out Per-vertex output vector: values from neighbor reduction
 * @param[in]  d_idata Per-vertex input vector: input to neighbor reduction
 * @param[in]  vgplan v-graph data structure
 * @param[in]  vgnrplan v-graph neighbor reduction algorithmic configuration
 */
void cudppVGDistributeExcessDispatch(const CUDPPVGraphPlan *vgplan,
                                     const CUDPPVGraphDEPlan *vgdeplan,
                                     void * d_out, const void * d_capacity,
                                     const void * d_excess)
{    
    // needs to dispatch based on vgplan's datatype
    vgDistributeExcess<int>(vgplan, vgdeplan,
                            (int *) d_out, (const int *) d_excess,
                            (const int *) d_capacity);
}

void cudppVGMinimumSpanningTreeDispatch(CUDPPVGraphPlan * vGraphHandle,
                                        CUDPPVGraphMSTPlan * vGraphMSTHandle,
                                        void *d_out)
{
    vgMinimumSpanningTree<float>(vGraphHandle, vGraphMSTHandle, 
                                 (unsigned int *) d_out);
}

#ifdef __cplusplus
}
#endif

/** @} */ // end v-graph functions
/** @} */ // end cudpp_app
