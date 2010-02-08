// -------------------------------------------------------------
// cuDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Source: $
// $Revision: $
// $Date: $
// -------------------------------------------------------------
// This source code is distributed under the terms of license.txt in
// the root directory of this source distribution.
// -------------------------------------------------------------

/**
 * @file
 * test_vgraph.cu
 *
 * @brief Host testrig routines to exercise cudpp's v-graph functionality.
 */

#include <stdio.h>
#include <cutil.h>
#include <time.h>
#include <limits.h>

#include "cudpp.h"
#include "vgraph_gold.h"

/**
 * testVGraphNR exercises CUDPP's neighbor-reduce routine with the
 * v-graph graph data structure.
 * Supports "global" options (see setOptions)
 * @param[in] argc Currently not used
 * @param[in] argv Currently not used
 * @return Number of tests that failed regression (0 for all pass)
 * @see CUDPPConfiguration, setOptions, cudppVGraph, cudppVGNeighborReduce
 */
int testVGraphNR(int argc, const char** argv)
{
    unsigned int timer;

    CUT_SAFE_CALL(cutCreateTimer(&timer));

    int retval = 0;

    testrigOptions testOptions;
    setOptions(argc, argv, testOptions);

    // allocate device memory input, output, and temp arrays

    int num_nodes, num_edges;

    int * d_out;                // num_nodes
    int * d_idata;              // num_nodes

    unsigned int * d_segment_descriptor;
    unsigned int * d_cross_pointers;
    unsigned int * d_head_flags;

    int * h_idata, * h_odata;
    unsigned int * h_segment_descriptor, * h_cross_pointers, * h_head_flags;
    int * reference = NULL;

    num_nodes = 5;
    num_edges = 6;
    cutGetCmdLineArgumenti(argc, (const char**) argv, "nodes", &num_nodes);
    cutGetCmdLineArgumenti(argc, (const char**) argv, "edges", &num_edges);
#ifndef _USE_BOOST_
    if ((num_nodes != 5) || (num_edges != 6))
    {
        fprintf(stderr, "Warning: without Boost, testing is only with a "
                "5-node, 6-edge test case\n");
        num_nodes = 5;
        num_edges = 6;
    }
#endif

    generateIntTestGraphNR(num_nodes, num_edges, &h_idata, &h_odata, 
                           &h_segment_descriptor,
                           &h_cross_pointers, &h_head_flags, &reference,
                           testOptions);

    CUDA_SAFE_CALL(cudaMalloc((void**) &d_out, num_nodes * sizeof(int))); 
    CUDA_SAFE_CALL(cudaMalloc((void**) &d_idata, num_nodes * sizeof(int))); 
    CUDA_SAFE_CALL(cudaMalloc((void**) &d_segment_descriptor,
                              num_nodes * sizeof(unsigned int))); 
    CUDA_SAFE_CALL(cudaMalloc((void**) &d_cross_pointers,
                              2 * num_edges * sizeof(unsigned int))); 
    CUDA_SAFE_CALL(cudaMalloc((void**) &d_head_flags,
                              2 * num_edges * sizeof(unsigned int)));

    CUDA_SAFE_CALL(cudaMemcpy(d_idata, h_idata, num_nodes * sizeof(int),
                              cudaMemcpyHostToDevice));

    CUDPPConfiguration config;
    config.datatype = CUDPP_INT;
    config.options = (CUDPPOption) 0;
    config.algorithm = CUDPP_SPMVMULT;

    CUDPPHandle vGraphHandle, vGraphNRHandle;
    CUDPPResult result = CUDPP_SUCCESS;

    result = cudppVGraph(&vGraphHandle, config, num_nodes, num_edges,
                         h_segment_descriptor, h_cross_pointers, h_head_flags,
                         (float *) NULL);

    if (result != CUDPP_SUCCESS)
    {
        fprintf(stderr, "Error creating v-graph object\n");
        return 1;
    }

    result = cudppVGraphNRPlan(&vGraphNRHandle, config, num_nodes, num_edges);

    if (result != CUDPP_SUCCESS)
    {
        fprintf(stderr, "Error creating v-graph neighbor-reduce plan\n");
        return 1;
    }

    // would like to call cudppVGraphNRAllocate() ...
    // this is not ideal
    unsigned int * d_temp, * d_temp2;
    CUDA_SAFE_CALL(cudaMalloc((void**) &d_temp,
                              2 * num_edges * sizeof(unsigned int))); 
    CUDA_SAFE_CALL(cudaMalloc((void**) &d_temp2, 
                              2 * num_edges * sizeof(unsigned int)));
    cudppSetVGTemps(vGraphHandle, d_temp, d_temp2);

    // Run it once to avoid timing startup overhead
    cudppVGNeighborReduce(vGraphHandle, vGraphNRHandle, d_out, d_idata);

    for (unsigned i = 0; i < testOptions.numIterations; i++)
    {
        cutStartTimer(timer);
        cudppVGNeighborReduce(vGraphHandle, vGraphNRHandle, d_out, d_idata);
        cudaThreadSynchronize();
        cutStopTimer(timer);
    }

    CUDA_SAFE_CALL(cudaMemcpy(h_odata, d_out, num_nodes * sizeof(int),
                              cudaMemcpyDeviceToHost));

    CUTBoolean vg_result = cutComparei(reference, h_odata, num_nodes);
    retval += (CUTTrue == vg_result) ? 0 : 1;

    if (testOptions.debug)
    {
        for (unsigned int i = 0; i < num_nodes; i++)
        {
            printf("i: %d\tref: %i\todata: %i\n", i, reference[i], h_odata[i]);
        }
    }

    printf("v-graph neighbor-reduce test %s\n", 
           (CUTTrue == vg_result) ? "PASSED" : "FAILED");
    printf("Average execution time: %f ms\n", 
           cutGetTimerValue(timer) / testOptions.numIterations);

    result = cudppDestroyVGraph(vGraphHandle);

    if (result != CUDPP_SUCCESS)
    {
        printf("Error destroying v-graph object\n");
    }

    result = cudppDestroyVGraphNRPlan(vGraphNRHandle);

    if (result != CUDPP_SUCCESS)
    {
        printf("Error destroying v-graph neighbor-reduce plan\n");
    }

    cutDeleteTimer(timer);

    free(reference);

    CUDA_SAFE_CALL(cudaFree(d_out));
    CUDA_SAFE_CALL(cudaFree(d_idata));
    CUDA_SAFE_CALL(cudaFree(d_segment_descriptor));
    CUDA_SAFE_CALL(cudaFree(d_cross_pointers));
    CUDA_SAFE_CALL(cudaFree(d_head_flags));

    return retval;
}

/**
 * testVGraphDE exercises CUDPP's distribute-excess routine with the
 * v-graph graph data structure.
 * Supports "global" options (see setOptions)
 * @param[in] argc Currently not used
 * @param[in] argv Currently not used
 * @return Number of tests that failed regression (0 for all pass)
 * @see CUDPPConfiguration, setOptions, cudppVGraph, cudppVGDistributeExcess
 */
int testVGraphDE(int argc, const char** argv)
{
    unsigned int timer;

    CUT_SAFE_CALL(cutCreateTimer(&timer));

    int retval = 0;

    testrigOptions testOptions;
    setOptions(argc, argv, testOptions);

    // allocate device memory input, output, and temp arrays

    int num_nodes, num_edges;

    int * d_out;                // num_nodes
    int * d_idata = NULL;       // num_nodes

    unsigned int * d_segment_descriptor;
    unsigned int * d_cross_pointers;
    unsigned int * d_head_flags;
    int * d_capacity, * d_excess;

    int * h_idata, * h_odata;
    unsigned int * h_segment_descriptor, * h_cross_pointers, * h_head_flags;
    int * h_capacity, * h_excess;
    int * reference = NULL;

    num_nodes = 2;
    num_edges = 6;
    cutGetCmdLineArgumenti(argc, (const char**) argv, "nodes", &num_nodes);
    cutGetCmdLineArgumenti(argc, (const char**) argv, "edges", &num_edges);
// #ifndef _USE_BOOST_
    if ((num_nodes != 2) || (num_edges != 6))
    {
        fprintf(stderr, "Warning: currently testing is only for a "
                "2-node, 6-edge test case\n");
        num_nodes = 2;
        num_edges = 6;
    }
// #endif

    generateIntExcessTestGraph(num_nodes, num_edges, &h_idata, &h_odata, 
                               &h_segment_descriptor,
                               &h_cross_pointers, &h_head_flags, 
                               &h_capacity, &h_excess,
                               &reference, testOptions);

    CUDA_SAFE_CALL(cudaMalloc((void**) &d_out, num_edges * sizeof(int))); 
    CUDA_SAFE_CALL(cudaMalloc((void**) &d_segment_descriptor,
                              num_nodes * sizeof(unsigned int))); 
    CUDA_SAFE_CALL(cudaMalloc((void**) &d_cross_pointers,
                              2 * num_edges * sizeof(unsigned int))); 
    CUDA_SAFE_CALL(cudaMalloc((void**) &d_head_flags,
                              num_edges * sizeof(unsigned int)));
    CUDA_SAFE_CALL(cudaMalloc((void**) &d_capacity, num_edges * sizeof(int))); 
    CUDA_SAFE_CALL(cudaMalloc((void**) &d_excess, num_nodes * sizeof(int))); 

    CUDPPConfiguration config;
    config.datatype = CUDPP_INT;
    config.options = (CUDPPOption) 0;
    config.algorithm = CUDPP_SPMVMULT;

    CUDPPHandle vGraphHandle, vGraphDEHandle;
    CUDPPResult result = CUDPP_SUCCESS;

    result = cudppVGraph(&vGraphHandle, config, num_nodes, num_edges,
                         h_segment_descriptor, h_cross_pointers, h_head_flags,
                         (float *) NULL);

    if (result != CUDPP_SUCCESS)
    {
        fprintf(stderr, "Error creating v-graph object\n");
        return 1;
    }

    result = cudppVGraphDEPlan(&vGraphDEHandle, config, num_nodes, num_edges);

    if (result != CUDPP_SUCCESS)
    {
        fprintf(stderr, "Error creating v-graph distribute-excess plan\n");
        return 1;
    }

    // would like to call cudppVGraphDEAllocate() ...
    // this is not ideal
    unsigned int * d_temp, * d_temp2;
    CUDA_SAFE_CALL(cudaMalloc((void**) &d_temp,
                              num_edges * sizeof(unsigned int))); 
    CUDA_SAFE_CALL(cudaMalloc((void**) &d_temp2, 
                              num_edges * sizeof(unsigned int)));
    cudppSetVGTemps(vGraphHandle, d_temp, d_temp2);

    // Run it once to avoid timing startup overhead
    cudppVGDistributeExcess(vGraphHandle, vGraphDEHandle, 
                            d_out, d_capacity, d_excess);

    for (unsigned i = 0; i < testOptions.numIterations; i++)
    {
        cutStartTimer(timer);
        cudppVGDistributeExcess(vGraphHandle, vGraphDEHandle, 
                                d_out, d_capacity, d_excess);
        cudaThreadSynchronize();
        cutStopTimer(timer);
    }

    CUDA_SAFE_CALL(cudaMemcpy(h_odata, d_out, num_edges * sizeof(int),
                              cudaMemcpyDeviceToHost));

    CUTBoolean vg_result = cutComparei(reference, h_odata, num_edges);
    retval += (CUTTrue == vg_result) ? 0 : 1;

    if (testOptions.debug)
    {
        for (unsigned int i = 0; i < num_edges; i++)
        {
            printf("i: %d\tref: %i\todata: %i\n", i, reference[i], h_odata[i]);
        }
    }

    printf("v-graph distribute-excess test %s\n", 
           (CUTTrue == vg_result) ? "PASSED" : "FAILED");
    printf("Average execution time: %f ms\n", 
           cutGetTimerValue(timer) / testOptions.numIterations);

    result = cudppDestroyVGraph(vGraphHandle);

    if (result != CUDPP_SUCCESS)
    {
        printf("Error destroying v-graph object\n");
    }

    result = cudppDestroyVGraphDEPlan(vGraphDEHandle);

    if (result != CUDPP_SUCCESS)
    {
        printf("Error destroying v-graph neighbor-reduce plan\n");
    }

    cutDeleteTimer(timer);

    free(reference);

    CUDA_SAFE_CALL(cudaFree(d_out));
    CUDA_SAFE_CALL(cudaFree(d_idata));
    CUDA_SAFE_CALL(cudaFree(d_segment_descriptor));
    CUDA_SAFE_CALL(cudaFree(d_cross_pointers));
    CUDA_SAFE_CALL(cudaFree(d_head_flags));

    return retval;
}

/**
 * testVGraphMST exercises CUDPP's minimum-spanning-tree routine with the
 * v-graph graph data structure.
 * Supports "global" options (see setOptions)
 * @param[in] argc Currently not used
 * @param[in] argv Currently not used
 * @return Number of tests that failed regression (0 for all pass)
 * @see CUDPPConfiguration, setOptions, cudppVGraph, cudppVGMinimumSpanningTree
 */
int testVGraphMST(int argc, const char** argv)
{
    unsigned int timer;

    CUT_SAFE_CALL(cutCreateTimer(&timer));

    int retval = 0;

    testrigOptions testOptions;
    setOptions(argc, argv, testOptions);

    // allocate device memory input, output, and temp arrays

    int num_nodes, num_edges;

    int * d_out;                // num_nodes
    int * d_idata;              // num_nodes

    int * h_idata, * h_odata;
    unsigned int * h_segment_descriptor, * h_cross_pointers, * h_head_flags;
    float * h_weights;
    int * reference = NULL;

    num_nodes = 5;
    num_edges = 6;
    cutGetCmdLineArgumenti(argc, (const char**) argv, "nodes", &num_nodes);
    cutGetCmdLineArgumenti(argc, (const char**) argv, "edges", &num_edges);
#ifndef _USE_BOOST_
    if ((num_nodes != 5) || (num_edges != 6))
    {
        fprintf(stderr, "Warning: without Boost, testing is only with a "
                "5-node, 6-edge test case\n");
        num_nodes = 5;
        num_edges = 6;
    }
#endif

    generateIntTestGraphMST(num_nodes, num_edges, &h_idata, &h_odata, 
                            &h_segment_descriptor,
                            &h_cross_pointers, &h_head_flags, &h_weights,
                            &reference, testOptions);

    CUDA_SAFE_CALL(cudaMalloc((void**) &d_out, num_nodes * sizeof(int))); 
    CUDA_SAFE_CALL(cudaMalloc((void**) &d_idata, num_nodes * sizeof(int))); 

    CUDA_SAFE_CALL(cudaMemcpy(d_idata, h_idata, num_nodes * sizeof(int),
                              cudaMemcpyHostToDevice));

    CUDPPConfiguration config;
    config.datatype = CUDPP_INT;
    config.options = (CUDPPOption) 0;
    config.algorithm = CUDPP_SPMVMULT;

    CUDPPHandle vGraphHandle, vGraphMSTHandle;
    CUDPPResult result = CUDPP_SUCCESS;

    result = cudppVGraph(&vGraphHandle, config, num_nodes, num_edges,
                         h_segment_descriptor, h_cross_pointers, h_head_flags,
                         h_weights);

    if (result != CUDPP_SUCCESS)
    {
        fprintf(stderr, "Error creating v-graph object\n");
        return 1;
    }

    result = cudppVGraphMSTPlan(&vGraphMSTHandle, config, num_nodes, num_edges);

    if (result != CUDPP_SUCCESS)
    {
        fprintf(stderr, "Error creating v-graph minimum-spannig-tree plan\n");
        return 1;
    }

    // would like to call cudppVGraphAllocateTemps() ...
    // this is not ideal
    unsigned int * d_temp, * d_temp2;
    CUDA_SAFE_CALL(cudaMalloc((void**) &d_temp,
                              2 * num_edges * sizeof(unsigned int))); 
    CUDA_SAFE_CALL(cudaMalloc((void**) &d_temp2, 
                              2 * num_edges * sizeof(unsigned int)));
    cudppSetVGTemps(vGraphHandle, d_temp, d_temp2);

    // would like to call cudppVGraphMSTAllocateTemps() ...
    // this is not ideal
    unsigned int * d_temp3, * d_temp4;
    CUDA_SAFE_CALL(cudaMalloc((void**) &d_temp3,
                              2 * num_edges * sizeof(float))); 
    CUDA_SAFE_CALL(cudaMalloc((void**) &d_temp4, 
                              2 * num_edges * sizeof(float)));
    cudppSetVGMSTTemps(vGraphMSTHandle, d_temp3, d_temp4);

    // Run it once to avoid timing startup overhead
    cudppVGMinimumSpanningTree(vGraphHandle, vGraphMSTHandle, d_out);

    for (unsigned i = 0; i < testOptions.numIterations; i++)
    {
        cutStartTimer(timer);
        cudppVGMinimumSpanningTree(vGraphHandle, vGraphMSTHandle, d_out);
        cudaThreadSynchronize();
        cutStopTimer(timer);
    }

    CUDA_SAFE_CALL(cudaMemcpy(h_odata, d_out, num_nodes * sizeof(int),
                              cudaMemcpyDeviceToHost));

    CUTBoolean vg_result = cutComparei(reference, h_odata, num_nodes);
    retval += (CUTTrue == vg_result) ? 0 : 1;

    if (testOptions.debug)
    {
        for (unsigned int i = 0; i < num_nodes; i++)
        {
            printf("i: %d\tref: %i\todata: %i\n", i, reference[i], h_odata[i]);
        }
    }

    printf("v-graph minimum-spanning-tree test %s\n", 
           (CUTTrue == vg_result) ? "PASSED" : "FAILED");
    printf("Average execution time: %f ms\n", 
           cutGetTimerValue(timer) / testOptions.numIterations);

    result = cudppDestroyVGraph(vGraphHandle);

    if (result != CUDPP_SUCCESS)
    {
        printf("Error destroying v-graph object\n");
    }

    result = cudppDestroyVGraphMSTPlan(vGraphMSTHandle);

    if (result != CUDPP_SUCCESS)
    {
        printf("Error destroying v-graph minimum-spanning-tree plan\n");
    }

    cutDeleteTimer(timer);

    free(reference);

    CUDA_SAFE_CALL(cudaFree(d_out));
    CUDA_SAFE_CALL(cudaFree(d_idata));

    return retval;
}
