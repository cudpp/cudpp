// -------------------------------------------------------------
// cuDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Source: $
// $Revision:  $
// $Date: $
// ------------------------------------------------------------- 
// This source code is distributed under the terms of license.txt in
// the root directory of this source distribution.
// ------------------------------------------------------------- 

/**
 * @file
 * cudpp_vgraph.h
 *
 * @brief v-graph functionality header file - contains CUDPP interface
 * (not public)
 */

#ifndef _CUDPP_VGRAPH_H_
#define _CUDPP_VGRAPH_H_

#include "cudpp_globals.h"
#include "cudpp.h"
#include "cudpp_plan.h"

// Functions
extern "C"
void cudppVGNeighborReduceDispatch (const CUDPPVGraphPlan *vgplan,
                                    const CUDPPVGraphNRPlan *vgnrplan,
                                    void * d_out, const void * d_idata);
template <class T, CUDPPOperator op>
void vgNeighborReduce(const CUDPPVGraphPlan *plan,
                      const CUDPPVGraphNRPlan *nrplan,
                      T * d_out, const T * d_idata);

extern "C"
void cudppVGDistributeExcessDispatch (const CUDPPVGraphPlan *vgplan,
                                      const CUDPPVGraphDEPlan *vgdeplan,
                                      void * d_out, const void * d_capacity,
                                      const void * d_excess);
template <class T>
void vgDistributeExcess(const CUDPPVGraphPlan *plan,
                        const CUDPPVGraphDEPlan *deplan,
                        void * d_out, const void * d_capacity,
                        const void * d_excess);

extern "C"
void cudppVGMinimumSpanningTreeDispatch(CUDPPVGraphPlan * vGraphHandle,
                                        CUDPPVGraphMSTPlan * vGraphMSTHandle,
                                        void * d_out);

template <class T>
void vgMinimumSpanningTree(const CUDPPVGraphPlan *plan,
                           const CUDPPVGraphMSTPlan *mstplan,
                           T * d_out);

extern "C"
void initializeVGraphStorage(CUDPPVGraphPlan *plan,
                             const unsigned int * h_segment_descriptor,
                             const unsigned int * h_cross_pointers,
                             const unsigned int * h_head_flags,
                             const float        * weights);

extern "C"
void freeVGraphStorage(CUDPPVGraphPlan *plan);

extern "C"
void initializeVGraphNRStorage(CUDPPVGraphNRPlan *plan);

extern "C"
void freeVGraphNRStorage(CUDPPVGraphNRPlan *plan);

extern "C"
void initializeVGraphDEStorage(CUDPPVGraphDEPlan *plan);

extern "C"
void freeVGraphDEStorage(CUDPPVGraphDEPlan *plan);

extern "C"
void initializeVGraphMSTStorage(CUDPPVGraphMSTPlan *plan);

extern "C"
void freeVGraphMSTStorage(CUDPPVGraphMSTPlan *plan);
#endif /* _CUDPP_VGRAPH_H_ */

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
