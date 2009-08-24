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
 * vgraph_gold.h
 *
 * @brief Header for generating and testing graphs for use by vgraph
 */

#include <stdio.h>
#include <cutil.h>

#include "cudpp.h"
#include "cudpp_testrig_options.h"

extern "C"
void generateIntTestGraphNR(const int num_nodes, const int num_edges,
                            int ** idata,
                            int ** odata,
                            unsigned int ** segment_descriptor,
                            unsigned int ** cross_pointers,
                            unsigned int ** head_flags,
                            int ** reference,
                            const testrigOptions & testOptions);

extern "C"
void generateIntExcessTestGraph(const int num_nodes, const int num_edges,
                                int ** idata,
                                int ** odata,
                                unsigned int ** segment_descriptor,
                                unsigned int ** cross_pointers,
                                unsigned int ** head_flags,
                                int ** capacity, int ** excess,
                                int ** reference,
                                const testrigOptions & testOptions);

extern "C"
void generateIntTestGraphMST(const int num_nodes, const int num_edges,
                             int ** idata,
                             int ** odata,
                             unsigned int ** segment_descriptor,
                             unsigned int ** cross_pointers,
                             unsigned int ** head_flags,
                             float ** weights,
                             int ** reference,
                             const testrigOptions & testOptions);

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
