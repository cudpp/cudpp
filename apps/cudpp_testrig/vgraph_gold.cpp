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
 * vgraph_gold.cpp
 *
 * @brief Generates and tests graphs for use by vgraph
 */

#include "vgraph_gold.h"
#include <iostream>

#ifdef _USE_BOOST_
/* can't include this in a .cu file since nvcc chokes on it */
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/kruskal_min_spanning_tree.hpp>
#include <boost/graph/random.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/graph/graphviz.hpp>
#include <boost/graph/graph_utility.hpp>
#include <boost/random/mersenne_twister.hpp>
using namespace boost;

// Would really like to use a typedef template (Graph<T>) but I don't
// think that's allowed in C++
typedef adjacency_list<listS, listS, undirectedS,
                       property<vertex_index_t, int>,         
                       property<edge_weight_t, float> > FGraph;
typedef adjacency_list<listS, listS, undirectedS,
                       property<vertex_index_t, int>,         
                       property<edge_weight_t, int> > IGraph;

#endif /* _USE_BOOST_ */

template<class T>
std::ostream & print_debug(std::ostream & os, const char * name, int count, 
                           T ** val)
{
    os << name << ": ";
    for (int i = 0; i < count; i++)
    {
        os << (*val)[i] << " ";
    }
    os << std::endl;
    return os;
}

#ifdef _USE_BOOST_
template<class T, class WeightsT>
void boostToVGraph(const FGraph & g,
                   const int num_nodes, const int num_edges, 
                   T ** idata,
                   T ** odata,
                   unsigned int ** segment_descriptor,
                   unsigned int ** cross_pointers,
                   unsigned int ** head_flags,
                   WeightsT ** weights,
                   T ** reference,
                   const testrigOptions & testOptions)
{
    typedef FGraph Graph;

    size_t nodes_ui_bytes = sizeof(unsigned int) * num_nodes;
    // size_t edges_ui_bytes = sizeof(unsigned int) * 2 * num_edges; /// XXX 2*
    // size_t nodes_T_bytes = sizeof(T) * num_nodes;
    // size_t edges_T_bytes = sizeof(T) * 2 * num_edges;
    
    /* ok, let's start filling the data structures */
    unsigned int * vtx_index = (unsigned int *) malloc(nodes_ui_bytes);
    typename graph_traits<Graph>::vertex_iterator ui, ui_end;
    typename graph_traits<Graph>::out_edge_iterator ei, ei_end;
    unsigned int v;             // vertex index
    unsigned int edges_seen;    // total number of edges seen
    for (tie(ui,ui_end) = vertices(g), v=0, edges_seen=0; 
         ui != ui_end;
         ++ui, ++v) {
        (*head_flags)[edges_seen] = 1; // assumes head_flags initialized to 0
        vtx_index[v] = edges_seen;
        if (testOptions.debug)
        {
            std::cout << get(vertex_index, g, *ui) << " <--> ";
        }
        unsigned int edges_this_vtx; // edges just for this vertex
        unsigned int neighbor_sum; // sum of neighbor vertex #s
        for (tie(ei,ei_end) = out_edges(*ui,g), edges_this_vtx = 0, 
                 neighbor_sum = 0;
             ei != ei_end;
             ++ei, ++edges_seen, ++edges_this_vtx)
        {
            // neighbor_sum += target(*ei,g);
            neighbor_sum += get(vertex_index, g, target(*ei, g));
            if (testOptions.debug)
            {
                std::cout << get(vertex_index, g, target(*ei, g)) << " ";
            }
        }
        if (testOptions.debug)
        {
            std::cout << std::endl;
        }
        (*segment_descriptor)[v] = edges_this_vtx;
        if (edges_this_vtx == 0)
        {
            std::cerr << std::endl << "Vertex " << v 
                      << " has no edges, exiting." << std::endl;
            exit(1);
        }
        (*idata)[v] = v;
        (*odata)[v] = -1;
        // this "reference" is for neighbor-reduce
        if (reference != NULL)
        {
            (*reference)[v] = neighbor_sum;
        }
    }
    
    /* this loop fills in cross pointers */
    for (tie(ui,ui_end) = vertices(g), v=0; ui != ui_end; ++ui, ++v)
    {
        for (tie(ei,ei_end) = out_edges(*ui,g); ei != ei_end; ++ei)
        {
            unsigned int remote_v = get(vertex_index, g, target(*ei, g));
            if (v < remote_v)
            {
                unsigned int start = vtx_index[v];
                unsigned int end = vtx_index[remote_v];
                (*cross_pointers)[start] = end;
                (*cross_pointers)[end] = start;
                if (weights != NULL)
                {
                    float weight = get(edge_weight, g, *ei);
                    (*weights)[start] = weight;
                    (*weights)[end] = weight;
                }
                vtx_index[v]++;
                vtx_index[remote_v]++;
            }
        }
    }
    
    free(vtx_index);
    if (testOptions.debug)
    {
        print_debug(std::cout, "idata     ", num_nodes, idata);
        print_debug(std::cout, "seg descr ", num_nodes, segment_descriptor);
        print_debug(std::cout, "cross ptrs", num_edges*2, cross_pointers);
        print_debug(std::cout, "head flags", num_edges*2, head_flags);
        if (weights != NULL)
        {
            print_debug(std::cout, "weights   ", num_edges*2, weights);
        }
        //// print_debug(std::cout, "reference ", num_nodes, reference);
    }
}
#endif

template<class T>
void generateTestGraphNR(const int num_nodes, const int num_edges,
                         T ** idata,
                         T ** odata,
                         unsigned int ** segment_descriptor,
                         unsigned int ** cross_pointers,
                         unsigned int ** head_flags,
                         T ** reference,
                         const testrigOptions & testOptions)
{
    size_t nodes_ui_bytes = sizeof(unsigned int) * num_nodes;
    size_t edges_ui_bytes = sizeof(unsigned int) * 2 * num_edges;
    size_t nodes_T_bytes = sizeof(T) * num_nodes;
    // size_t edges_T_bytes = sizeof(T) * 2 * num_edges;

    *idata = (T *) malloc(nodes_T_bytes);
    memset((void *) *idata, 0, nodes_T_bytes);
    *odata = (T *) malloc(nodes_T_bytes);
    memset((void *) *odata, 0, nodes_T_bytes);
    *segment_descriptor = (unsigned int *) malloc(nodes_ui_bytes);
    memset((void *) *segment_descriptor, 0, nodes_ui_bytes);
    *cross_pointers = (unsigned int *) malloc(edges_ui_bytes);
    memset((void *) *cross_pointers, 0, edges_ui_bytes);
    *head_flags = (unsigned int *) malloc(edges_ui_bytes);
    memset((void *) *head_flags, 0, edges_ui_bytes);
    *reference = (T *) malloc(nodes_T_bytes);
    memset((void *) *reference, 0, nodes_T_bytes);

#ifdef _USE_BOOST_
    typedef FGraph Graph;
    Graph g;
    
    /* create a random graph */
    mt19937 rng;                // defined in mersenne_twister - some
                                // random # generator - don't know
                                // what it is
    generate_random_graph(g, num_nodes, num_edges, rng);

    /* randomly set weights */
    boost::uniform_real<> ur(1,10); 
    boost::variate_generator<boost::mt19937 &, 
        boost::uniform_real<> > ew1rg(rng, ur);
    randomize_property<edge_weight_t>(g, ew1rg);

    /* set vertex ids */
    graph_traits<Graph>::vertex_iterator ui, ui_end;
    unsigned int v = 0;
    for (tie(ui,ui_end) = vertices(g); ui != ui_end; ++ui)
    {
        put(vertex_index, g, *ui, v++);
    }

    /* finished with creating random graph */

#if 0
    if (testOptions.debug)
    {
        // print, if necessary
        dynamic_properties dp;
        dp.property("id", get(vertex_index, g));
        dp.property("weight", get(edge_weight, g));
        /* I don't know why I need "id" in here but not "weight" */
        /* http://www.boost.org/doc/libs/1_35_0/libs/graph/example/graphviz.cpp */
        write_graphviz(std::cout, g, dp, std::string("id"));
    }
#endif
    
    boostToVGraph(g, num_nodes, num_edges, idata, odata, segment_descriptor, 
                  cross_pointers, head_flags, (T **) NULL, reference, 
                  testOptions);
    
#else
    if (testOptions.debug) {} // removes warning
    /* manually fills a test case with 5 nodes and 6 edges */
    int tiny_idata[] = {1,2,4,8,16};
    int tiny_odata[] = {1001,1002,1003,1004,1005};
    unsigned int tiny_segment_descriptor[] = {1,3,3,2,3};
    //                           vtx_index = {0,1,4,7,9};
    unsigned int tiny_cross_pointers[] = {1,0,4,9,2,7,10,5,11,3,6,8};
    unsigned int tiny_head_flags[] = {1,1,0,0,1,0,0,1,0,1,0,0};
    int tiny_reference[] = {2,21,26,20,14};
    memcpy(*idata, tiny_idata, nodes_T_bytes);
    memcpy(*odata, tiny_odata, nodes_T_bytes);
    memcpy(*segment_descriptor, tiny_segment_descriptor, nodes_ui_bytes);
    memcpy(*cross_pointers, tiny_cross_pointers, edges_ui_bytes);
    memcpy(*head_flags, tiny_head_flags, edges_ui_bytes);
    memcpy(*reference, tiny_reference, nodes_T_bytes);
#endif /* _USE_BOOST_ */
}

template<class T>
void generateExcessTestGraph(const int num_nodes, const int num_edges,
                             T ** idata,
                             T ** odata,
                             unsigned int ** segment_descriptor,
                             unsigned int ** cross_pointers,
                             unsigned int ** head_flags,
                             T ** capacity, T ** excess,
                             T ** reference,
                             const testrigOptions & testOptions)
{
    size_t nodes_ui_bytes = sizeof(unsigned int) * num_nodes;
    size_t edges_ui_bytes = sizeof(unsigned int) * num_edges;
    size_t nodes_T_bytes = sizeof(T) * num_nodes;
    size_t edges_T_bytes = sizeof(T) * num_edges;

    *odata = (T *) malloc(edges_T_bytes);
    memset((void *) *odata, 0, edges_T_bytes);
    *segment_descriptor = (unsigned int *) malloc(nodes_ui_bytes);
    memset((void *) *segment_descriptor, 0, nodes_ui_bytes);
    *head_flags = (unsigned int *) malloc(edges_ui_bytes);
    memset((void *) *head_flags, 0, edges_ui_bytes);
    *capacity = (T *) malloc(edges_T_bytes);
    memset((void *) *capacity, 0, edges_T_bytes);
    *excess = (T *) malloc(nodes_T_bytes);
    memset((void *) *excess, 0, nodes_T_bytes);
    *reference = (T *) malloc(edges_T_bytes);
    memset((void *) *reference, 0, edges_T_bytes);

    if (testOptions.debug)      // removes warning
    {
        idata=idata; 
        cross_pointers=cross_pointers;
    } 
    /* manually fills a test case with 2 nodes and 6 edges */
    int tiny_odata[] = {100,100,100,100,100,100};
    unsigned int tiny_segment_descriptor[] = {2,4};
    unsigned int tiny_head_flags[] = {1,0,1,0,0,0};
    int tiny_capacity[] = {13,5,7,11,6,14};
    int tiny_excess[] = {15,20};
    int tiny_reference[] = {13,2,7,11,2,0};
    memcpy(*odata, tiny_odata, edges_T_bytes);
    memcpy(*segment_descriptor, tiny_segment_descriptor, nodes_ui_bytes);
    memcpy(*head_flags, tiny_head_flags, edges_ui_bytes);
    memcpy(*capacity, tiny_capacity, edges_T_bytes);
    memcpy(*excess, tiny_excess, nodes_T_bytes);
    memcpy(*reference, tiny_reference, edges_T_bytes);
}

template<class T, class WeightsT>
void generateMSTGraph(const int num_nodes, const int num_edges,
                      T ** idata,
                      T ** odata,
                      unsigned int ** segment_descriptor,
                      unsigned int ** cross_pointers,
                      unsigned int ** head_flags,
                      WeightsT ** weights,
                      T ** reference,
                      const testrigOptions & testOptions)
{
    size_t nodes_ui_bytes = sizeof(unsigned int) * num_nodes;
    size_t edges_ui_bytes = sizeof(unsigned int) * 2 * num_edges;
    size_t nodes_T_bytes = sizeof(T) * num_nodes;
    // size_t edges_T_bytes = sizeof(T) * 2 * num_edges;
    size_t edges_WeightsT_bytes = sizeof(WeightsT) * 2 * num_edges;
    
    *idata = (T *) malloc(nodes_T_bytes);
    memset((void *) *idata, 0, nodes_T_bytes);
    *odata = (T *) malloc(nodes_T_bytes);
    memset((void *) *odata, 0, nodes_T_bytes);
    *segment_descriptor = (unsigned int *) malloc(nodes_ui_bytes);
    memset((void *) *segment_descriptor, 0, nodes_ui_bytes);
    *cross_pointers = (unsigned int *) malloc(edges_ui_bytes);
    memset((void *) *cross_pointers, 0, edges_ui_bytes);
    *head_flags = (unsigned int *) malloc(edges_ui_bytes);
    memset((void *) *head_flags, 0, edges_ui_bytes);
    *weights = (WeightsT *) malloc(edges_WeightsT_bytes);
    memset((void *) *weights, 0, edges_WeightsT_bytes);
    *reference = (T *) malloc(nodes_T_bytes);
    memset((void *) *reference, 0, nodes_T_bytes);
    
#ifdef _USE_BOOST_
    FGraph g;
    typedef FGraph Graph;
    
    /* create a random graph */
    mt19937 rng;                // defined in mersenne_twister - some
                                // random # generator - don't know
                                // what it is
    generate_random_graph(g, num_nodes, num_edges, rng);
    graph_traits<Graph>::vertex_iterator ui, ui_end;
    int v = 0;
    for (tie(ui,ui_end) = vertices(g); ui != ui_end; ++ui)
    {
        put(vertex_index, g, *ui, v++);
    }
    
    boost::uniform_real<> ur(1,10); 
    boost::variate_generator<boost::mt19937 &, 
        boost::uniform_real<> > ew1rg(rng, ur);
    randomize_property<edge_weight_t>(g, ew1rg);

    boostToVGraph(g, num_nodes, num_edges, idata, odata, segment_descriptor, 
                  cross_pointers, head_flags, weights, reference, testOptions);
    /* finished with creating random graph */
    
    typedef graph_traits < Graph >::edge_descriptor Edge;
    property_map < Graph, edge_weight_t >::type weight = get(edge_weight, g);
    std::vector < Edge > spanning_tree;
    
    kruskal_minimum_spanning_tree(g, std::back_inserter(spanning_tree));
    
    if (testOptions.debug)
    {
        std::cout << "Print the edges in the MST:" << std::endl;
        for (std::vector < Edge >::iterator ei = spanning_tree.begin();
             ei != spanning_tree.end(); ++ei) {
            std::cout << get(vertex_index, g, source(*ei, g)) << " <--> " 
                      << get(vertex_index, g, target(*ei, g))
                      << " with weight of " << weight[*ei]
                      << std::endl;
        }
    }
#endif // _USE_BOOST_
}



extern "C"
void generateIntTestGraphNR(const int num_nodes, const int num_edges,
                            int ** idata,
                            int ** odata,
                            unsigned int ** segment_descriptor,
                            unsigned int ** cross_pointers,
                            unsigned int ** head_flags,
                            int ** reference,
                            const testrigOptions & testOptions)
{
    generateTestGraphNR(num_nodes, num_edges, idata, odata, segment_descriptor,
                        cross_pointers, head_flags, reference, testOptions);
}

extern "C"
void generateIntExcessTestGraph(const int num_nodes, const int num_edges,
                                int ** idata,
                                int ** odata,
                                unsigned int ** segment_descriptor,
                                unsigned int ** cross_pointers,
                                unsigned int ** head_flags,
                                int ** capacity, int ** excess,
                                int ** reference,
                                const testrigOptions & testOptions)
{
    generateExcessTestGraph(num_nodes, num_edges, idata, odata,
                            segment_descriptor, cross_pointers, head_flags, 
                            capacity, excess, reference, testOptions);
}

extern "C"
void generateIntTestGraphMST(const int num_nodes, const int num_edges,
                             int ** idata,
                             int ** odata,
                             unsigned int ** segment_descriptor,
                             unsigned int ** cross_pointers,
                             unsigned int ** head_flags,
                             float ** weights,
                             int ** reference,
                             const testrigOptions & testOptions)
{
    generateMSTGraph(num_nodes, num_edges, idata, odata, segment_descriptor,
                     cross_pointers, head_flags, weights, reference, 
                     testOptions);
}


// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
