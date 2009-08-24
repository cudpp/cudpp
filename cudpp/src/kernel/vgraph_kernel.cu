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
 * vgraph_kernel.cu
 *
 * @brief CUDPP kernel-level vgraph routines
 */

/** \defgroup cudpp_kernel CUDPP Kernel-Level API
  * The CUDPP Kernel-Level API contains functions that run on the GPU 
  * device across a grid of Cooperative Thread Array (CTA, aka Thread
  * Block).  These kernels are declared \c __global__ so that they 
  * must be invoked from host (CPU) code.  They generally invoke GPU 
  * \c __device__ routines in the CUDPP \link cudpp_cta CTA-Level API\endlink. 
  * Kernel-Level API functions are used by CUDPP 
  * \link cudpp_app Application-Level\endlink functions to implement their 
  * functionality.
  * @{
  */

/** @name v-graph kernel-level functions
 * @{
 */

#include <cudpp_globals.h>
#include <cudpp_util.h>

/** @brief Specialized function for \a vgNeighborReduce that gathers
 * from \a srcaddr and permutes using \a destaddr
 *
 * Several steps encapsulated by a one-line function:
 *
 * 1. Each element gets its \a srcaddr address and subtracts 1 from it
 *    (because of the surely-premature optimization in \a
 *    vgNeighborReduce() ).
 *
 * 2. Using that address, gather from \a idata.
 *
 * 3. Permute that result into \a dest using the permutation address
 *    in \a destaddr.
 * 
 * @param[out] dest Output array
 * @param[in] destaddr Array of indices into \a dest; must be a permutation
 * @param[in] srcaddr Array of indices into \a idata for gather
 * @param[in] idata Source array for data; this data ends up in \a dest
 * @param[in] num_elements Total number of elements to process
 *
 * @see vgNeighborReduce
 */
template<class T>
__global__
void gatherAndPermuteMinus1(T * dest, 
                            const unsigned int * srcaddr, 
                            const unsigned int * destaddr, 
                            const T * idata, size_t num_elements)
{
    const int n      = blockDim.x;
    const int bid    = blockIdx.x;
    const int thid   = threadIdx.x;
    const int offset = __mul24(bid, n) + thid;

    if (offset < num_elements)
    {
        dest[destaddr[offset]] = idata[srcaddr[offset]-1];
    }
}

/** @brief For each elt, gathers from \a addr vector and sets to 0 if
 * not a head flag
 *
 * Templatized by shift; add the shift to the index.
 * 
 * Two steps encapsulated by a one-line function. For each element:
 *
 * 1. Gather from \a idata with address at \a addr into destination \a dest.
 *
 * 2. If the head flag (\a head_flags) at that element is zero, set \a
 *    dest to zero.
 *
 * Will not work in-place. \a dest must not be the same as \a src.
 * 
 * @param[out] dest Output array
 * @param[in] src Source array for data; this data ends up in \a dest
 * @param[in] addr Array of gather indices into \a src
 * @param[in] head_flags Per-element head flags
 * @param[in] num_elements Total number of elements to process
 *
 * @see vgNeighborReduce
 */
template<class T, int shift>
__global__
void vGraphGatherAndZero(T * dest, const T * src, const unsigned int * addr, 
                         const unsigned int * head_flags, size_t num_elements)
{
    const int n      = blockDim.x;
    const int bid    = blockIdx.x;
    const int thid   = threadIdx.x;
    const int offset = __mul24(bid, n) + thid;

    const T * idata = src + offset;
    const unsigned int * hf = head_flags + offset;
    const unsigned int * a = addr + offset;
    T * odata       = dest + offset;

    if (offset < num_elements)
    {
        int idx = int(*a) + shift;
        *odata = *hf ? idata[idx] : 0;
    }
}

/** @brief For each elt, gathers from end of its segment and sets to 0 if
 * not a head flag
 *
 * For each element:
 *
 * 1. Gather from \a idata with address at self + \a segment_length
 * into destination \a dest.
 *
 * 2. If the head flag (\a head_flags) at that element is zero, set \a
 *    dest to zero.
 *
 * Will not work in-place. \a dest must not be the same as \a src.
 * 
 * @param[out] dest Output array
 * @param[in] src Source array for data; this data ends up in \a dest
 * @param[in] segment_length How long is this segment?
 * @param[in] vertex_id Maps edges to vertices, modulated by shift
 * @param[in] head_flags Per-element head flags
 * @param[in] num_elements Total number of elements to process
 *
 * @see vgMinimumSpanningTree
 */
template<class T, int shift>
__global__
void vGraphMSTGatherAndZero(T * dest, const T * src, 
                            const unsigned int * segment_length, 
                            const int * vertex_id, 
                            const unsigned int * head_flags, 
                            size_t num_elements)
{
    const int n      = blockDim.x;
    const int bid    = blockIdx.x;
    const int thid   = threadIdx.x;
    const int offset = __mul24(bid, n) + thid;

    const T * idata = src + offset;
    const unsigned int * hf = head_flags + offset;
    T * odata       = dest + offset;

    if (offset < num_elements)
    {
        int my_vertex = vertex_id[offset] + shift;
        int last_in_segment_offset = segment_length[my_vertex] - 1;
        *odata = *hf ? idata[last_in_segment_offset] : T(0);
    }
}

/** @brief Copies src to dest (for debug purposes)
 *
 * Currently unused.
 *
 * @param[out] dest Output array
 * @param[in] src Input array
 * @param[in] num_elements Total number of elements to process
 */
template<class T>
__global__
void vGraphCopyDebug(T * dest, const T * src, size_t num_elements)
{
    const int n      = blockDim.x;
    const int bid    = blockIdx.x;
    const int thid   = threadIdx.x;
    const int offset = __mul24(bid, n) + thid;

    if (offset < num_elements)
    {
        dest[offset] = src[offset];
    }
}

/** @brief Scatter last data value in segment to dense output vector
 *
 * Am I at the end of a segment? If the next element is a head flag,
 * or we're at the end of the array, yes we are.
 *
 * If I'm at the end of the segment, write our source value (\a src)
 * into the destination array (\a dest) at the address marked by \a
 * addr MINUS ONE (see vgNeighborReduce())
 * 
 * @param[out] dest Output array
 * @param[in] src Source array for data; this data ends up in \a dest
 * @param[in] addr Array of scatter indices into \a src
 * @param[in] head_flags Per-element head flags
 * @param[in] num_elements Total number of elements to process
 *
 * @see vgNeighborReduce
 */
template<class T>
__global__
void gatherFromEndOfSegment(T * dest, const T * src, const unsigned int * addr,
                            const unsigned int * head_flags, 
                            size_t num_elements)
{
    const int n      = blockDim.x;
    const int bid    = blockIdx.x;
    const int thid   = threadIdx.x;
    const int offset = __mul24(bid, n) + thid;

    /* algorithm : if the next element has a head flag (or if we're at
     * the end), then scatter my src to dest[addr-1] (remember addr is
     * still 1 too big) */

    if ((offset < num_elements) &&
        ((offset == (num_elements - 1)) || head_flags[offset+1]))
    {
        dest[addr[offset]-1] = src[offset];
    }
}

template<class T>
__global__
void vGraphDistributeExcessKernel(T * dest, const T * capacity, 
                                  const T * temp_excess, 
                                  const T * temp_capacity,
                                  size_t num_elements)
{
    const int n      = blockDim.x;
    const int bid    = blockIdx.x;
    const int thid   = threadIdx.x;
    const int offset = __mul24(bid, n) + thid;

    if (offset < num_elements)
    {
        dest[offset] = 
            min(capacity[offset], 
                max(temp_excess[offset] - temp_capacity[offset], T(0)));
    }
}

// this was the random function used in JDO's dissertation
// it was from a Graphics Gem
// float frand(int s) { /* get random number from seed s */
//   s = shift(s, 13) ^ s;
//   s = (lo(s * (lo(lo(s * s) * 15731) + 789221)) + 1376312589) & 0x7fffffff;
//   // return (1.0 - itof(s)/1073741824.0);
//   return (1.0 - itof(s) * 0.000000000931322574615f);
// }

__global__
void random0Or1(unsigned int * out, const unsigned int * head_flags, int seed,
                size_t num_elements)
{
    const int n      = blockDim.x;
    const int bid    = blockIdx.x;
    const int thid   = threadIdx.x;
    const int offset = __mul24(bid, n) + thid;
    seed            += offset;

    if (offset < num_elements)
    {
        int s = (seed << 13) ^ seed;
        s = ((s * (s * s * 15731) + 789221) >> 7) & 0x1;
        /* only set the result of segment heads */
        out[offset] = s & head_flags[offset];
    }
}

__global__
void setHeadToOffset(unsigned int * out, unsigned int * endOfSegment,
                     const unsigned int * head_flags,
                     size_t num_elements)
{
    const int n      = blockDim.x;
    const int bid    = blockIdx.x;
    const int thid   = threadIdx.x;
    const int offset = __mul24(bid, n) + thid;

    if (offset < num_elements)
    {
        out[offset] = head_flags[offset] ? offset : 0;
        endOfSegment[offset] = 
            (offset == (num_elements - 1)) ? 1 : head_flags[offset + 1];
    }
}

template<class WeightsT>
__global__
void computeStarEdgesAndNeededSpace(unsigned int * neededspace,
                                    unsigned int * isStarEdge,
                                    int * starParent,
                                    const unsigned int * isParent,
                                    const unsigned int * cross_pointers,
                                    const WeightsT * minWeights,
                                    const WeightsT * weights,
                                    const unsigned int * myHead,
                                    const int * myVertexID,
                                    size_t num_elements)
{
    const int n      = blockDim.x;
    const int bid    = blockIdx.x;
    const int thid   = threadIdx.x;
    const int offset = __mul24(bid, n) + thid;

    if (offset < num_elements)
    {
        /* now star edges are where weight == mstplan->m_d_tmp AND the min edge 
         * is a child AND the opposite edge is a parent */
        unsigned int isStarEdgeLocal;
        isStarEdgeLocal = ((minWeights[offset] == weights[offset]) &&
                           (isParent[offset] == 0) &&
                           (isParent[cross_pointers[offset]] == 1)
                           );
        neededspace[offset] = ((isParent[offset] == 0) &&
                               (isStarEdgeLocal == 0));
        isStarEdge[offset] = isStarEdgeLocal;
        if (isStarEdgeLocal)
        {
            /* write into my head what my star parent is */
#ifdef __DEVICE_EMULATION__
#ifndef _MSC_VER                // MS Visual C++ does NOT like this printf
            printf("Edge with weight %f is part of MST\n", weights[offset]);
#endif
#endif
            starParent[myVertexID[offset]] = myVertexID[cross_pointers[offset]];
        }
        /* needs global sync after this */
    }
}

__global__
void markSurvivingEdges(unsigned int * survivingEdges,
                        const unsigned int * isStarEdge,
                        const int * myVertexID,
                        const unsigned int * isParent,
                        const int * starParent,
                        const unsigned int * cross_pointers,
                        size_t num_elements)
{
    const int n      = blockDim.x;
    const int bid    = blockIdx.x;
    const int thid   = threadIdx.x;
    const int offset = __mul24(bid, n) + thid;

    if (offset < num_elements)
    {
        unsigned int parent = isParent[offset];
        unsigned int survivor = 0;
        if (parent)
        {
            /* if I'm a parent, I survive if I'm NOT a star-edge */
            /* but for the purposes of this computation, just write 1,
             * because the next kernel will write over the star-edge vals */
            survivor = 1;
        }
        else if (isStarEdge[offset] == 0)
        {
            /* if I'm not a parent, I survive if I'm NOT a star-edge
             * AND (my pointer is neither into my own vertex nor my
             * starParent's edge)*/
            int myVtx = myVertexID[offset];
            unsigned int dest = myVertexID[cross_pointers[offset]];
            if ((dest != myVtx) && (dest != starParent[myVtx]))
            {
                survivor = 1;
            }
        } 
        else 
        {
            
        }
        survivingEdges[offset] = survivor;
    }
}


__global__
void clearStarParent(int * starParent,
                     size_t num_elements)
{
    const int n      = blockDim.x;
    const int bid    = blockIdx.x;
    const int thid   = threadIdx.x;
    const int offset = __mul24(bid, n) + thid;

    if (offset < num_elements)
    {
        starParent[offset] = -1;
    }
}

template<class T, int shift>
__global__
void vectorAdd(T * dest, const T * src, size_t num_elements)
{
    const int n      = blockDim.x;
    const int bid    = blockIdx.x;
    const int thid   = threadIdx.x;
    const int offset = __mul24(bid, n) + thid;

    if (offset < num_elements)
    {
        dest[offset] = src[offset] + shift;
    }
}



/** @} */ // end v-graph kernel functions
/** @} */ // end cudpp_kernel
