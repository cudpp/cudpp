// -------------------------------------------------------------
// CUDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision$
// $Date$
// ------------------------------------------------------------- 
// This source code is distributed under the terms of license.txt 
// in the root directory of this source distribution.
// ------------------------------------------------------------- 

/**
* @file
* cudpp_compress.h
*
* @brief Compress functionality header file - contains CUDPP interface (not public)
*/

#ifndef _CUDPP_COMPRESS_H_
#define _CUDPP_COMPRESS_H_

class CUDPPCompressPlan;

extern "C"
void allocCompressStorage(CUDPPCompressPlan* plan);

extern "C"
void freeCompressStorage(CUDPPCompressPlan* plan);

extern "C"
void cudppCompressDispatch(void *d_uncompressed,
                           void *d_bwtIndex,
                           void *d_histSize,
                           void *d_hist,
                           void *d_encodeOffset,
                           void *d_compressedSize,
                           void *d_compressed,
                           size_t numElements,
                           const CUDPPCompressPlan *plan);

#endif // _CUDPP_COMPRESS_H_