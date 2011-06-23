/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA) 
 * associated with this source code for terms and conditions that govern 
 * your use of this NVIDIA software.
 * 
 */

#if !defined(__CUDA_RUNTIME_DYNLINK_H__)
#define __CUDA_RUNTIME_DYNLINK_H__

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#include "host_config.h"

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#include "builtin_types.h"
#include "channel_descriptor_dynlink.h"
#include "cuda_runtime_api_dynlink.h"
#include "driver_functions.h"
#include "host_defines.h"
#include "vector_functions.h"

#if defined(__CUDACC__)

#include "common_functions_dynlink.h"
#include "cuda_texture_types_dynlink.h"
#include "device_functions_dynlink.h"
#include "device_launch_parameters.h"

#endif /* __CUDACC__ */

#if defined(__cplusplus)

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

template<class T>
__inline__ __host__ cudaError_t cudaSetupArgument(
  T      arg,
  size_t offset
)
{
    return dyn::cudaSetupArgument((const void*)&arg, sizeof(T), offset);
}

#if defined(__CUDACC__)

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

static __inline__ __host__ cudaError_t cudaMemcpyToSymbol(
        char                *symbol,
  const void                *src,
        size_t               count,
        size_t               offset = 0,
        enum cudaMemcpyKind  kind   = cudaMemcpyHostToDevice
)
{
  return cudaMemcpyToSymbol((const char*)symbol, src, count, offset, kind);
}

template<class T>
__inline__ __host__ cudaError_t cudaMemcpyToSymbol(
  const T                   &symbol,
  const void                *src,
        size_t               count,
        size_t               offset = 0,
        enum cudaMemcpyKind  kind   = cudaMemcpyHostToDevice
)
{
  return cudaMemcpyToSymbol((const char*)&symbol, src, count, offset, kind);
}

static __inline__ __host__ cudaError_t cudaMemcpyToSymbolAsync(
        char                *symbol,
  const void                *src,
        size_t               count,
        size_t               offset,
        enum cudaMemcpyKind  kind,
        cudaStream_t         stream
)
{
  return cudaMemcpyToSymbolAsync((const char*)symbol, src, count, offset, kind, stream);
}

template<class T>
__inline__ __host__ cudaError_t cudaMemcpyToSymbolAsync(
  const T                   &symbol,
  const void                *src,
        size_t               count,
        size_t               offset,
        enum cudaMemcpyKind  kind,
        cudaStream_t         stream
)
{
  return cudaMemcpyToSymbolAsync((const char*)&symbol, src, count, offset, kind, stream);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

static __inline__ __host__ cudaError_t cudaMemcpyFromSymbol(
  void                *dst,
  char                *symbol,
  size_t               count,
  size_t               offset = 0,
  enum cudaMemcpyKind  kind   = cudaMemcpyDeviceToHost
)
{
  return cudaMemcpyFromSymbol(dst, (const char*)symbol, count, offset, kind);
}

template<class T>
__inline__ __host__ cudaError_t cudaMemcpyFromSymbol(
        void                *dst,
  const T                   &symbol,
        size_t               count,
        size_t               offset = 0,
        enum cudaMemcpyKind  kind   = cudaMemcpyDeviceToHost
)
{
  return cudaMemcpyFromSymbol(dst, (const char*)&symbol, count, offset, kind);
}

static __inline__ __host__ cudaError_t cudaMemcpyFromSymbolAsync(
  void                *dst,
  char                *symbol,
  size_t               count,
  size_t               offset,
  enum cudaMemcpyKind  kind,
  cudaStream_t         stream
)
{
  return cudaMemcpyFromSymbolAsync(dst, (const char*)symbol, count, offset, kind, stream);
}

template<class T>
__inline__ __host__ cudaError_t cudaMemcpyFromSymbolAsync(
        void                *dst,
  const T                   &symbol,
        size_t               count,
        size_t               offset,
        enum cudaMemcpyKind  kind,
        cudaStream_t         stream
)
{
  return cudaMemcpyFromSymbolAsync(dst, (const char*)&symbol, count, offset, kind, stream);
}

static __inline__ __host__ cudaError_t cudaGetSymbolAddress(
  void **devPtr,
  char  *symbol
)
{
  return cudaGetSymbolAddress(devPtr, (const char*)symbol);
}

template<class T>
__inline__ __host__ cudaError_t cudaGetSymbolAddress(
        void **devPtr,
  const T     &symbol
)
{
  return cudaGetSymbolAddress(devPtr, (const char*)&symbol);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

static __inline__ __host__ cudaError_t cudaGetSymbolSize(
  size_t *size,
  char   *symbol
)
{
  return cudaGetSymbolSize(size, (const char*)symbol);
}

template<class T>
__inline__ __host__ cudaError_t cudaGetSymbolSize(
        size_t *size,
  const T      &symbol
)
{
  return cudaGetSymbolSize(size, (const char*)&symbol);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

template<class T, int dim, enum cudaTextureReadMode readMode>
__inline__ __host__ cudaError_t cudaBindTexture(
        size_t                           *offset,
  const struct texture<T, dim, readMode> &tex,
  const void                             *devPtr,
  const struct cudaChannelFormatDesc     &desc,
        size_t                            size = UINT_MAX
)
{
  return cudaBindTexture(offset, &tex, devPtr, &desc, size);
}

template<class T, int dim, enum cudaTextureReadMode readMode>
__inline__ __host__ cudaError_t cudaBindTexture(
        size_t                           *offset,
  const struct texture<T, dim, readMode> &tex,
  const void                             *devPtr,
        size_t                            size = UINT_MAX
)
{
  return cudaBindTexture(offset, tex, devPtr, tex.channelDesc, size);
}

template<class T, int dim, enum cudaTextureReadMode readMode>
__inline__ __host__ cudaError_t cudaBindTextureToArray(
  const struct texture<T, dim, readMode> &tex,
  const struct cudaArray                 *array,
  const struct cudaChannelFormatDesc     &desc
)
{
  return cudaBindTextureToArray(&tex, array, &desc);
}

template<class T, int dim, enum cudaTextureReadMode readMode>
__inline__ __host__ cudaError_t cudaBindTextureToArray(
  const struct texture<T, dim, readMode> &tex,
  const struct cudaArray                 *array
)
{
  struct cudaChannelFormatDesc desc;
  cudaError_t                  err = cudaGetChannelDesc(&desc, array);

  return err == cudaSuccess ? cudaBindTextureToArray(tex, array, desc) : err;
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

template<class T, int dim, enum cudaTextureReadMode readMode>
__inline__ __host__ cudaError_t cudaUnbindTexture(
  const struct texture<T, dim, readMode> &tex
)
{
  return cudaUnbindTexture(&tex);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

template<class T, int dim, enum cudaTextureReadMode readMode>
__inline__ __host__ cudaError_t cudaGetTextureAlignmentOffset(
        size_t                           *offset,
  const struct texture<T, dim, readMode> &tex
)
{
  return cudaGetTextureAlignmentOffset(offset, &tex);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

template<class T>
__inline__ __host__ cudaError_t cudaLaunch(
  T *symbol
)
{
  return cudaLaunch((const char*)symbol);
}

#endif /* __CUDACC__ */

#endif /* __cplusplus */

#endif /* !__CUDA_RUNTIME_DYNLINK_H__ */
