/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#if !defined(__CUDA_TEXTURE_TYPES_H__)
#define __CUDA_TEXTURE_TYPES_H__

#if defined(__cplusplus) && defined(__CUDACC__)

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#include "channel_descriptor_dynlink.h"
#include "driver_types.h"
#include "host_defines.h"
#include "texture_types.h"

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

/*TEXTURE_TYPE*/
template<class T, int dim = 1, enum cudaTextureReadMode = cudaReadModeElementType>
struct texture : public textureReference
{
  __host__ texture(int                         norm  = 0,
                   enum cudaTextureFilterMode  fMode = cudaFilterModePoint,
                   enum cudaTextureAddressMode aMode = cudaAddressModeClamp)
  {
    normalized     = norm;
    filterMode     = fMode;
    addressMode[0] = aMode;
    addressMode[1] = aMode;
    addressMode[2] = aMode;
    channelDesc    = cudaCreateChannelDesc<T>();
  }

  __host__ texture(int                          norm,
                   enum cudaTextureFilterMode   fMode,
                   enum cudaTextureAddressMode  aMode,
                   struct cudaChannelFormatDesc desc)
  {
    normalized     = norm;
    filterMode     = fMode;
    addressMode[0] = aMode;
    addressMode[1] = aMode;
    addressMode[2] = aMode;
    channelDesc    = desc;
  }
};

#endif /* __cplusplus && __CUDACC__ */

#endif /* !__CUDA_TEXTURE_TYPES_H__ */
