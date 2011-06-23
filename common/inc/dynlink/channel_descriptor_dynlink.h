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

#if !defined(__CHANNEL_DESCRIPTOR_H__)
#define __CHANNEL_DESCRIPTOR_H__

#if defined(__cplusplus)

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#include "driver_types.h"
#include "cuda_runtime_api_dynlink.h"
#include "host_defines.h"
#include "vector_types.h"

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

template<class T> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc(void)
{
	return dyn::cudaCreateChannelDesc(0, 0, 0, 0, cudaChannelFormatKindNone);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<char>(void)
{
  int e = (int)sizeof(char) * 8;

#if __SIGNED_CHARS__
  return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned);
#else
  return dyn::cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned);
#endif
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<signed char>(void)
{
  int e = (int)sizeof(signed char) * 8;

  return dyn::cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<unsigned char>(void)
{
  int e = (int)sizeof(unsigned char) * 8;

  return dyn::cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<char1>(void)
{
  int e = (int)sizeof(signed char) * 8;

  return dyn::cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<uchar1>(void)
{
  int e = (int)sizeof(unsigned char) * 8;

  return dyn::cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<char2>(void)
{
  int e = (int)sizeof(signed char) * 8;

  return dyn::cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindSigned);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<uchar2>(void)
{
  int e = (int)sizeof(unsigned char) * 8;

  return dyn::cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindUnsigned);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<char4>(void)
{
  int e = (int)sizeof(signed char) * 8;

  return dyn::cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindSigned);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<uchar4>(void)
{
  int e = (int)sizeof(unsigned char) * 8;

  return dyn::cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindUnsigned);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<short>(void)
{
  int e = (int)sizeof(short) * 8;

  return dyn::cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<unsigned short>(void)
{
  int e = (int)sizeof(unsigned short) * 8;

  return dyn::cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<short1>(void)
{
  int e = (int)sizeof(short) * 8;

  return dyn::cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<ushort1>(void)
{
  int e = (int)sizeof(unsigned short) * 8;

  return dyn::cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<short2>(void)
{
  int e = (int)sizeof(short) * 8;

  return dyn::cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindSigned);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<ushort2>(void)
{
  int e = (int)sizeof(unsigned short) * 8;

  return dyn::cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindUnsigned);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<short4>(void)
{
  int e = (int)sizeof(short) * 8;

  return dyn::cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindSigned);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<ushort4>(void)
{
  int e = (int)sizeof(unsigned short) * 8;

  return dyn::cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindUnsigned);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<int>(void)
{
  int e = (int)sizeof(int) * 8;

  return dyn::cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<unsigned int>(void)
{
  int e = (int)sizeof(unsigned int) * 8;

  return dyn::cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<int1>(void)
{
  int e = (int)sizeof(int) * 8;

  return dyn::cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<uint1>(void)
{
  int e = (int)sizeof(unsigned int) * 8;

  return dyn::cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<int2>(void)
{
  int e = (int)sizeof(int) * 8;

  return dyn::cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindSigned);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<uint2>(void)
{
  int e = (int)sizeof(unsigned int) * 8;

  return dyn::cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindUnsigned);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<int4>(void)
{
  int e = (int)sizeof(int) * 8;

  return dyn::cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindSigned);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<uint4>(void)
{
  int e = (int)sizeof(unsigned int) * 8;

  return dyn::cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindUnsigned);
}

#if !defined(__LP64__)

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<long>(void)
{
  int e = (int)sizeof(long) * 8;

  return dyn::cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<unsigned long>(void)
{
  int e = (int)sizeof(unsigned long) * 8;

  return dyn::cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<long1>(void)
{
  int e = (int)sizeof(long) * 8;

  return dyn::cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<ulong1>(void)
{
  int e = (int)sizeof(unsigned long) * 8;

  return dyn::cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<long2>(void)
{
  int e = (int)sizeof(long) * 8;

  return dyn::cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindSigned);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<ulong2>(void)
{
  int e = (int)sizeof(unsigned long) * 8;

  return dyn::cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindUnsigned);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<long4>(void)
{
  int e = (int)sizeof(long) * 8;

  return dyn::cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindSigned);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<ulong4>(void)
{
  int e = (int)sizeof(unsigned long) * 8;

  return dyn::cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindUnsigned);
}

#endif /* !__LP64__ */

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<float>(void)
{
  int e = (int)sizeof(float) * 8;

  return dyn::cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindFloat);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<float1>(void)
{
  int e = (int)sizeof(float) * 8;

  return dyn::cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindFloat);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<float2>(void)
{
  int e = (int)sizeof(float) * 8;

  return dyn::cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindFloat);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<float4>(void)
{
  int e = (int)sizeof(float) * 8;

  return dyn::cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindFloat);
}

#endif /* __cplusplus */

#endif /* !__CHANNEL_DESCRIPTOR_H__ */
