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

// ----------------------------------------------------------------------------
// 
// Main public header file for the CompUte Device Api
//
// ----------------------------------------------------------------------------

#ifndef __cuda_dynlink_h__
#define __cuda_dynlink_h__


/* CUDA API version number */
#define CUDA_VERSION 3000 /* 3.0 */

#ifdef __cplusplus
extern "C" {
#endif
    typedef unsigned int CUdeviceptr; 

    typedef int CUdevice; 
    typedef struct CUctx_st *CUcontext;
    typedef struct CUmod_st *CUmodule;
    typedef struct CUfunc_st *CUfunction;
    typedef struct CUarray_st *CUarray;
    typedef struct CUtexref_st *CUtexref;
    typedef struct CUevent_st *CUevent;
    typedef struct CUstream_st *CUstream;

/************************************
 **
 **    Enums
 **
 ***********************************/

//
// context creation flags
//
typedef enum CUctx_flags_enum {
    CU_CTX_SCHED_AUTO  = 0,
    CU_CTX_SCHED_SPIN  = 1,
    CU_CTX_SCHED_YIELD = 2,
    CU_CTX_SCHED_MASK  = 0x3,
    CU_CTX_FLAGS_MASK  = CU_CTX_SCHED_MASK
} CUctx_flags;

//
// array formats
//
typedef enum CUarray_format_enum {
    CU_AD_FORMAT_UNSIGNED_INT8  = 0x01,
    CU_AD_FORMAT_UNSIGNED_INT16 = 0x02,
    CU_AD_FORMAT_UNSIGNED_INT32 = 0x03,
    CU_AD_FORMAT_SIGNED_INT8    = 0x08,
    CU_AD_FORMAT_SIGNED_INT16   = 0x09,
    CU_AD_FORMAT_SIGNED_INT32   = 0x0a,
    CU_AD_FORMAT_HALF           = 0x10,
    CU_AD_FORMAT_FLOAT          = 0x20
} CUarray_format;

//
// Texture reference addressing modes
//
typedef enum CUaddress_mode_enum {
    CU_TR_ADDRESS_MODE_WRAP = 0,
    CU_TR_ADDRESS_MODE_CLAMP = 1,
    CU_TR_ADDRESS_MODE_MIRROR = 2,
} CUaddress_mode;

//
// Texture reference filtering modes
//
typedef enum CUfilter_mode_enum {
    CU_TR_FILTER_MODE_POINT = 0,
    CU_TR_FILTER_MODE_LINEAR = 1
} CUfilter_mode;

//
// Device properties
//
typedef enum CUdevice_attribute_enum {
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 2,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y = 3,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z = 4,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 5,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y = 6,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z = 7,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8,
    CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK = 8,      // Deprecated, use CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK
    CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY = 9,
    CU_DEVICE_ATTRIBUTE_WARP_SIZE = 10,
    CU_DEVICE_ATTRIBUTE_MAX_PITCH = 11,
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK = 12,
    CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK = 12,         // Deprecated, use CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK
    CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13,
    CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT = 14,

    CU_DEVICE_ATTRIBUTE_GPU_OVERLAP = 15,
    CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16,
    CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT = 17
} CUdevice_attribute;

//
// Legacy device properties
//
typedef struct CUdevprop_st {
    int maxThreadsPerBlock;
    int maxThreadsDim[3];
    int maxGridSize[3]; 
    int sharedMemPerBlock;
    int totalConstantMemory;
    int SIMDWidth;
    int memPitch;
    int regsPerBlock;
    int clockRate;
    int textureAlign;
} CUdevprop;

//
// Memory types
//
typedef enum CUmemorytype_enum {
    CU_MEMORYTYPE_HOST = 0x01,
    CU_MEMORYTYPE_DEVICE = 0x02,
    CU_MEMORYTYPE_ARRAY = 0x03
} CUmemorytype;


//
// Online compiler options
//
typedef enum CUjit_option_enum
{
    // CU_JIT_MAX_REGISTERS - Max number of registers that a thread may use.
    CU_JIT_MAX_REGISTERS            = 0,

    // CU_JIT_THREADS_PER_BLOCK -
    // IN: Specifies minimum number of threads per block to target compilation for
    // OUT: Returns the number of threads the compiler actually targeted.  This
    // restricts the resource utilization fo the compiler (e.g. max registers) such
    // that a block with the given number of threads should be able to launch based
    // on register limitations.  Note, this option does not currently take into
    // account any other resource limitations, such as shared memory utilization.
    CU_JIT_THREADS_PER_BLOCK,

    // CU_JIT_WALL_TIME - returns a float value in the option of the wall clock
    // time, in milliseconds, spent creating the cubin
    CU_JIT_WALL_TIME,

    // CU_JIT_INFO_LUG_BUFFER - pointer to a buffer in which to print any log
    // messsages from PTXAS that are informational in nature
    CU_JIT_INFO_LOG_BUFFER,

    // CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES -
    // IN: Log buffer size in bytes.  Log messages will be capped at this size
    // (including null terminator)
    // OUT: Amount of log buffer filled with messages
    CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,

    // CU_JIT_ERROR_LOG_BUFFER - pointer to a buffer in which to print any log
    // messages from PTXAS that reflect errors
    CU_JIT_ERROR_LOG_BUFFER,

    // CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES -
    // IN: Log buffer size in bytes.  Log messages will be capped at this size
    // (including null terminator)
    // OUT: Amount of log buffer filled with messages
    CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,

    // CU_JIT_OPTIMIZATION_LEVEL - level of optimizations to apply to generated
    // code (0 - 4), with 4 being the default and highest level of optimizations.
    CU_JIT_OPTIMIZATION_LEVEL,

    // CU_JIT_TARGET_FROM_CU_CONTEXT - no option value required.  Determines
    // the target based on the current attached context (default)
    CU_JIT_TARGET_FROM_CUCONTEXT,

    // CU_JIT_TARGET - target is chosen based on supplied CUjit_target_enum.
    CU_JIT_TARGET,

    // CU_JIT_FALLBACK_STRATEGY - specifies choice of fallback strategy if
    // matching cubin is not found.  Choice is based on supplied 
    // CUjit_fallback_enum.
    CU_JIT_FALLBACK_STRATEGY
    
} CUjit_option;

//
// Online compilation targets
//
typedef enum CUjit_target_enum
{
    CU_TARGET_COMPUTE_10            = 0,
    CU_TARGET_COMPUTE_11,
    CU_TARGET_COMPUTE_12,
    CU_TARGET_COMPUTE_13
} CUjit_target;

//
// Cubin matching fallback strategies
//
typedef enum CUjit_fallback_enum
{
    // prefer to compile ptx
    CU_PREFER_PTX                   = 0,

    // prefer to fall back to compatible binary code
    CU_PREFER_BINARY

} CUjit_fallback;

/************************************
 **
 **    Error codes
 **
 ***********************************/

typedef enum cudaError_enum {

    CUDA_SUCCESS                    = 0,
    CUDA_ERROR_INVALID_VALUE        = 1,
    CUDA_ERROR_OUT_OF_MEMORY        = 2,
    CUDA_ERROR_NOT_INITIALIZED      = 3,
    CUDA_ERROR_DEINITIALIZED        = 4,

    CUDA_ERROR_NO_DEVICE            = 100,
    CUDA_ERROR_INVALID_DEVICE       = 101,

    CUDA_ERROR_INVALID_IMAGE        = 200,
    CUDA_ERROR_INVALID_CONTEXT      = 201,
    CUDA_ERROR_CONTEXT_ALREADY_CURRENT = 202,
    CUDA_ERROR_MAP_FAILED           = 205,
    CUDA_ERROR_UNMAP_FAILED         = 206,
    CUDA_ERROR_ARRAY_IS_MAPPED      = 207,
    CUDA_ERROR_ALREADY_MAPPED       = 208,
    CUDA_ERROR_NO_BINARY_FOR_GPU    = 209,
    CUDA_ERROR_ALREADY_ACQUIRED     = 210,
    CUDA_ERROR_NOT_MAPPED           = 211,

    CUDA_ERROR_INVALID_SOURCE       = 300,
    CUDA_ERROR_FILE_NOT_FOUND       = 301,

    CUDA_ERROR_INVALID_HANDLE       = 400,

    CUDA_ERROR_NOT_FOUND            = 500,

    CUDA_ERROR_NOT_READY            = 600,

    CUDA_ERROR_LAUNCH_FAILED        = 700,
    CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES = 701,
    CUDA_ERROR_LAUNCH_TIMEOUT       = 702,
    CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING = 703,

    CUDA_ERROR_UNKNOWN              = 999
} CUresult;

#ifdef _WIN32
#define CUDAAPI __stdcall
#else
#define CUDAAPI 
#endif

    /*********************************
     ** Initialization
     *********************************/
    typedef CUresult  CUDAAPI tcuInit(unsigned int Flags);

    /************************************
     **
     **    Device management
     **
     ***********************************/
   
    typedef CUresult  CUDAAPI tcuDeviceGet(CUdevice *device, int ordinal);
    typedef CUresult  CUDAAPI tcuDeviceGetCount(int *count);
    typedef CUresult  CUDAAPI tcuDeviceGetName(char *name, int len, CUdevice dev);
    typedef CUresult  CUDAAPI tcuDeviceComputeCapability(int *major, int *minor, CUdevice dev);
    typedef CUresult  CUDAAPI tcuDeviceTotalMem(unsigned int *bytes, CUdevice dev);
    typedef CUresult  CUDAAPI tcuDeviceGetProperties(CUdevprop *prop, CUdevice dev);
    typedef CUresult  CUDAAPI tcuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev);
        
    /************************************
     **
     **    Context management
     **
     ***********************************/

    typedef CUresult  CUDAAPI tcuCtxCreate(CUcontext *pctx, unsigned int flags, CUdevice dev );
    typedef CUresult  CUDAAPI tcuCtxDestroy( CUcontext ctx );
    typedef CUresult  CUDAAPI tcuCtxAttach(CUcontext *pctx, unsigned int flags);
    typedef CUresult  CUDAAPI tcuCtxDetach(CUcontext ctx);
    typedef CUresult  CUDAAPI tcuCtxPushCurrent( CUcontext ctx );
    typedef CUresult  CUDAAPI tcuCtxPopCurrent( CUcontext *pctx );
    typedef CUresult  CUDAAPI tcuCtxGetDevice(CUdevice *device);
    typedef CUresult  CUDAAPI tcuCtxSynchronize(void);


    /************************************
     **
     **    Module management
     **
     ***********************************/
    
    typedef CUresult  CUDAAPI tcuModuleLoad(CUmodule *module, const char *fname);
    typedef CUresult  CUDAAPI tcuModuleLoadData(CUmodule *module, const void *image);
    typedef CUresult  CUDAAPI tcuModuleLoadDataEx(CUmodule *module, const void *image, unsigned int numOptions, CUjit_option *options, void **optionValues);
    typedef CUresult  CUDAAPI tcuModuleLoadFatBinary(CUmodule *module, const void *fatCubin);
    typedef CUresult  CUDAAPI tcuModuleUnload(CUmodule hmod);
    typedef CUresult  CUDAAPI tcuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name);
    typedef CUresult  CUDAAPI tcuModuleGetGlobal(CUdeviceptr *dptr, unsigned int *bytes, CUmodule hmod, const char *name);
    typedef CUresult  CUDAAPI tcuModuleGetTexRef(CUtexref *pTexRef, CUmodule hmod, const char *name);
    
    /************************************
     **
     **    Memory management
     **
     ***********************************/
    
    typedef CUresult CUDAAPI tcuMemGetInfo(unsigned int *free, unsigned int *total);

    typedef CUresult CUDAAPI tcuMemAlloc( CUdeviceptr *dptr, unsigned int bytesize);
    typedef CUresult CUDAAPI tcuMemAllocPitch( CUdeviceptr *dptr, 
                                      unsigned int *pPitch,
                                      unsigned int WidthInBytes, 
                                      unsigned int Height, 
                                      // size of biggest r/w to be performed by kernels on this memory
                                      // 4, 8 or 16 bytes
                                      unsigned int ElementSizeBytes
                                     );
    typedef CUresult CUDAAPI tcuMemFree(CUdeviceptr dptr);
    typedef CUresult CUDAAPI tcuMemGetAddressRange( CUdeviceptr *pbase, unsigned int *psize, CUdeviceptr dptr );

    typedef CUresult CUDAAPI tcuMemAllocHost(void **pp, unsigned int bytesize);
    typedef CUresult CUDAAPI tcuMemFreeHost(void *p);

    /************************************
     **
     **    Synchronous Memcpy
     **
     ** Intra-device memcpy's done with these functions may execute in parallel with the CPU,
     ** but if host memory is involved, they wait until the copy is done before returning.
     **
     ***********************************/

    // 1D functions
        // system <-> device memory
        typedef CUresult  CUDAAPI tcuMemcpyHtoD (CUdeviceptr dstDevice, const void *srcHost, unsigned int ByteCount );
        typedef CUresult  CUDAAPI tcuMemcpyDtoH (void *dstHost, CUdeviceptr srcDevice, unsigned int ByteCount );

        // device <-> device memory
        typedef CUresult  CUDAAPI tcuMemcpyDtoD (CUdeviceptr dstDevice, CUdeviceptr srcDevice, unsigned int ByteCount );

        // device <-> array memory
        typedef CUresult  CUDAAPI tcuMemcpyDtoA ( CUarray dstArray, unsigned int dstIndex, CUdeviceptr srcDevice, unsigned int ByteCount );
        typedef CUresult  CUDAAPI tcuMemcpyAtoD ( CUdeviceptr dstDevice, CUarray hSrc, unsigned int SrcIndex, unsigned int ByteCount );

        // system <-> array memory
        typedef CUresult  CUDAAPI tcuMemcpyHtoA( CUarray dstArray, unsigned int dstIndex, const void *pSrc, unsigned int ByteCount );
        typedef CUresult  CUDAAPI tcuMemcpyAtoH( void *dstHost, CUarray srcArray, unsigned int srcIndex, unsigned int ByteCount );

        // array <-> array memory
        typedef CUresult  CUDAAPI tcuMemcpyAtoA( CUarray dstArray, unsigned int dstIndex, CUarray srcArray, unsigned int srcIndex, unsigned int ByteCount );

    // 2D memcpy

        typedef struct CUDA_MEMCPY2D_st {

            unsigned int srcXInBytes, srcY;
            CUmemorytype srcMemoryType;
                const void *srcHost;
                CUdeviceptr srcDevice;
                CUarray srcArray;
                unsigned int srcPitch; // ignored when src is array

            unsigned int dstXInBytes, dstY;
            CUmemorytype dstMemoryType;
                void *dstHost;
                CUdeviceptr dstDevice;
                CUarray dstArray;
                unsigned int dstPitch; // ignored when dst is array

            unsigned int WidthInBytes;
            unsigned int Height;
        } CUDA_MEMCPY2D;
        typedef CUresult  CUDAAPI tcuMemcpy2D( const CUDA_MEMCPY2D *pCopy );
        typedef CUresult  CUDAAPI tcuMemcpy2DUnaligned( const CUDA_MEMCPY2D *pCopy );

    // 3D memcpy

        typedef struct CUDA_MEMCPY3D_st {

            unsigned int srcXInBytes, srcY, srcZ;
            unsigned int srcLOD;
            CUmemorytype srcMemoryType;
                const void *srcHost;
                CUdeviceptr srcDevice;
                CUarray srcArray;
                void *reserved0;        // must be NULL
                unsigned int srcPitch;  // ignored when src is array
                unsigned int srcHeight; // ignored when src is array; may be 0 if Depth==1

            unsigned int dstXInBytes, dstY, dstZ;
            unsigned int dstLOD;
            CUmemorytype dstMemoryType;
                void *dstHost;
                CUdeviceptr dstDevice;
                CUarray dstArray;
                void *reserved1;        // must be NULL
                unsigned int dstPitch;  // ignored when dst is array
                unsigned int dstHeight; // ignored when dst is array; may be 0 if Depth==1

            unsigned int WidthInBytes;
            unsigned int Height;
            unsigned int Depth;
        } CUDA_MEMCPY3D;
        typedef CUresult  CUDAAPI tcuMemcpy3D( const CUDA_MEMCPY3D *pCopy );

    /************************************
     **
     **    Asynchronous Memcpy
     **
     ** Any host memory involved must be DMA'able (e.g., allocated with cuMemAllocHost).
     ** memcpy's done with these functions execute in parallel with the CPU and, if
     ** the hardware is available, may execute in parallel with the GPU.
     ** Asynchronous memcpy must be accompanied by appropriate stream synchronization.
     **
     ***********************************/

    // 1D functions
        // system <-> device memory
        typedef CUresult  CUDAAPI tcuMemcpyHtoDAsync (CUdeviceptr dstDevice, 
            const void *srcHost, unsigned int ByteCount, CUstream hStream );
        typedef CUresult  CUDAAPI tcuMemcpyDtoHAsync (void *dstHost, 
            CUdeviceptr srcDevice, unsigned int ByteCount, CUstream hStream );

        // system <-> array memory
        typedef CUresult  CUDAAPI tcuMemcpyHtoAAsync( CUarray dstArray, unsigned int dstIndex, 
            const void *pSrc, unsigned int ByteCount, CUstream hStream );
        typedef CUresult  CUDAAPI tcuMemcpyAtoHAsync( void *dstHost, CUarray srcArray, unsigned int srcIndex, 
            unsigned int ByteCount, CUstream hStream );

        // 2D memcpy
        typedef CUresult  CUDAAPI tcuMemcpy2DAsync( const CUDA_MEMCPY2D *pCopy, CUstream hStream );

        // 3D memcpy
        typedef CUresult  CUDAAPI tcuMemcpy3DAsync( const CUDA_MEMCPY3D *pCopy, CUstream hStream );

    /************************************
     **
     **    Memset
     **
     ***********************************/
        typedef CUresult  CUDAAPI tcuMemsetD8( CUdeviceptr dstDevice, unsigned char uc, unsigned int N );
        typedef CUresult  CUDAAPI tcuMemsetD16( CUdeviceptr dstDevice, unsigned short us, unsigned int N );
        typedef CUresult  CUDAAPI tcuMemsetD32( CUdeviceptr dstDevice, unsigned int ui, unsigned int N );

        typedef CUresult  CUDAAPI tcuMemsetD2D8( CUdeviceptr dstDevice, unsigned int dstPitch, unsigned char uc, unsigned int Width, unsigned int Height );
        typedef CUresult  CUDAAPI tcuMemsetD2D16( CUdeviceptr dstDevice, unsigned int dstPitch, unsigned short us, unsigned int Width, unsigned int Height );
        typedef CUresult  CUDAAPI tcuMemsetD2D32( CUdeviceptr dstDevice, unsigned int dstPitch, unsigned int ui, unsigned int Width, unsigned int Height );

    /************************************
     **
     **    Function management
     **
     ***********************************/


    typedef CUresult CUDAAPI tcuFuncSetBlockShape (CUfunction hfunc, int x, int y, int z);
    typedef CUresult CUDAAPI tcuFuncSetSharedSize (CUfunction hfunc, unsigned int bytes);

    /************************************
     **
     **    Array management 
     **
     ***********************************/
   
    typedef struct
    {
        //
        // dimensions
        //            
            unsigned int Width;
            unsigned int Height;
            
        //
        // format
        //
            CUarray_format Format;
        
            // channels per array element
            unsigned int NumChannels;
    } CUDA_ARRAY_DESCRIPTOR;

    typedef CUresult  CUDAAPI tcuArrayCreate( CUarray *pHandle, const CUDA_ARRAY_DESCRIPTOR *pAllocateArray );
    typedef CUresult  CUDAAPI tcuArrayGetDescriptor( CUDA_ARRAY_DESCRIPTOR *pArrayDescriptor, CUarray hArray );
    typedef CUresult  CUDAAPI tcuArrayDestroy( CUarray hArray );

    typedef struct
    {
        //
        // dimensions
        //
            unsigned int Width;
            unsigned int Height;
            unsigned int Depth;
        //
        // format
        //
            CUarray_format Format;
        
            // channels per array element
            unsigned int NumChannels;
        //
        // flags
        //
            unsigned int Flags;

    } CUDA_ARRAY3D_DESCRIPTOR;
    typedef CUresult  CUDAAPI tcuArray3DCreate( CUarray *pHandle, const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray );
    typedef CUresult  CUDAAPI tcuArray3DGetDescriptor( CUDA_ARRAY3D_DESCRIPTOR *pArrayDescriptor, CUarray hArray );

    /************************************
     **
     **    Texture reference management
     **
     ***********************************/
    typedef CUresult  CUDAAPI tcuTexRefCreate( CUtexref *pTexRef );
    typedef CUresult  CUDAAPI tcuTexRefDestroy( CUtexref hTexRef );
    
    typedef CUresult  CUDAAPI tcuTexRefSetArray( CUtexref hTexRef, CUarray hArray, unsigned int Flags );
        // override the texref format with a format inferred from the array
        #define CU_TRSA_OVERRIDE_FORMAT 0x01
    typedef CUresult  CUDAAPI tcuTexRefSetAddress( unsigned int *ByteOffset, CUtexref hTexRef, CUdeviceptr dptr, unsigned int bytes );
    typedef CUresult  CUDAAPI tcuTexRefSetFormat( CUtexref hTexRef, CUarray_format fmt, int NumPackedComponents );
    
    typedef CUresult  CUDAAPI tcuTexRefSetAddressMode( CUtexref hTexRef, int dim, CUaddress_mode am );
    typedef CUresult  CUDAAPI tcuTexRefSetFilterMode( CUtexref hTexRef, CUfilter_mode fm );
    typedef CUresult  CUDAAPI tcuTexRefSetFlags( CUtexref hTexRef, unsigned int Flags );
        // read the texture as integers rather than promoting the values
        // to floats in the range [0,1]
        #define CU_TRSF_READ_AS_INTEGER         0x01

        // use normalized texture coordinates in the range [0,1) instead of [0,dim)
        #define CU_TRSF_NORMALIZED_COORDINATES  0x02

    typedef CUresult  CUDAAPI tcuTexRefGetAddress( CUdeviceptr *pdptr, CUtexref hTexRef );
    typedef CUresult  CUDAAPI tcuTexRefGetArray( CUarray *phArray, CUtexref hTexRef );
    typedef CUresult  CUDAAPI tcuTexRefGetAddressMode( CUaddress_mode *pam, CUtexref hTexRef, int dim );
    typedef CUresult  CUDAAPI tcuTexRefGetFilterMode( CUfilter_mode *pfm, CUtexref hTexRef );
    typedef CUresult  CUDAAPI tcuTexRefGetFormat( CUarray_format *pFormat, int *pNumChannels, CUtexref hTexRef );
    typedef CUresult  CUDAAPI tcuTexRefGetFlags( unsigned int *pFlags, CUtexref hTexRef );

    /************************************
     **
     **    Parameter management
     **
     ***********************************/

    typedef CUresult  CUDAAPI tcuParamSetSize (CUfunction hfunc, unsigned int numbytes);
    typedef CUresult  CUDAAPI tcuParamSeti    (CUfunction hfunc, int offset, unsigned int value);
    typedef CUresult  CUDAAPI tcuParamSetf    (CUfunction hfunc, int offset, float value);
    typedef CUresult  CUDAAPI tcuParamSetv    (CUfunction hfunc, int offset, void * ptr, unsigned int numbytes);
    typedef CUresult  CUDAAPI tcuParamSetTexRef(CUfunction hfunc, int texunit, CUtexref hTexRef);
        // for texture references loaded into the module,
        // use default texunit from texture reference
        #define CU_PARAM_TR_DEFAULT -1

    /************************************
     **
     **    Launch functions
     **
     ***********************************/

    typedef CUresult CUDAAPI tcuLaunch ( CUfunction f );
    typedef CUresult CUDAAPI tcuLaunchGrid (CUfunction f, int grid_width, int grid_height);
    typedef CUresult CUDAAPI tcuLaunchGridAsync( CUfunction f, int grid_width, int grid_height, CUstream hStream );

    /************************************
     **
     **    Events
     **
     ***********************************/
    typedef CUresult CUDAAPI tcuEventCreate( CUevent *phEvent, unsigned int Flags );
    typedef CUresult CUDAAPI tcuEventRecord( CUevent hEvent, CUstream hStream );
    typedef CUresult CUDAAPI tcuEventQuery( CUevent hEvent );
    typedef CUresult CUDAAPI tcuEventSynchronize( CUevent hEvent );
    typedef CUresult CUDAAPI tcuEventDestroy( CUevent hEvent );
    typedef CUresult CUDAAPI tcuEventElapsedTime( float *pMilliseconds, CUevent hStart, CUevent hEnd );

    /************************************
     **
     **    Streams
     **
     ***********************************/
    typedef CUresult CUDAAPI  tcuStreamCreate( CUstream *phStream, unsigned int Flags );
    typedef CUresult CUDAAPI  tcuStreamQuery( CUstream hStream );
    typedef CUresult CUDAAPI  tcuStreamSynchronize( CUstream hStream );
    typedef CUresult CUDAAPI  tcuStreamDestroy( CUstream hStream );


    extern tcuInit                         *cuInit;
    extern tcuDeviceGet                    *cuDeviceGet;
    extern tcuDeviceGetCount               *cuDeviceGetCount;
    extern tcuDeviceGetName                *cuDeviceGetName;
    extern tcuDeviceComputeCapability      *cuDeviceComputeCapability;
    extern tcuDeviceTotalMem               *cuDeviceTotalMem;
    extern tcuDeviceGetProperties          *cuDeviceGetProperties;
    extern tcuDeviceGetAttribute           *cuDeviceGetAttribute;
    extern tcuCtxCreate                    *cuCtxCreate;
    extern tcuCtxDestroy                   *cuCtxDestroy;
    extern tcuCtxAttach                    *cuCtxAttach;
    extern tcuCtxDetach                    *cuCtxDetach;
    extern tcuCtxPushCurrent               *cuCtxPushCurrent;
    extern tcuCtxPopCurrent                *cuCtxPopCurrent;
    extern tcuCtxGetDevice                 *cuCtxGetDevice;
    extern tcuCtxSynchronize               *cuCtxSynchronize;
    extern tcuModuleLoad                   *cuModuleLoad;
    extern tcuModuleLoadData               *cuModuleLoadData;
    extern tcuModuleLoadDataEx             *cuModuleLoadDataEx;
    extern tcuModuleLoadFatBinary          *cuModuleLoadFatBinary;
    extern tcuModuleUnload                 *cuModuleUnload;
    extern tcuModuleGetFunction            *cuModuleGetFunction;
    extern tcuModuleGetGlobal              *cuModuleGetGlobal;
    extern tcuModuleGetTexRef              *cuModuleGetTexRef;
    extern tcuMemGetInfo                   *cuMemGetInfo;
    extern tcuMemAlloc                     *cuMemAlloc;
    extern tcuMemAllocPitch                *cuMemAllocPitch;
    extern tcuMemFree                      *cuMemFree;
    extern tcuMemGetAddressRange           *cuMemGetAddressRange;
    extern tcuMemAllocHost                 *cuMemAllocHost;
    extern tcuMemFreeHost                  *cuMemFreeHost;
    extern tcuMemcpyHtoD                   *cuMemcpyHtoD;
    extern tcuMemcpyDtoH                   *cuMemcpyDtoH;
    extern tcuMemcpyDtoD                   *cuMemcpyDtoD;
    extern tcuMemcpyDtoA                   *cuMemcpyDtoA;
    extern tcuMemcpyAtoD                   *cuMemcpyAtoD;
    extern tcuMemcpyHtoA                   *cuMemcpyHtoA;
    extern tcuMemcpyAtoH                   *cuMemcpyAtoH;
    extern tcuMemcpyAtoA                   *cuMemcpyAtoA;
    extern tcuMemcpy2D                     *cuMemcpy2D;
    extern tcuMemcpy2DUnaligned            *cuMemcpy2DUnaligned;
    extern tcuMemcpy3D                     *cuMemcpy3D;
    extern tcuMemcpyHtoDAsync              *cuMemcpyHtoDAsync;
    extern tcuMemcpyDtoHAsync              *cuMemcpyDtoHAsync;
    extern tcuMemcpyHtoAAsync              *cuMemcpyHtoAAsync;
    extern tcuMemcpyAtoHAsync              *cuMemcpyAtoHAsync;
    extern tcuMemcpy2DAsync                *cuMemcpy2DAsync;
    extern tcuMemcpy3DAsync                *cuMemcpy3DAsync;
    extern tcuMemsetD8                     *cuMemsetD8;
    extern tcuMemsetD16                    *cuMemsetD16;
    extern tcuMemsetD32                    *cuMemsetD32;
    extern tcuMemsetD2D8                   *cuMemsetD2D8;
    extern tcuMemsetD2D16                  *cuMemsetD2D16;
    extern tcuMemsetD2D32                  *cuMemsetD2D32;
    extern tcuFuncSetBlockShape            *cuFuncSetBlockShape;
    extern tcuFuncSetSharedSize            *cuFuncSetSharedSize;
    extern tcuArrayCreate                  *cuArrayCreate;
    extern tcuArrayGetDescriptor           *cuArrayGetDescriptor;
    extern tcuArrayDestroy                 *cuArrayDestroy;
    extern tcuArray3DCreate                *cuArray3DCreate;
    extern tcuArray3DGetDescriptor         *cuArray3DGetDescriptor;
    extern tcuTexRefCreate                 *cuTexRefCreate;
    extern tcuTexRefDestroy                *cuTexRefDestroy;
    extern tcuTexRefSetArray               *cuTexRefSetArray;
    extern tcuTexRefSetAddress             *cuTexRefSetAddress;
    extern tcuTexRefSetFormat              *cuTexRefSetFormat;
    extern tcuTexRefSetAddressMode         *cuTexRefSetAddressMode;
    extern tcuTexRefSetFilterMode          *cuTexRefSetFilterMode;
    extern tcuTexRefSetFlags               *cuTexRefSetFlags;
    extern tcuTexRefGetAddress             *cuTexRefGetAddress;
    extern tcuTexRefGetArray               *cuTexRefGetArray;
    extern tcuTexRefGetAddressMode         *cuTexRefGetAddressMode;
    extern tcuTexRefGetFilterMode          *cuTexRefGetFilterMode;
    extern tcuTexRefGetFormat              *cuTexRefGetFormat;
    extern tcuTexRefGetFlags               *cuTexRefGetFlags;
    extern tcuParamSetSize                 *cuParamSetSize;
    extern tcuParamSeti                    *cuParamSeti;
    extern tcuParamSetf                    *cuParamSetf;
    extern tcuParamSetv                    *cuParamSetv;
    extern tcuParamSetTexRef               *cuParamSetTexRef;
    extern tcuLaunch                       *cuLaunch;
    extern tcuLaunchGrid                   *cuLaunchGrid;
    extern tcuLaunchGridAsync              *cuLaunchGridAsync;
    extern tcuEventCreate                  *cuEventCreate;
    extern tcuEventRecord                  *cuEventRecord;
    extern tcuEventQuery                   *cuEventQuery;
    extern tcuEventSynchronize             *cuEventSynchronize;
    extern tcuEventDestroy                 *cuEventDestroy;
    extern tcuEventElapsedTime             *cuEventElapsedTime;
    extern tcuStreamCreate                 *cuStreamCreate;
    extern tcuStreamQuery                  *cuStreamQuery;
    extern tcuStreamSynchronize            *cuStreamSynchronize;
    extern tcuStreamDestroy                *cuStreamDestroy;

    extern CUresult CUDAAPI cuDriverAPIdynload(void);

#ifdef __cplusplus
}
#endif

#endif /* __cuda_dynlink_h__ */
