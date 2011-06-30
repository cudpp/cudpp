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
 
#ifndef _CUTIL_INLINE_FUNCTIONS_DRVAPI_H_
#define _CUTIL_INLINE_FUNCTIONS_DRVAPI_H_

#include <stdio.h>
#include <string.h>
#include <stdlib.h>


// Error Code string definitions here
typedef struct
{
    char const *error_string;
    int  error_id;
} s_CudaErrorStr;

/**
 * Error codes
 */
static s_CudaErrorStr sCudaDrvErrorString[] =
{
    /**
     * The API call returned with no errors. In the case of query calls, this
     * can also mean that the operation being queried is complete (see
     * ::cuEventQuery() and ::cuStreamQuery()).
     */
    { "CUDA_SUCCESS", 0 },

    /**
     * This indicates that one or more of the parameters passed to the API call
     * is not within an acceptable range of values.
     */
    { "CUDA_ERROR_INVALID_VALUE", 1 },

    /**
     * The API call failed because it was unable to allocate enough memory to
     * perform the requested operation.
     */
    { "CUDA_ERROR_OUT_OF_MEMORY", 2 },

    /**
     * This indicates that the CUDA driver has not been initialized with
     * ::cuInit() or that initialization has failed.
     */
    { "CUDA_ERROR_NOT_INITIALIZED", 3 },

    /**
     * This indicates that the CUDA driver is in the process of shutting down.
     */
    { "CUDA_ERROR_DEINITIALIZED", 4 },

    /**
     * This indicates profiling APIs are called while application is running
     * in visual profiler mode. 
    */
    { "CUDA_ERROR_PROFILER_DISABLED", 5 },
    /**
     * This indicates profiling has not been initialized for this context. 
     * Call cuProfilerInitialize() to resolve this. 
    */
    { "CUDA_ERROR_PROFILER_NOT_INITIALIZED", 6 },
    /**
     * This indicates profiler has already been started and probably
     * cuProfilerStart() is incorrectly called.
    */
    { "CUDA_ERROR_PROFILER_ALREADY_STARTED", 7 },
    /**
     * This indicates profiler has already been stopped and probably
     * cuProfilerStop() is incorrectly called.
    */
    { "CUDA_ERROR_PROFILER_ALREADY_STOPPED", 8 },  
    /**
     * This indicates that no CUDA-capable devices were detected by the installed
     * CUDA driver.
     */
    { "CUDA_ERROR_NO_DEVICE (no CUDA-capable devices were detected)", 100 },

    /**
     * This indicates that the device ordinal supplied by the user does not
     * correspond to a valid CUDA device.
     */
    { "CUDA_ERROR_INVALID_DEVICE (device specified is not a valid CUDA device)", 101 },


    /**
     * This indicates that the device kernel image is invalid. This can also
     * indicate an invalid CUDA module.
     */
    { "CUDA_ERROR_INVALID_IMAGE", 200 },

    /**
     * This most frequently indicates that there is no context bound to the
     * current thread. This can also be returned if the context passed to an
     * API call is not a valid handle (such as a context that has had
     * ::cuCtxDestroy() invoked on it). This can also be returned if a user
     * mixes different API versions (i.e. 3010 context with 3020 API calls).
     * See ::cuCtxGetApiVersion() for more details.
     */
    { "CUDA_ERROR_INVALID_CONTEXT", 201 },

    /**
     * This indicated that the context being supplied as a parameter to the
     * API call was already the active context.
     * \deprecated
     * This error return is deprecated as of CUDA 3.2. It is no longer an
     * error to attempt to push the active context via ::cuCtxPushCurrent().
     */
    { "CUDA_ERROR_CONTEXT_ALREADY_CURRENT", 202 },

    /**
     * This indicates that a map or register operation has failed.
     */
    { "CUDA_ERROR_MAP_FAILED", 205 },

    /**
     * This indicates that an unmap or unregister operation has failed.
     */
    { "CUDA_ERROR_UNMAP_FAILED", 206 },

    /**
     * This indicates that the specified array is currently mapped and thus
     * cannot be destroyed.
     */
    { "CUDA_ERROR_ARRAY_IS_MAPPED", 207 },

    /**
     * This indicates that the resource is already mapped.
     */
    { "CUDA_ERROR_ALREADY_MAPPED", 208 },

    /**
     * This indicates that there is no kernel image available that is suitable
     * for the device. This can occur when a user specifies code generation
     * options for a particular CUDA source file that do not include the
     * corresponding device configuration.
     */
    { "CUDA_ERROR_NO_BINARY_FOR_GPU", 209 },

    /**
     * This indicates that a resource has already been acquired.
     */
    { "CUDA_ERROR_ALREADY_ACQUIRED", 210 },

    /**
     * This indicates that a resource is not mapped.
     */
    { "CUDA_ERROR_NOT_MAPPED", 211 },

    /**
     * This indicates that a mapped resource is not available for access as an
     * array.
     */
    { "CUDA_ERROR_NOT_MAPPED_AS_ARRAY", 212 },

    /**
     * This indicates that a mapped resource is not available for access as a
     * pointer.
     */
    { "CUDA_ERROR_NOT_MAPPED_AS_POINTER", 213 },

    /**
     * This indicates that an uncorrectable ECC error was detected during
     * execution.
     */
    { "CUDA_ERROR_ECC_UNCORRECTABLE", 214 },

    /**
     * This indicates that the ::CUlimit passed to the API call is not
     * supported by the active device.
     */
    { "CUDA_ERROR_UNSUPPORTED_LIMIT", 215 },

    /**
     * This indicates that the ::CUcontext passed to the API call can
     * only be bound to a single CPU thread at a time but is already 
     * bound to a CPU thread.
     */
    { "CUDA_ERROR_CONTEXT_ALREADY_IN_USE", 216 },

    /**
     * This indicates that the device kernel source is invalid.
     */
    { "CUDA_ERROR_INVALID_SOURCE", 300 },

    /**
     * This indicates that the file specified was not found.
     */
    { "CUDA_ERROR_FILE_NOT_FOUND", 301 },

    /**
     * This indicates that a link to a shared object failed to resolve.
     */
    { "CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND", 302 },

    /**
     * This indicates that initialization of a shared object failed.
     */
    { "CUDA_ERROR_SHARED_OBJECT_INIT_FAILED", 303 },

    /**
     * This indicates that an OS call failed.
     */
    { "CUDA_ERROR_OPERATING_SYSTEM", 304 },


    /**
     * This indicates that a resource handle passed to the API call was not
     * valid. Resource handles are opaque types like ::CUstream and ::CUevent.
     */
    { "CUDA_ERROR_INVALID_HANDLE", 400 },


    /**
     * This indicates that a named symbol was not found. Examples of symbols
     * are global/constant variable names, texture names }, and surface names.
     */
    { "CUDA_ERROR_NOT_FOUND", 500 },


    /**
     * This indicates that asynchronous operations issued previously have not
     * completed yet. This result is not actually an error, but must be indicated
     * differently than ::CUDA_SUCCESS (which indicates completion). Calls that
     * may return this value include ::cuEventQuery() and ::cuStreamQuery().
     */
    { "CUDA_ERROR_NOT_READY", 600 },


    /**
     * An exception occurred on the device while executing a kernel. Common
     * causes include dereferencing an invalid device pointer and accessing
     * out of bounds shared memory. The context cannot be used }, so it must
     * be destroyed (and a new one should be created). All existing device
     * memory allocations from this context are invalid and must be
     * reconstructed if the program is to continue using CUDA.
     */
    { "CUDA_ERROR_LAUNCH_FAILED", 700 },

    /**
     * This indicates that a launch did not occur because it did not have
     * appropriate resources. This error usually indicates that the user has
     * attempted to pass too many arguments to the device kernel, or the
     * kernel launch specifies too many threads for the kernel's register
     * count. Passing arguments of the wrong size (i.e. a 64-bit pointer
     * when a 32-bit int is expected) is equivalent to passing too many
     * arguments and can also result in this error.
     */
    { "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES", 701 },

    /**
     * This indicates that the device kernel took too long to execute. This can
     * only occur if timeouts are enabled - see the device attribute
     * ::CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT for more information. The
     * context cannot be used (and must be destroyed similar to
     * ::CUDA_ERROR_LAUNCH_FAILED). All existing device memory allocations from
     * this context are invalid and must be reconstructed if the program is to
     * continue using CUDA.
     */
    { "CUDA_ERROR_LAUNCH_TIMEOUT", 702 },

    /**
     * This error indicates a kernel launch that uses an incompatible texturing
     * mode.
     */
    { "CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING", 703 },
    
    /**
     * This error indicates that a call to ::cuCtxEnablePeerAccess() is
     * trying to re-enable peer access to a context which has already
     * had peer access to it enabled.
     */
    { "CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED", 704 },

    /**
     * This error indicates that ::cuCtxDisablePeerAccess() is 
     * trying to disable peer access which has not been enabled yet 
     * via ::cuCtxEnablePeerAccess(). 
     */
    { "CUDA_ERROR_PEER_ACCESS_NOT_ENABLED", 705 },

    /**
     * This error indicates that the primary context for the specified device
     * has already been initialized.
     */
    { "CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE", 708 },

    /**
     * This error indicates that the context current to the calling thread
     * has been destroyed using ::cuCtxDestroy }, or is a primary context which
     * has not yet been initialized.
     */
    { "CUDA_ERROR_CONTEXT_IS_DESTROYED", 709 },

    /**
     * A device-side assert triggered during kernel execution. The context
     * cannot be used anymore, and must be destroyed. All existing device 
     * memory allocations from this context are invalid and must be 
     * reconstructed if the program is to continue using CUDA.
     */
    { "CUDA_ERROR_ASSERT", 710 },

    /**
     * This indicates that an unknown internal error has occurred.
     */
    { "CUDA_ERROR_UNKNOWN", 999 },
    { NULL, -1 }
};

// This is just a linear search through the array, since the error_id's are not
// always ocurring consecutively
inline const char * getCudaDrvErrorString(CUresult error_id)
{
    int index = 0;
    while (sCudaDrvErrorString[index].error_id != error_id && 
           sCudaDrvErrorString[index].error_id != -1)
    {
        index++;
        return (const char *)sCudaDrvErrorString[index].error_string;
    }
    return (const char *)"CUDA_ERROR not found!";
}

// We define these calls here, so the user doesn't need to include __FILE__ and __LINE__
// The advantage is the developers gets to use the inline function so they can debug
#define cutilDrvSafeCallNoSync(err)     __cuSafeCallNoSync  (err, __FILE__, __LINE__)
#define cutilDrvSafeCall(err)           __cuSafeCall        (err, __FILE__, __LINE__)
#define cutilDrvCtxSync()               __cuCtxSync         (__FILE__, __LINE__)
#define cutilDrvCheckMsg(msg)           __cuCheckMsg        (msg, __FILE__, __LINE__)
#define cutilDrvAlignOffset(offset, alignment)  ( offset = (offset + (alignment-1)) & ~((alignment-1)) )

// These are the inline versions for all of the CUTIL functions
inline void __cuSafeCallNoSync( CUresult err, const char *file, const int line )
{
    if( CUDA_SUCCESS != err) {
        fprintf(stderr, "cuSafeCallNoSync() Driver API error = %04d \"%s\" from file <%s>, line %i.\n",
                err, getCudaDrvErrorString(err), file, line );
        exit(-1);
    }
}
inline void __cuSafeCall( CUresult err, const char *file, const int line )
{
    __cuSafeCallNoSync( err, file, line );
}

inline void __cuCtxSync(const char *file, const int line )
{
    CUresult err = cuCtxSynchronize();
    if( CUDA_SUCCESS != err ) {
        fprintf(stderr, "cuCtxSynchronize() API error = %04d \"%s\" from file <%s>, line %i.\n",
                err, getCudaDrvErrorString(err), file, line );
        exit(-1);
    }
}

#define MIN(a,b) ((a < b) ? a : b)
#define MAX(a,b) ((a > b) ? a : b)

// Beginning of GPU Architecture definitions
inline int _ConvertSMVer2CoresDrvApi(int major, int minor)
{
	// Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
	typedef struct {
		int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
		int Cores;
	} sSMtoCores;

        sSMtoCores nGpuArchCoresPerSM[] =
        { { 0x10,  8 },
          { 0x11,  8 },
          { 0x12,  8 },
          { 0x13,  8 },
          { 0x20, 32 },
          { 0x21, 48 },
          {   -1, -1 }
        };

	int index = 0;
	while (nGpuArchCoresPerSM[index].SM != -1) {
		if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor) ) {
			return nGpuArchCoresPerSM[index].Cores;
		}
		index++;
	}
	printf("MapSMtoCores undefined SMversion %d.%d!\n", major, minor);
	return -1;
}
// end of GPU Architecture definitions

// This function returns the best GPU based on performance
inline int cutilDrvGetMaxGflopsDeviceId()
{
    CUdevice current_device = 0, max_perf_device = 0;
    int device_count     = 0, sm_per_multiproc = 0;
    int max_compute_perf = 0, best_SM_arch     = 0;
    int major = 0, minor = 0, multiProcessorCount, clockRate;

    cuInit(0);
    cutilDrvSafeCallNoSync(cuDeviceGetCount(&device_count));

	// Find the best major SM Architecture GPU device
	while ( current_device < device_count ) {
		cutilDrvSafeCallNoSync( cuDeviceComputeCapability(&major, &minor, current_device ) );
		if (major > 0 && major < 9999) {
			best_SM_arch = MAX(best_SM_arch, major);
		}
		current_device++;
	}

    // Find the best CUDA capable GPU device
	current_device = 0;
	while( current_device < device_count ) {
		cutilDrvSafeCallNoSync( cuDeviceGetAttribute( &multiProcessorCount, 
                                                            CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, 
                                                            current_device ) );
        cutilDrvSafeCallNoSync( cuDeviceGetAttribute( &clockRate, 
                                                            CU_DEVICE_ATTRIBUTE_CLOCK_RATE, 
                                                            current_device ) );
		cutilDrvSafeCallNoSync( cuDeviceComputeCapability(&major, &minor, current_device ) );

		if (major == 9999 && minor == 9999) {
		    sm_per_multiproc = 1;
		} else {
		    sm_per_multiproc = _ConvertSMVer2CoresDrvApi(major, minor);
		}

		int compute_perf  = multiProcessorCount * sm_per_multiproc * clockRate;
		if( compute_perf  > max_compute_perf ) {
            // If we find GPU with SM major > 2, search only these
			if ( best_SM_arch > 2 ) {
				// If our device==dest_SM_arch, choose this, or else pass
				if (major == best_SM_arch) {	
                    max_compute_perf  = compute_perf;
                    max_perf_device   = current_device;
				}
			} else {
				max_compute_perf  = compute_perf;
				max_perf_device   = current_device;
			}
		}
		++current_device;
	}
	return max_perf_device;
}

// This function returns the best Graphics GPU based on performance
inline int cutilDrvGetMaxGflopsGraphicsDeviceId()
{
    CUdevice current_device = 0, max_perf_device = 0;
    int device_count     = 0, sm_per_multiproc = 0;
    int max_compute_perf = 0, best_SM_arch     = 0;
    int major = 0, minor = 0, multiProcessorCount, clockRate;
	int bTCC = 0;
	char deviceName[256];

    cuInit(0);
    cutilDrvSafeCallNoSync(cuDeviceGetCount(&device_count));

	// Find the best major SM Architecture GPU device that are graphics devices
	while ( current_device < device_count ) {
		cutilDrvSafeCallNoSync( cuDeviceGetName(deviceName, 256, current_device) );
		cutilDrvSafeCallNoSync( cuDeviceComputeCapability(&major, &minor, current_device ) );

#if CUDA_VERSION >= 3020
		cutilDrvSafeCallNoSync( cuDeviceGetAttribute( &bTCC,  CU_DEVICE_ATTRIBUTE_TCC_DRIVER, current_device ) );
#else
		// Assume a Tesla GPU is running in TCC if we are running CUDA 3.1
		if (deviceName[0] == 'T') bTCC = 1;
#endif
		if (!bTCC) {
			if (major > 0 && major < 9999) {
				best_SM_arch = MAX(best_SM_arch, major);
			}
		}
		current_device++;
	}

    // Find the best CUDA capable GPU device
	current_device = 0;
	while( current_device < device_count ) {
		cutilDrvSafeCallNoSync( cuDeviceGetAttribute( &multiProcessorCount, 
                                                            CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, 
                                                            current_device ) );
        cutilDrvSafeCallNoSync( cuDeviceGetAttribute( &clockRate, 
                                                            CU_DEVICE_ATTRIBUTE_CLOCK_RATE, 
                                                            current_device ) );
		cutilDrvSafeCallNoSync( cuDeviceComputeCapability(&major, &minor, current_device ) );

#if CUDA_VERSION >= 3020
		cutilDrvSafeCallNoSync( cuDeviceGetAttribute( &bTCC,  CU_DEVICE_ATTRIBUTE_TCC_DRIVER, current_device ) );
#else
		// Assume a Tesla GPU is running in TCC if we are running CUDA 3.1
		if (deviceName[0] == 'T') bTCC = 1;
#endif

		if (major == 9999 && minor == 9999) {
		    sm_per_multiproc = 1;
		} else {
		    sm_per_multiproc = _ConvertSMVer2CoresDrvApi(major, minor);
		}

		// If this is a Tesla based GPU and SM 2.0, and TCC is disabled, this is a contendor
		if (!bTCC) // Is this GPU running the TCC driver?  If so we pass on this
		{
			int compute_perf  = multiProcessorCount * sm_per_multiproc * clockRate;
			if( compute_perf  > max_compute_perf ) {
				// If we find GPU with SM major > 2, search only these
				if ( best_SM_arch > 2 ) {
					// If our device = dest_SM_arch, then we pick this one
					if (major == best_SM_arch) {	
                        max_compute_perf  = compute_perf;
                        max_perf_device   = current_device;
					}
				} else {
					max_compute_perf  = compute_perf;
					max_perf_device   = current_device;
				}
			}
		}
		++current_device;
	}
	return max_perf_device;
}

inline void __cuCheckMsg( const char * msg, const char *file, const int line )
{
    CUresult err = cuCtxSynchronize();
    if( CUDA_SUCCESS != err) {
		fprintf(stderr, "cutilDrvCheckMsg -> %s", msg);
        fprintf(stderr, "cutilDrvCheckMsg -> cuCtxSynchronize API error = %04d \"%s\" in file <%s>, line %i.\n",
                err, getCudaDrvErrorString(err), file, line );
        exit(-1);
    }
}


#if __DEVICE_EMULATION__
    inline int cutilDeviceInitDrv(int ARGC, char **ARGV) { } 
#else
    inline int cutilDeviceInitDrv(int ARGC, char ** ARGV) 
    {
        int cuDevice = 0;
        int deviceCount = 0;
        CUresult err = cuInit(0);
        if (CUDA_SUCCESS == err)
            cutilDrvSafeCallNoSync(cuDeviceGetCount(&deviceCount));
        if (deviceCount == 0) {
            fprintf(stderr, "CUTIL DeviceInitDrv error: no devices supporting CUDA\n");
            exit(-1);
        }
        int dev = 0;
        cutGetCmdLineArgumenti(ARGC, (const char **) ARGV, "device", &dev);
        if (dev < 0) dev = 0;
        if (dev > deviceCount-1) {
			fprintf(stderr, "\n");
			fprintf(stderr, ">> %d CUDA capable GPU device(s) detected. <<\n", deviceCount);
            fprintf(stderr, ">> cutilDeviceInit (-device=%d) is not a valid GPU device. <<\n", dev);
			fprintf(stderr, "\n");
            return -dev;
        }
        cutilDrvSafeCallNoSync(cuDeviceGet(&cuDevice, dev));
        char name[100];
        cuDeviceGetName(name, 100, cuDevice);
        if (cutCheckCmdLineFlag(ARGC, (const char **) ARGV, "quiet") == CUTFalse) {
           printf("> Using CUDA Device [%d]: %s\n", dev, name);
       	}
        return dev;
    }
#endif

    // General initialization call to pick the best CUDA Device
#if __DEVICE_EMULATION__
    inline CUdevice cutilChooseCudaDeviceDrv(int argc, char **argv, int *p_devID)
#else
    inline CUdevice cutilChooseCudaDeviceDrv(int argc, char **argv, int *p_devID)
    {
        CUdevice cuDevice;
        int devID = 0;
        // If the command-line has a device number specified, use it
        if( cutCheckCmdLineFlag(argc, (const char**)argv, "device") ) {
            devID = cutilDeviceInitDrv(argc, argv);
            if (devID < 0) {
                printf("exiting...\n");
                exit(0);
            }
        } else {
            // Otherwise pick the device with highest Gflops/s
            char name[100];
            devID = cutilDrvGetMaxGflopsDeviceId();
            cutilDrvSafeCallNoSync(cuDeviceGet(&cuDevice, devID));
            cuDeviceGetName(name, 100, cuDevice);
            printf("> Using CUDA Device [%d]: %s\n", devID, name);
        }
        cuDeviceGet(&cuDevice, devID);
        if (p_devID) *p_devID = devID;
        return cuDevice;
    }
#endif


//! Check for CUDA context lost
inline void cutilDrvCudaCheckCtxLost(const char *errorMessage, const char *file, const int line ) 
{
    CUresult err = cuCtxSynchronize();
    if( CUDA_ERROR_INVALID_CONTEXT != err) {
        fprintf(stderr, "Cuda error: %s in file '%s' in line %i\n",
                errorMessage, file, line );
        exit(-1);
    }
    err = cuCtxSynchronize();
    if( CUDA_SUCCESS != err) {
        fprintf(stderr, "Cuda error: %s in file '%s' in line %i\n",
                errorMessage, file, line );
        exit(-1);
    } 
}

#ifndef STRCASECMP
#ifdef _WIN32
#define STRCASECMP  _stricmp
#else
#define STRCASECMP  strcasecmp
#endif
#endif

#ifndef STRNCASECMP
#ifdef _WIN32
#define STRNCASECMP _strnicmp
#else
#define STRNCASECMP strncasecmp
#endif
#endif

inline void __cutilDrvQAFinish(int argc, char **argv, bool bStatus)
{
    const char *sStatus[] = { "FAILED", "PASSED", "WAIVED", NULL };

    bool bFlag = false;
    for (int i=1; i < argc; i++) {
        if (!STRCASECMP(argv[i], "-qatest") || !STRCASECMP(argv[i], "-noprompt")) {
            bFlag |= true;
        }
    }

    if (bFlag) {
        printf("&&&& %s %s", sStatus[bStatus], argv[0]);
        for (int i=1; i < argc; i++) printf(" %s", argv[i]);
    } else {
        printf("[%s] test result\n%s\n", argv[0], sStatus[bStatus]);
    }
}

// General check for CUDA GPU SM Capabilities for a specific device #
inline bool cutilDrvCudaDevCapabilities(int major_version, int minor_version, int deviceNum)
{
    int major, minor, dev;
    char device_name[256];

#ifdef __DEVICE_EMULATION__
    printf("> Compute Device Emulation Mode \n");
#endif

    cutilDrvSafeCallNoSync( cuDeviceGet(&dev, deviceNum) );
    cutilDrvSafeCallNoSync( cuDeviceComputeCapability(&major, &minor, dev));
    cutilDrvSafeCallNoSync( cuDeviceGetName(device_name, 256, dev) ); 

    if((major > major_version) ||
       (major == major_version && minor >= minor_version))
    {
        printf("> Device %d: < %s >, Compute SM %d.%d detected\n", dev, device_name, major, minor);
        return true;
    }
    else
    {
        printf("There is no device supporting CUDA compute capability %d.%d.\n", major_version, minor_version);
        return false;
    }
}

// General check for CUDA GPU SM Capabilities
inline bool cutilDrvCudaCapabilities(int major_version, int minor_version)
{
    return cutilDrvCudaDevCapabilities(major_version, minor_version,0);
}

#endif // _CUTIL_INLINE_FUNCTIONS_DRVAPI_H_
