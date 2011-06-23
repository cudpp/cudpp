/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 * 
 */
 
 #ifndef _CUTIL_INLINE_FUNCTIONS_DRVAPI_H_
#define _CUTIL_INLINE_FUNCTIONS_DRVAPI_H_

#include <stdio.h>
#include <string.h>
#include <stdlib.h>


// We define these calls here, so the user doesn't need to include __FILE__ and __LINE__
// The advantage is the developers gets to use the inline function so they can debug
#define cutilDrvSafeCallNoSync(err)     __cuSafeCallNoSync  (err, __FILE__, __LINE__)
#define cutilDrvSafeCall(err)           __cuSafeCall        (err, __FILE__, __LINE__)
#define cutilDrvCtxSync()               __cuCtxSync         (__FILE__, __LINE__)
#define cutilDrvCheckMsg(msg)           __cuCheckMsg        (msg, __FILE__, __LINE__)
#define cutilDrvAlignOffset(offset, alignment)  ( offset = (offset + (alignment-1)) & ~((alignment-1)) )

#define MIN(a,b) ((a < b) ? a : b)
#define MAX(a,b) ((a > b) ? a : b)

// This function returns the best GPU based on performance
inline int cutilDrvGetMaxGflopsDeviceId()
{
    CUdevice current_device = 0, max_perf_device = 0;
    int device_count     = 0, sm_per_multiproc = 0;
	int max_compute_perf = 0, best_SM_arch     = 0;
    int major = 0, minor = 0, multiProcessorCount, clockRate;
    int arch_cores_sm[3] = { 1, 8, 32 };

    cuInit(0);
    CU_SAFE_CALL_NO_SYNC(cuDeviceGetCount(&device_count));

	// Find the best major SM Architecture GPU device
	while ( current_device < device_count ) {
		CU_SAFE_CALL_NO_SYNC( cuDeviceComputeCapability(&major, &minor, current_device ) );
		if (major > 0 && major < 9999) {
			best_SM_arch = MAX(best_SM_arch, major);
		}
		current_device++;
	}

    // Find the best CUDA capable GPU device
	current_device = 0;
	while( current_device < device_count ) {
		CU_SAFE_CALL_NO_SYNC( cuDeviceGetAttribute( &multiProcessorCount, 
			                                        CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, 
													current_device ) );
		CU_SAFE_CALL_NO_SYNC( cuDeviceGetAttribute( &clockRate, 
			                                        CU_DEVICE_ATTRIBUTE_CLOCK_RATE, 
													current_device ) );

		if (major == 9999 && minor == 9999) {
		    sm_per_multiproc = 1;
		} else if (major <= 2) {
			sm_per_multiproc = arch_cores_sm[major];
		} else {
			sm_per_multiproc = arch_cores_sm[2];
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

// These are the inline versions for all of the CUTIL functions
inline void __cuSafeCallNoSync( CUresult err, const char *file, const int line )
{
    if( CUDA_SUCCESS != err) {
        fprintf(stderr, "cuSafeCallNoSync() Driver API error = %04d from file <%s>, line %i.\n",
                err, file, line );
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
        fprintf(stderr, "cuCtxSynchronize() API error = %04d in file <%s>, line %i.\n",
                err, file, line );
        exit(-1);
    }
}

inline void __cuCheckMsg( const char * msg, const char *file, const int line )
{
    CUresult err = cuCtxSynchronize();
    if( CUDA_SUCCESS != err) {
		fprintf(stderr, "cutilDrvCheckMsg -> %s", msg);
        fprintf(stderr, "cutilDrvCheckMsg -> cuCtxSynchronize API error = %04d in file <%s>, line %i.\n",
                err, file, line );
        exit(-1);
    }
}


#if __DEVICE_EMULATION__
    inline void cutilDeviceInitDrv(int cuDevice, int ARGC, char **ARGV) { } 
#else
    inline void cutilDeviceInitDrv(int cuDevice, int ARGC, char ** ARGV) 
    {
        cuDevice = 0;
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
        if (dev > deviceCount-1) dev = deviceCount - 1;
        cutilDrvSafeCallNoSync(cuDeviceGet(&cuDevice, dev));
        char name[100];
        cuDeviceGetName(name, 100, cuDevice);
        if (cutCheckCmdLineFlag(ARGC, (const char **) ARGV, "quiet") == CUTFalse)
            fprintf(stderr, "Using CUDA device [%d]: %s\n", dev, name);
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
        printf("> Compute SM %d.%d Device Detected\n", major, minor);
        printf("> Device %d: <%s>\n", dev, device_name);
        return true;
    }
    else
    {
        printf("There is no device supporting CUDA compute capability %d.%d.\n", major_version, minor_version);
        printf("PASSED\n");
        return false;
    }
}

// General check for CUDA GPU SM Capabilities
inline bool cutilDrvCudaCapabilities(int major_version, int minor_version)
{
	return cutilDrvCudaDevCapabilities(major_version, minor_version, 0);
}


#endif // _CUTIL_INLINE_FUNCTIONS_DRVAPI_H_
