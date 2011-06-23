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
 
#ifndef _CUTIL_INLINE_FUNCTIONS_RUNTIME_H_
#define _CUTIL_INLINE_FUNCTIONS_RUNTIME_H_

#ifdef _WIN32
#ifdef _DEBUG // Do this only in debug mode...
#  define WINDOWS_LEAN_AND_MEAN
#  include <windows.h>
#  include <stdlib.h>
#  undef min
#  undef max
#endif
#endif

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <cufft.h>

// We define these calls here, so the user doesn't need to include __FILE__ and __LINE__
// The advantage is the developers gets to use the inline function so they can debug
#define cutilSafeCallNoSync(err)     __cudaSafeCallNoSync(err, __FILE__, __LINE__)
#define cutilSafeCall(err)           __cudaSafeCall      (err, __FILE__, __LINE__)
#define cutilSafeThreadSync()        __cudaSafeThreadSync(__FILE__, __LINE__)
#define cufftSafeCall(err)           __cufftSafeCall     (err, __FILE__, __LINE__)
#define cutilCheckError(err)         __cutilCheckError   (err, __FILE__, __LINE__)
#define cutilCheckMsg(msg)           __cutilCheckMsg     (msg, __FILE__, __LINE__)
#define cutilSafeMalloc(mallocCall)  __cutilSafeMalloc   ((mallocCall), __FILE__, __LINE__)
#define cutilCondition(val)          __cutilCondition    (val, __FILE__, __LINE__)
#define cutilExit(argc, argv)        __cutilExit         (argc, argv)

inline void __cutilCondition(int val, char *file, int line) 
{
    if( CUTFalse == cutCheckCondition( val, file, line ) ) {
        exit(EXIT_FAILURE);
    }
}

inline void __cutilExit(int argc, char **argv)
{
    if (!cutCheckCmdLineFlag(argc, (const char**)argv, "noprompt")) {
        printf("\nPress ENTER to exit...\n");
        fflush( stdout);
        fflush( stderr);
        getchar();
    }
    exit(EXIT_SUCCESS);
}

#define MIN(a,b) ((a < b) ? a : b)
#define MAX(a,b) ((a > b) ? a : b)

// This function returns the best GPU (with maximum GFLOPS)
inline int cutGetMaxGflopsDeviceId()
{
	int current_device   = 0, sm_per_multiproc = 0;
	int max_compute_perf = 0, max_perf_device  = 0;
	int device_count     = 0, best_SM_arch     = 0;
    int arch_cores_sm[3] = { 1, 8, 32 };
	cudaDeviceProp deviceProp;

	cudaGetDeviceCount( &device_count );
	// Find the best major SM Architecture GPU device
	while ( current_device < device_count ) {
		cudaGetDeviceProperties( &deviceProp, current_device );
		if (deviceProp.major > 0 && deviceProp.major < 9999) {
			best_SM_arch = MAX(best_SM_arch, deviceProp.major);
		}
		current_device++;
	}

    // Find the best CUDA capable GPU device
	current_device = 0;
	while( current_device < device_count ) {
		cudaGetDeviceProperties( &deviceProp, current_device );
		if (deviceProp.major == 9999 && deviceProp.minor == 9999) {
		    sm_per_multiproc = 1;
		} else if (deviceProp.major <= 2) {
			sm_per_multiproc = arch_cores_sm[deviceProp.major];
		} else {
			sm_per_multiproc = arch_cores_sm[2];
		}

		int compute_perf  = deviceProp.multiProcessorCount * sm_per_multiproc * deviceProp.clockRate;
		if( compute_perf  > max_compute_perf ) {
            // If we find GPU with SM major > 2, search only these
			if ( best_SM_arch > 2 ) {
				// If our device==dest_SM_arch, choose this, or else pass
				if (deviceProp.major == best_SM_arch) {	
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

// Give a little more for Windows : the console window often disapears before we can read the message
#ifdef _WIN32
# if 1//ndef UNICODE
#  ifdef _DEBUG // Do this only in debug mode...
	inline void VSPrintf(FILE *file, LPCSTR fmt, ...)
	{
		size_t fmt2_sz	= 2048;
		char *fmt2		= (char*)malloc(fmt2_sz);
		va_list  vlist;
		va_start(vlist, fmt);
		while((_vsnprintf(fmt2, fmt2_sz, fmt, vlist)) < 0) // means there wasn't anough room
		{
			fmt2_sz *= 2;
			if(fmt2) free(fmt2);
			fmt2 = (char*)malloc(fmt2_sz);
		}
		OutputDebugStringA(fmt2);
		fprintf(file, fmt2);
		free(fmt2);
	}
#	define FPRINTF(a) VSPrintf a
#  else //debug
#	define FPRINTF(a) fprintf a
// For other than Win32
#  endif //debug
# else //unicode
// Unicode case... let's give-up for now and keep basic printf
#	define FPRINTF(a) fprintf a
# endif //unicode
#else //win32
#	define FPRINTF(a) fprintf a
#endif //win32

// NOTE: "%s(%i) : " allows Visual Studio to directly jump to the file at the right line
// when the user double clicks on the error line in the Output pane. Like any compile error.

inline void __cudaSafeCallNoSync( cudaError err, const char *file, const int line )
{
    if( cudaSuccess != err) {
        FPRINTF((stderr, "%s(%i) : cudaSafeCallNoSync() Runtime API error : %s.\n",
                file, line, cudaGetErrorString( err) ));
        exit(-1);
    }
}

inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
    if( cudaSuccess != err) {
		FPRINTF((stderr, "%s(%i) : cudaSafeCall() Runtime API error : %s.\n",
                file, line, cudaGetErrorString( err) ));
        exit(-1);
    }
}

inline void __cudaSafeThreadSync( const char *file, const int line )
{
    cudaError err = cudaThreadSynchronize();
    if ( cudaSuccess != err) {
        FPRINTF((stderr, "%s(%i) : cudaThreadSynchronize() Driver API error : %s.\n",
                file, line, cudaGetErrorString( err) ));
        exit(-1);
    }
}

inline void __cufftSafeCall( cufftResult err, const char *file, const int line )
{
    if( CUFFT_SUCCESS != err) {
        FPRINTF((stderr, "%s(%i) : cufftSafeCall() CUFFT error.\n",
                file, line));
        exit(-1);
    }
}

inline void __cutilCheckError( CUTBoolean err, const char *file, const int line )
{
    if( CUTTrue != err) {
        FPRINTF((stderr, "%s(%i) : CUTIL CUDA error.\n",
                file, line));
        exit(-1);
    }
}

inline void __cutilCheckMsg( const char *errorMessage, const char *file, const int line )
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) {
        FPRINTF((stderr, "%s(%i) : cutilCheckMsg() CUTIL CUDA error : %s : %s.\n",
                file, line, errorMessage, cudaGetErrorString( err) ));
        exit(-1);
    }
#ifdef _DEBUG
    err = cudaThreadSynchronize();
    if( cudaSuccess != err) {
		FPRINTF((stderr, "%s(%i) : cutilCheckMsg cudaThreadSynchronize error: %s : %s.\n",
                file, line, errorMessage, cudaGetErrorString( err) ));
        exit(-1);
    }
#endif
}
inline void __cutilSafeMalloc( void *pointer, const char *file, const int line )
{
    if( !(pointer)) {
        FPRINTF((stderr, "%s(%i) : cutilSafeMalloc host malloc failure\n",
                file, line));
        exit(-1);
    }
}

#if __DEVICE_EMULATION__
    inline void cutilDeviceInit(int ARGC, char **ARGV) { }
    inline void cutilChooseCudaDevice(int ARGC, char **ARGV) { }
#else
    inline void cutilDeviceInit(int ARGC, char **ARGV)
    {
        int deviceCount;
        cutilSafeCallNoSync(cudaGetDeviceCount(&deviceCount));
        if (deviceCount == 0) {
            FPRINTF((stderr, "CUTIL CUDA error: no devices supporting CUDA.\n"));
            exit(-1);
        }
        int dev = 0;
        cutGetCmdLineArgumenti(ARGC, (const char **) ARGV, "device", &dev);
	    if (dev < 0) dev = 0;\
        if (dev > deviceCount-1) dev = deviceCount - 1;
        cudaDeviceProp deviceProp;
        cutilSafeCallNoSync(cudaGetDeviceProperties(&deviceProp, dev));
        if (deviceProp.major < 1) {
            FPRINTF((stderr, "cutil error: GPU device does not support CUDA.\n"));
            exit(-1);                                                  \
        }
        if (cutCheckCmdLineFlag(ARGC, (const char **) ARGV, "quiet") == CUTFalse)
            FPRINTF((stderr, "Using CUDA device [%d]: %s\n", dev, deviceProp.name));
        cutilSafeCall(cudaSetDevice(dev));
    }

    // General initialization call to pick the best CUDA Device
    inline void cutilChooseCudaDevice(int argc, char **argv)
    {
        // If the command-line has a device number specified, use it
        if( cutCheckCmdLineFlag(argc, (const char**)argv, "device") ) {
            cutilDeviceInit(argc, argv);
        } else {
            // Otherwise pick the device with highest Gflops/s
            cudaSetDevice( cutGetMaxGflopsDeviceId() );
        }
    }
#endif


//! Check for CUDA context lost
inline void cutilCudaCheckCtxLost(const char *errorMessage, const char *file, const int line ) 
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) {
        FPRINTF((stderr, "%s(%i) : CUDA error: %s : %s.\n",
        file, line, errorMessage, cudaGetErrorString( err) ));
        exit(-1);
    }
    err = cudaThreadSynchronize();
    if( cudaSuccess != err) {
        FPRINTF((stderr, "%s(%i) : CCUDA error: %s : %s.\n",
        file, line, errorMessage, cudaGetErrorString( err) ));
        exit(-1);
    }
}

// General check for CUDA GPU SM Capabilities
inline bool cutilCudaCapabilities(int major_version, int minor_version)
{
    cudaDeviceProp deviceProp;
    deviceProp.major = 0;
    deviceProp.minor = 0;
    int dev;

#ifdef __DEVICE_EMULATION__
    printf("> Compute Device Emulation Mode \n");
#endif

    cutilSafeCall( cudaGetDevice(&dev) );
    cutilSafeCall( cudaGetDeviceProperties(&deviceProp, dev));

    if((deviceProp.major > major_version) ||
	   (deviceProp.major == major_version && deviceProp.minor >= minor_version))
    {
        printf("> Compute SM %d.%d Device Detected\n", deviceProp.major, deviceProp.minor);
        printf("> Device %d: <%s>\n", dev, deviceProp.name);
        return true;
    }
    else
    {
        printf("There is no device supporting CUDA compute capability %d.%d.\n", major_version, minor_version);
        printf("PASSED\n");
        return false;
    }
}

#endif // _CUTIL_INLINE_FUNCTIONS_RUNTIME_H_
