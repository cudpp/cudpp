CUDA Utility Library
====================

CUTIL is a simple utility library designed for use in the CUDA SDK samples.

It provides functions for:
- parsing command line arguments
- read and writing binary files and PPM format images
- comparing arrays of data (typically used for comparing GPU results with CPU)
- timers
- macros for checking error codes
- checking for shared memory bank conflicts

CUTIL is not part of CUDA
=========================

Note that CUTIL is not part of the CUDA Toolkit and is not supported by NVIDIA.
It exists only for the convenience of writing concise and platform-independent
example code. 

Library Functions
=================

Most of the functions should be self explanatory. The function parameters are
documented in the "cutil.h" file.

Macros
======

CUTIL includes a number of macros that can be used to easily initialize the
device, and automatically check the error codes returned by CUDA runtime
functions when debugging.

These macros are compiled out in release builds and so they will not affect
performance. Note that in debug mode they call cudaThreadSynchronize()
to ensure that kernel execution has completed, which can affect performance.

CUT_INIT_DEVICE
- this macro finds the first available CUDA device and initializes it. When
compiling for device emulation it has no effect.

CUT_EXIT
- this simply exits the program, prompting the user to press enter so that
the console window doesn't disappear too quickly under Windows. You can force
SDK samples to exit without a prompt by passing the "--noprompt" command line 
option.

CUDA_SAFE_CALL(call)
- this macro is intended to be wrapped around a CUDA runtime API call. It
checks the returned error code and exits with a message if there is an error.

CU_SAFE_CALL(call)
- as above, but designed for CUDA driver API calls

CUT_SAFE_CALL(call)
- as above, but for CUTIL functions.

CUT_CHECK_ERROR
- checks for CUDA runtime errors.

CUT_CHECK_ERROR_GL
- checks for OpenGL errors
