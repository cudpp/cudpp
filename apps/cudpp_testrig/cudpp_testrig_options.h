// -------------------------------------------------------------
// cuDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision: $
// $Date: $
// ------------------------------------------------------------- 
// This source code is distributed under the terms of license.txt in
// the root directory of this source distribution.
// ------------------------------------------------------------- 

/**
 * @file
 * cudpp_testrig_options.h
 * 
 * @brief Sets global options from command line for cudpp_testrig
 * 
 * Used to set options such as number of iterations, and RUNMODE for
 * testrig functions
 */

#ifndef __CUDPP_TESTRIG_OPTIONS_H__
#define __CUDPP_TESTRIG_OPTIONS_H__

#ifdef WIN32
#include <windows.h>
#endif

#include <cudpp.h>
#include <limits.h>
#include <cutil.h>

#define numTestIterations 100

/**
 * @brief "Global" testrig options set by command line.
 *
 * @param runMode 
 * @param op String containing name of OP (useful in e.g. scan)
 * @param dir String containing the path of the random number regression test files
 * @param numIterations Number of iterations to run
 * @param debug Application-dependent bool, set if --debug 
 * 
 */
struct testrigOptions
{
    char *runMode;     //!< Set automatically to "EMU" for emulation, "GPU" for hardware.
    char *op;          //!< String containing name of OP (useful in e.g. scan).
    char *datatype;    //!< String containing data type.
    char *dir;         //!< String containing the path of the random number regression test files
    int numIterations; //!< Number of iterations to run.
    bool debug;        //!< Application-dependent bool, set if --debug 
};

extern "C"
void setOptions(int argc, const char **argv, testrigOptions &testOptions);

#endif

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
