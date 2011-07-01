// -------------------------------------------------------------
// cuDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision$
// $Date$
// ------------------------------------------------------------- 
// This source code is distributed under the terms of license.txt
// in the root directory of this source distribution.
// ------------------------------------------------------------- 
#include "cudpp_testrig_options.h"
/**
 * Sets "global" options in testOptions given command line 
 * - --debug: sets bool <var>debug</var>. Usage is application-dependent.
 * - --op=OP: sets char * <var>op</var> to OP
 * - --iterations=#: sets int <var>numIterations</var> to #
 */

extern "C"
void setOptions(int argc, const char **argv, testrigOptions &testOptions)
{
    testOptions.debug = 
        (cutCheckCmdLineFlag(argc, (const char**) argv, "debug") == CUTTrue)
        ? true : false;

    cutGetCmdLineArgumentstr(argc, (const char**) argv, "op", &testOptions.op);
    if (testOptions.op == NULL)
    {
        testOptions.op = (char*)"sum";
    }

    testOptions.numIterations = numTestIterations;
    cutGetCmdLineArgumenti(argc, (const char**) argv, "iterations", 
                           &testOptions.numIterations);
    
    //get the rand path, if there is one
    cutGetCmdLineArgumentstr(argc, (const char**) argv, "dir", &testOptions.dir);
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
