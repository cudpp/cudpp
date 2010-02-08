// -------------------------------------------------------------
// cuDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision: 3716 $
// $Date: 2007-10-12 13:55:18 +0100 (Fri, 12 Oct 2007) $
// ------------------------------------------------------------- 
// This source code is distributed under the terms of license.txt in
// the root directory of this source distribution.
// ------------------------------------------------------------- 

/**
 * @file
 * cudpp_testrig.cu
 * 
 * @brief Main testing file for cudpp library.
 * 
 * Main testing file for cudpp library. Host code to link with cudpp
 * to exercise and test cudpp functionality. Contains regression
 * script to test cudpp calls. Simplest test to run regression is
 * "cudpp_testrig.exe --all".
 */

#include <cutil.h>
#include <memory.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include <cutil.h>
#include <cstdlib>
#include <cstdio>

#include "cudpp.h"

#ifndef __linux__
extern "C"
{
#endif
    
#ifndef __linux__
}
#endif
#include "arraycompare.h"

int testScan(int argc, const char ** argv, const CUDPPConfiguration *config);
int testSegmentedScan(int argc, const char ** argv, const CUDPPConfiguration *config);
int testMultiSumScan(int argc, const char ** argv);
int testCompact(int argc, const char ** argv, const CUDPPConfiguration *config);
int testRadixSort(int argc, const char ** argv, const CUDPPConfiguration *config);
int testSparseMatrixVectorMultiply(int argc, const char ** argv);
int testRandMD5(int argc, const char ** argv);
int testReduce(int argc, const char ** argv, const CUDPPConfiguration *config);
int testTridiagonal(int argc, const char** argv);

/**
 * main in cudpp_testrig is a dispatch routine to exercise cudpp functionality. 
 *
 * - --all calls every regression routine.
 *   - The scan regression calls forward and backward sum and max scans.
 * - --scan calls one scan regression routine (by default, forward sum-scan)
 *   - Use --backward and/or --op=max to change default
 * - --multiscan calls the multiscan regression routine
 * - --compact calls the compact regression routine
 * - --sort calls the sort regression routine
 * - --spmvmult calls the sparse matrix-vector routine
 * - --n=# sets the size of the dataset
 * - --numIterations=# sets the number of iterations to run
 */ 
int main(int argc, const char** argv)
{
    bool quiet = (CUTTrue == cutCheckCmdLineFlag(argc, argv, "quiet"));

    CUT_DEVICE_INIT(argc, argv);
    cudaDeviceProp prop;
    int dev = 0;
    cudaGetDevice(&dev);
    if (!quiet && cudaGetDeviceProperties(&prop, dev) == 0)
    {
#ifdef __DEVICE_EMULATION__
        printf("[EMU] ");
#endif

        printf("%s; global mem: %dB; compute v%d.%d; clock: %d kHz\n",
               prop.name, prop.totalGlobalMem, prop.major, prop.minor,
               prop.clockRate);
    }

    int retval = 0;

    if (argc == 1 || (CUTTrue == cutCheckCmdLineFlag(argc, argv, "help")))
    {
        printf("Usage: \"cudpp_testrig -<flag> -<option>=<value>\"\n\n");
        printf("--- Global Flags ---\n");
        printf("all: Run all tests\n");
        printf("scan: Run scan test(s)\n");
        printf("segscan; Run segmented scan test(s)\n");
        printf("multiscan: Run multi-row scan test(s)\n");
        printf("sort: Run sort test(s)\n");
        printf("compact: Run compact test(s)\n\n");
        printf("rand: Run random number generator test(s)\n\n");
        printf("--- Global Options ---\n");
        printf("iterations=<N>: Number of times to run each test\n");
        printf("n=<N>: Number of values to use in a single test\n");
        printf("r=<N>: Number of rows to scan (--multiscan only)\n\n");
        printf("--- Scan (Segmented and Unsegmented) Options ---\n");
        printf("backward: Run backward scans\n");
        printf("forward: Run backward scans (default)\n");
        printf("op=<OP>: Set scan operation to OP "
               "(OP=\"sum\", \"max\" \"min\" and \"multiply\"  currently. "
               "Default is sum)\n");
        printf("inclusive: Run inclusive scan (default)\n");
        printf("Exclusive: Run exclusive scan \n\n");
        printf("--- Radix Sort Options ---\n");
        printf("uint: Run radix sort on unsigned int keys (default)\n");
        printf("float: Run radix sort on float keys\n");
        printf("keyval: Run radix sort on key/value pairs (default)\n");
        printf("keysonly: Run radix sort on keys only\n");
        printf("keybits=<# bits>: Run radix sort on specified number "
               "of bits in the key (default is 32)\n");
        printf("--- Sparse Matrix-Vector Multiply Options ---\n");
        printf("mat=<File Name>: File containing sparse matrix in Matrix Market format\n");
        printf("--- Rand Options ---\n");
        printf("dir=<directory>: Directory containing all the random number regression tests\n");
    }

    bool testAll = (CUTTrue == cutCheckCmdLineFlag(argc, argv, "all"));
    

    if (testAll || (CUTTrue == cutCheckCmdLineFlag(argc, argv, "scan")))
    {
        if (testAll)
        {
            CUDPPConfiguration config;
            config.algorithm = CUDPP_SCAN;
            config.datatype = CUDPP_FLOAT;
            config.op = CUDPP_ADD;
            
            // Forward exclusive sum-scan            
            config.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_EXCLUSIVE;
            retval += testScan(argc, argv, &config);
            // Forward inclusive sum-scan            
            config.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_INCLUSIVE;
            retval += testScan(argc, argv, &config);
            // Backward exclusive sum-scan
            config.options = CUDPP_OPTION_BACKWARD | CUDPP_OPTION_EXCLUSIVE;
            retval += testScan(argc, argv, &config);
            // Backward inclusive sum-scan
            config.options = CUDPP_OPTION_BACKWARD | CUDPP_OPTION_INCLUSIVE;
            retval += testScan(argc, argv, &config);

            config.op = CUDPP_ADD;

            // Forward exclusive mul-scan            
            config.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_EXCLUSIVE;
            retval += testScan(argc, argv, &config);
            // Forward inclusive mul-scan            
            config.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_INCLUSIVE;
            retval += testScan(argc, argv, &config);
            // Backward exclusive mul-scan
            config.options = CUDPP_OPTION_BACKWARD | CUDPP_OPTION_EXCLUSIVE;
            retval += testScan(argc, argv, &config);
            // Backward inclusive mul-scan
            config.options = CUDPP_OPTION_BACKWARD | CUDPP_OPTION_INCLUSIVE;
            retval += testScan(argc, argv, &config);
                        
            config.op = CUDPP_MAX;

            // Forward exclusive max-scan            
            config.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_EXCLUSIVE;
            retval += testScan(argc, argv, &config);
            // Forward inclusive max-scan            
            config.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_INCLUSIVE;
            retval += testScan(argc, argv, &config);
            // Backward exclusive max-scan
            config.options = CUDPP_OPTION_BACKWARD | CUDPP_OPTION_EXCLUSIVE;
            retval += testScan(argc, argv, &config);
            // Backward inclusive max-scan
            config.options = CUDPP_OPTION_BACKWARD | CUDPP_OPTION_INCLUSIVE;
            retval += testScan(argc, argv, &config);

            config.op = CUDPP_MIN;

            // Forward exclusive min-scan            
            config.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_EXCLUSIVE;
            retval += testScan(argc, argv, &config);
            // Forward inclusive min-scan            
            config.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_INCLUSIVE;
            retval += testScan(argc, argv, &config);
            // Backward exclusive min-scan
            config.options = CUDPP_OPTION_BACKWARD | CUDPP_OPTION_EXCLUSIVE;
            retval += testScan(argc, argv, &config);
            // Backward inclusive min-scan
            config.options = CUDPP_OPTION_BACKWARD | CUDPP_OPTION_INCLUSIVE;
            retval += testScan(argc, argv, &config);
        }
        else
        {
            retval += testScan(argc, argv, NULL);
        }

    }

    if (testAll || (CUTTrue == cutCheckCmdLineFlag(argc, argv, "segscan")))
    {
        if (testAll)
        {
            CUDPPConfiguration config;
            config.algorithm = CUDPP_SEGMENTED_SCAN;
            config.datatype = CUDPP_FLOAT;
            config.op = CUDPP_ADD;                        

            // Forward exclusive sum-segscan            
            config.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_EXCLUSIVE;
            retval += testSegmentedScan(argc, argv, &config);
            // Forward inclusive sum-segscan            
            config.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_INCLUSIVE;
            retval += testSegmentedScan(argc, argv, &config);
            // Backward exclusive sum-segscan            
            config.options = CUDPP_OPTION_BACKWARD | CUDPP_OPTION_EXCLUSIVE;
            retval += testSegmentedScan(argc, argv, &config);
            // Backward inclusive sum-segscan            
            config.options = CUDPP_OPTION_BACKWARD | CUDPP_OPTION_INCLUSIVE;
            retval += testSegmentedScan(argc, argv, &config);

            config.op = CUDPP_MULTIPLY;

            // Forward exclusive mul-segscan            
            config.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_EXCLUSIVE;
            retval += testSegmentedScan(argc, argv, &config);
            // Forward inclusive mul-segscan            
            config.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_INCLUSIVE;
            retval += testSegmentedScan(argc, argv, &config);
            // Backward exclusive mul-segscan            
            config.options = CUDPP_OPTION_BACKWARD | CUDPP_OPTION_EXCLUSIVE;
            retval += testSegmentedScan(argc, argv, &config);
            // Backward inclusive mul-segscan            
            config.options = CUDPP_OPTION_BACKWARD | CUDPP_OPTION_INCLUSIVE;
            retval += testSegmentedScan(argc, argv, &config);

            config.op = CUDPP_MAX;                        

            // Forward exclusive max-segscan            
            config.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_EXCLUSIVE;
            retval += testSegmentedScan(argc, argv, &config);
            // Forward inclusive max-segscan            
            config.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_INCLUSIVE;
            retval += testSegmentedScan(argc, argv, &config);
            // Backward exclusive max-segscan            
            config.options = CUDPP_OPTION_BACKWARD | CUDPP_OPTION_EXCLUSIVE;
            retval += testSegmentedScan(argc, argv, &config);
            // Backward inclusive max-segscan            
            config.options = CUDPP_OPTION_BACKWARD | CUDPP_OPTION_INCLUSIVE;
            retval += testSegmentedScan(argc, argv, &config);

            config.op = CUDPP_MIN;

            // Forward exclusive min-segscan            
            config.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_EXCLUSIVE;
            retval += testSegmentedScan(argc, argv, &config);
            // Forward inclusive min-segscan            
            config.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_INCLUSIVE;
            retval += testSegmentedScan(argc, argv, &config);
            // Backward exclusive min-segscan            
            config.options = CUDPP_OPTION_BACKWARD | CUDPP_OPTION_EXCLUSIVE;
            retval += testSegmentedScan(argc, argv, &config);
            // Backward inclusive min-segscan            
            config.options = CUDPP_OPTION_BACKWARD | CUDPP_OPTION_INCLUSIVE;
            retval += testSegmentedScan(argc, argv, &config);
        }
        else
        {
            retval += testSegmentedScan(argc, argv, NULL);
        }
    }

    if (testAll || (CUTTrue == cutCheckCmdLineFlag(argc, argv, "multiscan")))
    {
        retval += testMultiSumScan(argc, argv);
    }
    

    if (testAll || (CUTTrue == cutCheckCmdLineFlag(argc, argv, "compact")))
    {
        if (testAll)
        {
            CUDPPConfiguration config;
            config.algorithm = CUDPP_COMPACT;
            config.options = CUDPP_OPTION_FORWARD;
            config.datatype = CUDPP_FLOAT;
            retval += testCompact(argc, argv, &config);
            config.options = CUDPP_OPTION_BACKWARD;
            retval += testCompact(argc, argv, &config);
        }
        else
            retval += testCompact(argc, argv, NULL);
    }

    if (testAll || (CUTTrue == cutCheckCmdLineFlag(argc, argv, "reduce")))
    {
        if (testAll)
        {
            CUDPPConfiguration config;
            config.options = 0;
            config.algorithm = CUDPP_REDUCE;
            config.op = CUDPP_ADD;
            config.datatype = CUDPP_FLOAT;
            retval += testReduce(argc, argv, &config);
            config.datatype = CUDPP_INT;
            retval += testReduce(argc, argv, &config);
            config.datatype = CUDPP_UINT;
            retval += testReduce(argc, argv, &config);
            config.datatype = CUDPP_DOUBLE;
            retval += testReduce(argc, argv, &config);

            config.op = CUDPP_MULTIPLY;
            config.datatype = CUDPP_FLOAT;
            retval += testReduce(argc, argv, &config);
            config.datatype = CUDPP_INT;
            retval += testReduce(argc, argv, &config);
            config.datatype = CUDPP_UINT;
            retval += testReduce(argc, argv, &config);
            config.datatype = CUDPP_DOUBLE;
            retval += testReduce(argc, argv, &config);

            config.op = CUDPP_MIN;
            config.datatype = CUDPP_FLOAT;
            retval += testReduce(argc, argv, &config);
            config.datatype = CUDPP_INT;
            retval += testReduce(argc, argv, &config);
            config.datatype = CUDPP_UINT;
            retval += testReduce(argc, argv, &config);
            config.datatype = CUDPP_DOUBLE;
            retval += testReduce(argc, argv, &config);

            config.op = CUDPP_MAX;
            config.datatype = CUDPP_FLOAT;
            retval += testReduce(argc, argv, &config);
            config.datatype = CUDPP_INT;
            retval += testReduce(argc, argv, &config);
            config.datatype = CUDPP_UINT;
            retval += testReduce(argc, argv, &config);
            config.datatype = CUDPP_DOUBLE;
            retval += testReduce(argc, argv, &config);
        }
        else
            retval += testReduce(argc, argv, NULL);
    }

    if (testAll || (CUTTrue == cutCheckCmdLineFlag(argc, argv, "sort")))
    {
        if(testAll)
        {
            CUDPPConfiguration config;
            config.algorithm = CUDPP_SORT_RADIX;                  
            config.datatype = CUDPP_UINT;
            config.op = CUDPP_ADD;
            config.options = 0;
            config.options = CUDPP_OPTION_KEY_VALUE_PAIRS;
            retval += testRadixSort(argc, argv, &config);
            config.options = CUDPP_OPTION_KEYS_ONLY;              
            retval += testRadixSort(argc, argv, &config);
            config.datatype = CUDPP_FLOAT;
            retval += testRadixSort(argc, argv, &config);
            config.options = CUDPP_OPTION_KEY_VALUE_PAIRS;                
            retval += testRadixSort(argc, argv, &config);
        }  
        else
            retval += testRadixSort(argc, argv, NULL);
    }    

    if ((CUTTrue == cutCheckCmdLineFlag(argc, argv, "spmvmult")))
    {
        retval += testSparseMatrixVectorMultiply(argc, argv);
    }    

    if (testAll ||(CUTTrue == cutCheckCmdLineFlag(argc, argv, "rand")))
    {
        //in the future we need to add so that it tests other random numbers as well
        retval += testRandMD5(argc, argv);
    }

    if (testAll ||(CUTTrue == cutCheckCmdLineFlag(argc, argv, "tridiagonal")))
    {
        retval += testTridiagonal(argc, argv);
    }

    if (retval)
    {
        if (!quiet)
            printf("%d tests failed\n", retval);
    }
    else
    {
        if (!quiet)
            printf("All tests passed.\n");
    }
    return 0;//retval;
}
