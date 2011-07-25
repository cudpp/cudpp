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

#include <memory.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cstdlib>
#include <cstdio>

#include <cuda_runtime_api.h>
#include "cudpp.h"
#include "cudpp_testrig_utils.h"
#include "cudpp_testrig_options.h"

#define CUDPP_APP_COMMON_IMPL
#include "stopwatch.h"
#include "findfile.h"
#include "commandline.h"

using namespace cudpp_app;

cudaDeviceProp devProps;

int testScan(int argc, const char ** argv, const CUDPPConfiguration *config, 
             bool multiRow, cudaDeviceProp props);
int testCompact(int argc, const char ** argv, const CUDPPConfiguration *config);
int testRadixSort(int argc, const char ** argv, const CUDPPConfiguration *config);
int testReduce(int argc, const char ** argv, const CUDPPConfiguration *config);
int testSparseMatrixVectorMultiply(int argc, const char ** argv);
int testRandMD5(int argc, const char ** argv);
int testTridiagonal(int argc, const char** argv, const CUDPPConfiguration *config);

int testAllDatatypes(int argc, 
                     const char** argv, 
                     CUDPPConfiguration & config, 
                     bool supportsDouble,
                     bool multiRow)
{
    int retval = 0;
    for (CUDPPDatatype dt = CUDPP_INT; dt != CUDPP_DATATYPE_INVALID; dt = CUDPPDatatype((int)dt+1))
    {
        config.datatype = dt;
        if (config.datatype != CUDPP_DOUBLE || supportsDouble) {
            switch (config.algorithm) {
                case CUDPP_SCAN:
                case CUDPP_SEGMENTED_SCAN:
                    retval += testScan(argc, argv, &config, multiRow, devProps);
                    break;
                case CUDPP_REDUCE:
                    retval += testReduce(argc, argv, &config);
                    break;
                case CUDPP_COMPACT:
                    retval += testCompact(argc, argv, &config);
                    break;
                case CUDPP_SORT_RADIX:
                    retval += testRadixSort(argc, argv, &config);
                default:
                    break;  
            }
        }
    }

    if (config.algorithm == CUDPP_TRIDIAGONAL)
    {
        config.datatype = CUDPP_FLOAT;
        retval += testTridiagonal(argc, argv, &config);

        if (supportsDouble)
        {
            config.datatype = CUDPP_DOUBLE;
            retval += testTridiagonal(argc, argv, &config);      
        }
    }
    
    return retval;
}

int testAllOptionsAndDatatypes(int argc, 
                               const char** argv, 
                               CUDPPConfiguration & config, 
                               bool supportsDouble,
                               bool multiRow = false)
{
    int retval = 0;
    
    if (config.algorithm == CUDPP_SORT_RADIX)
    {
        config.options = CUDPP_OPTION_KEY_VALUE_PAIRS | CUDPP_OPTION_FORWARD;
        retval += testAllDatatypes(argc, argv, config, supportsDouble, multiRow);
        config.options = CUDPP_OPTION_KEYS_ONLY | CUDPP_OPTION_FORWARD;
        retval += testAllDatatypes(argc, argv, config, supportsDouble, multiRow);              
        config.options = CUDPP_OPTION_KEY_VALUE_PAIRS | CUDPP_OPTION_BACKWARD;
        retval += testAllDatatypes(argc, argv, config, supportsDouble, multiRow);
        config.options = CUDPP_OPTION_KEYS_ONLY | CUDPP_OPTION_BACKWARD;
        retval += testAllDatatypes(argc, argv, config, supportsDouble, multiRow);              
        return retval;
    }
    
    config.op = CUDPP_ADD;
    
    bool runReduce = (config.algorithm == CUDPP_REDUCE);
    
    config.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_EXCLUSIVE;
    retval += testAllDatatypes(argc, argv, config, supportsDouble, multiRow);        
    
    if (!runReduce)
    {
        config.options = CUDPP_OPTION_BACKWARD | CUDPP_OPTION_EXCLUSIVE;
        retval += testAllDatatypes(argc, argv, config, supportsDouble, multiRow);        
    }

    if (config.algorithm == CUDPP_COMPACT) // only one operator for compact
        return retval;
        
    if (!runReduce)
    {
        config.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_INCLUSIVE;
        retval += testAllDatatypes(argc, argv, config, supportsDouble, multiRow);        
        config.options = CUDPP_OPTION_BACKWARD | CUDPP_OPTION_INCLUSIVE;
        retval += testAllDatatypes(argc, argv, config, supportsDouble, multiRow);            
    } 

    config.op = CUDPP_MULTIPLY;

    config.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_EXCLUSIVE;
    retval += testAllDatatypes(argc, argv, config, supportsDouble, multiRow);        
    if (!runReduce)
    {   
        config.options = CUDPP_OPTION_BACKWARD | CUDPP_OPTION_EXCLUSIVE;
        retval += testAllDatatypes(argc, argv, config, supportsDouble, multiRow);        
        config.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_INCLUSIVE;
        retval += testAllDatatypes(argc, argv, config, supportsDouble, multiRow);        
        config.options = CUDPP_OPTION_BACKWARD | CUDPP_OPTION_INCLUSIVE;
        retval += testAllDatatypes(argc, argv, config, supportsDouble, multiRow);        
    }                

    config.op = CUDPP_MAX;

    config.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_EXCLUSIVE;
    retval += testAllDatatypes(argc, argv, config, supportsDouble, multiRow);        
    if (!runReduce)
    {
        config.options = CUDPP_OPTION_BACKWARD | CUDPP_OPTION_EXCLUSIVE;
        retval += testAllDatatypes(argc, argv, config, supportsDouble, multiRow);        
        config.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_INCLUSIVE;
        retval += testAllDatatypes(argc, argv, config, supportsDouble, multiRow);        
        config.options = CUDPP_OPTION_BACKWARD | CUDPP_OPTION_INCLUSIVE;
        retval += testAllDatatypes(argc, argv, config, supportsDouble, multiRow);        
    }
    
    config.op = CUDPP_MIN;

    config.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_EXCLUSIVE;
    retval += testAllDatatypes(argc, argv, config, supportsDouble, multiRow);        
    if (!runReduce)
    {
        config.options = CUDPP_OPTION_BACKWARD | CUDPP_OPTION_EXCLUSIVE;
        retval += testAllDatatypes(argc, argv, config, supportsDouble, multiRow);
        config.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_INCLUSIVE;
        retval += testAllDatatypes(argc, argv, config, supportsDouble, multiRow);        
        config.options = CUDPP_OPTION_BACKWARD | CUDPP_OPTION_INCLUSIVE;
        retval += testAllDatatypes(argc, argv, config, supportsDouble, multiRow);
    }    
    
    return retval;
}

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
 * - --reduce calls the reduce regression routine
 * - --n=# sets the size of the dataset
 * - --iterations=# sets the number of iterations to run
 */ 
int main(int argc, const char** argv)
{
    bool quiet = checkCommandLineFlag(argc, argv, "quiet"); 

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        fprintf(stderr, "error: no devices supporting CUDA.\n");
        exit(EXIT_FAILURE);
    }
    int dev = 0;
    commandLineArg(dev, argc, argv, "device");
    if (dev < 0) dev = 0;
    if (dev > deviceCount-1) dev = deviceCount - 1;
    cudaSetDevice(dev);

    if (!quiet && cudaGetDeviceProperties(&devProps, dev) == 0)
    {
        printf("Using device %d:\n", dev);
        printf("%s; global mem: %dB; compute v%d.%d; clock: %d kHz\n",
               devProps.name, (int)devProps.totalGlobalMem, (int)devProps.major, 
               (int)devProps.minor, (int)devProps.clockRate);
    }

    int computeVersion = devProps.major * 10 + devProps.minor;
    bool supportsDouble = (computeVersion >= 13);

    int retval = 0;

    if (argc == 1 || checkCommandLineFlag(argc, argv, "help"))
    {
        printf("Usage: \"cudpp_testrig -<flag> -<option>=<value>\"\n\n");
        printf("--- Global Flags ---\n");
        printf("all: Run all tests\n");
        printf("scan: Run scan test(s)\n");
        printf("segscan; Run segmented scan test(s)\n");
        printf("multiscan: Run multi-row scan test(s)\n");
        printf("sort: Run sort test(s)\n");
        printf("compact: Run compact test(s)\n\n");
        printf("reduce: Run reduce test(s)\n\n");
        printf("rand: Run random number generator test(s)\n\n");
        printf("tridiagonal: Run tridiagonal solver test(s)\n\n");	
        printf("--- Global Options ---\n");
        printf("iterations=<N>: Number of times to run each test\n");
        printf("n=<N>: Number of values to use in a single test\n");
        printf("r=<N>: Number of rows to scan (--multiscan only)\n\n");
        printf("--- Scan (Segmented and Unsegmented) Options ---\n");
        printf("backward: Run backward scans\n");
        printf("forward: Run forward scans (default)\n");
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
        printf("forward: Run forward sorts (default)\n");
        printf("backward: Run backward sorts (DOES NOT WORK YET)\n");
        printf("--- Sparse Matrix-Vector Multiply Options ---\n");
        printf("mat=<File Name>: File containing sparse matrix in Matrix Market format\n");
        printf("--- Rand Options ---\n");
        printf("dir=<directory>: Directory containing all the random number regression tests\n");
    }
    
    bool runAll = checkCommandLineFlag(argc, argv, "all");
    bool runScan = runAll || checkCommandLineFlag(argc, argv, "scan");
    bool runSegScan = runAll || checkCommandLineFlag(argc, argv, "segscan");
    bool runMultiScan = runAll || checkCommandLineFlag(argc, argv, "multiscan");
    bool runCompact = runAll || checkCommandLineFlag(argc, argv, "compact");
    bool runReduce = runAll || checkCommandLineFlag(argc, argv, "reduce");
    bool runSort = runAll || checkCommandLineFlag(argc, argv, "sort");
    bool runRand = runAll || checkCommandLineFlag(argc, argv, "rand");
    bool runSpmv = checkCommandLineFlag(argc, argv, "spmv");
    bool runTridiagonal = checkCommandLineFlag(argc, argv, "tridiagonal");
    
    bool hasopts = hasOptions(argc, argv);

    if (hasopts) 
    {
        if (runScan)      retval += testScan(argc, argv, NULL, false, devProps);
        if (runSegScan)   retval += testScan(argc, argv, NULL, false, devProps);
        if (runCompact)   retval += testCompact(argc, argv, NULL);
        if (runReduce)    retval += testReduce(argc, argv, NULL);
        if (runSort)      retval += testRadixSort(argc, argv, NULL);
        if (runMultiScan) retval += testScan(argc, argv, NULL, true, devProps);
    }
    else
    {
        CUDPPConfiguration config;
        config.options = 0;
        
        if (runScan) {
            config.algorithm = CUDPP_SCAN;
            retval += testAllOptionsAndDatatypes(argc, argv, config, supportsDouble);        
        }
    
        if (runSegScan) {
            config.algorithm = CUDPP_SEGMENTED_SCAN;
            retval += testAllOptionsAndDatatypes(argc, argv, config, supportsDouble);                        
        }
        
        if (runCompact) {
            config.algorithm = CUDPP_COMPACT;
            retval += testAllOptionsAndDatatypes(argc, argv, config, supportsDouble);        
        }
        
        if (runReduce) {
            config.algorithm = CUDPP_REDUCE;
            retval += testAllOptionsAndDatatypes(argc, argv, config, supportsDouble);
        }
        
        if (runSort) {
            config.algorithm = CUDPP_SORT_RADIX;
            retval += testAllOptionsAndDatatypes(argc, argv, config, supportsDouble);
        }
        
        if (runMultiScan) {
            config.algorithm = CUDPP_SCAN;
            retval += testAllOptionsAndDatatypes(argc, argv, config, supportsDouble, true);        
        }
        
        if (runTridiagonal) {
            config.algorithm = CUDPP_TRIDIAGONAL;
            retval += testAllDatatypes(argc, argv, config, supportsDouble, false);
        }    

    }

    if (runSpmv)
    {
        retval += testSparseMatrixVectorMultiply(argc, argv);
    }    

    if (runRand)
    {
        //in the future we need to add so that it tests other random numbers as well
        retval += testRandMD5(argc, argv);
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


// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
