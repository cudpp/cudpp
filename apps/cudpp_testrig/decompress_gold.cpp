/*  decompress_gold.cpp
 *
 *  Contains functions to compress a file on the CPU. These functions are called
 *  as part of the test routines in test_decompress.cpp
 *
 *  Resulting compressed file is decompressed on the GPU to test the CUDPP
 *  decompression functionality.
 */

/*  Trevor Gibson, UC Davis Dept. of Electrical and Computer Engineering
 *  Date created: 8/20/14
 */

#include <iostream>

// Compresses a file on the CPU to test decompression on the GPU
int computeDecompressGold(/*  Params:
                           *     In:
                           *        - input file (uncompressed)
                           *     Out:
                           *        - output file (compressed)
                           */)
{
    /*  Steps:
     *     - Allocated memory
     *     - Run BWT
     *     - Run MTF transform
     *     - Run Huffman encoding
     */
}

// Run a Burrows-Wheeler Transform (BWT) for computeDecompressGold()
int computeBWT(/*  Params:
                *     In: - input data array (i_data: characters in input file, in order)
                *         - length of input data array (array_len)
                *     Out: 
                *         - BWT transformed data (o_data: same characters as
                *           input characters, just rearranged)
                */)
{
/*  Steps:
 *     - Allocated memory for all rotations
 *     - Calculate every rotation (loop 'array_len' times)
 *     - Sort rotations in lexigraphical order
 *     - Take last character from each rotation (in order) and add (in order)
 *       to output array
 */
}

// Run a Move-To-Front (MTF) Transform for computeDecompressGold()
int computeMTF(/*  Params:
                *     In: - input data array from BWT (i_data: original input
                            characters in random order)
                *         - length of input data array (array_len)
                *     Out: 
                *         - MTF transformed data (o_data, list of numbers)
                */)
