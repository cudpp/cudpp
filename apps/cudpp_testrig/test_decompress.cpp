// -------------------------------------------------------------
// CUDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision: $
// $Date: $
// -------------------------------------------------------------
// This source code is distributed under the terms of license.txt in
// the root directory of this source distribution.
// -------------------------------------------------------------

/** @file test_decompress.cpp
 *  @brief Host testrig routines to exercise cudpp's decompression functionality.
 */

#include <iostream>
#include "decompress_gold.cpp"

int main(int argc, char* argv[])
{
    int ret_val = 0;
    int length = 100;
    unsigned char* input;

    if (argc > 1 && argv[1] == string("rand")) {
        if (argc > 2 && argv[2] != string("q")) length = (int)*argv[2];
        //input = new unsigned char[length + 1];
        input = new unsigned char[length];

        for (int i=0; i<length; i++)
            input[i] = (rand() % 255) + 1;

        //input[length] = (unsigned char)*"\0";
    }
    //unsigned char input[] = "The quick brown fox jumps over the lazy dog.";
    size_t num_elements = length;
    ret_val = computeDecompressGold(input, num_elements, !(argc-1) || ((argv[argc-1] == string("q")) ? false : true));

    return ret_val;
}
