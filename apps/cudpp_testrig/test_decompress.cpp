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
#include <ctime>
#include "decompress_gold.cpp"

int main(int argc, char* argv[])
{
    srand(time(NULL));

    int ret_val = 0;
    int length = 44;
    unsigned char* input = new unsigned char[length];
    unsigned char input2[] = "The quick brown fox jumps over the lazy dog.";

    if (argc > 1 && argv[1] == string("rand")) {
        //input = new unsigned char[length + 1];
        input = new unsigned char[length];

        for (int i=0; i<length; i++) {
            input[i] = (rand() % 255) + 1;
        }
    }
    size_t num_elements = length;
    ret_val = (argc > 1 ? computeDecompressGold(input, num_elements, !(argc-1) || ((argv[argc-1] == string("q")) ? false : true)):
                          computeDecompressGold(input2, num_elements, !(argc-1) || ((argv[argc-1] == string("q")) ? false : true)));

    delete [] input;

    return ret_val;
}
