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
    bool verbose = true;
    unsigned char* input = new unsigned char[length+1];

    strcpy((char*)input, ("The quick brown fox jumps over the lazy dog."));

    if (argc > 1) {
        for (int i=1; i<argc; i++) {
            if (argv[i] == string("q"))verbose = false;
            else if (argv[i] == string("rand")) for (int j=0; j<length; j++) { input[j] = (rand() % 255) + 1; }
        }
    }

    input[length] = '\0';
    size_t num_elements = length;
    ret_val = computeDecompressGold(input, num_elements, verbose);

    delete [] input;

    return ret_val;
}
