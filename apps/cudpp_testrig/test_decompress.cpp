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

int main(int argc, char *argv[])
{
    char input[] = "The quick brown fox jumps over the lazy dog.";
    size_t num_elements = sizeof(input) - 1;
    return computeDecompressGold(input, num_elements);
}
