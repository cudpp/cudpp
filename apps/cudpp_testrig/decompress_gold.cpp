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
                           *        - final huffman tree object
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
     *     - Take last character from each rotation (in order) and add to
     *       output array in order
     */
}

// Run a Move-To-Front (MTF) Transform for computeDecompressGold()
int computeMTF(/*  Params:
                *     In: - input data array from BWT (i_data: original input
                *           characters in random order)
                *         - length of input data array (array_len)
                *     Out: 
                *         - MTF transformed data (o_data: list of numbers)
                */)
{
    /*  Steps:
     *     - Generate list of characters in array and sort (loop through)
     *     - Loop through array, this time assigning a number to each
     *       character corresponding to that charater's position in the MTF
     *       list. Then move that character to the front of the MTF list
     *     - Store those numbers in order in an array
     */
}

// Build a huffman tree for computeDecompressGold()
int computeHuffmanTree(/*  Params:
                        *     In: - input data array from MTF (i_data: array of
                        *           numbers)
                        *         - length of input data array (array_len)
                        *     Out: 
                        *         - huffman coded data (o_data: list of binary
                        *           numbers)
                        *         - huffman tree (huffman_tree: huffman tree
                        *           object)
                        */)
{
    /*  Steps:
     *     - Loop through input array and calculate frequency of numbers.
     *       Assign to "huffman pairs" (key(number)-value(frequency) tuple)
     *     - Sort array by values (frequencies) from least to most
     *     - Copy array for huffman tree
     *     - Create new huffman tree object. Add lowest 2 items from huffman
     *       tree array to the tree. Remove those two items from huffman tree
     *       array. Add new item with value = sum of previous 2 item
     *       frequencies. Sort huffman tree array.
     *     - Repeat until only 1 item remains in huffman tree array. Add root
     *       to huffman tree.
     */

    /*  Data required:
     *     - HuffmanTree object (contains nodes of "huffman pairs")
     *       Attributes:
     *          - Array of pointers to nodes
     *          - Pointer to root node
     *          - (int) Number of nodes
     *     - HuffmanNode object
     *       Attributes:
     *          - (enum) Type: Root, Internal, or Leaf
     *          - Pointers to Parent, Left Child, Right Child (NULL if
     *            attribute does not exist)
     *          - (int) Value: Number that the node represents (frequency from
     *            output of MTF transform, NULL if internal or root node)
}
