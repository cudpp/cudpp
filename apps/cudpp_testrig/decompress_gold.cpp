// -------------------------------------------------------------
// CUDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision: $
// $Date: $
// -------------------------------------------------------------
// This source code is distributed under the terms of license.txt in
// the root directory of this source distribution.
// -------------------------------------------------------------

/** @file decompress_gold.cpp
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

/*
 *  Custom data objects
 */

// Enumerated type for the different kinds of Huffman tree nodes
enum huffman_node_type {root, internal, leaf};

// Data structure for a Huffman tree node
struct HuffmanNode {
    huffman_node_type type;    // Root, Internal, or Leaf
    HuffmanNode* parent;       // Pointer to parent node (NULL if node is root)
    HuffmanNode* left_child;   // Pointer to left child node (NULL if node is a leaf)
    HuffmanNode* right_child;  // Pointer to right child node (NULL if node is a leaf)
    int value;                 // Value (character frequency) represented by the node (NULL if internal or root)
};

// Data structure for a Huffman tree
struct HuffmanTree {
    HuffmanNode* nodes;  // Pointer to array of all nodes in the tree
    HuffmanNode* root;   // Pointer to root node
    int num_nodes;       // Number of nodes in the tree
};

/** @brief Compresses a file on the CPU to test decompression on the GPU
 *
 *  @param[in]  i_data
 *  @param[out] o_data
 *  @param[out] 
                           *        - input file (uncompressed)
                           *     Out:
                           *        - output file (compressed)
                           *        - final huffman tree object
                           *)
 */
int computeDecompressGold()
{
    /*  Steps:
     *     - Allocate memory
     *     - Run BWT
     *     - Run MTF transform
     *     - Run Huffman encoding
     */
}

/** @brief Run a Burrows-Wheeler Transform (BWT) for computeDecompressGold()
 *
 *
 *  @param[in]  i_data        Pointer to input data array
 *  @param[in]  num_elements  Length of input data array
 *  @param[out] o_data        Pointer to output data array
 */
int computeBWT(char* i_data, char* o_data, int num_elements)
{
    /*  Steps:
     *     - Allocate memory for all rotations
     *     - Calculate every rotation (loop 'array_len' times)
     *     - Sort rotations in lexigraphical order
     *     - Take last character from each rotation (in order) and add to
     *       output array in order
     */
}

/** @brief Run a Move-To-Front (MTF) Transform for computeDecompressGold()
 *
 *
 *  @param[in]  i_data              Pointer to input data array
 *  @param[in]  num_input_elements  Length of input data array
 *  @param[out] o_data              Pointer to output data array
 *  @param[out] MTF_list_length     Length of MTF character list
 */
int computeMTF(char* i_data, char* o_data, int num_input_elements, int* MTF_list_length)
{
    /*  Steps:
     *     - Generate list of characters in array and sort (loop through)
     *     - Loop through array, this time assigning a number to each
     *       character corresponding to that charater's position in the MTF
     *       list. Then move that character to the front of the MTF list
     *     - Store those numbers in order in an array
     */
}

/** @brief Build a huffman tree for computeDecompressGold()
 *
 *
 *  @param[in]  i_data        Pointer to input data array
 *  @param[in]  num_elements  Length of input data array
 *  @param[out] o_data        Pointer to output data array
 *  @param[out] tree          Pointer to final Huffman tree
 */
int computeHuffmanTree(char* i_data, char* o_data, int num_elements, HuffmanTree* tree)
{
    /*  Steps:
     *     - Loop through input array and calculate frequency of numbers.
     *       Assign to "huffman pairs" (key(number)-value(frequency) tuple)
     *     - Sort array by values (frequencies) from least to most
     *     - Copy array for huffman treew
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
     */
}






int main(int argc, char *argv[]) {return 0;}
