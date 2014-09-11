// -------------------------------------------------------------
// CUDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision: $
// $Date: $
// -------------------------------------------------------------
// This source code is distributed under the terms of license.txt in
// the root directory of this source distribution.
// -------------------------------------------------------------

/*  Trevor Gibson, UC Davis Dept. of Electrical and Computer Engineering
 *  Date created: 8/20/14
 */

/** @file decompress_gold.cpp
 *  @brief Contains functions to compress a file on the CPU.
 *
 *  These functions are called as part of the test routines in test_decompress.cpp. Resulting compressed file is
 *  decompressed on the GPU to test the CUDPP decompression functionality.
 */

#include <iostream>
#include <string>
#include <cstring>
#include <algorithm>
#include <vector>

/** @namespace std */
using namespace std;

/** @enum huffman_node_type
 *  @brief Enumerated type for the different kinds of Huffman tree nodes
 */
enum huffman_node_type {root, internal, leaf};

/** @struct HuffmanNode
 *  @brief Data structure for a Huffman tree node
 */
struct HuffmanNode {
    huffman_node_type type;    // Root, Internal, or Leaf
    HuffmanNode* parent;       // Pointer to parent node (NULL if node is root)
    HuffmanNode* left_child;   // Pointer to left child node (NULL if node is a leaf)
    HuffmanNode* right_child;  // Pointer to right child node (NULL if node is a leaf)
    int value;                 // Value (character frequency) represented by the node (NULL if internal or root)
};

/** @struct HuffmanTree
 *  @brief Data structure for a Huffman tree
 */
struct HuffmanTree {
    HuffmanNode* nodes;  // Pointer to array of all nodes in the tree
    HuffmanNode* root;   // Pointer to root node
    int num_nodes;       // Number of nodes in the tree
};

/** @brief Run a Burrows-Wheeler Transform (BWT) for computeDecompressGold()
 *
 *  @param[in]  i_data        Pointer to input data array
 *  @param[in]  num_elements  Number of elements in the input array
 *  @param[out] o_data        Pointer to output data array
 */
int computeBWT(char* i_data, char* o_data, size_t num_elements)
{
    /*  Steps:
     *     - Allocate memory for all rotations
     *     - Calculate every rotation (loop 'array_len' times)
     *     - Sort rotations in lexigraphical order
     *     - Take last character from each rotation (in order) and add to
     *       output array in order
     */

    char* r = new char[num_elements];  // Allocate a temporary array to store one rotation of the input array
    string* rotations = new string[num_elements];  // Allocate an array of strings to store all input array rotations
    
    cout << endl;
    for (int i=0; i<num_elements; i++) {
        for (int j=0; j<num_elements; j++) {
            r[j] = i_data[(i+j) % num_elements];  /* Builds a rotation by iterating through the input array,
                                                    * looping back around to the beginning when reaching the end of the input array
                                                    */
        }

        string rot(r, num_elements);  // Convert the newly calcuated rotation from a char array into a string
        rotations[i] = rot;  // Store the string rotation for later use
	cout << rot << endl;
    }
    cout << endl;

    std::sort(rotations, rotations + num_elements);  // Sort all the rotations in lexigraphical order
    for (int i=0; i<num_elements; i++) {
        o_data[i] = rotations[i].back();  // Take the last character from each rotation and add it to the output array (in order)
	cout << rotations[i] << endl;
    }
    
    cout << endl << endl;
    if (o_data[num_elements-1] == 0) return -1;
    else return 0;
}

/** @brief Run a Move-To-Front (MTF) Transform for computeDecompressGold()
 *
 *  @param[in]  i_data        Pointer to input data array
 *  @param[in]  num_elements  Length of input data array
 *  @param[out] o_data        Pointer to output data array
 *  @param[out] MTF_list      MTF character list
 */
int computeMTF(char* i_data, int* o_data, int num_elements/*, string MTF_list*/)
{
    /*  Steps:
     *     - Generate list of characters in array and sort (loop through)
     *     - Loop through array, this time assigning a number to each
     *       character corresponding to that charater's position in the MTF
     *       list. Then move that character to the front of the MTF list
     *     - Store those numbers in order in an array
     */

    string MTF_list;
    bool found;  // Temporary boolean variable to determine if a character has already been discovered

    // Loop through the input array and build a list of unique characters
    for (int i=0; i<num_elements; i++){
        found = false;
        for (int j=0; j<MTF_list.size(); j++){
            if (i_data[i] == MTF_list[j]) {  // If the character has already been found, set the flag and exit the loop
                found = true;
                break;
            }
        }

        if (!found) MTF_list += i_data[i];  // If the character has not already been discovered, add it to the list
    }

    std::sort(MTF_list.begin(), MTF_list.end());  // Sort string of unique characters
    int position = 0;

    // Perform move-to-front transform
    for (int i=0; i<num_elements; i++) {
        position = MTF_list.find(i_data[i]);  // Find input character in list
        o_data[i] = position;  // Add input character position to output
        if (position) {  // If input character is not at front of MTF list, move it to front of MTF list
            MTF_list.erase(position, 1);
            MTF_list = i_data[i] + MTF_list;
        }
    }
    
    if (o_data[num_elements-1] == 0) return -1;
    else return 0;
}



















/** @brief Build a huffman tree for computeDecompressGold()
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
     */
return 0;}

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

int computeDecompressGold(char* input, size_t num_elements)
{
    /*  Steps:
     *     - Allocate memory
     *     - Run BWT
     *     - Run MTF transform
     *     - Run Huffman encoding
     */

    cout << "num_elements: " << num_elements << endl << endl;
    char* bwt_output = new char[num_elements];
    int* mtf_output = new int[num_elements];

    cout << "Input:       |" << input << "|" << endl;
    int ret_val = (computeBWT(input, bwt_output, num_elements) == 0 ? 0 : 1);
    cout << "BWT Output:  |" << bwt_output << "|" << endl;

    ret_val = (computeMTF(bwt_output, mtf_output, num_elements/*, *MTFList*/) == 0 ? 0 : 1);
    cout << "MTF Output: ";
    for (int i=0; i<num_elements; i++) { cout << mtf_output[i] << ","; }
    cout << endl;

    cout << endl << "ret_val: " << ret_val << endl;
    return ret_val;
}
