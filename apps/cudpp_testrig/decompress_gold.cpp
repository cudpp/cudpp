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

    vector<char> r(num_elements);  // Allocate a temporary array to store one rotation of the input array
    vector<vector<char>> rotations(num_elements, vector<char> (num_elements));  // Allocate an array of strings to store all input array rotations
    
    for (int i=0; i<num_elements; i++) {
        for (int j=0; j<num_elements; j++) {
            rotations[i][j] = i_data[(i+j) % num_elements];  /* Builds rotations by iterating through the input array,
                                                                looping back around to the beginning when reaching the end */
        }
    }

    sort(rotations.begin(), rotations.end());  // Sort all the rotations in lexigraphical order
    for (int i=0; i<num_elements; i++) {
        o_data[i] = rotations[i][num_elements-1];  // Take the last character from each rotation and add it to the output array (in order)
    }
    
    if (o_data[num_elements-1] == 0) return -1;  // If last character is bad, return an error
    else return 0;
}

/** @brief Run a Move-To-Front (MTF) Transform for computeDecompressGold()
 *
 *  @param[in]  i_data        Pointer to input data array
 *  @param[in]  num_elements  Length of input data array
 *  @param[out] o_data        Pointer to output data array
 *  @param[out] MTF_list      MTF character list
 */
int computeMTF(char* i_data, int* o_data, size_t num_elements, vector<char>* MTF_list)
{
    //string MTF_list; // String object used to store the unique characters in the input array
    bool found;  // Temporary boolean variable to determine if a character has already been discovered

    // Loop through the input array and build a list of unique characters
    for (int i=0; i<num_elements; i++){
        found = false;
        for (int j=0; j<(*MTF_list).size(); j++){
            if (i_data[i] == (*MTF_list)[j]) {  // If the character has already been found, set the flag and exit the loop
                found = true;
                break;
            }
        }
        if (!found) (*MTF_list).push_back(i_data[i]);  // If the character has not already been discovered, add it to the list
    }

    sort((*MTF_list).begin(), (*MTF_list).end());  // Sort MTF list (unique characters)
    string MTF((*MTF_list).begin(), (*MTF_list).end());
    int position = 0;  // Variable used to store the position of a character in the MTF list

    // Perform move-to-front transform
    for (int i=0; i<num_elements; i++) {
        position = MTF.find(i_data[i]);  // Find input character in list
        o_data[i] = position;  // Add input character position to output
        if (position) {  // If input character is not at front of MTF list, move it to front of MTF list
            MTF.erase(position, 1);
            MTF = i_data[i] + MTF;
        }
    }
    
    if (o_data[num_elements-1] == 0) return -1; // If last character is bad, return an error
    else return 0;
}



















/** @brief Build a huffman tree for computeDecompressGold()
 *
 *  @param[in]  i_data        Pointer to input data array
 *  @param[in]  num_elements  Length of input data array
 *  @param[out] o_data        Pointer to output data array
 *  @param[out] tree          Pointer to final Huffman tree
 */
int computeHuffmanTree(char* i_data, char* o_data, size_t num_elements, HuffmanTree* tree)
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
    vector<char>* MTF_list = new vector<char>(0);

    cout << "Input:       |" << input << "|" << endl;
    int ret_val = 0;
    if ((ret_val = (computeBWT(input, bwt_output, num_elements) == 0 ? 0 : 1)) != 0) {
        cout << "Error in BWT: " << ret_val << endl;
        return ret_val;
    }
    cout << "BWT Output:  |" << bwt_output << "|" << endl;

    if ((ret_val = (computeMTF(bwt_output, mtf_output, num_elements, MTF_list) == 0 ? 0 : 1)) != 0) {
        cout << "Error in MTF: " << ret_val << endl;
        return ret_val;
    }
    cout << "MTF Output:  ";
    for (int i=0; i<num_elements; i++) { cout << mtf_output[i] << ","; }

    cout << endl << "MTF List:    |";
    for (int i=0; i<(*MTF_list).size(); i++) { cout << (*MTF_list)[i]; }
    cout << "|" << endl;

    cout << endl << "ret_val: " << ret_val << endl;
    return ret_val;
}
