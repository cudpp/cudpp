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
    huffman_node_type type;    ///< Root, Internal, or Leaf
    HuffmanNode* parent;       ///< Pointer to parent node (NULL if node is root)
    HuffmanNode* left_child;   ///< Pointer to left child node (NULL if node is a leaf)
    HuffmanNode* right_child;  ///< Pointer to right child node (NULL if node is a leaf)
    int value;                 ///< Value (character frequency) represented by the node (NULL if internal or root)

    HuffmanNode() {  ///< Constructor that initializes pointers to NULL
        parent = NULL;
        left_child = NULL;
        right_child = NULL;
    }
};

/** @struct HuffmanTree
 *  @brief Data structure for a Huffman tree
 */
struct HuffmanTree {
    HuffmanNode* nodes[];  ///< Pointer to array of all nodes in the tree
    HuffmanNode* root;   ///< Pointer to root node
    int num_nodes;       ///< Number of nodes in the tree
};

/** @brief Run a Burrows-Wheeler Transform (BWT) for computeDecompressGold()
 *
 *  @param[in]  i_data        Pointer to input data array
 *  @param[in]  num_elements  Number of elements in the input array
 *  @param[out] o_data        Pointer to output data array
 *
 *  @return  Status. 0 = success, else = failure
 */
int computeBWT(char* i_data, char* o_data, size_t num_elements)
{
    vector<vector<char>> rotations(num_elements, vector<char> (num_elements));  // Allocate a 2D vector to store all input array rotations
    
    for (int i=0; i<num_elements; i++) {
        for (int j=0; j<num_elements; j++) {
            rotations[i][j] = i_data[(i+j) % num_elements];  /* Build rotations by iterating through the input array,
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
 *
 *  @return  Status. 0 = success, else = failure
 */
int computeMTF(char* i_data, int* o_data, size_t num_elements, vector<char>* MTF_list)
{
    bool found;  // Temporary boolean variable to determine if a character has already been discovered

    // Loop through the input array and build a list of unique characters
    for (int i=0; i<num_elements; i++){
        found = false;
        for (int j=0; j<(*MTF_list).size(); j++){
            if (i_data[i] == (*MTF_list)[j]) {  // If the character has already been discovered, set the flag and exit the loop
                found = true;
                break;
            }
        }
        if (!found) (*MTF_list).push_back(i_data[i]);  // If the character has not already been discovered, add it to the list
    }

    sort((*MTF_list).begin(), (*MTF_list).end());  // Sort MTF list (unique characters)
    string MTF((*MTF_list).begin(), (*MTF_list).end());  // Convert MTF list from vector to string (for searching functionality)
    int pos = 0;  // Temporary variable used to store the position of a character in the MTF list

    // Perform move-to-front transform
    for (int i=0; i<num_elements; i++) {
        pos = MTF.find(i_data[i]);  // Find input character in list
        o_data[i] = pos;  // Add input character position to output
        if (pos) {        // If input character is not at front of MTF list, move it to front of MTF list
            MTF.erase(pos, 1);
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
 *
 *  @return  Status. 0 = success, else = failure
 */
int computeHuffmanTree(char* i_data, char* o_data, size_t num_elements, HuffmanTree* tree)
{
    /*  Steps:
     *     - Loop through input array and calculate frequency of numbers. Assign to "huffman pairs" (key(number)-value(frequency) tuple)
     *     - Sort array by values (frequencies) from least to most
     *     - Copy array for huffman tree
     *     - Create new huffman tree object. Add lowest 2 items from huffman tree array to the tree. Remove those two items
     *       from huffman tree array. Add new item with value = sum of previous 2 item frequencies. Sort huffman tree array.
     *     - Repeat until only 1 item remains in huffman tree array. Add root to huffman tree.
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
     *          - Pointers to Parent, Left Child, Right Child (NULL if attribute does not exist)
     *          - (int) Value: Number that the node represents (frequency from output of MTF transform, NULL if internal or root node)
     */

// - 2D vector, storing tuples of (frequency of data, data (number from MTF transform))
// - Loop through input, calculating frequency of each number
// - Sort 2D array by frequency
// - Make Huffman nodes for the two lowest frequencies. Add the frequencies, remove them from the vector, and add a new vector item with the new sum
// - Add the two nodes to the Huffman tree (insert into nodes array, increment num_nodes)
// - Loop until there is only 1 item remaining in the vector, each time looking at the two lowest-valued (frequency) nodes
// - The remaining node is the root

    return 0;
}

/** @brief Compresses a file on the CPU using the bzip2 method.
 *
 *  The compressed file is then decompressed on the GPU and compared to the original file to verify CUDPP decompression functionality
 *
 *  @param[in]  input         Pointer to input data array
 *  @param[in]  num_elements  Length of input data array
 *  @param[in]  verbose       Optional input to print out intermediate data
 *
 *  @return  Status. 0 = success, else = failure
 */
int computeDecompressGold(char* input, size_t num_elements, bool verbose = false)
{
    /*  Steps:
     *     - Allocate memory
     *     - Run BWT
     *     - Run MTF transform
     *     - Run Huffman encoding
     */

    char* bwt_output = new char[num_elements];  // Pointer to char array that stores the output of the BWT operation
    int* mtf_output = new int[num_elements];  // Pointer to char array that stores the output of the MTF operation
    vector<char>* MTF_list = new vector<char>(0);  // Pointer to vector object that stores the list of unique characters
    int ret_val = 0;  // Variable to store return value (status)

    // ----- Print input array -----
    if (verbose) cout << "Number of Elements: " << num_elements << endl << endl;
    if (verbose) cout << "Input:       |" << input << "|" << endl;
    // -----------------------------

    // ----- Compute BWT -----
    if ((ret_val = (computeBWT(input, bwt_output, num_elements) == 0 ? 0 : 1)) != 0) {
        cout << "Error in BWT: " << ret_val << endl;
        delete [] bwt_output;
        delete [] mtf_output;
        delete MTF_list;
        return ret_val;
    }
    // -----------------------

    // ----- Print BWT output -----
    if (verbose) cout << "BWT Output:  |" << bwt_output << "|" << endl;

    // ----- Compute MTF transform -----
    if ((ret_val = (computeMTF(bwt_output, mtf_output, num_elements, MTF_list) == 0 ? 0 : 1)) != 0) {
        cout << "Error in MTF: " << ret_val << endl;
        delete [] bwt_output;
        delete [] mtf_output;
        delete MTF_list;
        return ret_val;
    }
    // ---------------------------------

    // ----- Print MTF output -----
    if (verbose) {
        cout << "MTF Output:  ";
        for (int i=0; i<num_elements; i++) { cout << mtf_output[i] << ","; }

        cout << endl << "MTF List:    |";
        for (int i=0; i<(*MTF_list).size(); i++) { cout << (*MTF_list)[i]; }
        cout << "|" << endl;
    }
    // ----------------------------

    cout << endl << "Return: " << ret_val << endl;

    delete [] bwt_output;
    delete [] mtf_output;
    delete MTF_list;

    return ret_val;
}
