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

/** @file decompress_gold.h
 *  @brief Contains struct and enum definitions for decompress_gold functions.
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
    int data;                  ///< Number (in MTF list) represented by the node (-1 if root, -2 or less if internal)
    int freq;                  ///< Frequency of node data (in MTF list)
    bool found;                ///< Variable used when generating Huffman codes to determine if a node has already been coded

    HuffmanNode();              ///< Default constructor that initializes pointers to NULL
    HuffmanNode(int d, int f);  ///< Constructor used when creating nodes to initialize data
    ~HuffmanNode();             ///< Destructor

    void swap();
};

/** @struct HuffmanTree
 *  @brief Data structure for a Huffman tree
 */
struct HuffmanTree {
    HuffmanNode* root;            ///< Pointer to root node
    vector<HuffmanNode*>* nodes;  ///< Pointer to a vector of all nodes in the tree
    HuffmanNode* nodes_array;     ///< Pointer to an array of all nodes in the tree

    HuffmanTree();   ///< Default constructor that initializes the root pointer to NULL
    ~HuffmanTree();  ///< Destructor
};

int computeDecompressGold(unsigned char* input, vector<bool>* output, size_t num_elements, bool verbose = false);
