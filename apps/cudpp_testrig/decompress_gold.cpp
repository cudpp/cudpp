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
    int data;                  ///< Number (in MTF list) represented by the node (-1 if internal, -2 if root)
    int freq;                  ///< Frequency of node data (in MTF list)

    HuffmanNode() {  ///< Default constructor that initializes pointers to NULL
        parent = NULL;
        left_child = NULL;
        right_child = NULL;
        data = -2;
        freq = 0;
        type = root;
    }

    HuffmanNode(int d, int f) {  ///< Constructor used when creating nodes to initialize data
        parent = NULL;
        left_child = NULL;
        right_child = NULL;
        data = d;
        freq = f;

        if (data < -1) type = root;
        else if (data < 0) type = huffman_node_type::internal;
        else type = leaf;
    }

    ~HuffmanNode() {  ///< Destructor
        delete parent;
        delete left_child;
        delete right_child;
    }
};

/** @struct HuffmanTree
 *  @brief Data structure for a Huffman tree
 */
struct HuffmanTree {
    HuffmanNode* root;           ///< Pointer to root node
    vector<HuffmanNode>* nodes;  ///< Pointer to a vector of all nodes in the tree
    int num_nodes;               ///< Number of nodes in the tree

    HuffmanTree() {  ///< Default constructor that initializes the root pointer to NULL
        root = NULL;
    }
 
    HuffmanTree(size_t n) {  ///< Constructor used to initialize nodes array
        root = NULL;
        nodes = new vector<HuffmanNode>();
        num_nodes = n;
    }

    ~HuffmanTree() {
        delete nodes;
        delete root;
    }
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
        for (int j=0; j<MTF_list->size(); j++){
            if (i_data[i] == (*MTF_list)[j]) {  // If the character has already been discovered, set the flag and exit the loop
                found = true;
                break;
            }
        }
        if (!found) MTF_list->push_back(i_data[i]);  // If the character has not already been discovered, add it to the list
    }

    sort(MTF_list->begin(), MTF_list->end());  // Sort MTF list (unique characters)
    string MTF(MTF_list->begin(), MTF_list->end());  // Convert MTF list from vector to string (for searching functionality)
    int pos;  // Temporary variable used to store the position of a character in the MTF list

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
int computeHuffmanTree(int* i_data, int* o_data, size_t num_elements, HuffmanTree* tree)
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

    vector<pair<int, int>> frequencies;
    bool exists; cout << endl;

    for (int i=0; i<num_elements; i++) {  // Generates pairs of MTF numbers and their corresponding frequencies in the input list
        bool exists = false;
        for (int j=0; j<frequencies.size(); j++) {
            if (i_data[i] == frequencies.at(j).second) {
                exists = true;
                (frequencies.at(j).first)++;
                break;
            }
        }
        if (!exists) frequencies.push_back(make_pair(1, i_data[i]));
    }

    tree = new HuffmanTree(frequencies.size());  // Initialize the Huffman Tree
    sort(frequencies.begin(), frequencies.end());  // Sort the nodes (this particular statement used only for cosmetic purposes when printing out the pairs)yy

    for (int j=0; j<frequencies.size(); j++) {
        cout << "{" << frequencies.at(j).first << ", " << frequencies.at(j).second << "}" << endl;
    }

    while (frequencies.size() > 1) {
        sort(frequencies.begin(), frequencies.end());  // Sort the nodes every time the loop starts over
        int a_num = frequencies.at(0).second;  // Shortcut to access the number represented by the pair
        int a_freq = frequencies.at(0).first;  // Shortcut to access the frequency of the number represented by the pair
        int b_num = frequencies.at(1).second;  // --------- SAME AS ^^^ -----------
        int b_freq = frequencies.at(1).first;  // ---------------------------------
cout << endl << "A: MTF number= " << a_num << ", frequency= " << a_freq << endl;
cout << "B: MTF number= " << b_num << ", frequency= " << b_freq << endl;
        HuffmanNode* a = new HuffmanNode(a_num, a_freq);  // Make a new HuffmanNode object for the two pairs at the beginning of the list
        HuffmanNode* b = new HuffmanNode(b_num, a_freq);  // ---------- ^^^ ----------

        if (a->type == huffman_node_type::internal) {  // If A is an internal node
            for (int i=0; i<tree->nodes->size(); i++) {  // Loop through all the existing nodes
                HuffmanNode* node1 = &tree->nodes->at(i);  // Shortcut to utilize a particular node
                if (node1->parent == NULL) {  // If a particular node does not have a parent node
                    if (a->left_child == NULL) {  // If A does not have a left child
cout << "A:" << endl;
cout << "\tData: " << node->data << " \tFreq: " << node->freq << " \tType: " << node->type << endl;
cout << "\tData: " << a->data << " \tFreq: " << a->freq << " \tType: " << a->type << endl;
                        a->left_child = node;  // Assign the node to be A's left child
                        node->parent = a;  // Assign A to be the node's parent
                    }
                    else if (a->right_child == NULL) {  // Otherwise, if A has a left child, but no right child
                        a->right_child = node;  // Assign the node to be A's right child
                        node->parent = a;  // Assign A to be the node's parent
                    }
                    else break;
                }
            }
        }

        if (b->type == huffman_node_type::internal) {  // See notes from A ^^^
            for (int i=0; i<tree->nodes->size(); i++) {
                HuffmanNode* node = &tree->nodes->at(i);
                if (node->parent == NULL) {
                    if (b->left_child == NULL) {
cout << "B:" << endl;
cout << "\tData: " << node->data << " \tFreq: " << node->freq << " \tType: " << node->type << endl;
cout << "\tData: " << a->data << " \tFreq: " << a->freq << " \tType: " << a->type << endl;
                        b->left_child = node;
                        node->parent = b;
                    }
                    else if (b->right_child == NULL) {
                        b->right_child = node;
                        node->parent = b;
                    }
                    else break;
                }
            }
        }    // ---------------- END NOTES ------------------

cout << "num nodes in tree->nodes: " << tree->nodes->size() << endl;
cout << "num nodes in frequencies: " << frequencies.size() << endl;
        tree->nodes->push_back(*a);  // Add A to the tree's list of nodes
        tree->nodes->push_back(*b);  // Add B to the tree's list of nodes

        int temp = a_freq + b_freq;  // Calculate the sum of the frequencies of A and B
        frequencies.erase(frequencies.begin(), frequencies.begin()+2);  // Delete the 2 pairs with the lowest frequencies from the frequency list
        frequencies.push_back(make_pair(temp, -1));  // Add a new value to the frequency list equal to the sum of the previous two lowest frequencies
    }

// Root node stuff...
    HuffmanNode* r = new HuffmanNode(-2, frequencies.at(0).first);
    tree->root = r;
    r->left_child = &tree->nodes->at(tree->nodes->size()-2);
    r->left_child = &tree->nodes->at(tree->nodes->size()-1);
    tree->nodes->at(tree->nodes->size()-2).parent = r;
    tree->nodes->at(tree->nodes->size()-1).parent = r;
// -----------------

/*  NOTES -----------------------
 *
 *  - This doesn't work
 *  - I suspect this is because it isn't quite fully recursive
 *  - In other words... I have yet to see it link an internal node to another internal node
 *  - Program just seg faults when trying to add a (seemingly random, although consistently the same) leaf node 
 *
 */
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

    int* huffman_output = new int[num_elements];  // Pointer to char array that stores the output of the Huffman operation
    HuffmanTree* myTree;

    vector<char>* MTF_list = new vector<char>();  // Pointer to vector object that stores the list of unique characters
    int ret_val = 0;  // Variable to store return value (status)

    // ----- Print input array -----
    if (verbose) cout << "Number of Elements: " << num_elements << endl << endl;
    if (verbose) cout << "Input:       |" << input << "|" << endl;
    // -----------------------------

    // ----- Compute BWT -----
    if (ret_val = computeBWT(input, bwt_output, num_elements)) {
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
    if (ret_val = computeMTF(bwt_output, mtf_output, num_elements, MTF_list)) {
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

    if (ret_val = computeHuffmanTree(mtf_output, huffman_output, num_elements, myTree)) { ; }

    cout << endl << "Return: " << ret_val << endl;

    delete [] bwt_output;
    delete [] mtf_output;
    delete [] huffman_output;
    delete MTF_list;
    delete myTree;

    return ret_val;
}
