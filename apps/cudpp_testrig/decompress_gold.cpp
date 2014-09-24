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

#define NUM_CHARS 10

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

    HuffmanNode() {  ///< Default constructor that initializes pointers to NULL
        parent = NULL;
        left_child = NULL;
        right_child = NULL;
        data = -1;
        freq = 0;
        type = root;
        found = false;
    }

    HuffmanNode(int d, int f) {  ///< Constructor used when creating nodes to initialize data
        parent = NULL;
        left_child = NULL;
        right_child = NULL;
        data = d;
        freq = f;
        found = false;

        if (data == -1) type = root;
        else if (data < -1) type = huffman_node_type::internal;
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
    HuffmanNode* root;            ///< Pointer to root node
    vector<HuffmanNode*>* nodes;  ///< Pointer to a vector of all nodes in the tree

    HuffmanTree() {  ///< Default constructor that initializes the root pointer to NULL
        root = NULL;
        nodes = new vector<HuffmanNode*>();
    }

    ~HuffmanTree() { delete nodes; }  ///< Destructor
};

void myInsert(char** b, int init, int loc) {
    char* temp = b[init];
    for (int i=init; i>loc; i--) { b[i] = b[i-1]; }
    b[loc] = temp;
}

void mySort(char** a, size_t num_elements) {
    bool out;
    for (int i=1; i<num_elements; i++) {
        out = false;
        for (int j=0; j<i; j++) {
            for (int k=0; k<num_elements; k++) {
                if (a[i][k] < a[j][k]) {
                    myInsert(a, i, j);
                    out = true;
                    break;
                }
                if (a[i][k] > a[j][k]) {
                    break;
                }
            }
            if (out) break;
        }
    }
}

/** @brief Run a Burrows-Wheeler Transform (BWT) for computeDecompressGold()
 *
 *  @param[in]  i_data        Pointer to input data array
 *  @param[in]  num_elements  Number of elements in the input array
 *  @param[out] o_data        Pointer to output data array
 *
 *  @return  Status. 0 = success, else = failure
 */
int computeBWT(unsigned char* i_data, unsigned char* o_data, size_t num_elements)
{
    o_data[num_elements] = '\0';  // Null-terminate the output array

    vector<vector<unsigned char>> rotations(num_elements, vector<unsigned char> (NUM_CHARS + 1));  // Allocate a 2D vector to store all input array rotations
    
    for (int i=0; i<num_elements; i++) {
        for (int j=0; j<NUM_CHARS; j++) {
            rotations[i][j] = i_data[(i+j) % num_elements];  /* Build rotations by iterating through the input array,
                                                                looping back around to the beginning when reaching the end */
        }
        rotations[i].back() = i_data[(i + num_elements - 1) % num_elements];
    }
    sort(rotations.begin(), rotations.end());
    for (int i=0; i<num_elements; i++) {
        o_data[i] = rotations[i].back();  // Take the last character from each rotation and add it to the output array (in order)
    }


    if (o_data[num_elements] != '\0') return -1;  // If output array is not null-terminated, return an error
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
int computeMTF(unsigned char* i_data, unsigned char* o_data, size_t num_elements, vector<unsigned char>* MTF_list)
{
    o_data[num_elements] = '\0';  // Null-terminate the output array

    bool found;  // Temporary boolean variable to determine if a character has already been discovered
    typedef std::basic_string<unsigned char> ustring;

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
    ustring MTF(MTF_list->begin(), MTF_list->end());  // Convert MTF list from vector to string (for searching functionality)
    unsigned char pos;  // Temporary variable used to store the position of a character in the MTF list

    // Perform move-to-front transform
    for (int i=0; i<num_elements; i++) {
        pos = MTF.find(i_data[i]);  // Find input character in list
        o_data[i] = pos;  // Add input character position to output
        if (pos) {        // If input character is not at front of MTF list, move it to front of MTF list
            MTF.erase(pos, 1);
            MTF = i_data[i] + MTF;
        }
    }

    if (o_data[num_elements] != '\0') return -1; // If output array is not null-terminated, return an error
    else return 0;
}

/** @brief Build a huffman tree for computeDecompressGold()
 *
 *  @param[in]  i_data        Pointer to input data array
 *  @param[in]  num_elements  Length of input data array
 *  @param[out] o_data        Pointer to output data vector
 *  @param[out] tree          Pointer to final Huffman tree
 *
 *  @return  Status. 0 = success, else = failure
 */
int computeHuffmanTree(unsigned char* i_data, vector<bool>* o_data, size_t num_elements, HuffmanTree* tree)
{
    vector<pair<int, int>> frequencies;
    bool exists;      // Variable that stores whether a pair exists for a given MTF number
    int largest = 0;  // 

    for (int i=0; i<num_elements; i++) {  // Generates pairs of MTF numbers and their corresponding frequencies in the input list
        bool exists = false;
        if (i_data[i] > largest) largest = i_data[i];  // Finds the largest MTF number (to use in generating Huffman codes)

        for (int j=0; j<frequencies.size(); j++) {        // Loop through the MTF output, generating (frequency, number) pairs for each unique number
            if (i_data[i] == frequencies.at(j).second) {  // If a pair has already been created for a particular number
                exists = true;                            // Mark it as "existing"
                (frequencies.at(j).first)++;              // Increment the frequency of the number
                break;                                    // Exit the inner loop
            }
        }
        if (!exists) frequencies.push_back(make_pair(1, i_data[i]));  // If a pair does not exist for a particular number, create one
    }

    int int_node_count = -2;  // Temporary variable used to assign a unique value to all internal nodes

    while (frequencies.size() > 1) {
        sort(frequencies.begin(), frequencies.end());  // Sort the nodes every time the loop starts over
        int a_num = frequencies.at(0).second;          // Shortcut to access the number represented by the pair
        int a_freq = frequencies.at(0).first;          // Shortcut to access the frequency of the number represented by the pair
        int b_num = frequencies.at(1).second;          // --------- SAME AS ^^^ -----------
        int b_freq = frequencies.at(1).first;          // ---------------------------------

        HuffmanNode* a = new HuffmanNode(a_num, a_freq);  // Make a new HuffmanNode object for the two pairs at the beginning of the list
        HuffmanNode* b = new HuffmanNode(b_num, b_freq);  // ---------- ^^^ ----------

        if (a->type == huffman_node_type::internal) {     // If A is an internal node
            for (int i=0; i<tree->nodes->size(); i++) {   // Loop through all the existing nodes
                HuffmanNode* node1 = tree->nodes->at(i);  // Shortcut to utilize a particular node
                if (node1->parent == NULL) {              // If a particular node does not have a parent node
                    if (a->left_child == NULL) {          // If A does not have a left child
                        a->left_child = node1;            // Assign the node to be A's left child
                        node1->parent = a;                // Assign A to be the node's parent
                    }
                    else if (a->right_child == NULL) {    // Otherwise, if A has a left child, but no right child
                        a->right_child = node1;           // Assign the node to be A's right child
                        node1->parent = a;                // Assign A to be the node's parent
                    }
                    else break;
                }
            }
        }

        if (b->type == huffman_node_type::internal) {  // See notes from A ^^^
            for (int i=0; i<tree->nodes->size(); i++) {
                HuffmanNode* node = tree->nodes->at(i);
                if (node->parent == NULL) {
                    if (b->left_child == NULL) {
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
        }

        tree->nodes->push_back(a);  // Add A to the tree's list of nodes
        tree->nodes->push_back(b);  // Add B to the tree's list of nodes

        int temp = a_freq + b_freq;                                     // Calculate the sum of the frequencies of A and B
        frequencies.erase(frequencies.begin(), frequencies.begin()+2);  // Delete the 2 pairs with the lowest frequencies from the frequency list
        frequencies.push_back(make_pair(temp, int_node_count--));       // Add a new value to the frequency list equal to the sum of the previous two lowest frequencies
    }

// Root node stuff...
    tree->root = new HuffmanNode(-1, frequencies.at(0).first);  // Set the last remaining node as the root
    HuffmanNode* r = tree->root;                                // Make a pointer to the root node so it's attributes can be modified more easily
    r->left_child = tree->nodes->at(tree->nodes->size()-2);     // Set the root's left child
    r->right_child = tree->nodes->back();                       // Set the root's right child
    tree->nodes->at(tree->nodes->size()-2)->parent = r;         // Set the root's left child parent to the root
    tree->nodes->back()->parent = r;                            // Set the root's right child parent to the root
    tree->nodes->push_back(r);
// -----------------

    vector<bool> code;                        // Create a vector to store the Huffman code for an individual character
    vector<vector<bool>> codes(largest + 1);  // Create vector to store all Huffman codes
    bool done = false;                        // Temporary variable used when generating Huffman codes to determine if the coding process is complete

    while (!done) {                      // Loop until all characters have been coded
        code.clear();                    // Clear the individual code vector
        HuffmanNode* node = tree->root;  // Make a pointer to the node we are working on, starting with the root
        while (true) {
            if (!(node->left_child->found)) {  // If all the subnodes of the left child node have not been completely coded yet
                code.push_back(0);             // Add a 0 to the end of the code for this sequence
                if (node->left_child->type == leaf) {      // If the left child is a leaf node
                    node->left_child->found = true;        // Mark the left child as found
                    codes[node->left_child->data] = code;  // Add the code to the vector containing all the codes
                    break;                                 // Go on to the next node
                }
                else if (node->left_child->type == huffman_node_type::internal) {  // If the left child is an internal node
                    node = node->left_child;                                       // Make the left child the current node
                    continue;                                                      // Loop recursively
                }
                else return -5;  // Error code for the case where a node does not have a type (leaf, internal, or root)
            }
            else if (!(node->right_child->found)) {  // If all the subnodes of the right child node have not been completely coded yet
                code.push_back(1);                   // Add a 1 to the end of the code for this sequence
                if (node->right_child->type == leaf) {      // If the right child is a leaf node
                    node->right_child->found = true;        // Mark the right child as found
                    node->found = true;                     // Mark the node as completely coded
                    codes[node->right_child->data] = code;  // Add the code to the vector containing all the codes
                    break;                                  // Go on to the next node
                }
                else if (node->right_child->type == huffman_node_type::internal) {  // If the right child is an internal node
                    node = node->right_child;                                       // Make the right child the current node
                    continue;                                                       // Loop recursively
                }
                else return -5;  // Error code for the case where a node does not have a type (leaf, internal, or root)
            }
            else {                    // Otherwise, all nodes from this point have been completely coded
                node->found = true;   // Mark the node as found
                code.pop_back();      // Remove the last digit from the code
                if (node->type == root) {  // If this is the root node, we're done with coding
                    done = true;           // Set done as true
                    break;                 // Exit the loop
                }
                node = node->parent;  // Make the parent node the current node and loop around again
            }
        }
    }

    for (int i=0; i<num_elements; i++) {  // Loops through all characters and adds the codes to the output vector
        for (int j=0; j<codes[i_data[i]].size(); j++) {  // For every digit in the code
            o_data->push_back(codes[i_data[i]][j]);      // Add the digit to the output vector
        }
    }

    return 0;
}

/** @brief Compresses a file on the CPU using the bzip2 method.
 *
 *  The compressed file is then decompressed on the GPU and compared to the original file to verify CUDPP decompression functionality
 *
 *  @param[in]  input         Pointer to input data array
 *  @param[in]  num_elements  Length of input data array
 *  @param[in]  verbose       Optional input to print out intermediate data
 *  @param[out] output        Pointer to output data vector
 *
 *  @return  Status. 0 = success, else = failure
 */
int computeDecompressGold(unsigned char* input, vector<bool>* output, size_t num_elements, bool verbose = false)
{
    HuffmanTree* myTree = new HuffmanTree();
    vector<unsigned char>* MTF_list = new vector<unsigned char>();  // Pointer to vector object that stores the list of unique characters

    int ret_val = 0;  // Variable to store return value (status)

    // ----- Print input array -----
    if (verbose) cout << "Number of Elements: " << num_elements << endl << endl;
    if (verbose) {
        cout << "Input:         |";
        for (int i=0; i<num_elements; i++) {
            unsigned char temp = input[i];
            if (temp == '\t') cout << "\\t";
            else if (temp == 28) cout << "\\s";
            else if (temp == '\n') cout << "\\n";
            else if (temp == '\b') cout << "\\b";
            else if (temp == '\v') cout << "\\v";
            else if (temp == '\r') cout << "\\r";
            else if (temp == '\f') cout << "\\f";
            else if (temp == '\0') cout << "Ø";
            else cout << temp;
        }
        cout << "|" << endl;
    }
    // -----------------------------

    unsigned char* bwt_output = new unsigned char[num_elements+1];  // Pointer to char array that stores the output of the BWT operation
    // ----- Compute BWT -----
    if (ret_val = computeBWT(input, bwt_output, num_elements)) {
        cout << "Error in BWT: " << ret_val << endl;
        delete [] bwt_output;
        delete MTF_list;
        return ret_val;
    }
    // -----------------------

    // ----- Print BWT output -----
    if (verbose) {
        cout << "BWT Output:    |";
        for (int i=0; i<num_elements; i++) {
            unsigned char temp = bwt_output[i];
            if (temp == '\t') cout << "\\t";
            else if (temp == 28) cout << "\\s";
            else if (temp == '\n') cout << "\\n";
            else if (temp == '\b') cout << "\\b";
            else if (temp == '\v') cout << "\\v";
            else if (temp == '\r') cout << "\\r";
            else if (temp == '\f') cout << "\\f";
            else if (temp == '\0') cout << "Ø";
            else cout << temp;
        }
        cout << "|" << endl;
    }

    unsigned char* mtf_output = new unsigned char[num_elements+1];  // Pointer to char array that stores the output of the MTF operation
    // ----- Compute MTF transform -----
    if (ret_val = computeMTF(bwt_output, mtf_output, num_elements, MTF_list)) {
        cout << "Error in MTF: " << ret_val << endl;
        delete [] bwt_output;
        delete [] mtf_output;
        delete MTF_list;
        return ret_val;
    }
    // ---------------------------------
    delete [] bwt_output;

    // ----- Print MTF output -----
    if (verbose) {
        cout << "MTF Output:    |";
        for (int i=0; i<num_elements-1; i++) { cout << (int)mtf_output[i] << ","; }
        cout << (int)mtf_output[num_elements-1] << "|" << endl;

        cout << "MTF List:      |";
        for (int i=0; i<MTF_list->size(); i++) {
            unsigned char temp = (*MTF_list)[i];
            if (temp == '\t') cout << "\\t";
            else if (temp == 28) cout << "\\s";
            else if (temp == '\n') cout << "\\n";
            else if (temp == '\b') cout << "\\b";
            else if (temp == '\v') cout << "\\v";
            else if (temp == '\r') cout << "\\r";
            else if (temp == '\f') cout << "\\f";
            else if (temp == '\0') cout << "Ø";
            else cout << temp;
        }
        cout << "|" << endl;
    }
    // ----------------------------
    delete MTF_list;

    // ----- Build Huffman code -----
    if (ret_val = computeHuffmanTree(mtf_output, output, num_elements, myTree)) {
        delete [] mtf_output;
        delete myTree;
        return ret_val;
    }
    // ------------------------------
    delete [] mtf_output;

    // ----- Print Huffman code -----
    if (verbose) {
        cout << "Huffman code:  |";
        for (int i=0; i<output->size(); i++) { cout << output->at(i); }
        cout << "|" << endl;
    }
    // ------------------------------
    delete myTree;

    //cout << endl << "Return: " << ret_val << endl;

    return ret_val;
}
