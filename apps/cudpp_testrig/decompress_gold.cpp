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

    vector<vector<unsigned char>> rotations(num_elements, vector<unsigned char> (num_elements));  // Allocate a 2D vector to store all input array rotations
    
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
int computeMTF(unsigned char* i_data, int* o_data, size_t num_elements, vector<unsigned char>* MTF_list)
{
    o_data[num_elements] = '\0';  // Null-terminate the output array

    bool found;  // Temporary boolean variable to determine if a character has already been discovered
    typedef std::basic_string <unsigned char> ustring;

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
int computeHuffmanTree(int* i_data, vector<bool>* o_data, size_t num_elements, HuffmanTree* tree)
{
    vector<pair<int, int>> frequencies;
    bool exists;
    int largest = 0;

    for (int i=0; i<num_elements; i++) {  // Generates pairs of MTF numbers and their corresponding frequencies in the input list
        bool exists = false;
        if (i_data[i] > largest) largest = i_data[i];  // Finds the largest MTF number (to use in generating Huffman codes)

        for (int j=0; j<frequencies.size(); j++) {
            if (i_data[i] == frequencies.at(j).second) {
                exists = true;
                (frequencies.at(j).first)++;
                break;
            }
        }
        if (!exists) frequencies.push_back(make_pair(1, i_data[i]));
    }

    int int_node_count = -2;  // Temporary variable used to assign a unique value to all internal nodes
//    sort(frequencies.begin(), frequencies.end());  // Sort the nodes (this particular statement used only for cosmetic purposes when printing out the pairs)yy

//    for (int j=0; j<frequencies.size(); j++) {
//        cout << "{" << frequencies.at(j).first << ", " << frequencies.at(j).second << "}" << endl;
//    }

    while (frequencies.size() > 1) {
        sort(frequencies.begin(), frequencies.end());  // Sort the nodes every time the loop starts over
        int a_num = frequencies.at(0).second;  // Shortcut to access the number represented by the pair
        int a_freq = frequencies.at(0).first;  // Shortcut to access the frequency of the number represented by the pair
        int b_num = frequencies.at(1).second;  // --------- SAME AS ^^^ -----------
        int b_freq = frequencies.at(1).first;  // ---------------------------------
//cout << endl << "A: MTF number= " << a_num << ", frequency= " << a_freq << endl;
//cout << "B: MTF number= " << b_num << ", frequency= " << b_freq << endl;
        HuffmanNode* a = new HuffmanNode(a_num, a_freq);  // Make a new HuffmanNode object for the two pairs at the beginning of the list
        HuffmanNode* b = new HuffmanNode(b_num, b_freq);  // ---------- ^^^ ----------

        if (a->type == huffman_node_type::internal) {  // If A is an internal node
            for (int i=0; i<tree->nodes->size(); i++) {  // Loop through all the existing nodes
                HuffmanNode* node1 = tree->nodes->at(i);  // Shortcut to utilize a particular node
                if (node1->parent == NULL) {  // If a particular node does not have a parent node
                    if (a->left_child == NULL) {  // If A does not have a left child
//cout << "A left child:" << endl;
//cout << "\tOld Node Data (child):  " << node1->data << " \tFreq: " << node1->freq << " \tType: " << (node1->type==2 ? "Leaf" : (node1->type==1 ? "Internal" : "Root")) << endl;
//cout << "\tNew Node Data (parent): " << a->data << " \tFreq: " << a->freq << " \tType: " << (a->type==2 ? "Leaf" : (a->type==1 ? "Internal" : "Root")) << endl;
                        a->left_child = node1;  // Assign the node to be A's left child
                        node1->parent = a;  // Assign A to be the node's parent
                    }
                    else if (a->right_child == NULL) {  // Otherwise, if A has a left child, but no right child
//cout << "A right child:" << endl;
//cout << "\tOld Node Data (child):  " << node1->data << " \tFreq: " << node1->freq << " \tType: " << (node1->type==2 ? "Leaf" : (node1->type==1 ? "Internal" : "Root")) << endl;
//cout << "\tNew Node Data (parent): " << a->data << " \tFreq: " << a->freq << " \tType: " << (a->type==2 ? "Leaf" : (a->type==1 ? "Internal" : "Root")) << endl;
                        a->right_child = node1;  // Assign the node to be A's right child
                        node1->parent = a;  // Assign A to be the node's parent
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
//cout << "B left child:" << endl;
//cout << "\tOld Node Data (child):  " << node->data << " \tFreq: " << node->freq << " \tType: " << (node->type==2 ? "Leaf" : (node->type==1 ? "Internal" : "Root")) << endl;
//cout << "\tNew Node Data (parent): " << b->data << " \tFreq: " << b->freq << " \tType: " << (b->type==2 ? "Leaf" : (b->type==1 ? "Internal" : "Root")) << endl;
                        b->left_child = node;
                        node->parent = b;
                    }
                    else if (b->right_child == NULL) {
//cout << "B right child:" << endl;
//cout << "\tOld Node Data (child):  " << node->data << " \tFreq: " << node->freq << " \tType: " << (node->type==2 ? "Leaf" : (node->type==1 ? "Internal" : "Root")) << endl;
//cout << "\tNew Node Data (parent): " << b->data << " \tFreq: " << b->freq << " \tType: " << (b->type==2 ? "Leaf" : (b->type==1 ? "Internal" : "Root")) << endl;
                        b->right_child = node;
                        node->parent = b;
                    }
                    else break;
                }
            }
        }    // ---------------- END NOTES ------------------

//cout << "num nodes in tree->nodes: " << tree->nodes->size() << endl;
//cout << "num nodes in frequencies: " << frequencies.size() << endl;
        tree->nodes->push_back(a);  // Add A to the tree's list of nodes
        tree->nodes->push_back(b);  // Add B to the tree's list of nodes

        int temp = a_freq + b_freq;  // Calculate the sum of the frequencies of A and B
        frequencies.erase(frequencies.begin(), frequencies.begin()+2);  // Delete the 2 pairs with the lowest frequencies from the frequency list
        frequencies.push_back(make_pair(temp, int_node_count--));  // Add a new value to the frequency list equal to the sum of the previous two lowest frequencies
    }

// Root node stuff...
    tree->root = new HuffmanNode(-1, frequencies.at(0).first);
    HuffmanNode* r = tree->root;
    r->left_child = tree->nodes->at(tree->nodes->size()-2);
    r->right_child = tree->nodes->back();
    tree->nodes->at(tree->nodes->size()-2)->parent = r;
    tree->nodes->back()->parent = r;
    tree->nodes->push_back(r);
// -----------------

    vector<bool> code;
    vector<vector<bool>> codes(largest + 1);  // Create vector to store Huffman codes
    bool done = false;  // Temporary variable used when generating Huffman codes to determine if the coding process is complete

    while (!done) {
        code.clear();
        HuffmanNode* node = tree->root;
        while (true) {
            if (!(node->left_child->found)) {  // If all the subnodes of the left child node have not been completely coded yet
                code.push_back(0);  // Add a 0 to the end of the code for this sequence
                //codes[node->left_child->data].second.push_back(0);  // Add a 0 to the end of the code for this sequence
                if (node->left_child->type == leaf) {  // If the left child node is a leaf node
                    node->left_child->found = true;  // Mark the node as coded
                    codes[node->left_child->data] = code;
//cout << "Data: " << node->left_child->data << "   \tcode: ";
//for (int i=0; i<code.size(); i++) { cout << code[i]; }
//cout << endl;
                    break;  // Go on to the next node
                }
                else if (node->left_child->type == huffman_node_type::internal) {  // If the left child is an internal node
                    node = node->left_child;  // Make the left child the current node
                    continue;  // Loop recursively
                }
                else return -5;
            }
            else if (!(node->right_child->found)) {  // If all the subnodes of the right child node have not been completely coded yet
                code.push_back(1);  // Add a 1 to the end of the code for this sequence
                //codes[node->right_child->data].second.push_back(1);  // Add a 1 to the end of the code for this sequence
                if (node->right_child->type == leaf) {
                    node->right_child->found = true;
                    node->found = true;
                    codes[node->right_child->data] = code;
//cout << "Data: " << node->right_child->data << "   \tcode: ";
//for (int i=0; i<code.size(); i++) { cout << code[i]; }
//cout << endl;
                    break;
                }
                else if (node->right_child->type == huffman_node_type::internal) {
                    node = node->right_child;
                    continue;
                }
                else return -5;
            }
            else {
                node->found = true;
                code.pop_back();
                if (node->type == root) {
                    done = true;
                    break;
                }
                node = node->parent;
            }
        }
    }

    for (int i=0; i<num_elements; i++) {
        for (int j=0; j<codes[i_data[i]].size(); j++) {
            o_data->push_back(codes[i_data[i]][j]);
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
 *
 *  @return  Status. 0 = success, else = failure
 */
int computeDecompressGold(unsigned char* input, size_t num_elements, bool verbose = false)
{
    /*  Steps:
     *     - Allocate memory
     *     - Run BWT
     *     - Run MTF transform
     *     - Run Huffman encoding
     */

    unsigned char* bwt_output = new unsigned char[num_elements+1];  // Pointer to char array that stores the output of the BWT operation
    int* mtf_output = new int[num_elements+1];  // Pointer to char array that stores the output of the MTF operation

    HuffmanTree* myTree = new HuffmanTree();
    vector<unsigned char>* MTF_list = new vector<unsigned char>();  // Pointer to vector object that stores the list of unique characters
    vector<bool>* huffman_output = new vector<bool>();  // Pointer to vector object that stores the huffman encoded data

    int ret_val = 0;  // Variable to store return value (status)

    // ----- Print input array -----
    if (verbose) cout << "Number of Elements: " << num_elements << endl << endl;
    if (verbose) {
        cout << "Input:         |";
        for (int i=0; i<num_elements; i++) {
            char temp = input[i];
            if (temp == '\t') cout << "\\t";
            else if (temp == '\n') cout << "\\n";
            else if (temp == '\b') cout << "\\b";
            else if (temp == '\v') cout << "\\v";
            else if (temp == '\r') cout << "\\r";
            else if (temp == '\0') cout << "Ø";
            else cout << temp;
        }
        cout << "|" << endl;
    }
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
    if (verbose) {
        cout << "BWT Output:    |";
        for (int i=0; i<num_elements; i++) {
            /*if (iscntrl(bwt_output[i])) putchar (bwt_output[i]);
            else cout << bwt_output[i];*/

            char temp = bwt_output[i];
            if (temp == '\t') cout << "\\t";
            else if (temp == '\n') cout << "\\n";
            else if (temp == '\b') cout << "\\b";
            else if (temp == '\v') cout << "\\v";
            else if (temp == '\r') cout << "\\r";
            else if (temp == '\0') cout << "Ø";
            else cout << temp;
        }
        cout << "|" << endl;
    }

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
        cout << "MTF Output:    |";
        for (int i=0; i<num_elements-1; i++) { cout << mtf_output[i] << ","; }
        cout << mtf_output[num_elements-1] << "|" << endl;

        cout << "MTF List:      |";
        for (int i=0; i<MTF_list->size(); i++) {
            char temp = (*MTF_list)[i];
            if (temp == '\t') cout << "\\t";
            else if (temp == '\n') cout << "\\n";
            else if (temp == '\b') cout << "\\b";
            else if (temp == '\v') cout << "\\v";
            else if (temp == '\r') cout << "\\r";
            else if (temp == '\0') cout << "Ø";
            else cout << temp;
        }
        cout << "|" << endl;
    }
    // ----------------------------

    if (ret_val = computeHuffmanTree(mtf_output, huffman_output, num_elements, myTree)) { ; }

    if (verbose) {
// ----------- DEBUG COUT STATEMENTS -------------
/*        for (int i=0; i<myTree->nodes->size(); i++) {
            HuffmanNode* node = (myTree->nodes->at(i));
            cout << endl << "Node (value, frequency): {" << node->data << "," << node->freq << "}, Type: " << (node->type==2 ? "Leaf" : (node->type==1 ? "Internal" : "Root"))  << endl;
            (node->left_child==NULL ? (cout << "\tNo left child" << endl) : (cout << "\tLeft Child (value, frequency): {" << node->left_child->data << "," << node->left_child->freq << "}, Type: " << (node->left_child->type==2 ? "Leaf" : (node->left_child->type==1 ? "Internal" : "Root")) << endl));
            (node->right_child==NULL ? (cout << "\tNo right child" << endl) : (cout << "\tRight Child (value, frequency): {" << node->right_child->data << "," << node->right_child->freq << "}, Type: " << (node->right_child->type==2 ? "Leaf" : (node->right_child->type==1 ? "Internal" : "Root")) << endl));
            //cout << "\tRight Child (value, frequency): {" << node.right_child->data << "," << node.right_child->freq << "}, Type: " << node.right_child->type << endl;
        }*/
// -----------------------------------------------

        cout << "Huffman code:  |";
        for (int i=0; i<huffman_output->size(); i++) { cout << huffman_output->at(i); }
        cout << "|" << endl;
    }

    cout << endl << "Return: " << ret_val << endl;

    delete [] bwt_output;
    delete [] mtf_output;
    delete huffman_output;
    delete MTF_list;
    delete myTree;

    return ret_val;
}
