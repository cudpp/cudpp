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
#include <fstream>
#include <ctime>
#include <cmath>
#include <string>
#include <exception>

#include "decompress_gold.cpp"
#include "cudpp.h"
#include "cuda_util.h"
#include "cuda_runtime_api.h"

class myError : exception {
    public:
    string* msg;

    myError(string str) {
        msg = new string(str);
    }
};

/** @brief Write the compressed data out to a file
 *
 *  @param[in] output  Vector of bits containing the compressed data
 *  @param[in] name    Name of the output file to create/overwrite
 */
void writeOutput(vector<bool>* output, char* name)
{
    ofstream output_file;                 // Output file object
    output_file.open(name, ios::binary);  // Opens the output file
    unsigned char temp;                   // Unsigned char object to store 8 bits of compressed data, since we can only write to a file 1 byte at a time
    for (int i=0; i<output->size(); i++) {    // Loop through the compressed data, bit by bit
        if (i%8 == 0) {                       // If we've reached the end of a byte
            if (i != 0) output_file << temp;  // Write the byte to the file (excluding the very first time since no data has been encoded in the byte yet)
            temp = 0;                         // Reset temp to 0 and start a new bype
        }
        temp += (unsigned char)(pow(2, (7-(i%8))) * output->at(i));  // Encode the binary data in an 8-bit number
    }

    output_file.close();
    if (output_file.is_open()) throw string("Error closing file: " + string(name));
}

/** @brief Test CUDPP decompress functionality
 *
 *  Tests the CUDPP decompress function by compressing data on the CPU, decompressing
 *  it on the GPU (using CUDPP), and comparing the results to the original data.
 *
 *  @param[in] argc  Number of input arguments
 *  @param[in] argv  Pointer to char array storing input arguments
 *
 *  @return Status. 0 = passed, else = failed
 */
int testDecompress(int argc, const char* argv[], const CUDPPConfiguration* init_config)
{
    int ret_val = 0;           // Stores the return value
    bool verbose = false;      // Determines whether the program prints output data or not
    bool failed = false;       // Stores whether the first try block has a failure and if so, prevents the second one from executing
    size_t num_elements = 44;  // Stores number of input array elements, initialized to number of input elements based on default input string. Changes if input comes from a file
    char* name = new char[23];                               // Default output file name. Changes if input comes from a file
    unsigned char* input = new unsigned char[num_elements];  // Input data array. Initialized for the default input string but is reinitialized if input comes from a different source
    vector<bool>* compressionOutput = new vector<bool>();    // Output data vector. Stores output data in binary form.
    HuffmanTreeArray* myTree = new HuffmanTreeArray();       // Tree object containing Huffman code for output data

    bool* h_compressionOutput;
    bool* d_compressionOutput;
    unsigned char* h_output;
    unsigned char* d_output;

    CUDPPConfiguration config;   // CUDPP configuration used to tell the CUDPP library to run decompression
    CUDPPHandle cudppLibrary;    // CUDPP handle for the CUDPP library
    CUDPPHandle decompressPlan;  // CUDPP handle for the decompress plan

    try {
//  Allocate input memory on host and populate input data
        srand(time(NULL));   // Used to generate random characters every time the code is run
        bool found = false;  // Stores whether an input source has already been found
        strcpy((char*)input, "The quick brown fox jumps over the lazy dog.");  // Default input string if nothing else is specified
        strcpy(name, "default_compressed.txt");                                // Default output file name if nothing else is specified

        if (argc > 1) {  // If there are any command-line arguments, process them
            for (int i=1; i<argc; i++) {  // Loop through all the command-line arguments
                if (argv[i] == string("v")) verbose = true;                              // "Verbose mode". Enables printing of output data
                else if (argv[i] == string("-decompress")) continue;
                else if (argv[i] == string("e")) throw myError("This is an error");      // Option to test try-catch functionality
                else if (argv[i] == string("f")) throw string("This is also an error");  // Another option to test try-catch funtionality
                else if (found) continue;   // If an input file has already been successfully scanned, process the other arguments
                else if (argv[i] == string("rand")) {
                    for (int j=0; j<num_elements; j++) { input[j] = (rand() % 126) + 1; }  // Generates random ASCII characters as input
                    found = true;                                                          // Mark that an input source has been found
                }
                else {  // If an option exists that isn't recognized, try and use it as a file name
                    cout << "\n\n" << argv[i] << "\n\n";
                    ifstream input_file(argv[i]);                                                       // Try and open the file
                    if (!input_file.is_open()) throw string("Unrecognized input: " + string(argv[i]));  // If the file doesn't exist, go to the next argument
                                                                                                        // If it's the last argument, throw an error

                    input_file.seekg(0, input_file.end);                  // Move the pointer to the end of the file
                    streamsize size = num_elements = input_file.tellg();  // Get the position of the end of the file (size of file)
                    input_file.seekg(0, input_file.beg);                  // Return the pointer to the beginning of the file so the file can be read

                    delete [] input;
                    input = new unsigned char[num_elements+1];  // Initialize input array to size of file plus 1 for null-termination
                    input_file.read((char*)input, size);        // Read file into input array
                    found = true;                               // Mark that an input source has been found

                    size_t name_length = strlen(argv[i]);
                    delete [] name;
                    name = new char[name_length + 12];
                    strncpy(name, argv[i], name_length - 4);
                    name[name_length - 4] = '\0';
                    strcat(name, "_compressed.txt");
                }
            }
        }
    }
    catch (myError& ex) { cout << *ex.msg << endl; failed = true; }  // Just some error handling...
    catch (string err) { cout << err << endl; failed = true; }
    catch (...) { cout << "Generic error in testDeompress\n"; failed = true; }

    try
    {
        if (failed) throw string("FAILED");
// ------------------------- TO DO --------------------------

//  Allocate input memory on device
        cudaMalloc((void**)&d_compressionOutput, num_elements * sizeof(bool));

//  Allocate output memory on device
        cudaMalloc((void**)&d_output, num_elements * sizeof(unsigned char));

//  Allocate temporary output memory on host
        h_compressionOutput = new bool[num_elements];

//  Allocate output memory on host
        h_output = new unsigned char[num_elements];

//  Initialize CUDPP  configuration
        if (init_config) config = *init_config;   // If no configuration is specified, initialize one
        else {
            config.algorithm = CUDPP_DECOMPRESS;  // Set algorithm to decompress
            config.options = 0;                   // Ensure no options are set
            config.datatype = CUDPP_UCHAR;        // Set data type to unsigned char
        }

//  Initialize CUDPP library
        if (cudppCreate(&cudppLibrary) != CUDPP_SUCCESS) {     // Try and initialize the CUDPP library
            ret_val = 1;
            throw string("Error initializing CUDPP library");  // If there was a problem initializing the library, throw an error
        }

//  Initialize CUDPP Plan
        if (cudppPlan(cudppLibrary, &decompressPlan, config, num_elements, 1, 0) != CUDPP_SUCCESS) {  // Try and initialize the CUDPP decompress plan
            ret_val = 1;
            throw string("Error creating decompress plan");  // If there was a problem initializing the decompress plan, throw an error
        }

//  Calculate decompress gold on host and store in temporary output memory
	if (ret_val = computeDecompressGold(input, compressionOutput, myTree, num_elements, verbose)) throw string("Error computing decompressGold");  // Run the compression code in decompress_gold.cpp
        for (int i=0; i<num_elements; i++) { h_compressionOutput[i] = compressionOutput->at(i); }

        writeOutput(compressionOutput, name);
    
//  Copy temporary output data from host to device
        cudaMemcpy(d_compressionOutput, h_compressionOutput, num_elements * sizeof(bool), cudaMemcpyHostToDevice);

//  Perform decompression on device
        if (cudppDecompress(decompressPlan, myTree, 0, 0) != CUDPP_SUCCESS) throw string("Error computing cudppDecompress");

//  Copy output data back from device
        cudaMemcpy(h_output, d_output, num_elements * sizeof(unsigned char), cudaMemcpyDeviceToHost);

//  Compare output from device to original input data

    }
    catch (string err) { cout << err << endl; }
    catch (...) { cout << "Generic error in testDeompress\n"; }

    cudppDestroyPlan(decompressPlan);
    cudppDestroy(cudppLibrary);

//  Free memory on host
    delete [] input;
    delete [] h_compressionOutput;
    delete [] h_output;
    delete compressionOutput;
    delete myTree;

//  Free memory on device
    cudaFree(d_compressionOutput);
    cudaFree(d_output);

    return ret_val;
}
