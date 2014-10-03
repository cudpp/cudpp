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

class myError : exception {
    public:
    string* msg;

    myError(string str) {
        msg = new string(str);
    }
};

/** @brief Process the input values for testDecompress()
 *
 *  Determine the source of input data, the location of output data, and whether to enable verbose mode
 *
 *  @param[in]  argc     Number of configuration options
 *  @param[in]  argv     Array storing configuration options
 *  @param[out] length   Length of the data input (number of characters)
 *  @param[out] input    Input data array (unsigned char)
 *  @param[out] name     Name of the output file
 *  @param[out] verbose  Enables verbose output
 */
void processInput(int argc, const char* argv[], int* length, unsigned char* input, char* name, bool* verbose)
{
    srand(time(NULL));  // Used to generate random characters every time the code is run

    bool found = false;                                                    // Stores whether a good input file has already been found
    strcpy((char*)input, "The quick brown fox jumps over the lazy dog.");  // Default input string if nothing else is specified
    *length = 44;                                                          // Default input string length
    strcpy(name, "default_compressed.txt");                                // Default output file name

    if (argc > 1) {  // If there are any command-line arguments, process them
        for (int i=1; i<argc; i++) {  // Loop through all the command-line arguments
            if (argv[i] == string("v")) *verbose = true;                             // "Verbose mode". Enables printing of output data
            else if (argv[i] == string("e")) throw myError("This is an error");      // Option to test try-catch functionality
            else if (argv[i] == string("f")) throw string("This is also an error");  // Another option to test try-catch funtionality
            else if (argv[i] == string("rand")) for (int j=0; j<(*length); j++) { input[j] = (rand() % 126) + 1; }  // Generates random ASCII characters as input
            else {  // If an option exists that isn't recognized, try and use it as a file name
                if (found) continue;           // If an input file has already been successfully scanned, process the other arguments
                ifstream input_file(argv[i]);  // Try and open the file
                if (!input_file.is_open()) throw string("Could not find any input files");  // If the file doesn't exist, go to the next argument
                                                                                            // If it's the last argument, throw an error

                input_file.seekg(0, ios::end);                     // Move the pointer to the end of the file
                streamsize size = (*length) = input_file.tellg();  // Return the position of the end of the file (size of file)
                input_file.seekg(0, ios::beg);                     // Return the pointer to the beginning of the file so the file can be read

                input = new unsigned char[(*length)+1];  // Initialize input array to size of file
                input_file.read((char*)input, size);     // Read file into input array
                found = true;                            // Mark that a good input file has been found

                name = new char[strlen(argv[i]) + 12];  // Store file name for use in generating an output file
                strncpy(name, argv[i], strlen(argv[i]) - 4);
                name[strlen(argv[i]) - 4] = '\0';
                strcat(name, "_compressed");
                strcat(name, ".txt");
	    }
	}
    }
}

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
int testDecompress(int argc, const char* argv[], CUDPPConfiguration* init_config)
{
    int ret_val = 0;            // Stores the return value
    int* length;                // Stores input array length. Changes if input comes from a file
    char* name = new char[23];  // Stores the name of an input file, if that's where the input is sourced from
    bool* verbose = false;      // Determines whether the program prints output data or not
    unsigned char* input = new unsigned char[*length];  // Input data array. Initialized for the default input string but is reinitialized if input comes from a different source
    vector<bool>* output = new vector<bool>();          // Output data vector. Stores output data in binary form.

    CUDPPConfiguration config;   // CUDPP configuration used to tell the CUDPP library to run decompression
    CUDPPHandle cudppLibrary;    // CUDPP handle for the CUDPP library
    CUDPPHandle decompressPlan;  // CUDPP handle for the decompress plan

    try
    {
// ------------------------- TO DO --------------------------

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

//  Allocate input memory on host and populate input data
//  Allocate temporary output memory on host
//  Calculate decompress gold on host and store in temporary output memory
//  Initialize CUDPP Plan
    if (cudppPlan(cudppLibrary, &decompressPlan, config, 1, 0) != CUDPP_SUCCESS) {  // Try and initialize the CUDPP decompress plan
        ret_val = 1;
        throw string("Error creating decompress plan");  // If there was a problem initializing the decompress plan, throw an error
    }

//  Allocate input memory on device
//  Allocate output memory on device
//  Allocate output memory on host
//  Copy temporary output data from host to device
//  Perform decompression on device
    computeDecompress();

//  Copy output data back from device
//  Free memory on device
//  Compare output from device to original input data
//  Free memory on host


    }
    catch (myError& ex) { cout << *ex.msg << endl; }  // Just some error handling...
    catch (string err) { cout << err << endl; }
    catch (...) { cout << "Generic error in testCompress\n"; }

    cudppDestroyPlan(decompressPlan);
    cudppDestroy(cudppLibrary);
    delete [] input;
    delete [] name;
    delete output;

    return ret_val;
}

/** @brief Run CUDPP decompress gold functionality
 *
 *  Runs the CUDPP decompress gold function by compressing data on the CPU.
 *
 *  @param[in] argc  Number of input arguments
 *  @param[in] argv  Pointer to char array storing input arguments
 *
 *  @return Status. 0 = passed, else = failed
 */
int testDecompressStandalone(int argc, const char* argv[])  // Rename to main() to run as standalone
{
    int ret_val = 0;            // Stores the return value
    int* length;                // Stores input array length. Changes if input comes from a file
    char* name = new char[23];  // Stores the name of an input file, if that's where the input is sourced from
    bool* verbose = false;      // Determines whether the program prints output data or not
    unsigned char* input = new unsigned char[*length];  // Input data array. Initialized for the default input string but is reinitialized if input comes from a different source
    vector<bool>* output = new vector<bool>();          // Output data vector. Stores output data in binary form.

    try
    {
        processInput(argc, argv, length, input, name, verbose);  // Process the input to determine configuration
	if (ret_val = computeDecompressGold(input, output, (size_t)(*length), *verbose)) throw string("Error computing decompressGold");  // Run the compression code in decompress_gold.cpp
        writeOutput(output, name);  // Write the output data out to a file
    }
    catch (myError& ex) { cout << *ex.msg << endl; }  // Just some error handling...
    catch (string err) { cout << err << endl; }
    catch (...) { cout << "Generic error in testCompress\n"; }

    delete [] input;
    delete [] name;
    delete output;

    return ret_val;
}
