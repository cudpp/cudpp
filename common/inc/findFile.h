// -------------------------------------------------------------
// cuDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
//  $Source$
//  $Revision: 3632 $
//  $Date: 2007-08-26 06:15:39 +0100 (Sun, 26 Aug 2007) $
// ------------------------------------------------------------- 
// This source code is distributed under the terms of license.txt in
// the root directory of this source distribution.
// ------------------------------------------------------------- 

/**
 * @file
 * tools.h
 * 
 * @brief Various utilities that are used in cudpp_testrig
 * 
 * Right now only Directory / File searching functions are inside this 
 * header file.  In the future useful tools across multiple tests will be 
 * added to this file
 */

#ifndef _TOOLS_H_
#define _TOOLS_H_

//Dir/File searching functions
//wrapper functions for the whole routine
extern "C"
{
int findDir(const char * startDir, const char * dirName, char * outputPath);
int findFile(const char * startDir, const char * dirName, char * outputPath);
}

#endif  //#ifndef _TOOLS_H_

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
