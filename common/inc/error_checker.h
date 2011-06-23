/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation and 
 * any modifications thereto.  Any use, reproduction, disclosure, or distribution 
 * of this software and related documentation without an express license 
 * agreement from NVIDIA Corporation is strictly prohibited.
 * 
 */

/* CUda UTility Library */

#ifndef _ERRORCHECKER_H_
#define _ERRORCHECKER_H_

// includes, system
#include <string>
#include <sstream>

// includes, project
#include <exception.h>

// typedefs
//typedef unsigned int GLuint;

//! Class providing the handler / tester functions for errors as static members
class ErrorChecker 
{
public:
    //! Check if a condition is true.
    //! @note In prinicple has the same functionality as assert but allows 
    //!       much better control this version prints an error and terminates
    //!       the program, no exception is thrown.
    inline static void condition( bool val, const char* file, const int line);
};

// functions, inlined

// includes, system
#include <iostream>

////////////////////////////////////////////////////////////////////////////////
//! Check if a condition is true.
//! @note In prinicple has the same functionality as assert but allows much
//!       better control this version prints an error and terminates the 
//!       program, no exception is thrown.
////////////////////////////////////////////////////////////////////////////////
/* static */ inline void
ErrorChecker::condition( bool val, const char* file, const int line) 
{
    if ( ! val) 
    {
        std::ostringstream os;
        os << "Condition failed: " << file << " in line " << line;
        RUNTIME_EXCEPTION( os.str() );
    }
}

#endif // _ERRORCHECKER_H_

