/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */
 
/* CUda UTility Library */

#ifndef _COMMAND_LINE_H_
#define _COMMAND_LINE_H_

#include <string>
#include <sstream>

namespace cudpp_app {

    ////////////////////////////////////////////////////////////////////////////////
    //! Conversion function for command line argument arrays
    //! @note This function is used each type for which no template specialization
    //!  exist (which will cause errors if the type does not fulfill the std::vector
    //!  interface).
    ////////////////////////////////////////////////////////////////////////////////
    template<class T>
    inline T convertTo( const std::string& element)
    {
        return (T)false;
    }

    ////////////////////////////////////////////////////////////////////////////////
    //! Conversion function for command line arguments of type int
    ////////////////////////////////////////////////////////////////////////////////
    template<>
    inline int convertTo<int>( const std::string& element) 
    {
        std::istringstream ios( element);
        int val;
        ios >> val;
        return val;
    }

    ////////////////////////////////////////////////////////////////////////////////
    //! Conversion function for command line arguments of type float
    ////////////////////////////////////////////////////////////////////////////////
    template<>
    inline float convertTo<float>( const std::string& element) 
    {
        std::istringstream ios( element);
        float val;
        ios >> val;
        return val;
    }

    ////////////////////////////////////////////////////////////////////////////////
    //! Conversion function for command line arguments of type double
    ////////////////////////////////////////////////////////////////////////////////
    template<>
    inline double convertTo<double>( const std::string& element) 
    {
        std::istringstream ios( element);
        double val;
        ios >> val;
        return val;
    }

    ////////////////////////////////////////////////////////////////////////////////
    //! Conversion function for command line arguments of type string
    ////////////////////////////////////////////////////////////////////////////////
    template<>
    inline std::string convertTo<std::string>( const std::string& element)
    {
        return element;
    }

    ////////////////////////////////////////////////////////////////////////////////
    //! Conversion function for command line arguments of type bool
    ////////////////////////////////////////////////////////////////////////////////
    template<>
    inline bool convertTo<bool>( const std::string& element) 
    {
        // check if value is given as string-type { true | false }
        if ( "true" == element) 
            return true;
        else if ( "false" == element) 
            return false;
        // check if argument is given as integer { 0 | 1 }
        else 
        {
            int tmp = convertTo<int>( element ); 
            return ( 1 == tmp);
        }

        return false;
    }

    template <typename T>
    bool commandLineArg(T& val, int argc, const char**argv, const char* name)
    {
        for( int i=1; i<argc; ++i) 
        {
            std::string arg = argv[i];
            size_t pos = arg.find(name);
            if (pos != std::string::npos && pos == 1 && arg[0] == '-') {
                std::string::size_type pos;
                // check if only flag or if a value is given
                if ( (pos = arg.find( '=')) == std::string::npos) 
                {  
                    val = convertTo<T>("true");                                  
                    return true;
                }
                else 
                {
                    val = convertTo<T>(std::string( arg, pos+1, arg.length()-1));
                    return true;
                }
            }
        }
        return false;
    }

    bool checkCommandLineFlag(int argc, const char** argv, const char* name);
}

#ifdef CUDPP_APP_COMMON_IMPL
#include "commandline.inl"
#endif

#endif // #ifndef _COMMAND_LINE_H_

