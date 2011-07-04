#ifdef CUDPP_APP_COMMON_IMPL

namespace cudpp_app {
    
    bool checkCommandLineFlag(int argc, const char** argv, const char* name)
    {
        bool val = false;
        commandLineArg(val, argc, argv, name);
        return val;
    }

}

#endif