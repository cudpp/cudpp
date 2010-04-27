################################################################################
#
# Common build script
#
################################################################################

.SUFFIXES : .cu .cu_dbg_o .c_dbg_o .cpp_dbg_o .cu_rel_o .c_rel_o .cpp_rel_o .cubin

CUDA_VERSION := $(shell nvcc --version | sed -n -e '/release/!d' -e 's/Cuda compilation tools, release \([0-9].[0-9]\), V[0-9].[0-9]*.[0-9]*/\1/p')

threeplus := 3.0
CUDA_VERSION_3PLUS := $(filter $(threeplus), $(firstword $(sort $(CUDA_VERSION) $(threeplus))))

CUDA_INSTALL_PATH ?= /usr/local/cuda

ifdef cuda-install
	CUDA_INSTALL_PATH := $(cuda-install)
endif

# detect OS
OSUPPER = $(shell uname -s 2>/dev/null | tr [:lower:] [:upper:])
OSLOWER = $(shell uname -s 2>/dev/null | tr [:upper:] [:lower:])

# 'linux' is output for Linux system, 'darwin' for OS X
DARWIN = $(strip $(findstring DARWIN, $(OSUPPER)))
ifneq ($(DARWIN),)
   SNOWLEOPARD = $(strip $(findstring 10.6, $(shell egrep "<string>10\.6" /System/Library/CoreServices/SystemVersion.plist)))
endif

# detect 32-bit or 64-bit platform
HP_64 = $(shell uname -m | grep 64)
OSARCH= $(shell uname -m)

# Basic directory setup for SDK
# (override directories only if they are not already defined)
SRCDIR     ?= 
CUSRCDIR   ?= 
ROOTDIR    ?= ..
ROOTBINDIR ?= $(ROOTDIR)/../bin
BINDIR     ?= $(ROOTBINDIR)/$(OSLOWER)
ROOTOBJDIR ?= obj
LIBDIR     := $(ROOTDIR)/../lib
COMMONDIR  := $(ROOTDIR)/../common
MACPORTSDIR:= /opt/local

# Compilers
NVCC       := nvcc 
CXX        := g++
CC         := gcc
LINK       := g++ -fPIC
ifneq ($(DARWIN),)
        # Additional command line param for declaring link path
        # http://www.cocoadev.com/index.pl?ApplicationLinking
	LINK +=  -Wl,-rpath,$(CUDA_INSTALL_PATH)/lib
endif

# Includes
INCLUDES  += -I. -I$(CUDA_INSTALL_PATH)/include -I$(COMMONDIR)/inc
ifneq ($(DARWIN),)
        # Link against macports install (currently for Boost)
	INCLUDES += -I$(MACPORTSDIR)/include
endif

# standard
ifneq ($(DARWIN),)
        LIB      += -L$(CUDA_INSTALL_PATH)/lib -L$(LIBDIR) -L$(COMMONDIR)/lib -L$(COMMONDIR)/lib/darwin -lcuda -lcudart ${OPENGLLIB}
else
        LIB      += -L$(CUDA_INSTALL_PATH)/lib -L$(LIBDIR) -L$(COMMONDIR)/lib -L$(COMMONDIR)/lib/linux  -lcuda -lcudart ${OPENGLLIB}
endif

# There's no standard installation place for glut on Mac. Default I'm
# using is the macports install.
ifneq ($(DARWIN),)
	DARWIN_GLUT_LIB ?= $(MACPORTSDIR)/lib
	LIB  += -L$(DARWIN_GLUT_LIB)
endif

# Warning flags
CXXWARN_FLAGS := \
	-W -Wall \
	-Wimplicit \
	-Wswitch \
	-Wformat \
	-Wchar-subscripts \
	-Wparentheses \
	-Wmultichar \
	-Wtrigraphs \
	-Wpointer-arith \
	-Wcast-align \
	-Wreturn-type \
	-Wno-unused-function \
	$(SPACE)

CWARN_FLAGS := $(CXXWARN_FLAGS) \
	-Wstrict-prototypes \
	-Wmissing-prototypes \
	-Wmissing-declarations \
	-Wnested-externs \
	-Wmain \

# Compiler-specific flags
# for CUDA 2.0, need this to compile on all platforms
#NVCCFLAGS := --host-compilation=C

CXXFLAGS  := $(CXXWARN_FLAGS)
CFLAGS    := $(CWARN_FLAGS)
LIB_ARCH  := $(OSARCH)

# Determining the necessary Cross-Compilation Flags
# 32-bit OS, but we target 64-bit cross compilation
ifeq ($(x86_64),1)
    NVCCFLAGS += -m64
    LIB_ARCH = x86_64

    ifneq ($(DARWIN),)
         CXX_ARCH_FLAGS += -arch x86_64
    else
         CXX_ARCH_FLAGS += -m64
    endif
else
# 64-bit OS, and we target 32-bit cross compilation
    ifeq ($(i386),1)
        NVCCFLAGS += -m32
        LIB_ARCH = i386
        ifneq ($(DARWIN),)
             CXX_ARCH_FLAGS += -arch i386
        else
             CXX_ARCH_FLAGS += -m32
        endif
    else
        ifneq ($(SNOWLEOPARD),)
             NVCCFLAGS += -m32
             CXX_ARCH_FLAGS += -arch i386 -m32
             LIB_ARCH  = i386
        endif
    endif
endif

# Compiler-specific flags
CXXFLAGS  := $(CXXWARN_FLAGS) $(CXX_ARCH_FLAGS)
CFLAGS    := $(CWARN_FLAGS) $(CXX_ARCH_FLAGS)
LINK      += $(CXX_ARCH_FLAGS)


# Common flags
COMMONFLAGS += $(INCLUDES) -DUNIX

# Debug/release configuration
ifeq ($(dbg),1)
	COMMONFLAGS += -g
	NVCCFLAGS   += -D_DEBUG
        CXXFLAGS    += -D_DEBUG
        CFLAGS      += -D_DEBUG
	BINSUBDIR   := debug
	LIBSUFFIX   := D
else 
	COMMONFLAGS += -O2 
	BINSUBDIR   := release
	LIBSUFFIX   :=
	NVCCFLAGS   += --compiler-options -fno-strict-aliasing
	CXXFLAGS    += -fno-strict-aliasing
	CFLAGS      += -fno-strict-aliasing
endif

ifeq ($(boost),1)
	COMMONFLAGS += -D_USE_BOOST_
endif

# append optional arch/SM version flags (such as -arch sm_11)
NVCCFLAGS += $(SMVERSIONFLAGS) 
NVCCFLAGS += -gencode=arch=compute_10,code=sm_10 

ifneq ($(CUDA_VERSION_3PLUS),)
	NVCCFLAGS += -gencode=arch=compute_20,code=sm_20 
endif


# architecture flag for cubin build
CUBIN_ARCH_FLAG := -m32

# OpenGL is used or not (if it is used, then it is necessary to include GLEW)
ifneq ($(DARWIN),)
	OPENGLLIB := -framework OpenGL
else
	OPENGLLIB := -lGL -lGLU
endif

# OpenGL is used or not (if it is used, then it is necessary to include GLEW)
ifeq ($(USEGLLIB),1)
	ifneq ($(DARWIN),)
		OPENGLLIB := -L/System/Library/Frameworks/OpenGL.framework/Libraries -lGL -lGLU -lGLEW
	else
		OPENGLLIB := -lGL -lGLU -lX11 -lXi -lXmu

		ifeq "$(strip $(HP_64))" ""
			OPENGLLIB += -lGLEW -L/usr/X11R6/lib
		else
			OPENGLLIB += -lGLEW_x86_64 -L/usr/X11R6/lib64
		endif
	endif

	CUBIN_ARCH_FLAG := -m64
endif

ifeq ($(USEGLUT),1)
	ifneq ($(DARWIN),)
		OPENGLLIB += -framework GLUT
	else
		OPENGLLIB += -lglut
	endif
endif

ifeq ($(USEPARAMGL),1)
	PARAMGLLIB := -lparamgl_$(LIB_ARCH)$(LIBSUFFIX)
endif

ifeq ($(USECUDPP), 1)
	# detect if 32 bit or 64 bit system
	HP_64 =	$(shell uname -m | grep 64)

	CUDPPLIB := -lcudpp_$(LIB_ARCH)$(LIBSUFFIX)

	ifeq ($(emu), 1)
		CUDPPLIB := $(CUDPPLIB)_emu
	endif
endif

# Libs
LIB       := -L$(CUDA_INSTALL_PATH)/lib -L$(LIBDIR) -L$(COMMONDIR)/lib -lcuda -lcudart ${OPENGLLIB} $(PARAMGLLIB) $(CUDPPLIB) ${LIB}


# Lib/exe configuration
ifneq ($(STATIC_LIB),)
	ifeq ($(emu), 1)
		NVCCFLAGS   += -deviceemu
		CUDACCFLAGS +=
		BINSUBDIR   := emu$(BINSUBDIR)
		LIBSUFFIX   := _$(LIB_ARCH)$(LIBSUFFIX)_emu
		# consistency, makes developing easier
		CXXFLAGS    += -D__DEVICE_EMULATION__
		CFLAGS	    += -D__DEVICE_EMULATION__		
	endif
	TARGETDIR := $(LIBDIR)
	TARGET   := $(subst .a,$(LIBSUFFIX).a,$(LIBDIR)/$(STATIC_LIB))
	LINKLINE  = ar qv $(TARGET) $(OBJS); ranlib $(TARGET)
else
	LIB += -lcutil$(LIBSUFFIX)
	# Device emulation configuration
	ifeq ($(emu), 1)
		NVCCFLAGS   += -deviceemu
		CUDACCFLAGS += 
		BINSUBDIR   := emu$(BINSUBDIR)
		# consistency, makes developing easier
		CXXFLAGS    += -D__DEVICE_EMULATION__
		CFLAGS	    += -D__DEVICE_EMULATION__
	endif
	TARGETDIR := $(BINDIR)/$(BINSUBDIR)
	TARGET    := $(TARGETDIR)/$(EXECUTABLE)
	LINKLINE  = $(LINK) -o $(TARGET) $(OBJS) $(LIB)
endif

# Lib/exe configuration
ifneq ($(CUDPP_STATIC_LIB),)
	ifeq ($(emu), 1)
		NVCCFLAGS   += -deviceemu
		CUDACCFLAGS +=
		LIBSUFFIX   := _$(LIB_ARCH)$(LIBSUFFIX)_emu
		# consistency, makes developing easier
		CXXFLAGS    += -D__DEVICE_EMULATION__
		CFLAGS	    += -D__DEVICE_EMULATION__		
        else
		LIBSUFFIX   := _$(LIB_ARCH)$(LIBSUFFIX)
	endif
	TARGETDIR := $(LIBDIR)

	# detect if 32 bit or 64 bit system
	HP_64 =	$(shell uname -m | grep 64)

	TARGET   := $(subst .a,$(LIBSUFFIX).a,$(LIBDIR)/$(CUDPP_STATIC_LIB))	

	LINKLINE  = ar qv $(TARGET) $(OBJS); ranlib $(TARGET)
endif

# check if verbose 
ifeq ($(verbose), 1)
	VERBOSE :=
else
	VERBOSE := @
endif

################################################################################
# Check for input flags and set compiler flags appropriately
################################################################################
ifeq ($(fastmath), 1)
	NVCCFLAGS += -use_fast_math
endif

ifeq ($(keep), 1)
	NVCCFLAGS += -keep
	NVCC_KEEP_CLEAN := *.i* *.cubin *.cu.c *.cudafe* *.fatbin.c *.ptx
endif

ifdef maxregisters
	NVCCFLAGS += -maxrregcount $(maxregisters)
endif

ifdef verboseptxas
	NVCCFLAGS += --ptxas-options=-v
endif

# Add cudacc flags
NVCCFLAGS += $(CUDACCFLAGS)

# Add common flags
NVCCFLAGS += $(COMMONFLAGS)
CXXFLAGS  += $(COMMONFLAGS)
CFLAGS    += $(COMMONFLAGS)

ifeq ($(nvcc_warn_verbose),1)
	NVCCFLAGS += $(addprefix --compiler-options ,$(CXXWARN_FLAGS)) 
	NVCCFLAGS += --compiler-options -fno-strict-aliasing
endif

################################################################################
# Set up object files
################################################################################
OBJDIR := $(ROOTOBJDIR)/$(LIB_ARCH)/$(BINSUBDIR)
OBJS +=  $(patsubst %.cpp,$(OBJDIR)/%.cpp_o,$(notdir $(CCFILES)))
OBJS +=  $(patsubst %.c,$(OBJDIR)/%.c_o,$(notdir $(CFILES)))
OBJS +=  $(patsubst %.cu,$(OBJDIR)/%.cu_o,$(notdir $(CUFILES)))

################################################################################
# Set up cubin files
################################################################################
CUBINDIR := $(SRCDIR)data
CUBINS +=  $(patsubst %.cu,$(CUBINDIR)/%.cubin,$(notdir $(CUBINFILES)))

################################################################################
# Rules
################################################################################
$(OBJDIR)/%.c_o : $(SRCDIR)%.c $(C_DEPS)
	$(VERBOSE)$(CC) $(CFLAGS) -o $@ -c $<

$(OBJDIR)/%.cpp_o : $(SRCDIR)%.cpp $(C_DEPS)
	$(VERBOSE)$(CXX) $(CXXFLAGS) -o $@ -c $<

$(OBJDIR)/%.cu_o : $(CUSRCDIR)%.cu $(CU_DEPS)
	$(VERBOSE)$(NVCC) -o $@ -c $< $(NVCCFLAGS)

$(CUBINDIR)/%.cubin : $(SRCDIR)%.cu cubindirectory
	$(VERBOSE)$(NVCC) $(CUBIN_ARCH_FLAG) -o $@ -cubin $< $(NVCCFLAGS)

$(TARGET): makedirectories $(OBJS) $(CUBINS) Makefile
	$(VERBOSE)$(LINKLINE)

cubindirectory:
	@mkdir -p $(CUBINDIR)

makedirectories:
	@mkdir -p $(LIBDIR)
	@mkdir -p $(OBJDIR)
	@mkdir -p $(TARGETDIR)

tidy :
	$(VERBOSE)find . | egrep "#" | xargs rm -f
	$(VERBOSE)find . | egrep "\~" | xargs rm -f

clean : tidy
	$(VERBOSE)rm -f $(OBJS)
	$(VERBOSE)rm -f $(CUBINS)
	$(VERBOSE)rm -f $(TARGET)
	$(VERBOSE)rm -f $(NVCC_KEEP_CLEAN)

clobber : clean
	$(VERBOSE)rm -rf $(ROOTOBJDIR)
