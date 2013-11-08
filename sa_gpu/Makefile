# Makefile for mpi-gather-test
CC			=	nvcc
NVCC		=		nvcc
LD			=	nvcc

NVCC = "$(shell which nvcc)"

EXE			:= 	bin/sa-gpu
#CPP_SRCS	:= 	CUDASampler-host.cpp \
                readfile.cpp    \
                main.cpp        \

#CU_SRCS     :=	CUDASampler.cu \
                internalsort.cu\
                aadmerge_kernel.cu

CPP_SRCS := mgpucontext.cpp  \
            format.cpp

CU_SRCS	:= main.cu \
           radix_sort.cu   \
           
#CU_SRCS	:= testlaunchbox.cu


CPP_OBJS	:=	${CPP_SRCS:.cpp=.o}
CU_OBJS	        :=	${CU_SRCS:.cu=.o}
OBJS			:=	$(CU_OBJS) $(CPP_OBJS)

# include directories
INC_DIRS 	:= 	"$(shell dirname $(NVCC))/../include" \
                "$(shell dirname $(NVCC))/../samples/common/inc" \
                "/home/laura/mylib/back40c" \
                "/home/laura/mylib/moderngpu" \
  	        "/home/yzh/mylib/b40c" \
	        "../back40computing-read-only"


# library directories
LIB_DIRS	:=	"$(shell dirname $(NVCC))/../lib"

# libraries
LIBS		:=	cuda

CCFLAGS		+=	-c -g $(foreach includedir, $(INC_DIRS), -I $(includedir))
LDFLAGS		+=	$(foreach librarydir, $(LIB_DIRS), -L $(librarydir))
LDFLAGS		+=	$(foreach library, $(LIBS), -l$(library))

$(CU_OBJS): EXTRA_FLAGS := -arch compute_20

.PHONY: all clean

$(EXE): $(OBJS)
	$(LD) $(OBJS) $(LDFLAGS) -o $(EXE) -O3

.cpp.o:
	$(CC) $(CCFLAGS) $<

%.o : %.cu
	$(NVCC) $(CCFLAGS) $(EXTRA_FLAGS) -c $< -o $@

clean:
	rm -vf $(EXE) $(OBJS)
