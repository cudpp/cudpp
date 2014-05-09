# Makefile for mpi-gather-test
CC			=	nvcc
NVCC		=		nvcc
LD			=	nvcc

NVCC = "$(shell which nvcc)"

EXE			:= 	bin/sa-gpu

CPP_SRCS := mgpuutil.cpp

CU_SRCS	:= main.cu \
           mgpucontext.cu \
           radix_sort.cu \
           skew.cu  
    	  

CPP_OBJS	:=	${CPP_SRCS:.cpp=.o}
CU_OBJS	        :=	${CU_SRCS:.cu=.o}
OBJS			:=	$(CU_OBJS) $(CPP_OBJS)

# include directories
INC_DIRS 	:= 	"$(shell dirname $(NVCC))/../include" \
                "$(shell dirname $(NVCC))/../samples/common/inc" \
                "../mylib/cub" \
                "../mylib/moderngpu/include" 

# library directories
LIB_DIRS	:=	"$(shell dirname $(NVCC))/../lib64"

# libraries
LIBS		:=	cuda

CCFLAGS		+=	-c -g $(foreach includedir, $(INC_DIRS), -I $(includedir))
LDFLAGS		+=	$(foreach librarydir, $(LIB_DIRS), -L $(librarydir))
LDFLAGS		+=	$(foreach library, $(LIBS), -l$(library))

$(CU_OBJS): EXTRA_FLAGS := -arch compute_35

.PHONY: all clean

$(EXE): $(OBJS)
	$(LD) $(OBJS) $(LDFLAGS) -o $(EXE) -O3

#%.cpp.o : %.cpp
.cpp.o:
	$(CC) $(CCFLAGS) $< #-o $@

%.o : %.cu
	$(NVCC) $(CCFLAGS) $(EXTRA_FLAGS) -c $< -o $@

clean:
	rm -vf $(EXE) $(OBJS)
