CXX=mpicxx
RM = rm -f
MKDIRS = mkdir -p

BUILD_GPU = 0
WARN = -pedantic -Wall -Wextra -Wcast-align -Wcast-qual -Wctor-dtor-privacy -Wdisabled-optimization -Wformat=2 -Winit-self -Wlogical-op -Wmissing-declarations -Wmissing-include-dirs -Wnoexcept -Wold-style-cast -Woverloaded-virtual -Wredundant-decls -Wshadow -Wsign-conversion -Wsign-promo -Wstrict-null-sentinel -Wstrict-overflow=5 -Wswitch-default -Wundef -Werror -Wno-unused
BINDIR = ./bin
SRCDIR = ./src
OBJDIR = ./obj
GPU_OBJDIR = ./obj
INCDIR = ./include -I${ACCFFT_DIR}/include/ -I${PNETCDF_DIR}/include -I${FFTW_DIR}/include -I${GLOG_DIR}/include
INCDIR = ./include -I./3rdparty/ -I${ACCFFT_DIR}/include/ -I${PNETCDF_DIR}/include -I${FFTW_DIR}/include -I${GLOG_DIR}/include
ifneq ($(CUDA_DIR),)
INCDIR+= -I$(CUDA_DIR)/include/
endif
APPDIR = ./app
TIMINGSDIR = ./3rdparty/timings/
INCDIR+= -I$(TIMINGSDIR)

PSC_INC = -I$(PETSC_DIR)/include -I$(PETSC_DIR)/$(PETSC_ARCH)/include  
PSC_LIB = -L$(PETSC_DIR)/lib -L$(PETSC_DIR)/$(PETSC_ARCH)/lib  -lpetsc

PSC_DBG_INC = -I$(PETSC_DBG_DIR)/include -I$(PETSC_DBG_DIR)/$(PETSC_DBG_ARCH)/include 
PSC_DBG_LIB = -L$(PETSC_DBG_DIR)/lib     -L$(PETSC_DBG_DIR)/$(PETSC_DBG_ARCH)/lib  -lpetsc

CXXFLAGS= -O3 -fopenmp -std=c++11 -DPVFMM_MEMDEBUG -DGAUSS_NEWTON  #-DENFORCE_POSITIVE_C #-DINVERT_RHO   -xhost 

N_FLAGS=-c  -O0 -gencode arch=compute_35,code=sm_35  -Xcompiler -fopenmp -DENABLE_GPU -lcudart 
N_INC= -I$(CUDA_DIR)/include -I./ -I./include/
N_LIB= -L$(CUDA_DIR)/lib64/

LDFLAGS=  -L${FFTW_DIR}/lib  -lfftw3 -lfftw3_threads -lfftw3f -lfftw3f_threads -L${ACCFFT_DIR}/lib
LDFLAGS+= -L${ACCFFT_DIR}/lib -laccfft -laccfft_utils -lfftw3 -lfftw3_threads  -L${PNETCDF_DIR}/lib -lpnetcdf 

ifeq ($(BUILD_GPU), 1)
LDFLAGS+=-L$(CUDA_DIR)/lib64 -lcudart
endif

TARGET_BIN= $(BINDIR)/forward
TARGET_BIN+= $(BINDIR)/inverse
TARGET_BIN+= $(BINDIR)/inversedata

SOURCES = $(wildcard $(SRCDIR)/*.cpp) $(TIMINGSDIR)/EventTimings.cpp
#SOURCES = $(SRCDIR)/DiffCoef.cpp\
		  $(SRCDIR)/ReacCoef.cpp \
		  $(SRCDIR)/Phi.cpp \
		  $(SRCDIR)/Obs.cpp \
		  $(SRCDIR)/Tumor.cpp \
		  $(SRCDIR)/MatProp.cpp \
		  $(SRCDIR)/DiffSolver.cpp \
		  $(SRCDIR)/PdeOperators.cpp \
		  $(SRCDIR)/DerivativeOperators.cpp \
		  $(SRCDIR)/InvSolver.cpp \
		  $(SRCDIR)/BLMVM.cpp \
		  $(SRCDIR)/TumorSolverInterface.cpp \
		  $(SRCDIR)/Utils.cpp \
		  $(TIMINGSDIR)/EventTimings.cpp \

GPU_SOURCES =		 
ifeq ($(BUILD_GPU), 1)
	SOURCES += $(SRCDIR)/gpu_interp3.cpp \
		  $(SRCDIR)/Interp3_Plan_GPU.cpp
GPU_SOURCES += $(SRCDIR)/gpu_interp3_kernels.cu
endif

OBJS = $(patsubst $(SRCDIR)/%.cpp,$(OBJDIR)/%.o,$(SOURCES))  # .cpp -> .o for all SOURCES
GPU_OBJS = $(patsubst $(SRCDIR)/%.cu,$(OBJDIR)/%.o,$(GPU_SOURCES))  # .cpp -> .o for all SOURCES

.SECONDARY: $(OBJS) $(GPU_OBJS)

all : $(TARGET_BIN) MISC

dbg : $(TARGET_BIN_DBG) MISC

$(BINDIR)/%: $(OBJDIR)/%.o  $(GPU_OBJS) $(OBJS)
	-@$(MKDIRS) $(dir $@) # if bin exists dont give an error
	$(CXX) $(CXXFLAGS) ${PSC_INC} $^ $(LDFLAGS) ${PSC_LIB} -o $@

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	-@$(MKDIRS) $(dir $@)
	$(CXX) $(CXXFLAGS) ${PSC_INC} -I$(INCDIR) -c $^ -o $@

$(GPU_OBJS): $(GPU_SOURCES)
	-@$(MKDIRS) $(dir $@)
	nvcc $(NFLAGS) -I$(N_INC) -c $^ -o $@

$(OBJDIR)/%.o: $(APPDIR)/%.cpp
	-@$(MKDIRS) $(dir $@)
	$(CXX) $(CXXFLAGS) ${PSC_INC} -I$(INCDIR) -c $^ -o $@

# DEBUG RULES

$(BINDIR)/%_dbg: $(OBJDIR)/%_dbg.o $(OBJS_DBG)
	-@$(MKDIRS) $(dir $@) # if bin exists dont give an error
	$(CXX) $(CXXFLAGS) ${PSC_DBG_INC} $^ $(LDFLAGS) ${PSC_DBG_LIB} -o $@

$(OBJDIR)/%_dbg.o: $(SRCDIR)/%.cpp
	-@$(MKDIRS) $(dir $@)
	$(CXX) $(CXXFLAGS) ${PSC_DBG_INC} -I$(INCDIR) -c $^ -o $@

$(OBJDIR)/%_dbg.o: $(APPDIR)/%.cpp
	-@$(MKDIRS) $(dir $@)
	$(CXX) $(CXXFLAGS) ${PSC_DBG_INC} -I$(INCDIR) -c $^ -o $@

MISC:
	@$(MKDIRS) results

run: $(TARGET_BIN)
	export OMP_NUM_THREADS=1; \
	ibrun -np 16 $(TARGET_BIN)  -pc_type none -ksp_monitor -hprec_ksp_monitor -heq_ksp_monitor -ksp_rtol 1e-6  -nx 512 -ny 512 -nz 512 -ndim 2 -dimx 4 -dimy 4
#-heq_ksp_monitor_true_residual -hprec_ksp_monitor -heq_pc_type none
run-dbg:
	ibrun -np 4  $(TARGET_BIN_DBG)  -heq_pc_type none -heq_ksp_monitor_true_residual -ksp_rtol 1e-6  -nx 64 -ny 64 -nz 64 -ndim 2 -dimx 2 -dimy 2


	#pc_type none
clean:
	$(RM) -r $(BINDIR)/* $(OBJDIR)/*
	$(RM) *~ */*~
