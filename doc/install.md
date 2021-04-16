## GLIA install
GLIA runs on CPUs and GPUs. This doc outlines the installation process for both architectures

## GPUs (recommended)
For medical imaging resolutions, a single GPU (16GB) is sufficient

### 1. Dependencies
* CUDA (<= 10.2 tested) and associated libs (cufft, cublas, thrust)
* MPI
* install **PETSc** (version <= 3.11 tested)
* install **pnetCDF** (if netcdf input files; recommended) or **niftilib** and **zlib** (if nifti input files)

### 2. Environment variables
* set CUDA_DIR
* set MPI_DIR
* set PETSC_DIR and PETSC_ARCH 
* set PNETCDF_DIR or (NIFTI_DIR and ZLIB_DIR) 

### 3. Compile and build
* compile.sh shows the scons script to compile the solver with default options
  * set use_gpu=yes, single_precision=yes (recommended, if no solver runs in double precision), multi_gpu=no (deprecated)
  * other options are defaulted
* scons --help to see all compile options

### 3. Clean binaries
* use clean.sh

## CPUs
Perfect scaling up to about 4000 cores. For medical imaging resolution, 256 mpi tasks should suffice

### 1. Dependencies
* MPI
* install **PETSc** (version <= 3.11 tested)
* install **pnetCDF** (if netcdf input files; recommended) or **niftilib** and **zlib** (if nifti input files)
* install **FFTW**
* install **accFFT** (scalable FFTs for multicore CPU machines)

### 2. Environment variables
* set MPI_DIR
* set PETSC_DIR and PETSC_ARCH 
* set PNETCDF_DIR or (NIFTI_DIR and ZLIB_DIR) 
* set FFTW_DIR and ACCFFT_DIR

### 3. Compile and build
* compile.sh shows the scons script to compile the solver with default options
  * set use_gpu=no, single_precision=yes (recommended, if no solver runs in double precision), multi_gpu=no (deprecated)
  * other options are defaulted
* scons --help to see all compile options

### 3. Clean binaries
* use clean.sh
