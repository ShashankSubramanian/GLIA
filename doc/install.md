## GLIA install
GLIA runs on CPUs and GPUs. This doc outlines the installation process for both architectures

## GPUs (recommended)
For medical imaging resolutions, a single GPU (16GB) is sufficient

### 1. Dependencies
* **C++11** compatible compiler
* **CUDA** (<= 10.2 tested) and associated libs (cufft, cublas, thrust)
* **MPI**
* install **PETSc** (3.7 <= version <= 3.11 tested; 3.11 recommended): [petsc 3.11](https://www.mcs.anl.gov/petsc/mirror/release-snapshots/petsc-lite-3.11.4.tar.gz)
* install **pnetCDF** (if netcdf input files; recommended): [pnetcdf 1.11](http://cucis.ece.northwestern.edu/projects/PnetCDF/Release/pnetcdf-1.11.1.tar.gz)
* install **niftilib** and **zlib** (if nifti input files): [nifti 2.0](https://sourceforge.net/projects/niftilib/files/nifticlib/nifticlib_2_0_0/nifticlib-2.0.0.tar.gz/download)

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

### 4. Run
* binary is build/last/tusolver
* see scripts/README on example run scripts

### 5. Clean binaries
* use clean.sh

## CPUs
Perfect scaling up to about 4000 cores. For medical imaging resolution, 256 mpi tasks should suffice

### 1. Dependencies
* **C++11** compatible compiler
* **MPI**
* install **PETSc** (3.7 <= version <= 3.11 tested; 3.11 recommended): [petsc 3.11](https://www.mcs.anl.gov/petsc/mirror/release-snapshots/petsc-lite-3.11.4.tar.gz)
* install **pnetCDF** (if netcdf input files; recommended): [pnetcdf 1.11](http://cucis.ece.northwestern.edu/projects/PnetCDF/Release/pnetcdf-1.11.1.tar.gz)
* install **niftilib** and **zlib** (if nifti input files): [nifti 2.0](https://sourceforge.net/projects/niftilib/files/nifticlib/nifticlib_2_0_0/nifticlib-2.0.0.tar.gz/download)
* install **FFTW**: see [http://accfft.org/articles/install/](http://accfft.org/articles/install/)
* install **accFFT** (scalable FFTs for multicore CPU machines): see [http://accfft.org/articles/install/](http://accfft.org/articles/install/)

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

### 4. Run
* binary is build/last/tusolver
* see scripts/README on example run scripts

### 5. Clean binaries
* use clean.sh
