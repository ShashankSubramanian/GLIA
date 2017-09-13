## PGLISTR

Parallel GLISTR is a parallel implementation of GLioma Image SegmenTation and Registration (GLISTR).
It uses novel mathematical algorithms to speedup the convergence, as well as highly optimized
parallel codes that are designed to have good scalability in modern supercomputers.

## Installation

- Install PETSC (version 3.6 or later, see https://dealii.org/developer/external-libs/petsc.html for installation guide)
- Install PNETCDF (see http://accfft.org/articles/install/ for installation guide)
- Install FFTW3 (version 3.3.4 or later, see http://accfft.org/articles/install/ for installation guide)
- Install AccFFT (see http://accfft.org/articles/install/ for installation guide)
- Install GLOG (see https://github.com/google/glog) 
- Set the following paths:
  - ACCFFT_DIR, ACCFFT_LIB, ACCFFT_INC 
  - PETSC_DIR, PETSC_LIB, PETSC_INC 
  - PNETCDF_DIR, PNETCDF_LIB, PNETCDF_INC 
  - FFTW_DIR, FFTW_LIB, FFTW_INC 
  - GLOG_DIR
- Use make to compile executables in bin directory.

## Documentation
The makefile will create two executables in the bin directory.
The first one is bin/Forward, which is a driver for the forward operator.
It accepts command line options, which can be found by running:
  ./bin/Forward -h

The second executable is bin/tao, which performs a PDE-constrained optimization
using TAO library. The command line options for this code can also be found by running:
  ./bin/tao -h

There is another README file in src/ directory which has a short explanation of
all of the source directory codes.

## License
PGLISTR is distributed under GNU GENERAL PUBLIC LICENSE Version 2.
Please see LICENSE file.

