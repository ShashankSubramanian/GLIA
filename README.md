## PGLISTR

Parallel GLISTR is a parallel implementation of GLioma Image SegmenTation and Registration (GLISTR). It is a framework to integrate biophysical tumor growth models with medical imaging.
It uses novel mathematical algorithms to speedup the convergence, as well as highly optimized
parallel codes that are designed to have good scalability in modern supercomputers. It also has support to 
run on GPUs

## Installation

- Install PETSC (version 3.7 or later, see https://dealii.org/developer/external-libs/petsc.html for installation guide)
- Install PNETCDF (see http://accfft.org/articles/install/ for installation guide)
- Install FFTW3 (version 3.3.4 or later, see http://accfft.org/articles/install/ for installation guide)
- Install AccFFT (see http://accfft.org/articles/install/ for installation guide)
- Set the following paths:
  - ACCFFT_DIR, ACCFFT_LIB, ACCFFT_INC 
  - PETSC_DIR, PETSC_LIB, PETSC_INC 
  - PNETCDF_DIR, PNETCDF_LIB, PNETCDF_INC 
  - FFTW_DIR, FFTW_LIB, FFTW_INC 
- Use scons to create the executables or the compile script. scons --help to see list of options available

## Documentation

Scons will create binaries in build/last. Use /scripts/TumorParams.py to set input parameters for the code.
Use /scripts/submit.py to submit jobs 

## License

PGLISTR is distributed under GNU GENERAL PUBLIC LICENSE Version 2.
Please see LICENSE file.

