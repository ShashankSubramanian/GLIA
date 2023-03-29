#!/usr/bin/python
''' This is an example petsc installation config file
    Installs petsc-gpu (on arch sm_70) with single precision
    Modify mpi, cuda directories with your paths
    run python3 <thisfile>.py to configure the petsc installation
    to install: make; make install (after petsc config is done
    with no errors)
'''
if __name__ == '__main__':
  import sys
  import os
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure_options = [
    '--with-cuda=1',
    '--download-cusp=yes',
    '--with-precision=single',
    '--download-f2cblaslapack=1',
    '--with-mpi=1',
    '--with-mpi-dir=/opt/ibm/spectrum_mpi',
#    '--download-mpich=yes',
    '--with-debugging=1',
    '--with-cuda-dir=/opt/apps/cuda/10.1',
    '--with-ssl=0',
#    '--with-64-bit-indices',
    '--with--shared',
    '--with-x=0',
    'COPTFLAGS="-O3"',
    'CXXOPTFLAGS="-O3"',
    'CUDAFLAGS="-arch=sm_80"'
  ]
  configure.petsc_configure(configure_options)
