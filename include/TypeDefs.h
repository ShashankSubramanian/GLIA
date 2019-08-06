#ifndef _TYPEDEFS_H
#define _TYPEDEFS_H

#include <petsc.h>
#include <mpi.h>

using ScalarType = PetscReal;
#ifdef CUDA
    #include "cuda.h"
    #include <cuda_runtime_api.h>
    #include "cublas_v2.h"
    #include "petsccuda.h"
    #include <accfft_gpu.h>
    #include <accfft_gpuf.h>
    #include <accfft_operators_gpu.h>
    #include <cuComplex.h>

    #ifdef SINGLE
        #define MPIType MPI_FLOAT
        using ComplexType = Complexf;
        using CudaComplexType = cuFloatComplex;
        using fft_plan = accfft_plan_gpuf;
        #define accfft_cleanup accfft_cleanup_gpuf
        #define accfft_plan_dft_3d_r2c accfft_plan_dft_3d_r2c_gpuf
        #define accfft_execute_r2c accfft_execute_r2c_gpuf
        #define accfft_execute_c2r accfft_execute_c2r_gpuf 
        #define makeCudaComplexType make_cuFloatComplex
    #else
        #define MPIType MPI_DOUBLE
        using ComplexType = Complex;
        using CudaComplexType = cuDoubleComplex;
        using fft_plan = accfft_plan_gpu;
        #define accfft_cleanup accfft_cleanup_gpu
        #define accfft_plan_dft_3d_r2c accfft_plan_dft_3d_r2c_gpu
        #define accfft_execute_r2c accfft_execute_r2c_gpu
        #define accfft_execute_c2r accfft_execute_c2r_gpu    
        #define makeCudaComplexType make_cuDoubleComplex
    #endif

    #define fft_free cudaFree

#else
    #include <accfft.h>
    #include <accfftf.h>
    #include <accfft_operators.h>

    // CPU accfft has function overloading for single and double precision; Not the GPU version
    #ifdef SINGLE
        using fft_plan = accfft_planf;
    #else
        using fft_plan = accfft_plan;
    #endif

    #define fft_free accfft_free
#endif

#endif