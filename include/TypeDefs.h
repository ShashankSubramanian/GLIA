#ifndef _TYPEDEFS_H
#define _TYPEDEFS_H

#include <petsc.h>
#include <mpi.h>
#include <cfloat>


// enums
enum {QDFS = 0, SLFS = 1};
enum {CONSTCOEF = 1, SINECOEF = 2, BRAIN = 0, BRAINNEARMF = 3, BRAINFARMF = 4};
enum {GAUSSNEWTON = 0, QUASINEWTON = 1};
enum {ARMIJO = 0, MT = 1};
enum {L1 = 0, L2 = 1, wL2 = 3, L2b = 4};
enum {INIT = 0, PRE_RD = 1, COSAMP_L1_INIT = 3, COSAMP_L1_THRES_GRAD = 4, COSAMP_L1_SOLVE_SUBSPACE = 5, COSAMP_L1_THRES_SOL = 6, FINAL_L2 = 7, POST_RD = 8, FINALIZE = 9};
enum {SEQ = 0, MPI = 1};


// datatypes and function aliases
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
        using CufftScalarType = cufftReal;
        using CufftComplexType = cufftComplex;
        using fft_plan = accfft_plan_gpuf;

        #define cufftExecuteR2C cufftExecR2C
        #define cufftExecuteC2R cufftExecC2R
        #define cuComplexMultiply cuCmulf
        #define cublasAXPY cublasSaxpy
        #define cublasScale cublasSscal
        #define cublasSum cublasSasum
        #define fft_local_size_dft_r2c accfft_local_size_dft_r2c_gpuf
        #define accfft_cleanup accfft_cleanup_gpuf
        #define fft_plan_dft_3d_r2c accfft_plan_dft_3d_r2c_gpuf
        #define fft_execute_r2c accfft_execute_r2c_gpuf
        #define fft_execute_c2r accfft_execute_c2r_gpuf 
        #define makeCudaComplexType make_cuFloatComplex

        #define accfftGrad accfft_grad_gpuf
        #define accfftDiv accfft_divergence_gpuf
    #else
        #define MPIType MPI_DOUBLE
        using ComplexType = Complex;
        using CudaComplexType = cuDoubleComplex;
        using CufftScalarType = cufftDoubleReal;
        using CufftComplexType = cufftDoubleComplex;
        using fft_plan = accfft_plan_gpu;

        #define cufftExecuteR2C cufftExecD2Z
        #define cufftExecuteC2R cufftExecZ2D
        #define cuComplexMultiply cuCmul
        #define cublasAXPY cublasDaxpy
        #define cublasScale cublasDscal
        #define cublasSum cublasDasum
        #define fft_local_size_dft_r2c accfft_local_size_dft_r2c_gpu
        #define accfft_cleanup accfft_cleanup_gpu
        #define fft_plan_dft_3d_r2c accfft_plan_dft_3d_r2c_gpu
        #define fft_execute_r2c accfft_execute_r2c_gpu
        #define fft_execute_c2r accfft_execute_c2r_gpu    
        #define makeCudaComplexType make_cuDoubleComplex

        #define accfftGrad accfft_grad_gpu
        #define accfftDiv accfft_divergence_gpu
    #endif

    #define fft_free cudaFree

#else
    #include <accfft.h>
    #include <accfftf.h>
    #include <accfft_operators.h>

    // CPU accfft has function overloading for single and double precision; Not the GPU version
    #ifdef SINGLE
        #define MPIType MPI_FLOAT
        using ComplexType = Complexf;
        using fft_plan = accfft_planf;
        #define fft_execute_r2c accfft_execute_r2cf
        #define fft_execute_c2r accfft_execute_c2rf
        #define fft_plan_dft_3d_r2c accfft_plan_dft_3d_r2cf
        #define fft_local_size_dft_r2c accfft_local_size_dft_r2cf

        #define accfftGrad accfft_gradf
        #define accfftDiv accfft_divergencef
    #else
        #define MPIType MPI_DOUBLE
        using ComplexType = Complex;
        using fft_plan = accfft_plan;
        #define fft_execute_r2c accfft_execute_r2c
        #define fft_execute_c2r accfft_execute_c2r
        #define fft_plan_dft_3d_r2c accfft_plan_dft_3d_r2c
        #define fft_local_size_dft_r2c accfft_local_size_dft_r2c

        #define accfftGrad accfft_grad
        #define accfftDiv accfft_divergence
    #endif

    #define fft_free accfft_free
#endif

#endif
