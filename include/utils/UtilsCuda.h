//cuda helpers
#ifndef _UTILSCUDA_H
#define _UTILSCUDA_H

#include "TypeDefs.h"

#include <complex>
#include "cuda.h"
#include <math_constants.h>
#include <cuda_runtime_api.h>
#include "cublas_v2.h"
#include "cufft.h"

#include <thrust/system_error.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/complex.h>
#include <thrust/extrema.h>

#include <cuda_runtime.h>

#define N_THREADS 512
#define N_THREADS_X 1
#define N_THREADS_Y 8
#define N_THREADS_Z 32

// Cuda error checking routines

#define cudaCheckError(ans) cudaAssert((ans), __FILE__, __LINE__,false)
#define cudaCheckKernelError() cudaCheckError(cudaPeekAtLastError())
#define cudaCheckLastError() cudaCheckError(cudaGetLastError())

inline int cudaAssert(cudaError_t code, const char *file, int line, bool abort=true) {
  if (code != cudaSuccess) {
    fprintf(stderr,"CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
    return code;
  }
  return 0;
}

inline void cudaPrintDeviceMemory(int dev=0) {
  size_t free_mem;
  size_t total_mem;

  cudaSetDevice(dev);
  cudaMemGetInfo(&free_mem, &total_mem);

  printf("GPU %i memory usage: used = %lf MiB, free = %lf MiB, total = %lf MiB\n",
    dev,
    static_cast<ScalarType>(total_mem - free_mem)/1048576.0,
    static_cast<ScalarType>(free_mem)/1048576.0,
    static_cast<ScalarType>(total_mem)/1048576.0);
}

//cublas error checking
inline const char* cublasGetErrorString (cublasStatus_t status) {
    switch(status) {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE"; 
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH"; 
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED"; 
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR"; 
        case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED"; 
        case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR"; 
    }
    return "unknown error";
}

inline const char* cufftGetErrorString (cufftResult error) {
    switch (error) {
        case CUFFT_SUCCESS: return "CUFFT_SUCCESS";
        case CUFFT_INVALID_PLAN: return "CUFFT_INVALID_PLAN";
        case CUFFT_ALLOC_FAILED: return "CUFFT_ALLOC_FAILED";
        case CUFFT_INVALID_TYPE: return "CUFFT_INVALID_TYPE";
        case CUFFT_INVALID_VALUE: return "CUFFT_INVALID_VALUE";
        case CUFFT_INTERNAL_ERROR: return "CUFFT_INTERNAL_ERROR";
        case CUFFT_EXEC_FAILED: return "CUFFT_EXEC_FAILED";
        case CUFFT_SETUP_FAILED: return "CUFFT_SETUP_FAILED";
        case CUFFT_INVALID_SIZE: return "CUFFT_INVALID_SIZE";
        case CUFFT_UNALIGNED_DATA: return "CUFFT_UNALIGNED_DATA";
        case CUFFT_INCOMPLETE_PARAMETER_LIST: return "CUFFT_INCOMPLETE_PARAMETER_LIST";
        case CUFFT_INVALID_DEVICE: return "CUFFT_INVALID_DEVICE";
        case CUFFT_PARSE_ERROR: return "CUFFT_PARSE_ERROR";
        case CUFFT_NO_WORKSPACE: return "CUFFT_NO_WORKSPACE";
        case CUFFT_NOT_IMPLEMENTED: return "CUFFT_NOT_IMPLEMENTED";
        case CUFFT_LICENSE_ERROR: return "CUFFT_LICENSE_ERROR";
        case CUFFT_NOT_SUPPORTED: return "CUFFT_NOT_SUPPORTED";
    }

    return "unknown error";
}


#define cublasCheckError(ans) cublasAssert((ans), __FILE__, __LINE__,false)
inline int cublasAssert (cublasStatus_t code, const char *file, int line, bool abort=true) {
	if (code != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr,"CUBLAS Error: %s %s %d\n", cublasGetErrorString (code), file, line);
		if (abort) exit(code);
    	return code;
  	}
  return 0;
}

#define cufftCheckError(ans) cufftAssert((ans), __FILE__, __LINE__,false)
inline int cufftAssert (cufftResult code, const char *file, int line, bool abort=true) {
  if (code != CUFFT_SUCCESS) {
    fprintf(stderr,"CUFFT Error: %s %s %d\n", cufftGetErrorString (code), file, line);
    if (abort) exit(code);
      return code;
    }
  return 0;
}

void initCudaConstants (int *isize, int *osize, int *istart, int *ostart, int *n);
void vecMaxCuda (ScalarType *x, int *loc, ScalarType *val, int sz);
void vecSumCuda(ScalarType *f, ScalarType *sum, int64_t sz);
void computeMagnitudeCuda (ScalarType *mag_ptr, ScalarType *x_ptr, ScalarType *y_ptr, ScalarType *z_ptr, int64_t sz);
void hadamardComplexProductCuda (CudaComplexType *y, CudaComplexType *x, int *sz);
void hadamardComplexProductCuda (CudaComplexType *y, ScalarType *x, int *sz);
void setCoordsCuda (ScalarType *x_ptr, ScalarType *y_ptr, ScalarType *z_ptr, int *sz);
void clipVectorCuda (ScalarType *x_ptr, int64_t sz);
void clipVectorAboveCuda (ScalarType *x_ptr, int64_t sz);
void copyDoubleToFloatCuda (float *dst, double *src, int64_t sz);
void copyFloatToDoubleCuda (double *dst, float *src, int64_t sz);
#endif
