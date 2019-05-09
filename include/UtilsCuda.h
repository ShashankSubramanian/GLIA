//cuda helpers
#ifndef _UTILSCUDA_H
#define _UTILSCUDA_H

#include <complex>
#include "cuda.h"
#include <math_constants.h>
#include <cuda_runtime_api.h>
#include "cublas_v2.h"

#include <cuda_runtime.h>


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
    static_cast<double>(total_mem - free_mem)/1048576.0,
    static_cast<double>(free_mem)/1048576.0,
    static_cast<double>(total_mem)/1048576.0);
}

//cublas error checking
inline const char* cublasGetErrorString (cublasStatus_t status)
{
    switch(status)
    {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE"; 
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH"; 
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED"; 
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR"; 
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


void computeWeierstrassFilterCuda (double *f, double *s, double sigma, int *isize, int *istart, int *n);
void hadamardComplexProductCuda (cuDoubleComplex *y, cuDoubleComplex *x, double *alph, int *sz);


#endif