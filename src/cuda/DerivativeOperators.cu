#include "DerivativeOperators.h"


__global__ void computeCrossEntropy(ScalarType *ce_ptr, ScalarType *d_ptr, ScalarType *c_ptr, ScalarType eps, int64_t sz) {
    int64_t i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < sz) {
        c_ptr[i] = (c_ptr[i] < eps) ? eps : c_ptr[i];
        c_ptr[i] = (c_ptr[i] > 1 - eps) ? 1 - eps : c_ptr[i];
        ce_ptr[i] = -(d_ptr[i] * log(c_ptr[i]) + (1 - d_ptr[i]) * log(1 - c_ptr[i]));
    }
}

__global__ void computeCrossEntropyAdjointIC(ScalarType *a_ptr, ScalarType *d_ptr, ScalarType *c_ptr, ScalarType eps, int64_t sz) {
    int64_t i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < sz) {
        c_ptr[i] = (c_ptr[i] < eps) ? eps : c_ptr[i];
        c_ptr[i] = (c_ptr[i] > 1 - eps) ? 1 - eps : c_ptr[i];
        a_ptr[i] = (d_ptr[i] / (c_ptr[i]) - (1 - d_ptr[i]) / (1 - c_ptr[i]));
    }
}

void computeCrossEntropyCuda(ScalarType *ce_ptr, ScalarType *d_ptr, ScalarType *c_ptr, ScalarType eps, int64_t sz) {
    int n_th = N_THREADS;
    computeCrossEntropy<<<(sz + n_th - 1)/n_th, n_th>>>(ce_ptr, d_ptr, c_ptr, eps, sz);
    cudaDeviceSynchronize();
    cudaCheckKernelError();
}

void computeCrossEntropyAdjointICCuda(ScalarType *a_ptr, ScalarType *d_ptr, ScalarType *c_ptr, ScalarType eps, int64_t sz) {
    int n_th = N_THREADS;
    computeCrossEntropyAdjointIC<<<(sz + n_th - 1)/n_th, n_th>>>(a_ptr, d_ptr, c_ptr, eps, sz);
    cudaDeviceSynchronize();
    cudaCheckKernelError();
}

