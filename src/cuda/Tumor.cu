#include "Tumor.h"


__global__ void nonlinearForceScaling (ScalarType *c_ptr, ScalarType *fx_ptr, ScalarType *fy_ptr, ScalarType *fz_ptr, int64_t sz) {
	int64_t i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i < sz) {
		fx_ptr[i] *= tanh (c_ptr[i]);
        fy_ptr[i] *= tanh (c_ptr[i]);
        fz_ptr[i] *= tanh (c_ptr[i]);
	}
}

__global__ void computeTumorSegmentation (ScalarType *bg_ptr, ScalarType *gm_ptr, ScalarType *wm_ptr, ScalarType *csf_ptr, ScalarType *glm_ptr, ScalarType *c_ptr, ScalarType *seg_ptr, int64_t sz) {
	int64_t i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i < sz) {
        // bg: 0, c: 1, wm: 2, gm: 3, csf: 4, glm: 5
        ScalarType max = bg_ptr[i];
        int ct = 0;
        if (c_ptr[i] > max) {max = c_ptr[i]; ct = 1;}
        if (wm_ptr[i] > max) {max = wm_ptr[i]; ct = 6;}
        if (gm_ptr[i] > max) {max = gm_ptr[i]; ct = 5;}
        if (csf_ptr[i] > max) {max = csf_ptr[i]; ct = 7;}
        if (glm_ptr[i] > max) {max = glm_ptr[i]; ct = 8;}
        seg_ptr[i] = ct;
    }
}

void nonlinearForceScalingCuda (ScalarType *c_ptr, ScalarType *fx_ptr, ScalarType *fy_ptr, ScalarType *fz_ptr, int64_t sz) {
	int n_th = N_THREADS;

	nonlinearForceScaling <<< (sz + n_th - 1) / n_th, n_th >>> (c_ptr, fx_ptr, fy_ptr, fz_ptr, sz);

	cudaDeviceSynchronize();
	cudaCheckKernelError ();
}

void computeTumorSegmentationCuda (ScalarType *bg_ptr, ScalarType *gm_ptr, ScalarType *wm_ptr, ScalarType *csf_ptr, ScalarType *glm_ptr, ScalarType *c_ptr, ScalarType *seg_ptr, int64_t sz) {
	int n_th = N_THREADS;

	computeTumorSegmentation <<< (sz + n_th - 1) / n_th, n_th >>> (bg_ptr, gm_ptr, wm_ptr, csf_ptr, glm_ptr, c_ptr, seg_ptr, sz);

	cudaDeviceSynchronize ();
	cudaCheckKernelError ();
}

