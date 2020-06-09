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
        ScalarType w, g, v, c;
        w = wm_ptr[i] * (1 - c_ptr[i]);
        g = gm_ptr[i] * (1 - c_ptr[i]);
        v = csf_ptr[i] * (1 - c_ptr[i]);
        c = glm_ptr[i] * (1 - c_ptr[i]);
        if (c_ptr[i] > max) {max = c_ptr[i]; ct = 1;}
        if (w > max) {max = w; ct = 6;}
        if (g > max) {max = g; ct = 5;}
        if (v > max) {max = v; ct = 7;}
        if (c > max) {max = c; ct = 8;}
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

