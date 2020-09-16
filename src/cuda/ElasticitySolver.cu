#include "ElasticitySolver.h"


__constant__ int n_cuda[3], ostart_cuda[3];

void initElasticityCudaConstants(int *n, int *ostart) {
	cudaMemcpyToSymbol (ostart_cuda, ostart, 3 * sizeof(int));
	cudaMemcpyToSymbol (n_cuda, n, 3 * sizeof(int));
}


__global__ void computeScreening (ScalarType *screen_ptr, ScalarType *c_ptr, ScalarType *bg_ptr, ScalarType screen_low, ScalarType screen_high, int64_t sz) {
	int64_t i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i < sz) {
		ScalarType c_threshold = 0.005;
		screen_ptr[i] = (c_ptr[i] >= c_threshold) ? screen_low : screen_high;
		if (bg_ptr[i] > 0.95) screen_ptr[i] = 1E6; // screen out the background completely to ensure no movement
	}
}

__global__ void computeTumorLame (ScalarType *mu_ptr, ScalarType *lam_ptr, ScalarType *c_ptr, ScalarType mu_tumor, ScalarType lam_tumor, int64_t sz) {
	int64_t i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i < sz) {
		// positivity clipping for mu, lam because tissues are no longer clipped
		mu_ptr[i] = (mu_ptr[i] > 0) ? mu_ptr[i] : 0;
        lam_ptr[i] = (lam_ptr[i] > 0) ? lam_ptr[i] : 0;

		mu_ptr[i] += (c_ptr[i] > 0) ? (mu_tumor * c_ptr[i]) : 0;
        lam_ptr[i] += (c_ptr[i] > 0) ? (lam_tumor * c_ptr[i]) : 0;
	}
}

__global__ void precFactorElasticity (CudaComplexType *ux_hat, CudaComplexType *uy_hat, CudaComplexType *uz_hat, CudaComplexType *fx_hat, CudaComplexType *fy_hat, CudaComplexType *fz_hat, ScalarType lam_avg, ScalarType mu_avg, ScalarType screen_avg, int* osize_cuda) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;
	int k = threadIdx.z + blockDim.z * blockIdx.z;

	int64_t ptr = i * osize_cuda[1] * osize_cuda[2] + j * osize_cuda[2] + k;

	if (i < osize_cuda[0] && j < osize_cuda[1] && k < osize_cuda[2]) {
		ScalarType s1, s2, s1_square, s3, scale;
	    int64_t wx, wy, wz;
	    ScalarType wTw, wTf_real, wTf_imag;
	    int64_t x_global, y_global, z_global;

	    ScalarType factor = 1.0 / (n_cuda[0] * n_cuda[1] * n_cuda[2]);
	    s2 = lam_avg + mu_avg;


		x_global = i + ostart_cuda[0];
		y_global = j + ostart_cuda[1];
		z_global = k + ostart_cuda[2];

		wx = x_global;
		if (x_global > n_cuda[0] / 2) // symmetric frequencies
			wx -= n_cuda[0];
		if (x_global == n_cuda[0] / 2) // nyquist frequency
			wx = 0;

		wy = y_global;
		if (y_global > n_cuda[1] / 2) // symmetric frequencies
			wy -= n_cuda[1];
		if (y_global == n_cuda[1] / 2) // nyquist frequency
			wy = 0;

		wz = z_global;
		if (z_global > n_cuda[2] / 2) // symmetric frequencies
			wz -= n_cuda[2];
		if (z_global == n_cuda[2] / 2) // nyquist frequency
			wz = 0;

		wTw = -1.0 * (wx * wx + wy * wy + wz * wz);

		s1 = -screen_avg + mu_avg * wTw;
		s1_square = s1 * s1;
		s3 = 1.0 / (1.0 + (wTw * s2) / s1);

		wTf_real = wx * fx_hat[ptr].x + wy * fy_hat[ptr].x + wz * fz_hat[ptr].x;
		wTf_imag = wx * fx_hat[ptr].y + wy * fy_hat[ptr].y + wz * fz_hat[ptr].y;

		// real part
		scale = -1.0 * wx * wTf_real;
		ux_hat[ptr].x = factor * (fx_hat[ptr].x * (1.0 / s1) - (1.0 / s1_square) * s2 * s3 * scale); 
		// imaginary part
		scale = -1.0 * wx * wTf_imag;
		ux_hat[ptr].y = factor * (fx_hat[ptr].y * (1.0 / s1) - (1.0 / s1_square) * s2 * s3 * scale); 

		// real part
		scale = -1.0 * wy * wTf_real;
		uy_hat[ptr].x = factor * (fy_hat[ptr].x * (1.0 / s1) - (1.0 / s1_square) * s2 * s3 * scale); 
		// imaginary part
		scale = -1.0 * wy * wTf_imag;
		uy_hat[ptr].y = factor * (fy_hat[ptr].y * (1.0 / s1) - (1.0 / s1_square) * s2 * s3 * scale); 

		// real part
		scale = -1.0 * wz * wTf_real;
		uz_hat[ptr].x = factor * (fz_hat[ptr].x * (1.0 / s1) - (1.0 / s1_square) * s2 * s3 * scale); 
		// imaginary part
		scale = -1.0 * wz * wTf_imag;
		uz_hat[ptr].y = factor * (fz_hat[ptr].y * (1.0 / s1) - (1.0 / s1_square) * s2 * s3 * scale); 
	}
}

void computeScreeningCuda (ScalarType *screen_ptr, ScalarType *c_ptr, ScalarType *bg_ptr, ScalarType screen_low, ScalarType screen_high, int64_t sz) {
	int n_th = N_THREADS;

	computeScreening <<< (sz + n_th - 1) / n_th, n_th >>> (screen_ptr, c_ptr, bg_ptr, screen_low, screen_high, sz);

	cudaDeviceSynchronize();
	cudaCheckKernelError ();
}

void computeTumorLameCuda (ScalarType *mu_ptr, ScalarType *lam_ptr, ScalarType *c_ptr, ScalarType mu_tumor, ScalarType lam_tumor, int64_t sz) {
	int n_th = N_THREADS;

	computeTumorLame <<< (sz + n_th - 1) / n_th, n_th >>> (mu_ptr, lam_ptr, c_ptr, mu_tumor, lam_tumor, sz);

	cudaDeviceSynchronize();
	cudaCheckKernelError ();
}

void precFactorElasticityCuda (CudaComplexType *ux_hat, CudaComplexType *uy_hat, CudaComplexType *uz_hat, CudaComplexType *fx_hat, CudaComplexType *fy_hat, CudaComplexType *fz_hat, ScalarType lam_avg, ScalarType mu_avg, ScalarType screen_avg, int *sz) {
	int n_th_x = N_THREADS_X;
	int n_th_y = N_THREADS_Y;
	int n_th_z = N_THREADS_Z;
	dim3 n_threads (n_th_x, n_th_y, n_th_z);
	dim3 n_blocks ((sz[0] + n_th_x - 1) / n_th_x, (sz[1] + n_th_y - 1) / n_th_y, (sz[2] + n_th_z - 1) / n_th_z);

	precFactorElasticity <<< n_blocks, n_threads >>> (ux_hat, uy_hat, uz_hat, fx_hat, fy_hat, fz_hat, lam_avg, mu_avg, screen_avg, sz);

	cudaDeviceSynchronize();
	cudaCheckKernelError ();
}
