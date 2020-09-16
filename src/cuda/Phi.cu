#include "Phi.h"

__constant__ int n_cuda[3], istart_cuda[3];

void initPhiCudaConstants(int *n, int *istart) {
	cudaMemcpyToSymbol (istart_cuda, istart, 3 * sizeof(int));
	cudaMemcpyToSymbol (n_cuda, n, 3 * sizeof(int));
}

__global__ void initializeGaussian (ScalarType *out, ScalarType sigma, ScalarType xc, ScalarType yc, ScalarType zc, int *isize_cuda) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;
	int k = threadIdx.z + blockDim.z * blockIdx.z;

	int64_t ptr = i * isize_cuda[1] * isize_cuda[2] + j * isize_cuda[2] + k;

	if (i < isize_cuda[0] && j < isize_cuda[1] && k < isize_cuda[2]) {
		ScalarType twopi = 2.0 * CUDART_PI;
	    ScalarType R, c;
	    int64_t X, Y, Z;
	    ScalarType r, ratio;

	    ScalarType hx = twopi / n_cuda[0], hy = twopi / n_cuda[1], hz = twopi / n_cuda[2];
	    ScalarType dx = 0, dy = 0, dz = 0, o = 0;

	    // PHI = GAUSSIAN
	    R = sqrt(2.) * sigma;
	    c = 1.;                       // 1./(sigma_ * std::sqrt(2*M_PI));

	    // PHI = BUMP
	    // R = sigma_ * sigma_;
	    // c = exp(1);

	    X = istart_cuda[0] + i;
        Y = istart_cuda[1] + j;
        Z = istart_cuda[2] + k;

        dx = hx * X - xc;
        dy = hy * Y - yc;
        dz = hz * Z - zc;

        // PHI = GAUSSIAN
        r = sqrt(dx*dx + dy*dy + dz*dz);
        ratio = r / R;
        o = c * exp(-ratio * ratio);

        // PHI = BUMP
        // r = sqrt(dx*dx/R + dy*dy/R + dz*dz/R);
        // o = (r < 1.)        ? c * exp(-1./(1.-r*r))   : 0.0;

        out[ptr] = o;
    }
}

__global__ void truncateGaussian (ScalarType *out, ScalarType sigma, ScalarType xc, ScalarType yc, ScalarType zc, int *isize_cuda) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;
	int k = threadIdx.z + blockDim.z * blockIdx.z;

	int64_t ptr = i * isize_cuda[1] * isize_cuda[2] + j * isize_cuda[2] + k;

	if (i < isize_cuda[0] && j < isize_cuda[1] && k < isize_cuda[2]) {
		ScalarType twopi = 2.0 * CUDART_PI;
	    int64_t X, Y, Z;
	    ScalarType r;

	    ScalarType hx = twopi / n_cuda[0], hy = twopi / n_cuda[1], hz = twopi / n_cuda[2];
	    ScalarType dx = 0, dy = 0, dz = 0;

	    X = istart_cuda[0] + i;
        Y = istart_cuda[1] + j;
        Z = istart_cuda[2] + k;

        dx = hx * X - xc;
        dy = hy * Y - yc;
        dz = hz * Z - zc;

        r = sqrt(dx*dx + dy*dy + dz*dz);
        // truncate to zero after radius 5*sigma
        out[ptr] = (r/sigma <= 5) ? out[ptr] : 0.0;
    }
}

void initializeGaussianCuda (ScalarType *out, ScalarType sigma, ScalarType xc, ScalarType yc, ScalarType zc, int *sz) {
	int n_th_x = N_THREADS_X;
	int n_th_y = N_THREADS_Y;
	int n_th_z = N_THREADS_Z;
	dim3 n_threads (n_th_x, n_th_y, n_th_z);
	dim3 n_blocks ((sz[0] + n_th_x - 1) / n_th_x, (sz[1] + n_th_y - 1) / n_th_y, (sz[2] + n_th_z - 1) / n_th_z);

	initializeGaussian <<< n_blocks, n_threads >>> (out, sigma, xc, yc, zc, sz);

	cudaDeviceSynchronize();
	cudaCheckKernelError ();
}

void truncateGaussianCuda (ScalarType *out, ScalarType sigma, ScalarType xc, ScalarType yc, ScalarType zc, int* sz) {
	int n_th_x = N_THREADS_X;
	int n_th_y = N_THREADS_Y;
	int n_th_z = N_THREADS_Z;
	dim3 n_threads (n_th_x, n_th_y, n_th_z);
	dim3 n_blocks ((sz[0] + n_th_x - 1) / n_th_x, (sz[1] + n_th_y - 1) / n_th_y, (sz[2] + n_th_z - 1) / n_th_z);

	truncateGaussian <<< n_blocks, n_threads >>> (out, sigma, xc, yc, zc, sz);

	cudaDeviceSynchronize();
	cudaCheckKernelError ();
}

