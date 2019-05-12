#include "UtilsCuda.h"

__global__ void computeWeierstrassFilter (double *f, double *s, double sigma) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;
	int k = threadIdx.z + blockDim.z * blockIdx.z;

	double X, Y, Z, Xp, Yp, Zp, twopi;
	int hx, hy, hz;
	int64_t ptr;
	twopi = 2. * CUDART_PI;
	hx = twopi / n_cuda[0];
	hy = twopi / n_cuda[1];
	hz = twopi / n_cuda[2];

	X = (istart_cuda[0] + i) * hx;
	Xp = X - twopi;
	Y = (istart_cuda[1] + j) * hy;
	Yp = Y - twopi;
	Z = (istart_cuda[2] + k) * hz;
	Zp = Z - twopi;
	ptr = i * isize_cuda[1] * isize_cuda[2] + j * isize_cuda[2] + k;
	f[ptr] = exp((-X * X - Y * Y - Z * Z) / sigma / sigma / 2.0)
			+ exp((-Xp * Xp - Yp * Yp - Zp * Zp) / sigma / sigma / 2.0);

	f[ptr] += exp((-Xp * Xp - Y * Y - Z * Z) / sigma / sigma / 2.0)
			+ exp((-X * X - Yp * Yp - Z * Z) / sigma / sigma / 2.0);

	f[ptr] += exp((-X * X - Y * Y - Zp * Zp) / sigma / sigma / 2.0)
			+ exp((-Xp * Xp - Yp * Yp - Z * Z) / sigma / sigma / 2.0);

	f[ptr] += exp((-Xp * Xp - Y * Y - Zp * Zp) / sigma / sigma / 2.0)
			+ exp((-X * X - Yp * Yp - Zp * Zp) / sigma / sigma / 2.0);

	if (f[ptr] != f[ptr])
		f[ptr] = 0.; // To avoid Nan
	(*s) += f[ptr];
}

__global__ void hadamardComplexProduct (cuDoubleComplex *y, cuDoubleComplex *x) {
	int i = threadIdx.x;
	y[i] = cuCmul (y[i], x[i]);
}

__global__ void precFactorDiffusion (double *precfactor, double *work) {
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int z = threadIdx.z + blockDim.z * blockIdx.z;

	double X, Y, Z;
	X = ostart_cuda[0] + x;
    Y = ostart_cuda[1] + y;
    Z = ostart_cuda[2] + z;

    double wx, wy, wz;
    wx = X;
    wy = Y;
    wz = Z;

    if (X > n_cuda[0] / 2.0)
        wx -= n_cuda[0];
    if (X == n_cuda[0] / 2.0)
        wx = 0;

    if (Y > n_cuda[1] / 2.0)
        wy -= n_cuda[1];
    if (Y == n_cuda[1] / 2.0)
        wy = 0;

    if (Z > n_cuda[2] / 2.0)
        wz -= n_cuda[2];
    if (Z == n_cuda[2] / 2.0)
        wz = 0;

    double dt = work[0];
    double kxx_avg = work[1];
    double kxy_avg = work[2];
    double kxz_avg = work[3];
    double kyz_avg = work[4];
    double kyy_avg = work[5];
    double kzz_avg = work[6];
    double factor = work[7];
    
    int64_t index = x * osize_cuda[1] * osize_cuda[2] + y * osize_cuda[2] + z;
    precfactor[index] = (1. + 0.25 * dt * (kxx_avg * wx * wx + 2.0 * kxy_avg * wx * wy
                            + 2.0 * kxz_avg * wx * wz + 2.0 * kyz_avg * wy * wz + kyy_avg * wy * wy
                                            + kzz_avg * wz *wz));
    if (precfactor[index] == 0)
        precfactor[index] = factor;
    else
        precfactor[index] = factor / precfactor[index];
}

void precFactorDiffusionCuda (double *precfactor, double *work) {
	int n_th_x = 32;
	int n_th_y = 8;
	int n_th_z = 1;
	dim3 n_threads (n_th_x, n_th_y, n_th_z);
	dim3 n_blocks (osize_cuda[0] / n_th_x, osize_cuda[1] / n_th_y, osize_cuda[2] / n_th_z);

	precFactorDiffusion <<< n_blocks, n_threads >>> (precfactor, work);

	cudaDeviceSynchronize();
	cudaCheckKernelError ();
}

void computeWeierstrassFilterCuda (double *f, double *s, double sigma) {
	int n_th_x = 32;
	int n_th_y = 8;
	int n_th_z = 1;
	dim3 n_threads (n_th_x, n_th_y, n_th_z);
	dim3 n_blocks (isize_cuda[0] / n_th_x, isize_cuda[1] / n_th_y, isize_cuda[2] / n_th_z);

	computeWeierstrassFilter <<< n_blocks, n_threads >>> (f, s, sigma);

	cudaDeviceSynchronize();
	cudaCheckKernelError ();
}

void hadamardComplexProductCuda (cuDoubleComplex *y, cuDoubleComplex *x) {
	int n_th = 512;

	hadamardComplexProduct <<< (osize_cuda[0] * osize_cuda[1] * osize_cuda[2]) / n_th, n_th >>> (y, x);

	cudaDeviceSynchronize();
	cudaCheckKernelError ();
}

