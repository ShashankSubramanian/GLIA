#include "UtilsCuda.h"


__global__ void computeWeierstrassFilter (double *f, double *s, double sigma, 
	int *isize, int *istart, int *n) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;
	int k = threadIdx.z + blockDim.z * blockIdx.z;

	double X, Y, Z, Xp, Yp, Zp, twopi;
	int hx, hy, hz;
	int64_t ptr;
	twopi = 2. * CUDART_PI;
	hx = twopi / n[0];
	hy = twopi / n[1];
	hz = twopi / n[2];

	X = (istart[0] + i) * hx;
	Xp = X - twopi;
	Y = (istart[1] + j) * hy;
	Yp = Y - twopi;
	Z = (istart[2] + k) * hz;
	Zp = Z - twopi;
	ptr = i * isize[1] * isize[2] + j * isize[2] + k;
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
	s += f[ptr];
}

__global__ void hadamardComplexProduct (std::complex<double> *y, std::complex<double> *x, double *alph) {
	int i = threadIdx.x;
	y[i] *= (x[i] * (*alph));
}

void computeWeierstrassFilterCuda (double *f, double *s, double sigma, int *isize, int *istart, int *n) {
	int n_th_x = 32;
	int n_th_y = 8;
	int n_th_z = 1;

	dim3 n_threads (n_th_x, n_th_y, n_th_z);
	dim3 n_blocks (isize[0] / n_th_x, isize[1] / n_th_y, isize[2] / n_th_z);
	computeWeierstrassFilter <<< n_blocks, n_threads >>> (f, s, sigma, isize, istart, n);
}

void hadamardComplexProductCuda (std::complex<double> *y, std::complex<double> *x, double *alph, int *osize) {
	int n_th = 512;
	hadamardComplexProduct <<< (osize[0] * osize[1] * osize[2]) / n_th, n_th >>> (y, x, alph);
}

