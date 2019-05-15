#include "UtilsCuda.h"

__constant__ int isize_cuda[3], istart_cuda[3], osize_cuda[3], ostart_cuda[3], n_cuda[3];

void initCudaConstants (int *isize, int *osize, int *istart, int *ostart, int *n) {
	cudaMemcpyToSymbol (isize_cuda, isize, 3 * sizeof(int));
	cudaMemcpyToSymbol (osize_cuda, osize, 3 * sizeof(int));
	cudaMemcpyToSymbol (istart_cuda, istart, 3 * sizeof(int));
	cudaMemcpyToSymbol (ostart_cuda, ostart, 3 * sizeof(int));
	cudaMemcpyToSymbol (n_cuda, n, 3 * sizeof(int));
}

__global__ void computeWeierstrassFilter (double *f, double sigma) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;
	int k = threadIdx.z + blockDim.z * blockIdx.z;

	double X, Y, Z, Xp, Yp, Zp, twopi;
	double hx, hy, hz;
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
}

__global__ void hadamardComplexProduct (cuDoubleComplex *y, double *x) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	y[i] = cuCmul (y[i], make_cuDoubleComplex(x[i], 0.));
}

__global__ void hadamardComplexProduct (cuDoubleComplex *y, cuDoubleComplex *x) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	y[i] = cuCmul (y[i], x[i]);
}

__global__ void precFactorDiffusion (double *precfactor, double *work) {
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int z = threadIdx.z + blockDim.z * blockIdx.z;

	double factor = 1.0 / (n_cuda[0] * n_cuda[1] * n_cuda[2]);

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

    int64_t index = x * osize_cuda[1] * osize_cuda[2] + y * osize_cuda[2] + z;
    precfactor[index] = (1. + 0.25 * dt * (kxx_avg * wx * wx + 2.0 * kxy_avg * wx * wy
                            + 2.0 * kxz_avg * wx * wz + 2.0 * kyz_avg * wy * wz + kyy_avg * wy * wy
                                            + kzz_avg * wz *wz));
    if (precfactor[index] == 0)
        precfactor[index] = factor;
    else
        precfactor[index] = factor / precfactor[index];
}

void precFactorDiffusionCuda (double *precfactor, double *work, int *sz) {
	int n_th_x = N_THREADS_X;
	int n_th_y = N_THREADS_Y;
	int n_th_z = N_THREADS_Z;
	dim3 n_threads (n_th_x, n_th_y, n_th_z);
	dim3 n_blocks (sz[0] / n_th_x, sz[1] / n_th_y, sz[2] / n_th_z);

	precFactorDiffusion <<< n_blocks, n_threads >>> (precfactor, work);

	cudaDeviceSynchronize();
	cudaCheckKernelError ();
}

void computeWeierstrassFilterCuda (double *f, double *sum, double sigma, int *sz) {
	int n_th_x = N_THREADS_X;
	int n_th_y = N_THREADS_Y;
	int n_th_z = N_THREADS_Z;
	dim3 n_threads (n_th_x, n_th_y, n_th_z);
	dim3 n_blocks (sz[0] / n_th_x, sz[1] / n_th_y, sz[2] / n_th_z);

	computeWeierstrassFilter <<< n_blocks, n_threads >>> (f, sigma);

	cudaDeviceSynchronize();
	cudaCheckKernelError ();

	// use thrust for reduction
	try {
		thrust::device_ptr<double> f_thrust;
		f_thrust = thrust::device_pointer_cast (f);
		(*sum) = thrust::reduce (f_thrust, f_thrust + (sz[0] * sz[1] * sz[2]));
	} catch (thrust::system_error &e) {
		std::cerr << "Thrust reduce error: " << e.what() << std::endl;
	}

	cudaDeviceSynchronize();
}

void hadamardComplexProductCuda (cuDoubleComplex *y, double *x, int *sz) {
	int n_th = N_THREADS;

	hadamardComplexProduct <<< (sz[0] * sz[1] * sz[2]) / n_th, n_th >>> (y, x);

	cudaDeviceSynchronize();
	cudaCheckKernelError ();
}

void hadamardComplexProductCuda (cuDoubleComplex *y, cuDoubleComplex *x, int *sz) {
	try	{
		thrust::device_ptr<thrust::complex<double>> y_thrust, x_thrust;
	    y_thrust = thrust::device_pointer_cast ((thrust::complex<double>*)y);
	    x_thrust = thrust::device_pointer_cast ((thrust::complex<double>*)x);

	    thrust::transform(y_thrust, y_thrust + (sz[0] * sz[1] * sz[2]), x_thrust, y_thrust, thrust::multiplies<thrust::complex<double>>());
	} catch (thrust::system_error &e) {
		std::cerr << "Thrust reduce error: " << e.what() << std::endl;
	}

	cudaDeviceSynchronize();
}

__global__ void logisticReactionLinearized (double *c_t_ptr, double *rho_ptr, double *c_ptr, double dt) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	double factor = 0., alph = 0.;

    factor = exp (rho_ptr[i] * dt);
    alph = (c_ptr[i] * factor + 1.0 - c_ptr[i]);
    c_t_ptr[i] = c_t_ptr[i] * factor / (alph * alph);
    
}

__global__ void logisticReaction (double *c_t_ptr, double *rho_ptr, double *c_ptr, double dt) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	double factor = 0., alph = 0.;

    factor = exp (rho_ptr[i] * dt);
    alph = (1.0 - c_t_ptr[i]) / c_t_ptr[i];
    c_t_ptr[i] = factor / (factor + alph);
    
}

void logisticReactionCuda (double *c_t_ptr, double *rho_ptr, double *c_ptr, double dt, int sz, int linearized) {
	int n_th = N_THREADS;

	if (linearized == 0)
		logisticReaction <<< sz / n_th, n_th >>> (c_t_ptr, rho_ptr, c_ptr, dt);
	else
		logisticReactionLinearized <<< sz / n_th, n_th >>> (c_t_ptr, rho_ptr, c_ptr, dt);

	cudaDeviceSynchronize();
	cudaCheckKernelError ();
}
