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

	int64_t ptr = i * isize_cuda[1] * isize_cuda[2] + j * isize_cuda[2] + k;

	if (ptr < isize_cuda[0] * isize_cuda[1] * isize_cuda[2]) {
		double X, Y, Z, Xp, Yp, Zp, twopi;
		double hx, hy, hz;
		
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
}

__global__ void hadamardComplexProduct (cuDoubleComplex *y, double *x) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i < osize_cuda[0] * osize_cuda[1] * osize_cuda[2]) 
		y[i] = cuCmul (y[i], make_cuDoubleComplex(x[i], 0.));
}

__global__ void hadamardComplexProduct (cuDoubleComplex *y, cuDoubleComplex *x) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i < osize_cuda[0] * osize_cuda[1] * osize_cuda[2]) 
		y[i] = cuCmul (y[i], x[i]);
}

__global__ void precFactorDiffusion (double *precfactor, double *work) {
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int z = threadIdx.z + blockDim.z * blockIdx.z;

	int64_t index = x * osize_cuda[1] * osize_cuda[2] + y * osize_cuda[2] + z;

	if (index < osize_cuda[0] * osize_cuda[1] * osize_cuda[2]) {
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

	    
	    precfactor[index] = (1. + 0.25 * dt * (kxx_avg * wx * wx + 2.0 * kxy_avg * wx * wy
	                            + 2.0 * kxz_avg * wx * wz + 2.0 * kyz_avg * wy * wz + kyy_avg * wy * wy
	                                            + kzz_avg * wz *wz));
	    if (precfactor[index] == 0)
	        precfactor[index] = factor;
	    else
	        precfactor[index] = factor / precfactor[index];
	}
}

__global__ void logisticReactionLinearized (double *c_t_ptr, double *rho_ptr, double *c_ptr, double dt) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i < isize_cuda[0] * isize_cuda[1] * isize_cuda[2]) {
		double factor = 0., alph = 0.;
	    factor = exp (rho_ptr[i] * dt);
	    alph = (c_ptr[i] * factor + 1.0 - c_ptr[i]);
	    c_t_ptr[i] = c_t_ptr[i] * factor / (alph * alph);
	}
}

__global__ void logisticReaction (double *c_t_ptr, double *rho_ptr, double *c_ptr, double dt) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i < isize_cuda[0] * isize_cuda[1] * isize_cuda[2]) {
		double factor = 0., alph = 0.;
	    factor = exp (rho_ptr[i] * dt);
	    alph = (1.0 - c_t_ptr[i]) / c_t_ptr[i];
	    c_t_ptr[i] = factor / (factor + alph);
	}
}

__global__ void multiplyXWaveNumber (cuDoubleComplex *w_f, cuDoubleComplex *f) {
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int z = threadIdx.z + blockDim.z * blockIdx.z;

	int64_t index = x * osize_cuda[1] * osize_cuda[2] + y * osize_cuda[2] + z;

	if (index < osize_cuda[0] * osize_cuda[1] * osize_cuda[2]) {
		double factor = 1.0 / (n_cuda[0] * n_cuda[1] * n_cuda[2]);
		double X;
		X = ostart_cuda[0] + x;

	    double wx;
	    wx = X;

	    if (X > n_cuda[0] / 2.0)
	        wx -= n_cuda[0];
	    if (X == n_cuda[0] / 2.0)
	        wx = 0;

	    w_f[index].x = -factor *  wx * f[index].y;
	    w_f[index].y = factor * wx * f[index].x;
	}
}

__global__ void multiplyYWaveNumber (cuDoubleComplex *w_f, cuDoubleComplex *f) {
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int z = threadIdx.z + blockDim.z * blockIdx.z;

	int64_t index = x * osize_cuda[1] * osize_cuda[2] + y * osize_cuda[2] + z;

	if (index < osize_cuda[0] * osize_cuda[1] * osize_cuda[2]) {
		double factor = 1.0 / (n_cuda[0] * n_cuda[1] * n_cuda[2]);
		double Y;
		Y = ostart_cuda[1] + y;

	    double wy;
	    wy = Y;

	    if (Y > n_cuda[1] / 2.0)
	        wy -= n_cuda[1];
	    if (Y == n_cuda[1] / 2.0)
	        wy = 0;

	    w_f[index].x = -factor *  wy * f[index].y;
	    w_f[index].y = factor * wy * f[index].x;
	}
}

__global__ void multiplyZWaveNumber (cuDoubleComplex *w_f, cuDoubleComplex *f) {
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int z = threadIdx.z + blockDim.z * blockIdx.z;

	int64_t index = x * osize_cuda[1] * osize_cuda[2] + y * osize_cuda[2] + z;

	if (index < osize_cuda[0] * osize_cuda[1] * osize_cuda[2]) {
		double factor = 1.0 / (n_cuda[0] * n_cuda[1] * n_cuda[2]);
		double Z;
		Z = ostart_cuda[2] + z;

	    double wz;
	    wz = Z;

	    if (Z > n_cuda[2] / 2.0)
	        wz -= n_cuda[2];
	    if (Z == n_cuda[2] / 2.0)
	        wz = 0;

	    w_f[index].x = -factor *  wz * f[index].y;
	    w_f[index].y = factor * wz * f[index].x;
	}
}

__global__ void computeEulerPoints (double *query_ptr, double *vx_ptr, double *vy_ptr, double *vz_ptr, double dt) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;
	int k = threadIdx.z + blockDim.z * blockIdx.z;

	int64_t ptr = i * isize_cuda[1] * isize_cuda[2] + j * isize_cuda[2] + k;

	if (ptr < isize_cuda[0] * isize_cuda[1] * isize_cuda[2]) {
		double hx, hy, hz, x1, x2, x3;
		double twopi = 2. * CUDART_PI;
		hx = 1. / n_cuda[0];
		hy = 1. / n_cuda[1];
		hz = 1. / n_cuda[2];

		x1 = hx * static_cast<double> (i + istart_cuda[0]);
        x2 = hy * static_cast<double> (j + istart_cuda[1]);
        x3 = hz * static_cast<double> (k + istart_cuda[2]);

        dt /= twopi;

        // compute the Euler points: xstar = x - dt * vel.
        // coords are normalized - requirement from interpolation
        query_ptr[ptr * 3 + 0] = (x1 - dt * vx_ptr[ptr]);   
        query_ptr[ptr * 3 + 1] = (x2 - dt * vy_ptr[ptr]);   
        query_ptr[ptr * 3 + 2] = (x3 - dt * vz_ptr[ptr]);  
    }
}

__global__ void computeSecondOrderEulerPoints (double *query_ptr, double *vx_ptr, double *vy_ptr, double *vz_ptr, double *wx_ptr, double *wy_ptr, double *wz_ptr, double dt) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;
	int k = threadIdx.z + blockDim.z * blockIdx.z;

	int64_t ptr = i * isize_cuda[1] * isize_cuda[2] + j * isize_cuda[2] + k;

	if (ptr < isize_cuda[0] * isize_cuda[1] * isize_cuda[2]) {
		double hx, hy, hz, x1, x2, x3;
		double twopi = 2. * CUDART_PI;
		hx = 1. / n_cuda[0];
		hy = 1. / n_cuda[1];
		hz = 1. / n_cuda[2];

		x1 = hx * static_cast<double> (i + istart_cuda[0]);
        x2 = hy * static_cast<double> (j + istart_cuda[1]);
        x3 = hz * static_cast<double> (k + istart_cuda[2]);

        dt /= twopi;

        // compute query points
        query_ptr[ptr * 3 + 0] = (x1 - 0.5 * dt * (vx_ptr[ptr] + wx_ptr[ptr]));   
        query_ptr[ptr * 3 + 1] = (x2 - 0.5 * dt * (vy_ptr[ptr] + wy_ptr[ptr]));   
        query_ptr[ptr * 3 + 2] = (x3 - 0.5 * dt * (vz_ptr[ptr] + wz_ptr[ptr]));    
    }
}

__global__ void precFactorElasticity (cuDoubleComplex *ux_hat, cuDoubleComplex *uy_hat, cuDoubleComplex *uz_hat, cuDoubleComplex *fx_hat, cuDoubleComplex *fy_hat, cuDoubleComplex *fz_hat, double lam_avg, double mu_avg, double screen_avg) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;
	int k = threadIdx.z + blockDim.z * blockIdx.z;

	int64_t ptr = i * osize_cuda[1] * osize_cuda[2] + j * osize_cuda[2] + k;

	if (ptr < osize_cuda[0] * osize_cuda[1] * osize_cuda[2]) {
		double s1, s2, s1_square, s3, scale;
	    int64_t wx, wy, wz;
	    double wTw, wTf_real, wTf_imag;
	    int64_t x_global, y_global, z_global;

	    double factor = 1.0 / (n_cuda[0] * n_cuda[1] * n_cuda[2]);
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

__global__ void computeMagnitude (double *mag_ptr, double *x_ptr, double *y_ptr, double *z_ptr, int sz) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i < sz)  
		mag_ptr[i] = sqrt (x_ptr[i] * x_ptr[i] + y_ptr[i] * y_ptr[i] + z_ptr[i] * z_ptr[i]);
}

__global__ void nonlinearForceScaling (double *c_ptr, double *fx_ptr, double *fy_ptr, double *fz_ptr, double fac, int sz) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i < sz) {
		fx_ptr[i] *= fac * tanh (c_ptr[i]);
        fy_ptr[i] *= fac * tanh (c_ptr[i]);
        fz_ptr[i] *= fac * tanh (c_ptr[i]);
	}
}

__global__ void setCoords (double *x_ptr, double *y_ptr, double *z_ptr) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;
	int k = threadIdx.z + blockDim.z * blockIdx.z;

	int64_t ptr = i * isize_cuda[1] * isize_cuda[2] + j * isize_cuda[2] + k;

	if (ptr < isize_cuda[0] * isize_cuda[1] * isize_cuda[2]) {
		double hx, hy, hz;
		hx = 1. / n_cuda[0];
		hy = 1. / n_cuda[1];
		hz = 1. / n_cuda[2];

		x_ptr[ptr] = hx * static_cast<double> (i + istart_cuda[0]);
        y_ptr[ptr] = hy * static_cast<double> (j + istart_cuda[1]);
        z_ptr[ptr] = hz * static_cast<double> (k + istart_cuda[2]);    
    }
}

void setCoordsCuda (double *x_ptr, double *y_ptr, double *z_ptr, int *sz) {
	int n_th_x = N_THREADS_X;
	int n_th_y = N_THREADS_Y;
	int n_th_z = N_THREADS_Z;
	dim3 n_threads (n_th_x, n_th_y, n_th_z);
	dim3 n_blocks (std::ceil(sz[0] / n_th_x), std::ceil(sz[1] / n_th_y), std::ceil(sz[2] / n_th_z));

	setCoords <<< n_blocks, n_threads >>> (x_ptr, y_ptr, z_ptr);

	cudaDeviceSynchronize();
	cudaCheckKernelError ();
}

void nonlinearForceScalingCuda (double *c_ptr, double *fx_ptr, double *fy_ptr, double *fz_ptr, double fac, int sz) {
	int n_th = N_THREADS;

	nonlinearForceScaling <<< std::ceil(sz / n_th), n_th >>> (c_ptr, fx_ptr, fy_ptr, fz_ptr, fac, sz);

	cudaDeviceSynchronize();
	cudaCheckKernelError ();
}

void computeMagnitudeCuda (double *mag_ptr, double *x_ptr, double *y_ptr, double *z_ptr, int sz) {
	int n_th = N_THREADS;

	computeMagnitude <<< std::ceil(sz / n_th), n_th >>> (mag_ptr, x_ptr, y_ptr, z_ptr, sz);

	cudaDeviceSynchronize();
	cudaCheckKernelError ();
}

void precFactorElasticityCuda (cuDoubleComplex *ux_hat, cuDoubleComplex *uy_hat, cuDoubleComplex *uz_hat, cuDoubleComplex *fx_hat, 
                              cuDoubleComplex *fy_hat, cuDoubleComplex *fz_hat, double lam_avg, double mu_avg, double screen_avg, int *sz) {
	int n_th_x = N_THREADS_X;
	int n_th_y = N_THREADS_Y;
	int n_th_z = N_THREADS_Z;
	dim3 n_threads (n_th_x, n_th_y, n_th_z);
	dim3 n_blocks (std::ceil(sz[0] / n_th_x), std::ceil(sz[1] / n_th_y), std::ceil(sz[2] / n_th_z));

	precFactorElasticity <<< n_blocks, n_threads >>> (ux_hat, uy_hat, uz_hat, fx_hat, fy_hat, fz_hat, lam_avg, mu_avg, screen_avg);

	cudaDeviceSynchronize();
	cudaCheckKernelError ();
}

void computeSecondOrderEulerPointsCuda (double *query_ptr, double *vx_ptr, double *vy_ptr, double *vz_ptr,
          double *wx_ptr, double *wy_ptr, double *wz_ptr, double dt, int *sz) {
	int n_th_x = N_THREADS_X;
	int n_th_y = N_THREADS_Y;
	int n_th_z = N_THREADS_Z;
	dim3 n_threads (n_th_x, n_th_y, n_th_z);
	dim3 n_blocks (std::ceil(sz[0] / n_th_x), std::ceil(sz[1] / n_th_y), std::ceil(sz[2] / n_th_z));

	computeSecondOrderEulerPoints <<< n_blocks, n_threads >>> (query_ptr, vx_ptr, vy_ptr, vz_ptr, wx_ptr, wy_ptr, wz_ptr, dt);

	cudaDeviceSynchronize();
	cudaCheckKernelError ();
}

void computeEulerPointsCuda (double *query_ptr, double *vx_ptr, double *vy_ptr, double *vz_ptr, double dt, int *sz) {
	int n_th_x = N_THREADS_X;
	int n_th_y = N_THREADS_Y;
	int n_th_z = N_THREADS_Z;
	dim3 n_threads (n_th_x, n_th_y, n_th_z);
	dim3 n_blocks (std::ceil(sz[0] / n_th_x), std::ceil(sz[1] / n_th_y), std::ceil(sz[2] / n_th_z));

	computeEulerPoints <<< n_blocks, n_threads >>> (query_ptr, vx_ptr, vy_ptr, vz_ptr, dt);

	cudaDeviceSynchronize();
	cudaCheckKernelError ();
}


void multiplyXWaveNumberCuda (cuDoubleComplex *w_f, cuDoubleComplex *f, int *sz) {
	int n_th_x = N_THREADS_X;
	int n_th_y = N_THREADS_Y;
	int n_th_z = N_THREADS_Z;
	dim3 n_threads (n_th_x, n_th_y, n_th_z);
	dim3 n_blocks (std::ceil(sz[0] / n_th_x), std::ceil(sz[1] / n_th_y), std::ceil(sz[2] / n_th_z));

	multiplyXWaveNumber <<< n_blocks, n_threads >>> (w_f, f);

	cudaDeviceSynchronize();
	cudaCheckKernelError ();
}

void multiplyYWaveNumberCuda (cuDoubleComplex *w_f, cuDoubleComplex *f, int *sz) {
	int n_th_x = N_THREADS_X;
	int n_th_y = N_THREADS_Y;
	int n_th_z = N_THREADS_Z;
	dim3 n_threads (n_th_x, n_th_y, n_th_z);
	dim3 n_blocks (std::ceil(sz[0] / n_th_x), std::ceil(sz[1] / n_th_y), std::ceil(sz[2] / n_th_z));

	multiplyYWaveNumber <<< n_blocks, n_threads >>> (w_f, f);

	cudaDeviceSynchronize();
	cudaCheckKernelError ();
}

void multiplyZWaveNumberCuda (cuDoubleComplex *w_f, cuDoubleComplex *f, int *sz) {
	int n_th_x = N_THREADS_X;
	int n_th_y = N_THREADS_Y;
	int n_th_z = N_THREADS_Z;
	dim3 n_threads (n_th_x, n_th_y, n_th_z);
	dim3 n_blocks (std::ceil(sz[0] / n_th_x), std::ceil(sz[1] / n_th_y), std::ceil(sz[2] / n_th_z));

	multiplyZWaveNumber <<< n_blocks, n_threads >>> (w_f, f);

	cudaDeviceSynchronize();
	cudaCheckKernelError ();
}


void precFactorDiffusionCuda (double *precfactor, double *work, int *sz) {
	int n_th_x = N_THREADS_X;
	int n_th_y = N_THREADS_Y;
	int n_th_z = N_THREADS_Z;
	dim3 n_threads (n_th_x, n_th_y, n_th_z);
	dim3 n_blocks (std::ceil(sz[0] / n_th_x), std::ceil(sz[1] / n_th_y), std::ceil(sz[2] / n_th_z));

	precFactorDiffusion <<< n_blocks, n_threads >>> (precfactor, work);

	cudaDeviceSynchronize();
	cudaCheckKernelError ();
}

void computeWeierstrassFilterCuda (double *f, double *sum, double sigma, int *sz) {
	int n_th_x = N_THREADS_X;
	int n_th_y = N_THREADS_Y;
	int n_th_z = N_THREADS_Z;
	dim3 n_threads (n_th_x, n_th_y, n_th_z);
	dim3 n_blocks (std::ceil(sz[0] / n_th_x), std::ceil(sz[1] / n_th_y), std::ceil(sz[2] / n_th_z));

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

	hadamardComplexProduct <<< std::ceil((sz[0] * sz[1] * sz[2]) / n_th), n_th >>> (y, x);

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


void logisticReactionCuda (double *c_t_ptr, double *rho_ptr, double *c_ptr, double dt, int sz, int linearized) {
	int n_th = N_THREADS;

	if (linearized == 0)
		logisticReaction <<< std::ceil(sz / n_th), n_th >>> (c_t_ptr, rho_ptr, c_ptr, dt);
	else
		logisticReactionLinearized <<< std::ceil(sz / n_th), n_th >>> (c_t_ptr, rho_ptr, c_ptr, dt);

	cudaDeviceSynchronize();
	cudaCheckKernelError ();
}
