#include "SpectralOperators.h"
__constant__ int n_cuda[3], istart_cuda[3], ostart_cuda[3];

void initSpecOpsCudaConstants(int *n, int *istart, int *ostart) {
	cudaMemcpyToSymbol (istart_cuda, istart, 3 * sizeof(int));
	cudaMemcpyToSymbol (ostart_cuda, ostart, 3 * sizeof(int));
	cudaMemcpyToSymbol (n_cuda, n, 3 * sizeof(int));
}

__global__ void computeWeierstrassFilter (ScalarType *f, ScalarType sigma, int *isize_cuda) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;
	int k = threadIdx.z + blockDim.z * blockIdx.z;

	int64_t ptr = i * isize_cuda[1] * isize_cuda[2] + j * isize_cuda[2] + k;

	if (ptr < isize_cuda[0] * isize_cuda[1] * isize_cuda[2]) {
		ScalarType X, Y, Z, Xp, Yp, Zp, twopi;
		ScalarType hx, hy, hz;
		
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
		
		// exp -> expf 
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

// No inplace multiply - be careful
__global__ void multiplyXWaveNumber (CudaComplexType *w_f, CudaComplexType *f, int *osize_cuda) {
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int z = threadIdx.z + blockDim.z * blockIdx.z;

	int64_t index = x * osize_cuda[1] * osize_cuda[2] + y * osize_cuda[2] + z;

	if (x < osize_cuda[0] && y < osize_cuda[1] && z < osize_cuda[2]) {
		ScalarType factor = 1.0 / (n_cuda[0] * n_cuda[1] * n_cuda[2]);
		int64_t X;
		X = ostart_cuda[0] + x;

	    int64_t wx;
	    wx = X;

	    if (X > n_cuda[0] / 2.0)
	        wx -= n_cuda[0];
	    if (X == n_cuda[0] / 2.0)
	        wx = 0;

	    w_f[index].x = -factor *  wx * f[index].y;
	    w_f[index].y = factor * wx * f[index].x;
	}
}

__global__ void multiplyYWaveNumber (CudaComplexType *w_f, CudaComplexType *f, int *osize_cuda) {
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int z = threadIdx.z + blockDim.z * blockIdx.z;

	int64_t index = x * osize_cuda[1] * osize_cuda[2] + y * osize_cuda[2] + z;

	if (x < osize_cuda[0] && y < osize_cuda[1] && z < osize_cuda[2]) {
		ScalarType factor = 1.0 / (n_cuda[0] * n_cuda[1] * n_cuda[2]);
		int64_t Y;
		Y = ostart_cuda[1] + y;

	    int64_t wy;
	    wy = Y;

	    if (Y > n_cuda[1] / 2.0)
	        wy -= n_cuda[1];
	    if (Y == n_cuda[1] / 2.0)
	        wy = 0;

	    w_f[index].x = -factor *  wy * f[index].y;
	    w_f[index].y = factor * wy * f[index].x;
	}
}

__global__ void multiplyZWaveNumber (CudaComplexType *w_f, CudaComplexType *f, int *osize_cuda) {
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int z = threadIdx.z + blockDim.z * blockIdx.z;

	int64_t index = x * osize_cuda[1] * osize_cuda[2] + y * osize_cuda[2] + z;

	if (x < osize_cuda[0] && y < osize_cuda[1] && z < osize_cuda[2]) {
		ScalarType factor = 1.0 / (n_cuda[0] * n_cuda[1] * n_cuda[2]);
		int64_t Z;
		Z = ostart_cuda[2] + z;

	    int64_t wz;
	    wz = Z;

	    if (Z > n_cuda[2] / 2.0)
	        wz -= n_cuda[2];
	    if (Z == n_cuda[2] / 2.0)
	        wz = 0;

	    w_f[index].x = -factor *  wz * f[index].y;
	    w_f[index].y = factor * wz * f[index].x;
	}
}



void computeWeierstrassFilterCuda (ScalarType *f, ScalarType *sum, ScalarType sigma, int *sz) {
	int n_th_x = N_THREADS_X;
	int n_th_y = N_THREADS_Y;
	int n_th_z = N_THREADS_Z;
	dim3 n_threads (n_th_x, n_th_y, n_th_z);
	dim3 n_blocks ((sz[0] + n_th_x - 1) / n_th_x, (sz[1] + n_th_y - 1) / n_th_y, (sz[2] + n_th_z - 1) / n_th_z);

	computeWeierstrassFilter <<< n_blocks, n_threads >>> (f, sigma, sz);

	cudaDeviceSynchronize();
	cudaCheckKernelError();

    cublasStatus_t status;
    cublasHandle_t handle;
    PetscCUBLASGetHandle (&handle);
    status = cublasSum (handle, sz[0]*sz[1]*sz[2], f, 1, sum);
    cublasCheckError (status);
	// use thrust for reduction
//	try {
//		thrust::device_ptr<ScalarType> f_thrust;
//		f_thrust = thrust::device_pointer_cast (f);
//		(*sum) = thrust::reduce (f_thrust, f_thrust + (sz[0] * sz[1] * sz[2]));
//	} catch (thrust::system_error &e) {
//		std::cerr << "Thrust reduce error: " << e.what() << std::endl;
//	}

	cudaDeviceSynchronize();
}

void multiplyXWaveNumberCuda (CudaComplexType *w_f, CudaComplexType *f, int *sz) {
	int n_th_x = N_THREADS_X;
	int n_th_y = N_THREADS_Y;
	int n_th_z = N_THREADS_Z;
	dim3 n_threads (n_th_x, n_th_y, n_th_z);
	dim3 n_blocks ((sz[0] + n_th_x - 1) / n_th_x, (sz[1] + n_th_y - 1) / n_th_y, (sz[2] + n_th_z - 1) / n_th_z);

	multiplyXWaveNumber <<< n_blocks, n_threads >>> (w_f, f, sz);

	cudaDeviceSynchronize();
	cudaCheckKernelError ();
}

void multiplyYWaveNumberCuda (CudaComplexType *w_f, CudaComplexType *f, int *sz) {
	int n_th_x = N_THREADS_X;
	int n_th_y = N_THREADS_Y;
	int n_th_z = N_THREADS_Z;
	dim3 n_threads (n_th_x, n_th_y, n_th_z);
	dim3 n_blocks ((sz[0] + n_th_x - 1) / n_th_x, (sz[1] + n_th_y - 1) / n_th_y, (sz[2] + n_th_z - 1) / n_th_z);

	multiplyYWaveNumber <<< n_blocks, n_threads >>> (w_f, f, sz);

	cudaDeviceSynchronize();
	cudaCheckKernelError ();
}

void multiplyZWaveNumberCuda (CudaComplexType *w_f, CudaComplexType *f, int *sz) {
	int n_th_x = N_THREADS_X;
	int n_th_y = N_THREADS_Y;
	int n_th_z = N_THREADS_Z;
	dim3 n_threads (n_th_x, n_th_y, n_th_z);
	dim3 n_blocks ((sz[0] + n_th_x - 1) / n_th_x, (sz[1] + n_th_y - 1) / n_th_y, (sz[2] + n_th_z - 1) / n_th_z);

	multiplyZWaveNumber <<< n_blocks, n_threads >>> (w_f, f, sz);

	cudaDeviceSynchronize();
	cudaCheckKernelError ();
}

