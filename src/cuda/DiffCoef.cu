#include "DiffCoef.h"

__constant__ int n_cuda[3], ostart_cuda[3];

void initDiffCoefCudaConstants(int *n, int *ostart) {
	cudaMemcpyToSymbol (ostart_cuda, ostart, 3 * sizeof(int));
	cudaMemcpyToSymbol (n_cuda, n, 3 * sizeof(int));
}

__global__ void precFactorDiffusion (ScalarType *precfactor, ScalarType *work, int *osize_cuda) {
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int z = threadIdx.z + blockDim.z * blockIdx.z;

	int64_t index = x * osize_cuda[1] * osize_cuda[2] + y * osize_cuda[2] + z;

	if (x < osize_cuda[0] && y < osize_cuda[1] && z < osize_cuda[2]) {
		ScalarType factor = 1.0 / (n_cuda[0] * n_cuda[1] * n_cuda[2]);

		ScalarType X, Y, Z;
		X = ostart_cuda[0] + x;
	    Y = ostart_cuda[1] + y;
	    Z = ostart_cuda[2] + z;

	    ScalarType wx, wy, wz;
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

	    ScalarType dt = work[0];
	    ScalarType kxx_avg = work[1];
	    ScalarType kxy_avg = work[2];
	    ScalarType kxz_avg = work[3];
	    ScalarType kyz_avg = work[4];
	    ScalarType kyy_avg = work[5];
	    ScalarType kzz_avg = work[6];

	    
	    precfactor[index] = (1. + 0.25 * dt * (kxx_avg * wx * wx + 2.0 * kxy_avg * wx * wy
	                            + 2.0 * kxz_avg * wx * wz + 2.0 * kyz_avg * wy * wz + kyy_avg * wy * wy
	                                            + kzz_avg * wz *wz));
	    if (precfactor[index] == 0)
	        precfactor[index] = factor;
	    else
	        precfactor[index] = factor / precfactor[index];
	}
}


void precFactorDiffusionCuda (ScalarType *precfactor, ScalarType *work, int *sz) {
	int n_th_x = N_THREADS_X;
	int n_th_y = N_THREADS_Y;
	int n_th_z = N_THREADS_Z;
	dim3 n_threads (n_th_x, n_th_y, n_th_z);
	dim3 n_blocks ((sz[0] + n_th_x - 1) / n_th_x, (sz[1] + n_th_y - 1) / n_th_y, (sz[2] + n_th_z - 1) / n_th_z);

	precFactorDiffusion <<< n_blocks, n_threads >>> (precfactor, work, sz);

	cudaDeviceSynchronize();
	cudaCheckKernelError ();
}
