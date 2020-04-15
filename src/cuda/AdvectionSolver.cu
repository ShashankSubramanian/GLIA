#include "AdvectionSolver.h"

__constant__ int n_cuda[3], istart_cuda[3], ostart_cuda[3];

void initAdvectionCudaConstants(int *n, int *istart, int *ostart) {
	cudaMemcpyToSymbol (istart_cuda, istart, 3 * sizeof(int));
	cudaMemcpyToSymbol (ostart_cuda, ostart, 3 * sizeof(int));
	cudaMemcpyToSymbol (n_cuda, n, 3 * sizeof(int));
}

__global__ void computeEulerPoints (ScalarType *query_ptr, ScalarType *vx_ptr, ScalarType *vy_ptr, ScalarType *vz_ptr, ScalarType dt, int *isize_cuda) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;
	int k = threadIdx.z + blockDim.z * blockIdx.z;

	int64_t ptr = i * isize_cuda[1] * isize_cuda[2] + j * isize_cuda[2] + k;

	if (ptr < isize_cuda[0] * isize_cuda[1] * isize_cuda[2]) {
		ScalarType hx, hy, hz, x1, x2, x3;
		ScalarType twopi = 2. * CUDART_PI;
		hx = 1. / n_cuda[0];
		hy = 1. / n_cuda[1];
		hz = 1. / n_cuda[2];

		x1 = hx * static_cast<ScalarType> (i + istart_cuda[0]);
        x2 = hy * static_cast<ScalarType> (j + istart_cuda[1]);
        x3 = hz * static_cast<ScalarType> (k + istart_cuda[2]);

        dt /= twopi;

        // compute the Euler points: xstar = x - dt * vel.
        // coords are normalized - requirement from interpolation
        query_ptr[ptr * 3 + 0] = (x1 - dt * vx_ptr[ptr]);   
        query_ptr[ptr * 3 + 1] = (x2 - dt * vy_ptr[ptr]);   
        query_ptr[ptr * 3 + 2] = (x3 - dt * vz_ptr[ptr]);  
    }
}

__global__ void computeSecondOrderEulerPoints (ScalarType *query_ptr, ScalarType *vx_ptr, ScalarType *vy_ptr, ScalarType *vz_ptr, ScalarType *wx_ptr, ScalarType *wy_ptr, ScalarType *wz_ptr, ScalarType dt, int *isize_cuda) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;
	int k = threadIdx.z + blockDim.z * blockIdx.z;

	int64_t ptr = i * isize_cuda[1] * isize_cuda[2] + j * isize_cuda[2] + k;

	if (ptr < isize_cuda[0] * isize_cuda[1] * isize_cuda[2]) {
		ScalarType hx, hy, hz, x1, x2, x3;
		ScalarType twopi = 2. * CUDART_PI;
		hx = 1. / n_cuda[0];
		hy = 1. / n_cuda[1];
		hz = 1. / n_cuda[2];

		x1 = hx * static_cast<ScalarType> (i + istart_cuda[0]);
        x2 = hy * static_cast<ScalarType> (j + istart_cuda[1]);
        x3 = hz * static_cast<ScalarType> (k + istart_cuda[2]);

        dt /= twopi;

        // compute query points
        query_ptr[ptr * 3 + 0] = (x1 - 0.5 * dt * (vx_ptr[ptr] + wx_ptr[ptr]));   
        query_ptr[ptr * 3 + 1] = (x2 - 0.5 * dt * (vy_ptr[ptr] + wy_ptr[ptr]));   
        query_ptr[ptr * 3 + 2] = (x3 - 0.5 * dt * (vz_ptr[ptr] + wz_ptr[ptr]));    
    }
}


void computeSecondOrderEulerPointsCuda (ScalarType *query_ptr, ScalarType *vx_ptr, ScalarType *vy_ptr, ScalarType *vz_ptr, ScalarType *wx_ptr, ScalarType *wy_ptr, ScalarType *wz_ptr, ScalarType dt, int *sz) {
	int n_th_x = N_THREADS_X;
	int n_th_y = N_THREADS_Y;
	int n_th_z = N_THREADS_Z;
	dim3 n_threads (n_th_x, n_th_y, n_th_z);
	dim3 n_blocks ((sz[0] + n_th_x - 1) / n_th_x, (sz[1] + n_th_y - 1) / n_th_y, (sz[2] + n_th_z - 1) / n_th_z);

	computeSecondOrderEulerPoints <<< n_blocks, n_threads >>> (query_ptr, vx_ptr, vy_ptr, vz_ptr, wx_ptr, wy_ptr, wz_ptr, dt, sz);

	cudaDeviceSynchronize();
	cudaCheckKernelError ();
}

void computeEulerPointsCuda (ScalarType *query_ptr, ScalarType *vx_ptr, ScalarType *vy_ptr, ScalarType *vz_ptr, ScalarType dt, int *sz) {
	int n_th_x = N_THREADS_X;
	int n_th_y = N_THREADS_Y;
	int n_th_z = N_THREADS_Z;
	dim3 n_threads (n_th_x, n_th_y, n_th_z);
	dim3 n_blocks ((sz[0] + n_th_x - 1) / n_th_x, (sz[1] + n_th_y - 1) / n_th_y, (sz[2] + n_th_z - 1) / n_th_z);

	computeEulerPoints <<< n_blocks, n_threads >>> (query_ptr, vx_ptr, vy_ptr, vz_ptr, dt, sz);

	cudaDeviceSynchronize();
	cudaCheckKernelError ();
}
