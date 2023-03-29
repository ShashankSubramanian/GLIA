#include "UtilsCuda.h"
//#include "Common.cuh"

__constant__ int isize_cuda[3], istart_cuda[3], osize_cuda[3], ostart_cuda[3], n_cuda[3];

void initCudaConstants (int *isize, int *osize, int *istart, int *ostart, int *n) {
	cudaMemcpyToSymbol (isize_cuda, isize, 3 * sizeof(int));
	cudaMemcpyToSymbol (osize_cuda, osize, 3 * sizeof(int));
	cudaMemcpyToSymbol (istart_cuda, istart, 3 * sizeof(int));
	cudaMemcpyToSymbol (ostart_cuda, ostart, 3 * sizeof(int));
	cudaMemcpyToSymbol (n_cuda, n, 3 * sizeof(int));
}

__global__ void hadamardComplexProduct (CudaComplexType *y, ScalarType *x) {
	int64_t i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i < osize_cuda[0] * osize_cuda[1] * osize_cuda[2]) 
		y[i] = cuComplexMultiply (y[i], makeCudaComplexType(x[i], 0.));
}

__global__ void hadamardComplexProduct (CudaComplexType *y, CudaComplexType *x) {
	int64_t i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i < osize_cuda[0] * osize_cuda[1] * osize_cuda[2]) 
		y[i] = cuComplexMultiply (y[i], x[i]);
}

__global__ void computeMagnitude (ScalarType *mag_ptr, ScalarType *x_ptr, ScalarType *y_ptr, ScalarType *z_ptr, int64_t sz) {
	int64_t i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i < sz)  
		mag_ptr[i] = sqrt (x_ptr[i] * x_ptr[i] + y_ptr[i] * y_ptr[i] + z_ptr[i] * z_ptr[i]);
}

__global__ void setCoords (ScalarType *x_ptr, ScalarType *y_ptr, ScalarType *z_ptr) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;
	int k = threadIdx.z + blockDim.z * blockIdx.z;

	int64_t ptr = i * isize_cuda[1] * isize_cuda[2] + j * isize_cuda[2] + k;

	if (ptr < isize_cuda[0] * isize_cuda[1] * isize_cuda[2]) {
		// ScalarType hx, hy, hz;
		// ScalarType twopi = 2. * CUDART_PI;
		// hx = twopi / n_cuda[0];
		// hy = twopi / n_cuda[1];
		// hz = twopi / n_cuda[2];

		x_ptr[ptr] = static_cast<ScalarType> (i + istart_cuda[0]);
        y_ptr[ptr] = static_cast<ScalarType> (j + istart_cuda[1]);
        z_ptr[ptr] = static_cast<ScalarType> (k + istart_cuda[2]);    
    }
}

__global__ void clipVector (ScalarType *x_ptr) {
	int64_t i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i < isize_cuda[0] * isize_cuda[1] * isize_cuda[2]) {
		x_ptr[i] = (x_ptr[i] <= 0.) ? 0. : x_ptr[i];
	}
}

__global__ void clipVectorAbove (ScalarType *x_ptr) {
	int64_t i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i < isize_cuda[0] * isize_cuda[1] * isize_cuda[2]) {
		x_ptr[i] = (x_ptr[i] > 1.) ? 1. : x_ptr[i];
	}
}

void setCoordsCuda (ScalarType *x_ptr, ScalarType *y_ptr, ScalarType *z_ptr, int *sz) {
	int n_th_x = N_THREADS_X;
	int n_th_y = N_THREADS_Y;
	int n_th_z = N_THREADS_Z;
	dim3 n_threads (n_th_x, n_th_y, n_th_z);
	dim3 n_blocks ((sz[0] + n_th_x - 1) / n_th_x, (sz[1] + n_th_y - 1) / n_th_y, (sz[2] + n_th_z - 1) / n_th_z);

	setCoords <<< n_blocks, n_threads >>> (x_ptr, y_ptr, z_ptr);

	cudaDeviceSynchronize();
	cudaCheckKernelError ();
}

void computeMagnitudeCuda (ScalarType *mag_ptr, ScalarType *x_ptr, ScalarType *y_ptr, ScalarType *z_ptr, int64_t sz) {
	int n_th = N_THREADS;

	computeMagnitude <<< (sz + n_th - 1) / n_th, n_th >>> (mag_ptr, x_ptr, y_ptr, z_ptr, sz);

	cudaDeviceSynchronize();
	cudaCheckKernelError ();
}

void hadamardComplexProductCuda (CudaComplexType *y, ScalarType *x, int *sz) {
	int n_th = N_THREADS;

	hadamardComplexProduct <<< ((sz[0] * sz[1] * sz[2]) + n_th - 1)/ n_th, n_th >>> (y, x);

	cudaDeviceSynchronize();
	cudaCheckKernelError ();
}

void hadamardComplexProductCuda (CudaComplexType *y, CudaComplexType *x, int *sz) {
	try	{
		thrust::device_ptr<thrust::complex<ScalarType>> y_thrust, x_thrust;
	    y_thrust = thrust::device_pointer_cast ((thrust::complex<ScalarType>*)y);
	    x_thrust = thrust::device_pointer_cast ((thrust::complex<ScalarType>*)x);

	    thrust::transform(y_thrust, y_thrust + (sz[0] * sz[1] * sz[2]), x_thrust, y_thrust, thrust::multiplies<thrust::complex<ScalarType>>());
	} catch (thrust::system_error &e) {
		std::cerr << "Thrust reduce error: " << e.what() << std::endl;
	}

	cudaDeviceSynchronize();
}

void clipVectorCuda (ScalarType *x_ptr, int64_t sz) {
	int n_th = N_THREADS;

	clipVector <<< (sz + n_th - 1) / n_th, n_th >>> (x_ptr);

	cudaDeviceSynchronize();
	cudaCheckKernelError ();	
}

void clipVectorAboveCuda (ScalarType *x_ptr, int64_t sz) {
	int n_th = N_THREADS;

	clipVectorAbove <<< (sz + n_th - 1) / n_th, n_th >>> (x_ptr);

	cudaDeviceSynchronize();
	cudaCheckKernelError ();	
}

void vecMaxCuda (ScalarType *x, int *loc, ScalarType *val, int sz) {
	// use thrust for vec max
	try {
		thrust::device_ptr<ScalarType> x_thrust;
		x_thrust = thrust::device_pointer_cast (x);
		// find the max itr
		thrust::device_vector<ScalarType>::iterator it = thrust::max_element(x_thrust, x_thrust + sz);
		// find the position
		thrust::device_ptr<ScalarType> max_pos = thrust::device_pointer_cast(&it[0]);
		if (loc != NULL)
			*loc = max_pos - x_thrust;
		*val = *it;
	} catch (thrust::system_error &e) {
		std::cerr << "Thrust vector maximum error: " << e.what() << std::endl;
	}

	cudaDeviceSynchronize();
}

void vecScatterCuda(ScalarType *f, ScalarType *f_scatter, ScalarType *seq, int64_t sz) {
	try {
		thrust::device_ptr<ScalarType> f_thrust;
		f_thrust = thrust::device_pointer_cast (f);
		thrust::device_ptr<ScalarType> f_scat_thrust;
		f_scat_thrust = thrust::device_pointer_cast (f_scatter);
    thrust::device_ptr<ScalarType> seq_thrust;
    seq_thrust = thrust::device_pointer_cast (seq);
		thrust::scatter(f_thrust, f_thrust + sz, seq_thrust, f_scat_thrust);
	} catch (thrust::system_error &e) {
		std::cerr << "Thrust scatter error: " << e.what() << std::endl;
	}

	cudaDeviceSynchronize();

}

void vecSortCuda(ScalarType *f, int64_t sz) {
	// use thrust for sort
	try {
		thrust::device_ptr<ScalarType> f_thrust;
		f_thrust = thrust::device_pointer_cast (f);
//		if (seq == NULL) {
		thrust::sort (f_thrust, f_thrust + sz);
//		} else {
//      thrust::device_ptr<ScalarType> seq_thrust;
//      seq_thrust = thrust::device_pointer_cast (seq);
//		  thrust::sort_by_key(f_thrust, f_thrust + sz, seq_thrust);
//    }
	} catch (thrust::system_error &e) {
		std::cerr << "Thrust sorting error: " << e.what() << std::endl;
	}

	cudaDeviceSynchronize();
}

void setSequenceCuda(ScalarType *f, int64_t sz) {
	// use thrust for to set sequence
	try {
		thrust::device_ptr<ScalarType> f_thrust;
		f_thrust = thrust::device_pointer_cast (f);
		thrust::sequence(f_thrust, f_thrust + sz);
	} catch (thrust::system_error &e) {
		std::cerr << "Thrust sequence set error: " << e.what() << std::endl;
	}

	cudaDeviceSynchronize();
}

void vecSumCuda(ScalarType *f, ScalarType *sum, int64_t sz) {
	// use thrust for reduction
	try {
		thrust::device_ptr<ScalarType> f_thrust;
		f_thrust = thrust::device_pointer_cast (f);
		(*sum) = thrust::reduce (f_thrust, f_thrust + sz);
	} catch (thrust::system_error &e) {
		std::cerr << "Thrust reduce error: " << e.what() << std::endl;
	}

	cudaDeviceSynchronize();
}

__global__ void copyDoubleToFloat(float *dst, double *src, int64_t sz) {
    int64_t i = threadIdx.x + blockDim.x * blockIdx.x;

    if (i < sz) {
        dst[i] = (float)src[i];
    }
}

void copyDoubleToFloatCuda (float *dst, double *src, int64_t sz) {
    int n_th = N_THREADS;

    copyDoubleToFloat <<<  (sz + n_th - 1) / n_th, n_th >>> (dst, src, sz);

    cudaDeviceSynchronize();
    cudaCheckKernelError();
}

__global__ void copyFloatToDouble(double *dst, float *src, int64_t sz) {
        int64_t i = threadIdx.x + blockDim.x * blockIdx.x;

        if (i < sz) {
            dst[i] = (double)src[i];
        }
}

__global__ void computeIndicatorFunction(ScalarType *i_ptr, ScalarType *x_ptr, ScalarType x_star, ScalarType threshold, int64_t sz) {

  int64_t i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < sz) {
    if (abs(x_ptr[i] - x_star) < threshold) {
      i_ptr[i] = 1;
    } else {
      i_ptr[i] = 0;
    }
  }
}
  
void copyFloatToDoubleCuda (double *dst, float *src, int64_t sz) {
    int n_th = N_THREADS;

    copyFloatToDouble <<<  (sz + n_th - 1) / n_th, n_th >>> (dst, src, sz);

    cudaDeviceSynchronize();
    cudaCheckKernelError();
}

void computeIndicatorFunctionCuda(ScalarType *i_ptr, ScalarType *x_ptr, ScalarType x_star, ScalarType threshold, int64_t sz) {
    int n_th = N_THREADS;

    computeIndicatorFunction <<<  (sz + n_th - 1) / n_th, n_th >>> (i_ptr, x_ptr, x_star, threshold, sz);

    cudaDeviceSynchronize();
    cudaCheckKernelError();
}

__global__ void smoothHeavisideFunction(ScalarType *x_ptr, ScalarType *y_ptr, ScalarType shapeFactor int64_t sz) {
  
  int64_t i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < sz) {
    y_ptr[i] = 1 / (1 + std::exp(- shapeFactor * x_ptr[i]))
  }
  
}

void smoothHeavisideFunctionCuda(ScalarType *x_ptr, ScalarType *y_ptr, ScalarType shapeFactor, int64_t sz) {
  int n_th = N_THREADS; 
  
  smoothHeavisideFunction <<< (sz + n_th - 1) / n_th, n_th >>> (x_ptr, y_ptr, shapeFactor, sz); 

  cudaDeviceSynchronize(); 
  cudaCheckKernelError();
  
}

__global__ void multispeciesObsOperator (ScalarType *p_ptr, ScalarType *n_ptr, ScalarType *i_ptr, ScalarType *w_ptr, ScalarType *g_ptr, ScalarType *f_ptr, ScalarType *Oc_ptr, ScalarType *Op_ptr, ScalarType *On_ptr, ScalarType *Ol_ptr, ScalarType shapeFactorEdema, ScalarType shapeFactor, ScalarType thresEdema, int64_t sz} {

  int64_t i = threadIdx.x + blockDim.x * blockIdx.x;

  if (i < sz) {
    ScalarType c = p_ptr[i] + n_ptr[i] + i_ptr[i];
    
    Oc_ptr[i] = (1/(1+std::exp(-shapeFactor * (c - w_ptr[i])))) * 
                (1/(1+std::exp(-shapeFactor * (c - g_ptr[i])))) * 
                (1/(1+std::exp(-shapeFactor * (c - f_ptr[i]))));
    
    Op_ptr[i] = (1/(1+std::exp(-shapeFactor * (p_ptr[i] - n_ptr[i])))) * 
                (1/(1+std::exp(-shapeFactor * (p_ptr[i] - i_ptr[i])))) * 
                Oc_ptr[i];
    
    On_ptr[i] = (1/(1+std::exp(-shapeFactor * (n_ptr[i] - p_ptr[i])))) *
                (1/(1+std::exp(-shapeFactor * (n_ptr[i] - i_ptr[i])))) *
                Oc_ptr[i];
    
    Ol_ptr[i] = (1 - Op_ptr[i] - On_ptr[i]) * (1/(1 + std::exp(-shapeFactorEdema * (i_ptr[i] - thresEdema))))

  } 
}


void multispeciesObsOperatorsCuda (ScalarType *p_ptr, ScalarType *n_ptr, ScalarType *i_ptr, ScalarType *w_ptr, ScalarType *g_ptr, ScalarType *f_ptr, ScalarType *Oc_ptr, ScalarType *Op_ptr, ScalarType *On_ptr, ScalarType *Ol_ptr, ScalarType shapeFactorEdema, ScalarType shapeFactor, ScalarType thresEdema, int64_t sz) {

  int n_th = N_THREADS;
  
  multispeciesObsOperators <<< (sz + n_th - 1) / n_th, n_th >>> (p_ptr, n_ptr, i_ptr, w_ptr, g_ptr, f_ptr, Oc_ptr, Op_ptr, On_ptr, Ol_ptr, shapeFactorEdema, shapeFactor, thresEdema, sz); 

  cudaDeviceSynchronize();
  cudaCheckKernelError();

}


