#include "UtilsCuda.h"

__constant__ int isize_cuda[3], istart_cuda[3], osize_cuda[3], ostart_cuda[3], n_cuda[3];

void initCudaConstants (int *isize, int *osize, int *istart, int *ostart, int *n) {
	cudaMemcpyToSymbol (isize_cuda, isize, 3 * sizeof(int));
	cudaMemcpyToSymbol (osize_cuda, osize, 3 * sizeof(int));
	cudaMemcpyToSymbol (istart_cuda, istart, 3 * sizeof(int));
	cudaMemcpyToSymbol (ostart_cuda, ostart, 3 * sizeof(int));
	cudaMemcpyToSymbol (n_cuda, n, 3 * sizeof(int));
}

__global__ void computeWeierstrassFilter (ScalarType *f, ScalarType sigma) {
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

__global__ void precFactorDiffusion (ScalarType *precfactor, ScalarType *work) {
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

__global__ void logisticReactionLinearized (ScalarType *c_t_ptr, ScalarType *rho_ptr, ScalarType *c_ptr, ScalarType dt) {
	int64_t i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i < isize_cuda[0] * isize_cuda[1] * isize_cuda[2]) {
		ScalarType factor = 0., alph = 0.;
	    factor = exp (rho_ptr[i] * dt);
	    alph = (c_ptr[i] * factor + 1.0 - c_ptr[i]);
	    c_t_ptr[i] = c_t_ptr[i] * factor / (alph * alph);
	}
}

__global__ void logisticReaction (ScalarType *c_t_ptr, ScalarType *rho_ptr, ScalarType *c_ptr, ScalarType dt) {
	int64_t i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i < isize_cuda[0] * isize_cuda[1] * isize_cuda[2]) {
		ScalarType factor = 0., alph = 0.;
	    factor = exp (rho_ptr[i] * dt);
	    alph = c_t_ptr[i] / (1.0 - c_t_ptr[i]);
	    if (isinf(alph)) c_t_ptr[i] = 1.0;
	    else c_t_ptr[i] = alph * factor / (alph * factor + 1.0);
	}
}

// No inplace multiply - be careful
__global__ void multiplyXWaveNumber (CudaComplexType *w_f, CudaComplexType *f) {
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

__global__ void multiplyYWaveNumber (CudaComplexType *w_f, CudaComplexType *f) {
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

__global__ void multiplyZWaveNumber (CudaComplexType *w_f, CudaComplexType *f) {
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

__global__ void computeEulerPoints (ScalarType *query_ptr, ScalarType *vx_ptr, ScalarType *vy_ptr, ScalarType *vz_ptr, ScalarType dt) {
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

__global__ void computeSecondOrderEulerPoints (ScalarType *query_ptr, ScalarType *vx_ptr, ScalarType *vy_ptr, ScalarType *vz_ptr, ScalarType *wx_ptr, ScalarType *wy_ptr, ScalarType *wz_ptr, ScalarType dt) {
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

__global__ void precFactorElasticity (CudaComplexType *ux_hat, CudaComplexType *uy_hat, CudaComplexType *uz_hat, CudaComplexType *fx_hat, CudaComplexType *fy_hat, CudaComplexType *fz_hat, ScalarType lam_avg, ScalarType mu_avg, ScalarType screen_avg) {
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

__global__ void computeMagnitude (ScalarType *mag_ptr, ScalarType *x_ptr, ScalarType *y_ptr, ScalarType *z_ptr, int64_t sz) {
	int64_t i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i < sz)  
		mag_ptr[i] = sqrt (x_ptr[i] * x_ptr[i] + y_ptr[i] * y_ptr[i] + z_ptr[i] * z_ptr[i]);
}

__global__ void nonlinearForceScaling (ScalarType *c_ptr, ScalarType *fx_ptr, ScalarType *fy_ptr, ScalarType *fz_ptr, ScalarType fac, int64_t sz) {
	int64_t i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i < sz) {
		fx_ptr[i] *= fac * tanh (c_ptr[i]);
        fy_ptr[i] *= fac * tanh (c_ptr[i]);
        fz_ptr[i] *= fac * tanh (c_ptr[i]);
	}
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

__global__ void conserveHealthyTissues (ScalarType *gm_ptr, ScalarType *wm_ptr, ScalarType *sum_ptr, ScalarType *scale_gm_ptr, ScalarType *scale_wm_ptr, ScalarType dt) {
	int64_t i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i < isize_cuda[0] * isize_cuda[1] * isize_cuda[2]) {
		scale_gm_ptr[i] = 0.0;
        scale_wm_ptr[i] = 0.0;

        if (gm_ptr[i] > 0.01 || wm_ptr[i] > 0.01) {
            scale_gm_ptr[i] = -1.0 * dt * gm_ptr[i] / (gm_ptr[i] + wm_ptr[i]);
            scale_wm_ptr[i] = -1.0 * dt * wm_ptr[i] / (gm_ptr[i] + wm_ptr[i]);
        }

        scale_gm_ptr[i] = (isnan (scale_gm_ptr[i])) ? 0.0 : scale_gm_ptr[i];
        scale_wm_ptr[i] = (isnan (scale_wm_ptr[i])) ? 0.0 : scale_wm_ptr[i];

        gm_ptr[i] += scale_gm_ptr[i] * sum_ptr[i];
        wm_ptr[i] += scale_wm_ptr[i] * sum_ptr[i];
	}
}


__global__ void computeReactionRate (ScalarType *m_ptr, ScalarType *ox_ptr, ScalarType *rho_ptr, ScalarType ox_inv, ScalarType ox_mit) {
	int64_t i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i < isize_cuda[0] * isize_cuda[1] * isize_cuda[2]) {
		if (ox_ptr[i] > ox_inv) m_ptr[i] = rho_ptr[i];
        else if (ox_ptr[i] <= ox_inv && ox_ptr[i] >= ox_mit) 
            m_ptr[i] = rho_ptr[i] * (ox_ptr[i] - ox_mit) / (ox_inv - ox_mit);
        else
            m_ptr[i] = 0.;
	}
}

__global__ void computeTransition (ScalarType *alpha_ptr, ScalarType *beta_ptr, ScalarType *ox_ptr, ScalarType *p_ptr, ScalarType *i_ptr, ScalarType alpha_0, ScalarType beta_0, ScalarType ox_inv, ScalarType thres) {
	int64_t i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i < isize_cuda[0] * isize_cuda[1] * isize_cuda[2]) {
		alpha_ptr[i] = alpha_0 * 0.5 * (1 + tanh (500 * (ox_inv - ox_ptr[i])));
        beta_ptr[i] = beta_0 * 0.5 * (1 + tanh (500 * (thres - p_ptr[i] - i_ptr[i]))) * ox_ptr[i];
	}
}

__global__ void computeThesholder (ScalarType *h_ptr, ScalarType *ox_ptr, ScalarType ox_hypoxia) {
	int64_t i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i < isize_cuda[0] * isize_cuda[1] * isize_cuda[2]) {
		h_ptr[i] = 0.5 * (1 + tanh (500 * (ox_hypoxia - ox_ptr[i])));
	}
}

__global__ void computeSources (ScalarType *p_ptr, ScalarType *i_ptr, ScalarType *n_ptr, ScalarType *m_ptr, ScalarType *al_ptr, ScalarType *bet_ptr, ScalarType *h_ptr, ScalarType *gm_ptr, ScalarType *wm_ptr, ScalarType *ox_ptr,
						ScalarType * di_ptr, ScalarType dt, ScalarType death_rate, ScalarType ox_source, ScalarType ox_consumption) {
	int64_t i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i < isize_cuda[0] * isize_cuda[1] * isize_cuda[2]) {
		ScalarType p_temp, i_temp, frac_1, frac_2;
	    ScalarType ox_heal = 1.;
	    ScalarType reac_ratio = 0.1;
	    ScalarType death_ratio = 0.3;		

	    p_temp = p_ptr[i]; i_temp = i_ptr[i];
        p_ptr[i] += dt * (m_ptr[i] * p_ptr[i] * (1. - p_ptr[i]) - al_ptr[i] * p_ptr[i] + bet_ptr[i] * i_ptr[i] - 
                            death_rate * h_ptr[i] * p_ptr[i]);
        i_ptr[i] += dt * (reac_ratio * m_ptr[i] * i_ptr[i] * (1. - i_ptr[i]) + al_ptr[i] * p_temp - bet_ptr[i] * i_ptr[i] - 
                            death_ratio * death_rate * h_ptr[i] * i_ptr[i]);
        n_ptr[i] += dt * (h_ptr[i] * death_rate * (p_temp + death_ratio * i_temp + gm_ptr[i] + wm_ptr[i]));
        ox_ptr[i] += dt * (-ox_consumption * p_temp + ox_source * (ox_heal - ox_ptr[i]) * (gm_ptr[i] + wm_ptr[i]));
        // ox_ptr[i] = (ox_ptr[i] <= 0.) ? 0. : ox_ptr[i];

        // conserve healthy cells
        if (gm_ptr[i] > 0.01 || wm_ptr[i] > 0.01) {
            frac_1 = gm_ptr[i] / (gm_ptr[i] + wm_ptr[i]); frac_2 = wm_ptr[i] / (gm_ptr[i] + wm_ptr[i]);
        } else {
            frac_1 = 0.; frac_2 = 0.;
        }
        frac_1 = (isnan(frac_1)) ? 0. : frac_1;
        frac_2 = (isnan(frac_2)) ? 0. : frac_2;
        gm_ptr[i] += -dt * (frac_1 * (m_ptr[i] * p_ptr[i] * (1. - p_ptr[i]) + reac_ratio * m_ptr[i] * i_ptr[i] * (1. - i_ptr[i]) + di_ptr[i])
                         + h_ptr[i] * death_rate * gm_ptr[i]); 
        wm_ptr[i] += -dt * (frac_2 * (m_ptr[i] * p_ptr[i] * (1. - p_ptr[i]) + reac_ratio * m_ptr[i] * i_ptr[i] * (1. - i_ptr[i]) + di_ptr[i])
                         + h_ptr[i] * death_rate * wm_ptr[i]); 
	}
}

__global__ void computeScreening (ScalarType *screen_ptr, ScalarType *c_ptr, ScalarType *bg_ptr, ScalarType screen_low, ScalarType screen_high) {
	int64_t i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i < isize_cuda[0] * isize_cuda[1] * isize_cuda[2]) {
		ScalarType c_threshold = 0.005;
		screen_ptr[i] = (c_ptr[i] >= c_threshold) ? screen_low : screen_high;
		if (bg_ptr[i] > 0.95) screen_ptr[i] = 1E6; // screen out the background completely to ensure no movement
	}
}

__global__ void computeTumorLame (ScalarType *mu_ptr, ScalarType *lam_ptr, ScalarType *c_ptr, ScalarType mu_tumor, ScalarType lam_tumor) {
	int64_t i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i < isize_cuda[0] * isize_cuda[1] * isize_cuda[2]) {
		mu_ptr[i] += (c_ptr[i] > 0) ? (mu_tumor * c_ptr[i]) : 0;
        lam_ptr[i] += (c_ptr[i] > 0) ? (lam_tumor * c_ptr[i]) : 0;
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


__global__ void clipHealthyTissues (ScalarType *gm_ptr, ScalarType *wm_ptr, ScalarType *csf_ptr) {
	int64_t i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i < isize_cuda[0] * isize_cuda[1] * isize_cuda[2]) {
		gm_ptr[i] = (gm_ptr[i] <= 0.) ? 0. : gm_ptr[i];
        wm_ptr[i] = (wm_ptr[i] <= 0.) ? 0. : wm_ptr[i];
        csf_ptr[i] = (csf_ptr[i] <= 0.) ? 0. : csf_ptr[i];
	}
}

__global__ void initializeGaussian (ScalarType *out, ScalarType sigma, ScalarType xc, ScalarType yc, ScalarType zc) {
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

__global__ void truncateGaussian (ScalarType *out, ScalarType sigma, ScalarType xc, ScalarType yc, ScalarType zc) {
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

void nonlinearForceScalingCuda (ScalarType *c_ptr, ScalarType *fx_ptr, ScalarType *fy_ptr, ScalarType *fz_ptr, ScalarType fac, int64_t sz) {
	int n_th = N_THREADS;

	nonlinearForceScaling <<< (sz + n_th - 1) / n_th, n_th >>> (c_ptr, fx_ptr, fy_ptr, fz_ptr, fac, sz);

	cudaDeviceSynchronize();
	cudaCheckKernelError ();
}

void computeMagnitudeCuda (ScalarType *mag_ptr, ScalarType *x_ptr, ScalarType *y_ptr, ScalarType *z_ptr, int64_t sz) {
	int n_th = N_THREADS;

	computeMagnitude <<< (sz + n_th - 1) / n_th, n_th >>> (mag_ptr, x_ptr, y_ptr, z_ptr, sz);

	cudaDeviceSynchronize();
	cudaCheckKernelError ();
}

void precFactorElasticityCuda (CudaComplexType *ux_hat, CudaComplexType *uy_hat, CudaComplexType *uz_hat, CudaComplexType *fx_hat, 
                              CudaComplexType *fy_hat, CudaComplexType *fz_hat, ScalarType lam_avg, ScalarType mu_avg, ScalarType screen_avg, int *sz) {
	int n_th_x = N_THREADS_X;
	int n_th_y = N_THREADS_Y;
	int n_th_z = N_THREADS_Z;
	dim3 n_threads (n_th_x, n_th_y, n_th_z);
	dim3 n_blocks ((sz[0] + n_th_x - 1) / n_th_x, (sz[1] + n_th_y - 1) / n_th_y, (sz[2] + n_th_z - 1) / n_th_z);

	precFactorElasticity <<< n_blocks, n_threads >>> (ux_hat, uy_hat, uz_hat, fx_hat, fy_hat, fz_hat, lam_avg, mu_avg, screen_avg);

	cudaDeviceSynchronize();
	cudaCheckKernelError ();
}

void computeSecondOrderEulerPointsCuda (ScalarType *query_ptr, ScalarType *vx_ptr, ScalarType *vy_ptr, ScalarType *vz_ptr,
          ScalarType *wx_ptr, ScalarType *wy_ptr, ScalarType *wz_ptr, ScalarType dt, int *sz) {
	int n_th_x = N_THREADS_X;
	int n_th_y = N_THREADS_Y;
	int n_th_z = N_THREADS_Z;
	dim3 n_threads (n_th_x, n_th_y, n_th_z);
	dim3 n_blocks ((sz[0] + n_th_x - 1) / n_th_x, (sz[1] + n_th_y - 1) / n_th_y, (sz[2] + n_th_z - 1) / n_th_z);

	computeSecondOrderEulerPoints <<< n_blocks, n_threads >>> (query_ptr, vx_ptr, vy_ptr, vz_ptr, wx_ptr, wy_ptr, wz_ptr, dt);

	cudaDeviceSynchronize();
	cudaCheckKernelError ();
}

void computeEulerPointsCuda (ScalarType *query_ptr, ScalarType *vx_ptr, ScalarType *vy_ptr, ScalarType *vz_ptr, ScalarType dt, int *sz) {
	int n_th_x = N_THREADS_X;
	int n_th_y = N_THREADS_Y;
	int n_th_z = N_THREADS_Z;
	dim3 n_threads (n_th_x, n_th_y, n_th_z);
	dim3 n_blocks ((sz[0] + n_th_x - 1) / n_th_x, (sz[1] + n_th_y - 1) / n_th_y, (sz[2] + n_th_z - 1) / n_th_z);

	computeEulerPoints <<< n_blocks, n_threads >>> (query_ptr, vx_ptr, vy_ptr, vz_ptr, dt);

	cudaDeviceSynchronize();
	cudaCheckKernelError ();
}


void multiplyXWaveNumberCuda (CudaComplexType *w_f, CudaComplexType *f, int *sz) {
	int n_th_x = N_THREADS_X;
	int n_th_y = N_THREADS_Y;
	int n_th_z = N_THREADS_Z;
	dim3 n_threads (n_th_x, n_th_y, n_th_z);
	dim3 n_blocks ((sz[0] + n_th_x - 1) / n_th_x, (sz[1] + n_th_y - 1) / n_th_y, (sz[2] + n_th_z - 1) / n_th_z);

	multiplyXWaveNumber <<< n_blocks, n_threads >>> (w_f, f);

	cudaDeviceSynchronize();
	cudaCheckKernelError ();
}

void multiplyYWaveNumberCuda (CudaComplexType *w_f, CudaComplexType *f, int *sz) {
	int n_th_x = N_THREADS_X;
	int n_th_y = N_THREADS_Y;
	int n_th_z = N_THREADS_Z;
	dim3 n_threads (n_th_x, n_th_y, n_th_z);
	dim3 n_blocks ((sz[0] + n_th_x - 1) / n_th_x, (sz[1] + n_th_y - 1) / n_th_y, (sz[2] + n_th_z - 1) / n_th_z);

	multiplyYWaveNumber <<< n_blocks, n_threads >>> (w_f, f);

	cudaDeviceSynchronize();
	cudaCheckKernelError ();
}

void multiplyZWaveNumberCuda (CudaComplexType *w_f, CudaComplexType *f, int *sz) {
	int n_th_x = N_THREADS_X;
	int n_th_y = N_THREADS_Y;
	int n_th_z = N_THREADS_Z;
	dim3 n_threads (n_th_x, n_th_y, n_th_z);
	dim3 n_blocks ((sz[0] + n_th_x - 1) / n_th_x, (sz[1] + n_th_y - 1) / n_th_y, (sz[2] + n_th_z - 1) / n_th_z);

	multiplyZWaveNumber <<< n_blocks, n_threads >>> (w_f, f);

	cudaDeviceSynchronize();
	cudaCheckKernelError ();
}


void precFactorDiffusionCuda (ScalarType *precfactor, ScalarType *work, int *sz) {
	int n_th_x = N_THREADS_X;
	int n_th_y = N_THREADS_Y;
	int n_th_z = N_THREADS_Z;
	dim3 n_threads (n_th_x, n_th_y, n_th_z);
	dim3 n_blocks ((sz[0] + n_th_x - 1) / n_th_x, (sz[1] + n_th_y - 1) / n_th_y, (sz[2] + n_th_z - 1) / n_th_z);

	precFactorDiffusion <<< n_blocks, n_threads >>> (precfactor, work);

	cudaDeviceSynchronize();
	cudaCheckKernelError ();
}

void computeWeierstrassFilterCuda (ScalarType *f, ScalarType *sum, ScalarType sigma, int *sz) {
	int n_th_x = N_THREADS_X;
	int n_th_y = N_THREADS_Y;
	int n_th_z = N_THREADS_Z;
	dim3 n_threads (n_th_x, n_th_y, n_th_z);
	dim3 n_blocks ((sz[0] + n_th_x - 1) / n_th_x, (sz[1] + n_th_y - 1) / n_th_y, (sz[2] + n_th_z - 1) / n_th_z);

	computeWeierstrassFilter <<< n_blocks, n_threads >>> (f, sigma);

	cudaDeviceSynchronize();
	cudaCheckKernelError ();

	// use thrust for reduction
	try {
		thrust::device_ptr<ScalarType> f_thrust;
		f_thrust = thrust::device_pointer_cast (f);
		(*sum) = thrust::reduce (f_thrust, f_thrust + (sz[0] * sz[1] * sz[2]));
	} catch (thrust::system_error &e) {
		std::cerr << "Thrust reduce error: " << e.what() << std::endl;
	}

	cudaDeviceSynchronize();
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


void logisticReactionCuda (ScalarType *c_t_ptr, ScalarType *rho_ptr, ScalarType *c_ptr, ScalarType dt, int64_t sz, int linearized) {
	int n_th = N_THREADS;

	if (linearized == 0)
		logisticReaction <<< (sz + n_th - 1) / n_th, n_th >>> (c_t_ptr, rho_ptr, c_ptr, dt);
	else
		logisticReactionLinearized <<< (sz + n_th - 1) / n_th, n_th >>> (c_t_ptr, rho_ptr, c_ptr, dt);

	cudaDeviceSynchronize();
	cudaCheckKernelError ();
}

void conserveHealthyTissuesCuda (ScalarType *gm_ptr, ScalarType *wm_ptr, ScalarType *sum_ptr, ScalarType *scale_gm_ptr, ScalarType *scale_wm_ptr, ScalarType dt, int64_t sz) {
	int n_th = N_THREADS;

	conserveHealthyTissues <<< (sz + n_th - 1) / n_th, n_th >>> (gm_ptr, wm_ptr, sum_ptr, scale_gm_ptr, scale_wm_ptr, dt);

	cudaDeviceSynchronize();
	cudaCheckKernelError ();
}

void computeReactionRateCuda (ScalarType *m_ptr, ScalarType *ox_ptr, ScalarType *rho_ptr, ScalarType ox_inv, ScalarType ox_mit, int64_t sz) {
	int n_th = N_THREADS;

	computeReactionRate <<< (sz + n_th - 1) / n_th, n_th >>> (m_ptr, ox_ptr, rho_ptr, ox_inv, ox_mit);

	cudaDeviceSynchronize();
	cudaCheckKernelError ();
}

void computeTransitionCuda (ScalarType *alpha_ptr, ScalarType *beta_ptr, ScalarType *ox_ptr, ScalarType *p_ptr, ScalarType *i_ptr, ScalarType alpha_0, ScalarType beta_0, ScalarType ox_inv, ScalarType thres, int64_t sz) {
	int n_th = N_THREADS;

	computeTransition <<< (sz + n_th - 1) / n_th, n_th >>> (alpha_ptr, beta_ptr, ox_ptr, p_ptr, i_ptr, alpha_0, beta_0, ox_inv, thres);

	cudaDeviceSynchronize();
	cudaCheckKernelError ();
}

void computeThesholderCuda (ScalarType *h_ptr, ScalarType *ox_ptr, ScalarType ox_hypoxia, int64_t sz) {
	int n_th = N_THREADS;

	computeThesholder <<< (sz + n_th - 1) / n_th, n_th >>> (h_ptr, ox_ptr, ox_hypoxia);

	cudaDeviceSynchronize();
	cudaCheckKernelError ();
}

void computeSourcesCuda (ScalarType *p_ptr, ScalarType *i_ptr, ScalarType *n_ptr, ScalarType *m_ptr, ScalarType *al_ptr, ScalarType *bet_ptr, ScalarType *h_ptr, ScalarType *gm_ptr, ScalarType *wm_ptr, ScalarType *ox_ptr, ScalarType * di_ptr, ScalarType dt, ScalarType death_rate, ScalarType ox_source, ScalarType ox_consumption, int64_t sz) {
	int n_th = N_THREADS;

	computeSources <<< (sz + n_th - 1) / n_th, n_th >>> (p_ptr, i_ptr, n_ptr, m_ptr, al_ptr, bet_ptr, h_ptr, gm_ptr, wm_ptr, ox_ptr, di_ptr, dt, death_rate, ox_source, ox_consumption);

	cudaDeviceSynchronize();
	cudaCheckKernelError ();
}

void computeScreeningCuda (ScalarType *screen_ptr, ScalarType *c_ptr, ScalarType *bg_ptr, ScalarType screen_low, ScalarType screen_high, int64_t sz) {
	int n_th = N_THREADS;

	computeScreening <<< (sz + n_th - 1) / n_th, n_th >>> (screen_ptr, c_ptr, bg_ptr, screen_low, screen_high);

	cudaDeviceSynchronize();
	cudaCheckKernelError ();
}

void computeTumorLameCuda (ScalarType *mu_ptr, ScalarType *lam_ptr, ScalarType *c_ptr, ScalarType mu_tumor, ScalarType lam_tumor, int64_t sz) {
	int n_th = N_THREADS;

	computeTumorLame <<< (sz + n_th - 1) / n_th, n_th >>> (mu_ptr, lam_ptr, c_ptr, mu_tumor, lam_tumor);

	cudaDeviceSynchronize();
	cudaCheckKernelError ();
}

void clipHealthyTissuesCuda (ScalarType *gm_ptr, ScalarType *wm_ptr, ScalarType *csf_ptr, int64_t sz) {
	int n_th = N_THREADS;

	clipHealthyTissues <<< (sz + n_th - 1) / n_th, n_th >>> (gm_ptr, wm_ptr, csf_ptr);

	cudaDeviceSynchronize();
	cudaCheckKernelError ();	
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


void initializeGaussianCuda (ScalarType *out, ScalarType sigma, ScalarType xc, ScalarType yc, ScalarType zc, int *sz) {
	int n_th_x = N_THREADS_X;
	int n_th_y = N_THREADS_Y;
	int n_th_z = N_THREADS_Z;
	dim3 n_threads (n_th_x, n_th_y, n_th_z);
	dim3 n_blocks ((sz[0] + n_th_x - 1) / n_th_x, (sz[1] + n_th_y - 1) / n_th_y, (sz[2] + n_th_z - 1) / n_th_z);

	initializeGaussian <<< n_blocks, n_threads >>> (out, sigma, xc, yc, zc);

	cudaDeviceSynchronize();
	cudaCheckKernelError ();
}

void truncateGaussianCuda (ScalarType *out, ScalarType sigma, ScalarType xc, ScalarType yc, ScalarType zc, int* sz) {
	int n_th_x = N_THREADS_X;
	int n_th_y = N_THREADS_Y;
	int n_th_z = N_THREADS_Z;
	dim3 n_threads (n_th_x, n_th_y, n_th_z);
	dim3 n_blocks ((sz[0] + n_th_x - 1) / n_th_x, (sz[1] + n_th_y - 1) / n_th_y, (sz[2] + n_th_z - 1) / n_th_z);

	truncateGaussian <<< n_blocks, n_threads >>> (out, sigma, xc, yc, zc);

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



























