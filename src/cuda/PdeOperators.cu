#include "PdeOperators.h"


__global__ void logisticReactionLinearized (ScalarType *c_t_ptr, ScalarType *rho_ptr, ScalarType *c_ptr, ScalarType dt, int64_t sz) {
	int64_t i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i < sz) {
		ScalarType factor = 0., alph = 0.;
	    factor = exp (rho_ptr[i] * dt);
	    alph = (c_ptr[i] * factor + 1.0 - c_ptr[i]);
	    c_t_ptr[i] = c_t_ptr[i] * factor / (alph * alph);
	}
}

__global__ void logisticReaction (ScalarType *c_t_ptr, ScalarType *rho_ptr, ScalarType *c_ptr, ScalarType dt, int64_t sz) {
	int64_t i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i < sz) {
		ScalarType factor = 0., alph = 0.;
	    factor = exp (rho_ptr[i] * dt);
	    alph = c_t_ptr[i] / (1.0 - c_t_ptr[i]);
	    if (isinf(alph)) c_t_ptr[i] = 1.0;
	    else c_t_ptr[i] = alph * factor / (alph * factor + 1.0);
	}
}

__global__ void conserveHealthyTissues (ScalarType *gm_ptr, ScalarType *wm_ptr, ScalarType *sum_ptr, ScalarType *scale_gm_ptr, ScalarType *scale_wm_ptr, ScalarType dt, int64_t sz) {
	int64_t i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i < sz) {
		ScalarType threshold = 1E-3;
		scale_gm_ptr[i] = 0;
        scale_wm_ptr[i] = 0;

        if ((gm_ptr[i] > threshold || wm_ptr[i] > threshold) && (wm_ptr[i] + gm_ptr[i] > threshold)) {
            scale_gm_ptr[i] = -dt * gm_ptr[i] / (gm_ptr[i] + wm_ptr[i]);
            scale_wm_ptr[i] = -dt * wm_ptr[i] / (gm_ptr[i] + wm_ptr[i]);
        }

        gm_ptr[i] += scale_gm_ptr[i] * sum_ptr[i];
        wm_ptr[i] += scale_wm_ptr[i] * sum_ptr[i];
        // wm_ptr[i] = wm_ptr[i] * exp(-sum_ptr[i] * dt);
	}
}

/*
__global__ void computeReactionRate (ScalarType *m_ptr, ScalarType *ox_ptr, ScalarType *rho_ptr, ScalarType ox_hypoxia, int64_t sz) {
	int64_t i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i < sz) {
		// if (ox_ptr[i] > ox_inv) m_ptr[i] = rho_ptr[i];
  //       else if (ox_ptr[i] <= ox_inv && ox_ptr[i] >= ox_mit) 
  //           m_ptr[i] = rho_ptr[i] * (ox_ptr[i] - ox_mit) / (ox_inv - ox_mit);
  //       else
  //           m_ptr[i] = 0.;
		m_ptr[i] = rho_ptr[i] * (1 / (1 + exp(-100 * (ox_ptr[i] - ox_hypoxia))));
	}
}
*/


__global__ void computeReactionRate (ScalarType *m_ptr, ScalarType *ox_ptr, ScalarType *rho_ptr, ScalarType ox_hypoxia, int64_t sz, ScalarType ox_inv) {
  int64_t i = threadIdx.x + blockDim.x * blockIdx.x;

  ScalarType ox_mit = (ox_hypoxia + ox_inv)/2;
  if (i < sz) {
    if (ox_ptr[i] > ox_inv) m_ptr[i] = rho_ptr[i];
    if (ox_ptr[i] <= ox_inv && ox_ptr[i] >= ox_mit) m_ptr[i] = rho_ptr[i] * (ox_ptr[i] - ox_mit) / (ox_inv - ox_mit);
    if (ox_ptr[i] <= ox_mit) m_ptr[i] = 0;
    //m_ptr[i] = rho_ptr[i] * (1 / (1 + exp(-100 * (ox_ptr[i] - ox_hypoxia))));
  }
}


__global__ void computeTransition (ScalarType *alpha_ptr, ScalarType *beta_ptr, ScalarType *ox_ptr, ScalarType *p_ptr, ScalarType *i_ptr, ScalarType alpha_0, ScalarType beta_0, ScalarType ox_inv, ScalarType sigma_b, int64_t sz) {
	int64_t i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i < sz) {
		// alpha_ptr[i] = alpha_0 * 0.5 * (1 + tanh (500 * (ox_inv - ox_ptr[i])));
  //       beta_ptr[i] = beta_0 * 0.5 * (1 + tanh (500 * (thres - p_ptr[i] - i_ptr[i]))) * ox_ptr[i];
	//	alpha_ptr[i] = alpha_0 * (1 / (1 + exp(100 * (ox_ptr[i] - ox_inv))));
  //      beta_ptr[i] = beta_0 * ox_ptr[i];
  //alpha_ptr[i] = alpha_0 * (1 / (1 + exp(100 * (ox_ptr[i] - ox_inv))));
  //beta_ptr[i] = beta_0 * ox_ptr[i] * (1 / (1 + exp(100 * (i_ptr[i] + p_ptr[i] - sigma_b))));
  alpha_ptr[i] = alpha_0 * (1 / (1 + exp(100 * (ox_ptr[i] - ox_inv))));
  //beta_ptr[i] = beta_0 * ox_ptr[i] ;
  beta_ptr[i] = beta_0 * (1 / (1 + exp(100 * (ox_inv - ox_ptr[i])))) ;
	}
}

__global__ void computeThesholder (ScalarType *h_ptr, ScalarType *ox_ptr, ScalarType ox_hypoxia, int64_t sz) {
	int64_t i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i < sz) {
		// h_ptr[i] = 0.5 * (1 + tanh (500 * (ox_hypoxia - ox_ptr[i])));
		h_ptr[i] = (1 / (1 + exp(100 * (ox_ptr[i] - ox_hypoxia))));
	}
}

__global__ void computeSources (ScalarType *p_ptr, ScalarType *i_ptr, ScalarType *n_ptr, ScalarType *m_ptr, ScalarType *al_ptr, ScalarType *bet_ptr, ScalarType *h_ptr, ScalarType *gm_ptr, ScalarType *wm_ptr, ScalarType *ox_ptr, ScalarType * di_ptr, ScalarType dt, ScalarType death_rate, ScalarType ox_source, ScalarType ox_consumption, int64_t sz) {
	int64_t i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i < sz) {
		ScalarType p_temp, i_temp, frac_1, frac_2;
	    ScalarType ox_heal = 1;
	    ScalarType reac_ratio = 1;
	    ScalarType death_ratio = 1;		


	    p_temp = p_ptr[i]; i_temp = i_ptr[i];

        //p_ptr[i] += dt * (m_ptr[i] * p_ptr[i] * (1. - p_ptr[i]) - al_ptr[i] * p_ptr[i] + bet_ptr[i] * i_ptr[i] - 
        //                    death_rate * h_ptr[i] * p_ptr[i]);
        p_ptr[i] += dt * (m_ptr[i] * p_temp * (1. - p_temp) - al_ptr[i] * p_temp + bet_ptr[i] * i_temp - 
                            death_rate * h_ptr[i] * p_temp);
        if (p_ptr[i] < 0.0) p_ptr[i] = 0.0;
        if (p_ptr[i] > 1.0) p_ptr[i] = 1.0;

        //i_ptr[i] += dt * (reac_ratio * m_ptr[i] * i_ptr[i] * (1. - i_ptr[i]) + al_ptr[i] * p_temp - bet_ptr[i] * i_ptr[i] - 
        //                    death_ratio * death_rate * h_ptr[i] * i_ptr[i]);
        i_ptr[i] += dt * (reac_ratio * m_ptr[i] * i_temp * (1. - i_temp) + al_ptr[i] * p_temp - bet_ptr[i] * i_temp - 
                            death_ratio * death_rate * h_ptr[i] * i_temp);
        if (i_ptr[i] < 0.0) i_ptr[i] = 0.0;
        if (i_ptr[i] > 1.0) i_ptr[i] = 1.0;

        n_ptr[i] += dt * (h_ptr[i] * death_rate * (p_temp + death_ratio * i_temp + gm_ptr[i] + wm_ptr[i]));
        if (n_ptr[i] < 0.0) n_ptr[i] = 0.0;
        if (n_ptr[i] > 1.0) n_ptr[i] = 1.0;

        ox_ptr[i] += dt * (-ox_consumption * p_temp + ox_source * (ox_heal - ox_ptr[i]) * (gm_ptr[i] + wm_ptr[i]));
        ox_ptr[i] = (ox_ptr[i] <= 0.) ? 0. : ox_ptr[i];
        if (ox_ptr[i] < 0.0) ox_ptr[i] = 0.0;
        if (ox_ptr[i] > 1.0) ox_ptr[i] = 1.0;

        // conserve healthy cells
        if (gm_ptr[i] > 0.01 || wm_ptr[i] > 0.01) {
            frac_1 = gm_ptr[i] / (gm_ptr[i] + wm_ptr[i]); frac_2 = wm_ptr[i] / (gm_ptr[i] + wm_ptr[i]);
        } else {
            frac_1 = 0.; frac_2 = 0.;
        }
        frac_1 = (isnan(frac_1)) ? 0. : frac_1;
        frac_2 = (isnan(frac_2)) ? 0. : frac_2;
        gm_ptr[i] += -dt * (frac_1 * (m_ptr[i] * p_temp * (1. - p_temp) + reac_ratio * m_ptr[i] * i_temp * (1. - i_temp) + di_ptr[i])
                         + h_ptr[i] * death_rate * gm_ptr[i]); 

        if (gm_ptr[i] < 0.0) gm_ptr[i] = 0.0;
        if (gm_ptr[i] > 1.0) gm_ptr[i] = 1.0;

        wm_ptr[i] += -dt * (frac_2 * (m_ptr[i] * p_temp * (1. - p_temp) + reac_ratio * m_ptr[i] * i_temp * (1. - i_temp) + di_ptr[i])
                         + h_ptr[i] * death_rate * wm_ptr[i]);
        if (wm_ptr[i] < 0.0) wm_ptr[i] = 0.0;
        if (wm_ptr[i] > 1.0) wm_ptr[i] = 1.0;
        
	}
}

__global__ void updateReacAndDiffCoefficients (ScalarType *rho_ptr, ScalarType *k_ptr, ScalarType *bg_ptr, ScalarType *gm_ptr, ScalarType *vt_ptr, ScalarType *csf_ptr, ScalarType rho, ScalarType k,  ScalarType gm_r_scale, ScalarType gm_k_scale, int64_t sz) {
	int64_t i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < sz) {
		ScalarType temp;
    temp = (1 - (bg_ptr[i] + gm_r_scale * gm_ptr[i] + vt_ptr[i] + csf_ptr[i]));
    temp = (temp < 0) ? 0 : temp;
    rho_ptr[i] = temp * rho;
    temp = (1 - (bg_ptr[i] + gm_k_scale * gm_ptr[i] + vt_ptr[i] + csf_ptr[i]));
    temp = (temp < 0) ? 0 : temp;
    k_ptr[i] = temp * k;
	}
}




void logisticReactionCuda (ScalarType *c_t_ptr, ScalarType *rho_ptr, ScalarType *c_ptr, ScalarType dt, int64_t sz, int linearized) {
	int n_th = N_THREADS;

	if (linearized == 0)
		logisticReaction <<< (sz + n_th - 1) / n_th, n_th >>> (c_t_ptr, rho_ptr, c_ptr, dt, sz);
	else
		logisticReactionLinearized <<< (sz + n_th - 1) / n_th, n_th >>> (c_t_ptr, rho_ptr, c_ptr, dt, sz);

	cudaDeviceSynchronize();
	cudaCheckKernelError ();
}

void conserveHealthyTissuesCuda (ScalarType *gm_ptr, ScalarType *wm_ptr, ScalarType *sum_ptr, ScalarType *scale_gm_ptr, ScalarType *scale_wm_ptr, ScalarType dt, int64_t sz) {
	int n_th = N_THREADS;

	conserveHealthyTissues <<< (sz + n_th - 1) / n_th, n_th >>> (gm_ptr, wm_ptr, sum_ptr, scale_gm_ptr, scale_wm_ptr, dt, sz);

	cudaDeviceSynchronize();
	cudaCheckKernelError ();
}

void computeReactionRateCuda (ScalarType *m_ptr, ScalarType *ox_ptr, ScalarType *rho_ptr, ScalarType ox_hypoxia, int64_t sz, ScalarType ox_inv) {
	int n_th = N_THREADS;

	computeReactionRate <<< (sz + n_th - 1) / n_th, n_th >>> (m_ptr, ox_ptr, rho_ptr, ox_hypoxia, sz, ox_inv);

	cudaDeviceSynchronize();
	cudaCheckKernelError ();
}

void computeTransitionCuda (ScalarType *alpha_ptr, ScalarType *beta_ptr, ScalarType *ox_ptr, ScalarType *p_ptr, ScalarType *i_ptr, ScalarType alpha_0, ScalarType beta_0, ScalarType ox_inv, ScalarType sigma_b, int64_t sz) {
	int n_th = N_THREADS;

	computeTransition <<< (sz + n_th - 1) / n_th, n_th >>> (alpha_ptr, beta_ptr, ox_ptr, p_ptr, i_ptr, alpha_0, beta_0, ox_inv, sigma_b, sz);

	cudaDeviceSynchronize();
	cudaCheckKernelError ();
}

void computeThesholderCuda (ScalarType *h_ptr, ScalarType *ox_ptr, ScalarType ox_hypoxia, int64_t sz) {
	int n_th = N_THREADS;

	computeThesholder <<< (sz + n_th - 1) / n_th, n_th >>> (h_ptr, ox_ptr, ox_hypoxia, sz);

	cudaDeviceSynchronize();
	cudaCheckKernelError ();
}

void computeSourcesCuda (ScalarType *p_ptr, ScalarType *i_ptr, ScalarType *n_ptr, ScalarType *m_ptr, ScalarType *al_ptr, ScalarType *bet_ptr, ScalarType *h_ptr, ScalarType *gm_ptr, ScalarType *wm_ptr, ScalarType *ox_ptr, ScalarType * di_ptr, ScalarType dt, ScalarType death_rate, ScalarType ox_source, ScalarType ox_consumption, int64_t sz) {
	int n_th = N_THREADS;

	computeSources <<< (sz + n_th - 1) / n_th, n_th >>> (p_ptr, i_ptr, n_ptr, m_ptr, al_ptr, bet_ptr, h_ptr, gm_ptr, wm_ptr, ox_ptr, di_ptr, dt, death_rate, ox_source, ox_consumption, sz);

	cudaDeviceSynchronize();
	cudaCheckKernelError ();
}

void updateReacAndDiffCoefficientsCuda (ScalarType *rho_ptr, ScalarType *k_ptr, ScalarType *bg_ptr, ScalarType *gm_ptr, ScalarType *vt_ptr, ScalarType *csf_ptr, ScalarType rho, ScalarType k, ScalarType gm_r_scale, ScalarType gm_k_scale, int64_t sz) {
	int n_th = N_THREADS;

	updateReacAndDiffCoefficients <<< (sz + n_th - 1) / n_th, n_th >>> (rho_ptr, k_ptr, bg_ptr, gm_ptr, vt_ptr, csf_ptr, rho, k, gm_r_scale, gm_k_scale, sz);

	cudaDeviceSynchronize ();
	cudaCheckKernelError ();
}


