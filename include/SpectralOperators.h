#ifndef _SPECTRALOPERATORS_H
#define _SPECTRALOPERATORS_H

#include "Utils.h"

class SpectralOperators {
	public:
		SpectralOperators (int fft_mode = ACCFFT) {
			fft_mode_ = fft_mode;
		}

		int fft_mode_;
		int *isize_, *istart_, *osize_, *ostart_, *n_;
		fft_plan *plan_;

		#ifdef CUDA
			cufftHandle plan_r2c_;
			cufftHandle plan_c2r_;
		#endif
		int64_t alloc_max_;

		ComplexType *x_hat_, *wx_hat_;
		ScalarType *d1_ptr_, *d2_ptr_;

		void setup (int *n, int *isize, int *istart, int *osize, int *ostart, MPI_Comm c_comm);
		void executeFFTR2C (ScalarType *f, ComplexType *f_hat);
		void executeFFTC2R (ComplexType *f_hat, ScalarType *f);

		PetscErrorCode computeGradient (Vec grad_x, Vec grad_y, Vec grad_z, Vec x, std::bitset<3> *pXYZ, ScalarType *timers);
		PetscErrorCode computeDivergence (Vec div, Vec dx, Vec dy, Vec dz, ScalarType *timers);

		PetscErrorCode weierstrassSmoother (Vec Wc, Vec c, std::shared_ptr<NMisc> n_misc, ScalarType sigma);
		int weierstrassSmoother (ScalarType * Wc, ScalarType *c, std::shared_ptr<NMisc> n_misc, ScalarType sigma);

		~SpectralOperators ();
};


#endif