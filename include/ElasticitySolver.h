#ifndef ELASTICITYSOLVER_H_
#define ELASTICITYSOLVER_H_

#include "Utils.h"
#include "Tumor.h"
#include "SpectralOperators.h"
#include <mpi.h>
#include <omp.h>

struct CtxElasticity {
	CtxElasticity (std::shared_ptr<NMisc> n_misc, std::shared_ptr<Tumor> tumor, std::shared_ptr<SpectralOperators> spec_ops) : 
	n_misc_ (n_misc), tumor_ (tumor), spec_ops_ (spec_ops) {
		PetscErrorCode ierr = 0;
		ierr = VecDuplicate (tumor_->mat_prop_->gm_, &mu_); 
		ierr = VecDuplicate (mu_, &lam_);
		ierr = VecDuplicate (mu_, &screen_);

		temp_.resize (3);
		for (int i = 0; i < 3; i++) {
			ierr = VecDuplicate (mu_, &temp_[i]);
			ierr = VecSet (temp_[i], 0.);
		}

		ierr = VecSet (mu_, 0.); 
		ierr = VecSet (lam_, 0.);
		ierr = VecSet (screen_, 0.);
	}
	std::shared_ptr<NMisc> n_misc_;
	std::shared_ptr<Tumor> tumor_;
	std::shared_ptr<SpectralOperators> spec_ops_;

	PetscScalar mu_avg_, lam_avg_, screen_avg_;

	Complex *fx_hat_, *fy_hat_, *fz_hat_;
	Complex *ux_hat_, *uy_hat_, *uz_hat_;

	Vec mu_;
	Vec lam_;
	Vec screen_;
	std::vector<Vec> temp_;

	double computeMu (double E, double nu) {
		return (E / (2 * (1 + nu)));
	}

	double computeLam (double E, double nu) {
		return (nu * E / ((1 + nu) * (1 - 2 * nu)));
	}

	~CtxElasticity () {
		PetscErrorCode ierr = 0;
		ierr = VecDestroy (&mu_);
		ierr = VecDestroy (&lam_);
		ierr = VecDestroy (&screen_);

		for (int i = 0; i < 3; i++) {
			ierr = VecDestroy (&temp_[i]);
		}
		temp_.clear ();

		fft_free (ux_hat_);
	    fft_free (uy_hat_);
	    fft_free (uz_hat_);
	    fft_free (fx_hat_);
	    fft_free (fy_hat_);
	    fft_free (fz_hat_);
	}
};

class ElasticitySolver {
	public:
		ElasticitySolver (std::shared_ptr<NMisc> n_misc, std::shared_ptr<Tumor> tumor, std::shared_ptr<SpectralOperators> spec_ops);   //tumor is needed for its work vectors

		KSP ksp_;
		Mat A_;
		Vec rhs_;

		PC pc_;

		std::shared_ptr<CtxElasticity> ctx_;

		virtual PetscErrorCode solve (std::shared_ptr<VecField> displacement, std::shared_ptr<VecField> rhs) = 0;

		virtual ~ElasticitySolver ();

};

class VariableLinearElasticitySolver : public ElasticitySolver {
	public:
		VariableLinearElasticitySolver (std::shared_ptr<NMisc> n_misc, std::shared_ptr<Tumor> tumor, std::shared_ptr<SpectralOperators> spec_ops) : ElasticitySolver (n_misc, tumor, spec_ops) {}
		PetscErrorCode computeMaterialProperties ();
		virtual PetscErrorCode solve (std::shared_ptr<VecField> displacement, std::shared_ptr<VecField> rhs);
		virtual ~VariableLinearElasticitySolver () {}
};

//Helper functions 
PetscErrorCode operatorConstantCoefficients (PC pc, Vec x, Vec y); // this is used as a preconditioner 
PetscErrorCode operatorVariableCoefficients (Mat A, Vec x, Vec y);
PetscErrorCode operatorCreateVecsElas (Mat A, Vec *left, Vec *right);

#endif
