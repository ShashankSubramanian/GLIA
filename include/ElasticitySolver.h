#ifndef ELASTICITYSOLVER_H_
#define ELASTICITYSOLVER_H_

#include "Utils.h"
#include "Tumor.h"
#include <mpi.h>
#include <omp.h>

struct CtxElasticity {
	CtxElasticity (std::shared_ptr<NMisc> n_misc, std::shared_ptr<Tumor> tumor) : n_misc_ (n_misc), tumor_ (tumor) {
		PetscErrorCode ierr = 0;
		ierr = VecDuplicate (tumor_->mat_prop_->gm_, &mu_); 
		ierr = VecDuplicate (mu, &lam_);
		ierr = VecDuplicate (mu, &screen_);
		ierr = VecDuplicate (mu, &temp_);

		ierr = VecSet (mu_, 0.); 
		ierr = VecSet (lam_, 0.);
		ierr = VecSet (screen_, 0.);
		ierr = VecSet (temp_, 0.);
	}
	std::shared_ptr<NMisc> n_misc_;
	std::shared_ptr<Tumor> tumor_;

	PetscScalar mu_avg_, lam_avg_, screen_avg_;

	Vec mu_;
	Vec lam_;
	Vec screen_;
	Vec temp_;

	double computeMu (double E, double nu) {
		return (E / (2 * (1 + nu)));
	}

	double computeLam (double E, double nu) {
		return (nu * E / ((1 + nu) * (1 - 2 * nu)));
	}

	~CtxElasticity () {
		ierr = VecDestroy (&mu_);
		ierr = VecDestroy (&lam_);
		ierr = VecDestroy (&screen_);
		ierr = VecDestroy (&temp_);
	}
};

class ElasticitySolver {
	public:
		ElasticitySolver (std::shared_ptr<NMisc> n_misc, std::shared_ptr<Tumor> tumor);   //tumor is needed for its work vectors

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
		VariableLinearElasticitySolver (std::shared_ptr<NMisc> n_misc, std::shared_ptr<Tumor> tumor) : ElasticitySolver (n_misc, tumor) {}
		PetscErrorCode computeMaterialProperties ();
		virtual PetscErrorCode solve (std::shared_ptr<VecField> displacement, std::shared_ptr<VecField> rhs);
		virtual ~VariableLinearElasticitySolver () {}
};

//Helper functions 
PetscErrorCode operatorConstantCoefficients (PC pc, Vec x, Vec y); // this is used as a preconditioner 
PetscErrorCode operatorVariableCoefficients (Mat A, Vec x, Vec y);

#endif
