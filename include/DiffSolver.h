#ifndef DIFFSOLVER_H_
#define DIFFSOLVER_H_

#include "Utils.h"
#include "DiffCoef.h"
#include "SpectralOperators.h"

struct Ctx {
	std::shared_ptr<DiffCoef> k_;
	std::shared_ptr<NMisc> n_misc_;
	std::shared_ptr<SpectralOperators> spec_ops_;
	fft_plan *plan_;
	ScalarType dt_;
	Vec temp_;
	ScalarType *precfactor_;
	ScalarType *work_cuda_;

	ComplexType *c_hat_;

	~Ctx () {
		fft_free (c_hat_);
	}
};

class DiffSolver {
	public:
		DiffSolver (std::shared_ptr<NMisc> n_misc, std::shared_ptr<SpectralOperators> spec_ops, std::shared_ptr<DiffCoef> k);

		KSP ksp_;
		Mat A_;

		PC pc_;

		Vec rhs_;

		int ksp_itr_;

		std::shared_ptr<Ctx> ctx_;

		PetscErrorCode solve (Vec c, ScalarType dt);
		PetscErrorCode precFactor ();

		virtual ~DiffSolver ();

};

//Helper functions for KSP solve
PetscErrorCode operatorA (Mat A, Vec x, Vec y);
PetscErrorCode operatorCreateVecs (Mat A, Vec *left, Vec *right);
PetscErrorCode precFactor (ScalarType *precfactor, std::shared_ptr<Ctx> ctx);
PetscErrorCode applyPC (PC pc, Vec x, Vec y);

#endif
