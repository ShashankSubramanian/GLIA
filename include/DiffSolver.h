#ifndef DIFFSOLVER_H_
#define DIFFSOLVER_H_

#include "Utils.h"
#include "DiffCoef.h"
#include <mpi.h>
#include <omp.h>

struct Ctx {
	std::shared_ptr<DiffCoef> k_;
	std::shared_ptr<NMisc> n_misc_;
	accfft_plan *plan_;
	double dt_;
	Vec temp_;
	double *precfactor_;
};

class DiffSolver {
	public:
		DiffSolver (std::shared_ptr<NMisc> n_misc, std::shared_ptr<DiffCoef> k);

		KSP ksp_;
		Mat A_;

		PC pc_;

		Vec rhs_;

		std::shared_ptr<Ctx> ctx_;

		PetscErrorCode solve (Vec c, double dt);

		virtual ~DiffSolver ();

};

//Helper functions for KSP solve
PetscErrorCode operatorA (Mat A, Vec x, Vec y);
PetscErrorCode precFactor (double *precfactor, std::shared_ptr<Ctx> ctx);
PetscErrorCode applyPC (PC pc, Vec x, Vec y);

#endif
