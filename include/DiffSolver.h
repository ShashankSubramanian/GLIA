#ifndef DIFFSOLVER_H_
#define DIFFSOLVER_H_

#include "Utils.h"
#include "DiffCoef.h"
#include <mpi.h>
#include <omp.h>

struct Ctx {
	std::shared_ptr<DiffCoef> k_;
	accfft_plan *plan_;
	double dt_;
	Vec temp_;
};

class DiffSolver {
	public:
		DiffSolver (std::shared_ptr<NMisc> n_misc, std::shared_ptr<DiffCoef> k);

		KSP ksp_;
		Mat A_;

		Vec rhs_;

		static PetscErrorCode operatorA (Mat A, Vec x, Vec y);
		PetscErrorCode solve (Vec c, double dt);

		virtual ~DiffSolver ();
		
};

#endif