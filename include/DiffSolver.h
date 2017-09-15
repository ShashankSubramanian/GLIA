#ifndef DIFFSOLVER_H_
#define DIFFSOLVER_H_

#include "Utils.h"
#include "DiffCoef.h"
#include <mpi.h>
#include <omp.h>

struct Ctx {
	DiffCoef *k_;
	accfft_plan *plan_;
	double dt_;
	Vec temp_;
};

class DiffSolver {
	public:
		DiffSolver (NMisc *n_misc, DiffCoef *k);

		KSP ksp_;
		Mat A_;

		Vec rhs_;

		static PetscErrorCode operatorA (Mat A, Vec x, Vec y);
		PetscErrorCode solve (Vec c, double dt);

		~DiffSolver ();
		
};

#endif