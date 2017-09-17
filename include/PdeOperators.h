#ifndef PDEOPERATORS_H_
#define PDEOPERATORS_H_

#include "Utils.h"
#include "Tumor.h"
#include "DiffSolver.h"

#include <mpi.h>
#include <omp.h>

PetscErrorCode reaction (Vec c_t, NMisc *n_misc, Tumor *tumor, double dt);

class PdeOperators {
	public:
		PdeOperators (Tumor *tumor, NMisc *n_misc) : tumor_(tumor) {
			diff_solver_ = new DiffSolver (n_misc, tumor->k_);
		}

		Tumor *tumor_;
		DiffSolver *diff_solver_;

		virtual PetscErrorCode solveState (NMisc *n_misc) = 0;
		// virtual PetscErrorCode solveAdjoint (NMisc *n_misc) = 0;

		~PdeOperators () {
			delete (diff_solver_);
		}
};

class PdeOperatorsRD : public PdeOperators {
	public:
		PdeOperatorsRD (Tumor *tumor, NMisc *n_misc) : PdeOperators (tumor, n_misc) {}

		PetscErrorCode solveState (NMisc *n_misc);
		// PetscErrorCode solveAdjoint (NMisc *n_misc);

		~PdeOperatorsRD () {}
};


#endif