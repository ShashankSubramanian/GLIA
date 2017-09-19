#ifndef PDEOPERATORS_H_
#define PDEOPERATORS_H_

#include "Utils.h"
#include "Tumor.h"
#include "DiffSolver.h"

#include <mpi.h>
#include <omp.h>

PetscErrorCode reaction (Vec c_t, std::shared_ptr<NMisc> n_misc, std::shared_ptr<Tumor> tumor, double dt);

class PdeOperators {
	public:
		PdeOperators (std::shared_ptr<Tumor> tumor, std::shared_ptr<NMisc> n_misc) : tumor_(tumor), n_misc_(n_misc) {
			// diff_solver_ = std::make_shared<DiffSolver> (n_misc, tumor->k_);
		}

		std::shared_ptr<Tumor> tumor_;
		// std::shared_ptr<DiffSolver> diff_solver_;
		std::shared_ptr<NMisc> n_misc_;

		virtual PetscErrorCode solveState () = 0;
		// virtual PetscErrorCode solveAdjoint () = 0;

		virtual ~PdeOperators () {}
};

class PdeOperatorsRD : public PdeOperators {
	public:
		PdeOperatorsRD (std::shared_ptr<Tumor> tumor, std::shared_ptr<NMisc> n_misc) : PdeOperators (tumor, n_misc) {}

		PetscErrorCode solveState ();
		// PetscErrorCode solveAdjoint ();

		~PdeOperatorsRD () {}
};


#endif