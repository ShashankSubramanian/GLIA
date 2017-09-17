#ifndef TUMORSOLVERINTERFACE_H_
#define TUMORSOLVERINTERFACE_H_

#include "Utils.h"
#include "Tumor.h"
#include "PdeOperators.h"

class TumorSolverInterface {
	public :
		TumorSolverInterface (Tumor *tumor) : tumor_(tumor) {}

		Tumor *tumor_;

		PetscErrorCode solveForward (PdeOperators *pde_operators, NMisc *n_misc);
		// PetscErrorCode solveInverse (DiffOperators *diff_operators, NMisc *n_misc);

		~TumorSolverInterface () {}
};

#endif