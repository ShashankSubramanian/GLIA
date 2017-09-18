#ifndef TUMORSOLVERINTERFACE_H_
#define TUMORSOLVERINTERFACE_H_

#include "Utils.h"
#include "Tumor.h"
#include "PdeOperators.h"

class TumorSolverInterface {
	public :
		TumorSolverInterface (Tumor *tumor, PdeOperators *pde_operators)
								: tumor_(tumor) 
								, pde_operators_(pde_operators) {}

		PetscErrorCode solveForward (NMisc *n_misc);

		~TumorSolverInterface () {}
	private :
		Tumor *tumor_;
		PdeOperators *pde_operators_;
};

#endif