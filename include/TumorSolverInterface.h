#ifndef TUMORSOLVERINTERFACE_H_
#define TUMORSOLVERINTERFACE_H_

#include "Utils.h"
#include "Tumor.h"
#include "PdeOperators.h"

class TumorSolverInterface {
	public :
		TumorSolverInterface (NMisc *n_misc);

		PetscErrorCode solveForward (NMisc *n_misc);

		~TumorSolverInterface ();
	private :
		Tumor *tumor_;
		PdeOperators *pde_operators_;
};

#endif