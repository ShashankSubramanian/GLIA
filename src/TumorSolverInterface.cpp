#include "TumorSolverInterface.h"

PetscErrorCode TumorSolverInterface::solveForward (PdeOperators *pde_operators, NMisc *n_misc) {
	PetscErrorCode ierr = 0;
	ierr = pde_operators->solveState (n_misc);
	return ierr;
}

