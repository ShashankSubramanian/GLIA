#include "TumorSolverInterface.h"

PetscErrorCode TumorSolverInterface::solveForward (NMisc *n_misc) {
	PetscErrorCode ierr = 0;
	ierr = pde_operators_->solveState (n_misc);
	return ierr;
}

