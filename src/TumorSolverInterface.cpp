#include "TumorSolverInterface.h"


TumorSolverInterface::TumorSolverInterface (NMisc *n_misc) {
	tumor_ = new Tumor (n_misc);
	PetscErrorCode ierr;
	Vec p;
	ierr = VecCreate (PETSC_COMM_WORLD, &p); 							
	ierr = VecSetSizes (p, PETSC_DECIDE, n_misc->np_);					
	ierr = VecSetFromOptions (p);										
	ierr = VecSet (p, n_misc->p_scale_);											

	ierr = tumor_->setValues (n_misc->k_, n_misc->rho_, n_misc->user_cm_, p, n_misc);		

	if (n_misc->rd_)
		pde_operators_ = new PdeOperatorsRD (tumor_, n_misc); 

	ierr = VecDestroy (&p);
}

PetscErrorCode TumorSolverInterface::solveForward (NMisc *n_misc) {
	PetscErrorCode ierr = 0;
	ierr = pde_operators_->solveState (n_misc);
	return ierr;
}

TumorSolverInterface::~TumorSolverInterface () {
	delete (pde_operators_);
	delete (tumor_);
}

