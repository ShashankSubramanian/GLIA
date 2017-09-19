#include "TumorSolverInterface.h"


TumorSolverInterface::TumorSolverInterface (std::shared_ptr<NMisc> n_misc) : n_misc_ (n_misc) {
	PetscErrorCode ierr = 0;
	tumor_ = std::make_shared<Tumor> (n_misc);

	Vec p;
	ierr = VecCreate (PETSC_COMM_WORLD, &p); 							
	ierr = VecSetSizes (p, PETSC_DECIDE, n_misc->np_);					
	ierr = VecSetFromOptions (p);										
	ierr = VecSet (p, n_misc->p_scale_);											

	ierr = tumor_->setValues (p, n_misc);		

	if (n_misc->rd_) {
		pde_operators_ = std::make_shared<PdeOperatorsRD> (tumor_, n_misc); 
		derivative_operators_ = std::make_shared<DerivativeOperatorsRD> (pde_operators_, n_misc);
	}

	inv_solver_ = std::make_shared<InvSolver> (derivative_operators_, n_misc);

	//TODO : Is this required ?!
	// ierr = VecDestroy (&p);
}

PetscErrorCode TumorSolverInterface::solveForward () {
	PetscErrorCode ierr = 0;
	ierr = pde_operators_->solveState ();
	return ierr;
}

PetscErrorCode TumorSolverInterface::solveInverse () {
	PetscErrorCode ierr = 0;
	ierr = inv_solver_->solve ();
	return ierr;
}


