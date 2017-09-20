#include "TumorSolverInterface.h"


TumorSolverInterface::TumorSolverInterface (std::shared_ptr<NMisc> n_misc)
:
initialized_(false),
n_misc_ (n_misc),
tumor_(),
pde_operators_(),
derivative_operators_(),
inv_solver_() {

	PetscErrorCode ierr = 0;
	if(n_misc != nullptr)
	  initialize(n_misc);
}

PetscErrorCode TumorSolverInterface::initialize (std::shared_ptr<NMisc> n_misc) {
	PetscFunctionBegin;
	PetscErrorCode ierr = 0;

	tumor_ = std::make_shared<Tumor> (n_misc);

  // set up vector p (should also add option to pass a p vec, that is used to initialize tumor)
	Vec p;
	ierr = VecCreate (PETSC_COMM_WORLD, &p);
	ierr = VecSetSizes (p, PETSC_DECIDE, n_misc->np_);
	ierr = VecSetFromOptions (p);
	ierr = VecSet (p, n_misc->p_scale_);

	ierr = tumor_->initialize (p, n_misc);

  // create pde and derivative operators
	if (n_misc->rd_) {
		pde_operators_ = std::make_shared<PdeOperatorsRD> (tumor_, n_misc);
		derivative_operators_ = std::make_shared<DerivativeOperatorsRD> (pde_operators_, n_misc, tumor_);
	}
  // create tumor inverse solver
	inv_solver_ = std::make_shared<InvSolver> (derivative_operators_, n_misc);
  initialized_ = true;

  // cleanup
  ierr = VecDestroy (&p); CHKERRQ(ierr);
	PetscFunctionReturn(0);
}

PetscErrorCode TumorSolverInterface::solveForward (Vec c0, Vec cT) {
	PetscErrorCode ierr = 0;
	// TODO: use c0, use cT COPY
	ierr = pde_operators_->solveState (0);
	PetscFunctionReturn(0);
}

PetscErrorCode TumorSolverInterface::solveInverse (Vec d1, Vec p_rec) {
	PetscErrorCode ierr = 0;
	// TODO: COPY copy d1 to data, take care of p_rec
	ierr = inv_solver_->solve ();
	PetscFunctionReturn(0);
}
