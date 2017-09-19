#include "Tumor.h"

Tumor::Tumor (std::shared_ptr<NMisc> n_misc) {
	PetscErrorCode ierr = 0;

	mat_prop_ = std::make_shared<MatProp> (n_misc);
	k_ = std::make_shared<DiffCoef> (n_misc);
	rho_ = std::make_shared<ReacCoef> (n_misc);
	phi_ = std::make_shared<Phi> (n_misc);

	ierr = VecCreate (PETSC_COMM_WORLD, &c_t_);
	ierr = VecSetSizes (c_t_, n_misc->n_local_, n_misc->n_global_);
	ierr = VecSetFromOptions (c_t_);
	ierr = VecDuplicate (c_t_, &c_0_);

	ierr = VecSet (c_t_, 0);
	ierr = VecSet (c_0_, 0);
}

PetscErrorCode Tumor::initialize (Vec p, std::shared_ptr<NMisc> n_misc) {
	PetscErrorCode ierr = 0;
	ierr = mat_prop_->setValues (n_misc);
	ierr = k_->setValues (n_misc->k_, mat_prop_, n_misc);
	ierr = rho_->setValues (n_misc->rho_, mat_prop_, n_misc);

	ierr = VecDuplicate (p, &p_);                CHKERRQ (ierr);
	ierr = VecCopy (p, p_);                      CHKERRQ (ierr);

	ierr = phi_->setValues (n_misc->user_cm_, mat_prop_, n_misc);
	ierr = phi_->apply(c_0_, p_);

  if(n_misc->writeOutput_)
	  dataOut (c_0_, n_misc, "results/C0.nc");

	return PetscFunctionReturn;
}

Tumor::~Tumor () {
	PetscErrorCode ierr;
	ierr = VecDestroy (&c_t_);
	ierr = VecDestroy (&c_0_);
	ierr = VecDestroy (&p_);
}
