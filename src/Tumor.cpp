#include "Tumor.h"

Tumor::Tumor (NMisc *n_misc) {
	PetscErrorCode ierr = 0;
	mat_prop_ = new MatProp (n_misc);
	k_ = new DiffCoef (n_misc);
	rho_ = new ReacCoef (n_misc);

	phi_ = new Phi (n_misc);

	ierr = VecCreate (PETSC_COMM_WORLD, &c_t_); 						
	ierr = VecSetSizes (c_t_, n_misc->n_local_, n_misc->n_global_);		
	ierr = VecSetFromOptions (c_t_);									

	ierr = VecDuplicate (c_t_, &c_0_);									

	ierr = VecSet (c_t_, 0);											
	ierr = VecSet (c_0_, 0);											
}

PetscErrorCode Tumor::setValues (double k, double rho, double *user_cm, Vec p, NMisc *n_misc) {
	PetscErrorCode ierr = 0;
	ierr = mat_prop_->setValues (n_misc);
	ierr = k_->setValues (k, mat_prop_, n_misc);
	ierr = rho_->setValues (rho, mat_prop_, n_misc);

	ierr = VecDuplicate (p, &p_);										CHKERRQ (ierr);
	ierr = VecCopy (p, p_);												CHKERRQ (ierr);

	ierr = phi_->setValues (user_cm, mat_prop_, n_misc);
	ierr = phi_->apply(c_0_, p_, n_misc);

	dataOut (c_0_, n_misc, "results/C0.nc");

	return ierr;
}

Tumor::~Tumor () {
	PetscErrorCode ierr;
	ierr = VecDestroy (&c_t_);											
	ierr = VecDestroy (&c_0_);											
	delete (k_);
	delete (rho_);
	delete (mat_prop_);
	delete (phi_);
} 



