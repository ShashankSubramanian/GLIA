#include "ReacCoef.h"

ReacCoef::ReacCoef (NMisc *n_misc) {
	PetscErrorCode ierr;
	ierr = VecCreate (PETSC_COMM_WORLD, &rho_vec_);							
	ierr = VecSetSizes (rho_vec_, n_misc->n_local_, n_misc->n_global_);		
	ierr = VecSetFromOptions (rho_vec_);									
	ierr = VecSet(rho_vec_, 0);												

	smooth_flag_ = 0;
}

PetscErrorCode ReacCoef::setValues (double rho_scale, MatProp *mat_prop, NMisc *n_misc) {
    PetscErrorCode ierr;
    rho_scale_ = rho_scale;
	double dr_dm_gm = rho_scale;        //GM
	double dr_dm_wm = rho_scale;		//WM
	double dr_dm_glm = 0.0;              //GLM

	ierr = VecAXPY (rho_vec_, dr_dm_gm, mat_prop->gm_);						CHKERRQ (ierr);
	ierr = VecAXPY (rho_vec_, dr_dm_wm, mat_prop->wm_);						CHKERRQ (ierr);
	ierr = VecAXPY (rho_vec_, dr_dm_glm, mat_prop->glm_);					CHKERRQ (ierr);

	if (smooth_flag_)
 		this->smooth (n_misc);

 	return ierr;
}

PetscErrorCode ReacCoef::smooth (NMisc *n_misc) {
	PetscErrorCode ierr;
	double sigma = 2.0 * M_PI / n_misc->n_[0];

	double *rho_vec_ptr;
	ierr = VecGetArray (rho_vec_, &rho_vec_ptr);							CHKERRQ (ierr);
	// ierr = weierstrassSmoother (rho_vec_ptr, rho_vec_ptr, &n_misc, sigma); 
	ierr = VecRestoreArray (rho_vec_, &rho_vec_ptr);							CHKERRQ (ierr);

	return ierr;
}

ReacCoef::~ReacCoef () {
	PetscErrorCode ierr;
	ierr = VecDestroy (&rho_vec_);											
}
