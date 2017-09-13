#include "DiffCoef.h"

DiffCoef::DiffCoef (NMisc *n_misc) {
	PetscErrorCode ierr;
	ierr = VecCreate (PETSC_COMM_WORLD, &kxx_);							
	ierr = VecSetSizes (kxx_, n_misc->n_local_, n_misc->n_global_);		
	ierr = VecSetFromOptions (kxx_);									

	ierr = VecDuplicate (kxx_, &kxy_);									
	ierr = VecDuplicate (kxx_, &kxz_);									
	ierr = VecDuplicate (kxx_, &kyy_);									
	ierr = VecDuplicate (kxx_, &kyz_);									
	ierr = VecDuplicate (kxx_, &kzz_);	

	ierr = VecSet (kxx_ , 0);											
	ierr = VecSet (kxy_ , 0);											
	ierr = VecSet (kxz_ , 0);											 
	ierr = VecSet (kyy_ , 0);											
	ierr = VecSet (kyz_ , 0);											
	ierr = VecSet (kzz_ , 0);			

	smooth_flag_ = 0;
}

PetscErrorCode DiffCoef::setValues (double k_scale, MatProp *mat_prop, NMisc *n_misc) {
    PetscErrorCode ierr;
    k_scale_ = k_scale;
	double dk_dm_gm =  k_scale / 5.0;        //GM
	double dk_dm_wm = k_scale;				 //WM
	double dk_dm_glm = 3.0 / 5.0 * k_scale;	 //GLM

	ierr = VecAXPY (kxx_, dk_dm_gm, mat_prop->gm_);						CHKERRQ (ierr);
	ierr = VecAXPY (kxx_, dk_dm_wm, mat_prop->wm_);						CHKERRQ (ierr);
	ierr = VecAXPY (kxx_, dk_dm_glm, mat_prop->glm_);					CHKERRQ (ierr);

	ierr = VecCopy (kxx_, kyy_);										CHKERRQ (ierr);
 	ierr = VecCopy (kxx_, kzz_);										CHKERRQ (ierr);

 	if (smooth_flag_)
 		this->smooth (n_misc);

 	return ierr;
} 

PetscErrorCode DiffCoef::smooth (NMisc *n_misc) {
	PetscErrorCode ierr;
	double sigma = 2.0 * M_PI / n_misc->n_[0];
	double *kxx_ptr, *kxy_ptr, *kxz_ptr, *kyy_ptr, *kyz_ptr, *kzz_ptr;

	ierr = VecGetArray (kxx_, &kxx_ptr);								CHKERRQ (ierr);
	ierr = VecGetArray (kxy_, &kxy_ptr);								CHKERRQ (ierr);
	ierr = VecGetArray (kxz_, &kxz_ptr);								CHKERRQ (ierr);
	ierr = VecGetArray (kyy_, &kyy_ptr);								CHKERRQ (ierr);
	ierr = VecGetArray (kyz_, &kyz_ptr);								CHKERRQ (ierr);
	ierr = VecGetArray (kzz_, &kzz_ptr);								CHKERRQ (ierr);

	// ierr = weierstrass_smoother (kxx_ptr, kxx_ptr, &n_misc, sigma);
 //    ierr = weierstrass_smoother (kxy_ptr, kxy_ptr, &n_misc, sigma);
 //    ierr = weierstrass_smoother (kxz_ptr, kxz_ptr, &n_misc, sigma);
 //    ierr = weierstrass_smoother (kyy_ptr, kyy_ptr, &n_misc, sigma);
 //    ierr = weierstrass_smoother (kyz_ptr, kyz_ptr, &n_misc, sigma);
 //    ierr = weierstrass_smoother (kzz_ptr, kzz_ptr, &n_misc, sigma);

    ierr = VecRestoreArray (kxx_, &kxx_ptr);								CHKERRQ (ierr);
	ierr = VecRestoreArray (kxy_, &kxy_ptr);								CHKERRQ (ierr);
	ierr = VecRestoreArray (kxz_, &kxz_ptr);								CHKERRQ (ierr);
	ierr = VecRestoreArray (kyy_, &kyy_ptr);								CHKERRQ (ierr);
	ierr = VecRestoreArray (kyz_, &kyz_ptr);								CHKERRQ (ierr);
	ierr = VecRestoreArray (kzz_, &kzz_ptr);								CHKERRQ (ierr);

	return ierr;
}

DiffCoef::~DiffCoef () {
	PetscErrorCode ierr;
	ierr = VecDestroy (&kxx_);											
	ierr = VecDestroy (&kxy_);											
	ierr = VecDestroy (&kxz_);											
	ierr = VecDestroy (&kyy_);											
	ierr = VecDestroy (&kyz_);											
	ierr = VecDestroy (&kzz_);											
}
