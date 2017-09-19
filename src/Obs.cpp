#include "Obs.h"

Obs::Obs (std::shared_ptr<NMisc> n_misc, Vec data, double obs_thr) {
	threshold_ = obs_thr;

	PetscErrorCode ierr;
	ierr = VecCreate (PETSC_COMM_WORLD, &filter_);							
	ierr = VecSetSizes (filter_, n_misc->n_local_, n_misc->n_global_);		
	ierr = VecSetFromOptions (filter_);										

	double *filter_ptr, *data_ptr;
	ierr = VecGetArray (filter_, &filter_ptr);								
	ierr = VecGetArray (data, &data_ptr);									
	for (int i = 0; i < n_misc->n_local_; i++) {
		filter_ptr[i] = double (data_ptr[i] > threshold_);
	}
	ierr = VecRestoreArray (filter_, &filter_ptr);							
	ierr = VecRestoreArray (data, &data_ptr);								
}	

PetscErrorCode Obs::setFilter (Vec &custom_filter) {
	PetscErrorCode ierr;
    ierr = VecCopy (custom_filter, filter_);								CHKERRQ (ierr);
	return ierr;
}

PetscErrorCode Obs::apply(Vec &y, Vec x) {
	PetscErrorCode ierr;
	ierr = VecPointwiseMult (y, x, filter_);								CHKERRQ (ierr);
	return ierr;
}

Obs::~Obs () {
	PetscErrorCode ierr;
	ierr = VecDestroy (&filter_);											
}