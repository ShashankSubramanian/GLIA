#include "MatProp.h"

MatProp::MatProp (std::shared_ptr<NMisc> n_misc) {
	PetscErrorCode ierr;
	ierr = VecCreate (PETSC_COMM_WORLD, &gm_);
	ierr = VecSetSizes (gm_, n_misc->n_local_, n_misc->n_global_);
	ierr = VecSetFromOptions (gm_);

	ierr = VecDuplicate (gm_, &wm_);
	ierr = VecDuplicate (gm_, &csf_);
	ierr = VecDuplicate (gm_, &glm_);
	ierr = VecDuplicate (gm_, &filter_);

	ierr = VecSet (gm_ , 0);
	ierr = VecSet (wm_ , 0);
	ierr = VecSet (csf_ , 0);
	ierr = VecSet (glm_ , 0);
	ierr = VecSet (filter_ , 0);
}

PetscErrorCode MatProp::setValues (std::shared_ptr<NMisc> n_misc) {
	PetscFunctionBegin;
	PetscErrorCode ierr;
	double *gm_ptr, *wm_ptr, *csf_ptr, *glm_ptr, *filter_ptr;
	#ifndef BRAIN
		ierr = VecSet (filter_, 1.0);						  CHKERRQ (ierr);
	#else
		ierr = VecGetArray (gm_, &gm_ptr);                    CHKERRQ (ierr);
		ierr = VecGetArray (wm_, &wm_ptr);                    CHKERRQ (ierr);
		ierr = VecGetArray (csf_, &csf_ptr);                  CHKERRQ (ierr);
		ierr = VecGetArray (glm_, &glm_ptr);                  CHKERRQ (ierr);
		ierr = VecGetArray (filter_, &filter_ptr);            CHKERRQ (ierr);


		dataIn (gm_ptr, n_misc, "gray_matter.nc");
		dataIn (wm_ptr, n_misc, "white_matter.nc");
		dataIn (csf_ptr, n_misc, "csf.nc");
		dataIn (glm_ptr, n_misc, "glial_matter.nc");
		dataIn (filter_ptr, n_misc, "filter_zero.nc");

		if(n_misc->writeOutput_) {
			dataOut (gm_ptr, n_misc, "results/gray_matter.nc");
			dataOut (wm_ptr, n_misc, "results/white_matter.nc");
			dataOut (csf_ptr, n_misc, "results/csf.nc");
			dataOut (glm_ptr, n_misc, "results/glial_matter.nc");
			dataOut (filter_ptr, n_misc, "results/filter_zero.nc");
		}

		ierr = VecRestoreArray (gm_, &gm_ptr);                    CHKERRQ (ierr);
		ierr = VecRestoreArray (wm_, &wm_ptr);                    CHKERRQ (ierr);
		ierr = VecRestoreArray (csf_, &csf_ptr);                  CHKERRQ (ierr);
		ierr = VecRestoreArray (glm_, &glm_ptr);                  CHKERRQ (ierr);
		ierr = VecRestoreArray (filter_, &filter_ptr);            CHKERRQ (ierr);
	#endif

	PetscFunctionReturn(0);
}

MatProp::~MatProp() {
	PetscErrorCode ierr;
	ierr = VecDestroy (&gm_);
	ierr = VecDestroy (&wm_);
	ierr = VecDestroy (&csf_);
	ierr = VecDestroy (&glm_);
	ierr = VecDestroy (&filter_);
}
