#include "Utils.h"
#include "PdeOperators.h"
#include "Tumor.h"

PetscErrorCode reaction (Vec c_t, std::shared_ptr<NMisc> n_misc, std::shared_ptr<Tumor> tumor, double dt) {
	PetscErrorCode ierr = 0;
	double *c_t_ptr, *rho_ptr;
	double factor, alph;
	ierr = VecGetArray (c_t, &c_t_ptr);							CHKERRQ (ierr);
	ierr = VecGetArray (tumor->rho_->rho_vec_, &rho_ptr);		CHKERRQ (ierr);

	for (int i = 0; i < n_misc->n_local_; i++) {
		factor = std::exp (rho_ptr[i] * dt);
		alph = (1.0 - c_t_ptr[i]) / c_t_ptr[i];
		c_t_ptr[i] = factor / (factor + alph);

	}

	ierr = VecRestoreArray (c_t, &c_t_ptr);							CHKERRQ (ierr);
	ierr = VecRestoreArray (tumor->rho_->rho_vec_, &rho_ptr);		CHKERRQ (ierr);

	return ierr;
}

