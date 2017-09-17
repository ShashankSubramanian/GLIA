#include "PdeOperators.h"

PetscErrorCode PdeOperatorsRD::solveState (NMisc *n_misc) {
	PetscErrorCode ierr = 0;
	double dt = n_misc->dt_;
	int nt = n_misc->time_horizon_ / dt;

	ierr = VecCopy (tumor_->c_0_, tumor_->c_t_);										CHKERRQ (ierr);
	for (int i = 0; i < nt; i++) {
		diff_solver_->solve (tumor_->c_t_, dt / 2.0);
		ierr = reaction (tumor_->c_t_, n_misc, tumor_, dt);
		diff_solver_->solve (tumor_->c_t_, dt / 2.0);
	}

	dataOut (tumor_->c_t_, n_misc, "results/CT.nc");

	return ierr;
}