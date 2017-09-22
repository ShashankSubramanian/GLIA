#include "Obs.h"

Obs::Obs (std::shared_ptr<NMisc> n_misc) : n_misc_ (n_misc) {
    threshold_ = n_misc->obs_threshold_;
    PetscErrorCode ierr;
    ierr = VecCreate (PETSC_COMM_WORLD, &filter_);
    ierr = VecSetSizes (filter_, n_misc->n_local_, n_misc->n_global_);
    ierr = VecSetFromOptions (filter_);
    ierr = VecSet (filter_, 1.0);
}

PetscErrorCode Obs::setDefaultFilter (Vec data) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    double *filter_ptr, *data_ptr;
    ierr = VecGetArray (filter_, &filter_ptr);                              CHKERRQ (ierr);
    ierr = VecGetArray (data, &data_ptr);                                   CHKERRQ (ierr);
    for (int i = 0; i < n_misc_->n_local_; i++) {
        filter_ptr[i] = double (data_ptr[i] > threshold_);
    }
    ierr = VecRestoreArray (filter_, &filter_ptr);                          CHKERRQ (ierr);
    ierr = VecRestoreArray (data, &data_ptr);                               CHKERRQ (ierr);
    PetscFunctionReturn (0);
}

PetscErrorCode Obs::setCustomFilter (Vec custom_filter) {
    PetscFunctionBegin;
    PetscErrorCode ierr;
    ierr = VecCopy (custom_filter, filter_);                                CHKERRQ (ierr);
    PetscFunctionReturn(0);
}

PetscErrorCode Obs::apply(Vec y, Vec x) {
    PetscFunctionBegin;
    PetscErrorCode ierr;
    ierr = VecPointwiseMult (y, x, filter_);                                CHKERRQ (ierr);
    PetscFunctionReturn(0);
}

Obs::~Obs () {
    PetscErrorCode ierr;
    ierr = VecDestroy (&filter_);
}
