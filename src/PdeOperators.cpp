#include "PdeOperators.h"

PdeOperatorsRD::PdeOperatorsRD (std::shared_ptr<Tumor> tumor, std::shared_ptr<NMisc> n_misc)
        : PdeOperators (tumor, n_misc) {

    PetscErrorCode ierr = 0;
    double dt = n_misc_->dt_;
    int nt = n_misc->nt_;
    c_ = (Vec *) malloc (sizeof (Vec *) * (nt));    //Stores all (nt) tumor concentrations for adjoint eqns
    ierr = VecCreate (PETSC_COMM_WORLD, &c_[0]);
    ierr = VecSetSizes (c_[0], n_misc->n_local_, n_misc->n_global_);
    ierr = VecSetFromOptions (c_[0]);
    for (int i = 1; i < nt; i++) {
        ierr = VecDuplicate (c_[0], &c_[i]);
    }
    for (int i = 0; i < nt; i++) {
        ierr = VecSet (c_[i], 0);
    }
}

PetscErrorCode PdeOperatorsRD::reaction (int linearized, int iter) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    double *c_t_ptr, *rho_ptr;
    double *c_ptr;
    double factor, alph;
    double dt = n_misc_->dt_;
    ierr = VecGetArray (tumor_->c_t_, &c_t_ptr);                    CHKERRQ (ierr);
    ierr = VecGetArray (tumor_->rho_->rho_vec_, &rho_ptr);          CHKERRQ (ierr);

    ierr = VecGetArray (c_[iter], &c_ptr);                          CHKERRQ (ierr);

    for (int i = 0; i < n_misc_->n_local_; i++) {
        if (linearized == 0) {
            factor = std::exp (rho_ptr[i] * dt);
            alph = (1.0 - c_t_ptr[i]) / c_t_ptr[i];
            c_t_ptr[i] = factor / (factor + alph);
        }
        else {
            factor = std::exp (rho_ptr[i] * dt);
            alph = (c_ptr[i] * factor + 1.0 - c_ptr[i]);
            c_t_ptr[i] = c_t_ptr[i] * factor / (alph * alph);
        }
    }

    ierr = VecRestoreArray (tumor_->c_t_, &c_t_ptr);                CHKERRQ (ierr);
    ierr = VecRestoreArray (tumor_->rho_->rho_vec_, &rho_ptr);      CHKERRQ (ierr);

    ierr = VecRestoreArray (c_[iter], &c_ptr);                      CHKERRQ (ierr);

    PetscFunctionReturn (0);
}

PetscErrorCode PdeOperatorsRD::solveState (int linearized) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    double dt = n_misc_->dt_;
    int nt = n_misc_->nt_;
    n_misc_->statistics_.nb_state_solves++;

    //enforce positivity : hack
    if (!linearized) {
        #ifdef POSITIVITY
            ierr = enforcePositivity (tumor_->c_0_, n_misc_);
        #endif
    }

    ierr = VecCopy (tumor_->c_0_, tumor_->c_t_);                                        CHKERRQ (ierr);

    for (int i = 0; i < nt; i++) {
        diff_solver_->solve (tumor_->c_t_, dt / 2.0);
        //Copy current conc to use for the adjoint equation
        if (linearized == 0) {
            ierr = VecCopy (tumor_->c_t_, c_[i]);                                       CHKERRQ (ierr);
        }
        ierr = reaction (linearized, i);
        diff_solver_->solve (tumor_->c_t_, dt / 2.0);

        //enforce positivity : hack
        if (!linearized) {
            #ifdef POSITIVITY
                ierr = enforcePositivity (tumor_->c_t_, n_misc_);
            #endif
        }
    }

    PetscFunctionReturn (0);
}

PetscErrorCode PdeOperatorsRD::reactionAdjoint (int linearized, int iter) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    double *p_0_ptr, *rho_ptr;
    double *c_ptr;
    double factor, alph;
    double dt = n_misc_->dt_;
    ierr = VecGetArray (tumor_->p_0_, &p_0_ptr);                     CHKERRQ (ierr);
    ierr = VecGetArray (tumor_->rho_->rho_vec_, &rho_ptr);           CHKERRQ (ierr);
    ierr = VecGetArray (c_[iter], &c_ptr);                           CHKERRQ (ierr);

    for (int i = 0; i < n_misc_->n_local_; i++) {
        if (linearized == 1) {
            factor = std::exp (rho_ptr[i] * dt);
            alph = (c_ptr[i] * factor + 1.0 - c_ptr[i]);
            p_0_ptr[i] = p_0_ptr[i] * factor / (alph * alph);
        }
        else { //Gauss - Newton method
            factor = std::exp (rho_ptr[i] * dt);
            alph = (c_ptr[i] * factor + 1.0 - c_ptr[i]);
            p_0_ptr[i] = p_0_ptr[i] * factor / (alph * alph);
        }
    }

    ierr = VecRestoreArray (tumor_->p_0_, &p_0_ptr);                 CHKERRQ (ierr);
    ierr = VecRestoreArray (tumor_->rho_->rho_vec_, &rho_ptr);       CHKERRQ (ierr);

    ierr = VecRestoreArray (c_[iter], &c_ptr);                       CHKERRQ (ierr);
    PetscFunctionReturn (0);
}

PetscErrorCode PdeOperatorsRD::solveAdjoint (int linearized) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    double dt = n_misc_->dt_;
    int nt = n_misc_->nt_;
    n_misc_->statistics_.nb_adjoint_solves++;

    ierr = VecCopy (tumor_->p_t_, tumor_->p_0_);                     CHKERRQ (ierr);
    for (int i = 0; i < nt; i++) {
        diff_solver_->solve (tumor_->p_0_, dt / 2.0);
        ierr = reactionAdjoint (linearized, nt - i - 1);
        diff_solver_->solve (tumor_->p_0_, dt / 2.0);
    }

    PetscFunctionReturn (0);
}

PdeOperatorsRD::~PdeOperatorsRD () {
    PetscErrorCode ierr = 0;
    for (int i = 0; i < this->nt_; i++) {
        ierr = VecDestroy (&c_[i]);
    }
}

PetscErrorCode enforcePositivity (Vec c, std::shared_ptr<NMisc> n_misc) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    double *c_ptr;
    ierr = VecGetArray (c, &c_ptr);                                 CHKERRQ (ierr);
    for (int i = 0; i < n_misc->n_local_; i++) {
        c_ptr[i] = (c_ptr[i] < 0.0) ? 0.0 : c_ptr[i];
        c_ptr[i] = (c_ptr[i] > 1.0) ? 1.0 : c_ptr[i];
    }
    ierr = VecRestoreArray (c, &c_ptr);                             CHKERRQ (ierr);
    PetscFunctionReturn (0);
}
