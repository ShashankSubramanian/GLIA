#include "DerivativeOperators.h"

PetscErrorCode DerivativeOperatorsRD::evaluateObjective (PetscReal *J, Vec x, Vec data) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    //TODO:  Data not added
    ierr = tumor_->phi_->apply (tumor_->c_0_, x);                   CHKERRQ (ierr);
    ierr = pde_operators_->solveState (0);
    ierr = tumor_->obs_->apply (temp_, tumor_->c_t_);               CHKERRQ (ierr); 
    ierr = VecAXPY (temp_, -1.0, data);                             CHKERRQ (ierr);
    ierr = VecDot (temp_, temp_, J);                                CHKERRQ (ierr);

    PetscReal reg;
    ierr = VecDot (tumor_->c_0_, tumor_->c_0_, &reg);               CHKERRQ (ierr);
    reg *= 0.5 * n_misc_->beta_;

    (*J) *= 0.5;
    (*J) += reg;

    PetscFunctionReturn(0);
}
PetscErrorCode DerivativeOperatorsRD::evaluateGradient (Vec dJ, Vec x, Vec data){
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    ierr = tumor_->phi_->apply (tumor_->c_0_, x);                   CHKERRQ (ierr);
    ierr = pde_operators_->solveState (0);

    ierr = tumor_->obs_->apply (temp_, tumor_->c_t_);               CHKERRQ (ierr);
    ierr = VecAXPY (temp_, -1.0, data);                             CHKERRQ (ierr);

    ierr = tumor_->obs_->apply (tumor_->p_t_, temp_);               CHKERRQ (ierr);
    ierr = VecScale (tumor_->p_t_, -1.0);                           CHKERRQ (ierr);

    ierr = pde_operators_->solveAdjoint (1);

    ierr = tumor_->phi_->applyTranspose (ptemp_, tumor_->p_0_);
    ierr = tumor_->phi_->applyTranspose (dJ, tumor_->c_0_);
    ierr = VecScale (dJ, n_misc_->beta_);                           CHKERRQ (ierr);
    ierr = VecAXPY (dJ, -1.0, ptemp_);                              CHKERRQ (ierr);

    PetscFunctionReturn(0);
}
PetscErrorCode DerivativeOperatorsRD::evaluateHessian (Vec y, Vec x){
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    ierr = tumor_->phi_->apply (tumor_->c_0_, x);                   CHKERRQ (ierr);
    ierr = pde_operators_->solveState (1);

    ierr = tumor_->obs_->apply (temp_, tumor_->c_t_);               CHKERRQ (ierr);
    ierr = tumor_->obs_->apply (tumor_->p_t_, temp_);               CHKERRQ (ierr);
    ierr = VecScale (tumor_->p_t_, -1.0);                           CHKERRQ (ierr);

    ierr = pde_operators_->solveAdjoint (2);

    ierr = tumor_->phi_->applyTranspose (ptemp_, tumor_->p_0_);
    ierr = tumor_->phi_->applyTranspose (y, tumor_->c_0_);
    ierr = VecScale (y, n_misc_->beta_);                            CHKERRQ (ierr);
    ierr = VecAXPY (y, -1.0, ptemp_);                               CHKERRQ (ierr);

    PetscFunctionReturn(0);
}
