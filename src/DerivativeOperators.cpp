#include "DerivativeOperators.h"

PetscErrorCode DerivativeOperatorsRD::evaluateObjective (PetscReal *J, Vec x) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    //TODO: Beta, Data not added
    //CHECK: Is it okay that tumor->c_0_/1_ is overwritten?

    ierr = tumor_->phi_->apply (tumor_->c_0_, x);                   CHKERRQ (ierr);
    ierr = pde_operators_->solveState (0);
    ierr = tumor_->obs_->apply (temp_, tumor_->c_1_);               CHKERRQ (ierr); //TODO : declare obs_ member in tumor
    ierr = VecAXPY (temp_, -1.0, ADDDATA);                          CHKERRQ (ierr);
    ierr = VecNorm (temp_, NORM_2, J);                              CHKERRQ (ierr);

    PetscReal reg;
    ierr = VecNorm (tumor_->c_0_, NORM_2, &reg);                    CHKERRQ (ierr);
    reg *= 0.5 * n_misc_->beta_;

    (*J) *= 0.5;
    (*J) += reg;

    PetscFunctionReturn(0);
}
PetscErrorCode DerivativeOperatorsRD::evaluateGradient (Vec dJ, Vec x){
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    ierr = tumor_->phi_->apply (tumor_->c_0_, x);                   CHKERRQ (ierr);
    ierr = pde_operators_->solveState (0);

    ierr = tumor_->obs_->apply (temp_, tumor_->c_1_);               CHKERRQ (ierr);
    ierr = VecAXPY (temp_, -1.0, ADDDATA);                          CHKERRQ (ierr);

    ierr = tumor_->obs_->apply (tumor_->p_t_, temp_);               CHKERRQ (ierr);
    ierr = VecScale (tumor_->p_t_, -1.0);                           CHKERRQ (ierr);

    ierr = pde_operators_->solveAdjoint (1);   

    ierr = tumor_->phi_->applyTranspose (ptemp_, tumor_->p_0_);
    ierr = tumor_->phi_->applyTranspose (dJ, tumor_->c_0_);     
    ierr = VecScale (dJ, n_misc_->beta_);                           CHKERRQ (ierr);
    ierr = VecAXPY (dJ, -1.0, ptemp_);                              CHKERRQ (ierr);

    PetscFunctionReturn(0);
}
PetscErrorCode DerivativeOperatorsRD::evaluateHessian (Vec x, Vec y){
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    PetscFunctionReturn(0);
}
