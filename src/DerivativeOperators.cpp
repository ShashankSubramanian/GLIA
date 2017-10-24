#include "DerivativeOperators.h"

PetscErrorCode DerivativeOperatorsRD::evaluateObjective (PetscReal *J, Vec x, Vec data) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

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

//POSITIVITY
/* ------------------------------------------------------------------------------------------ */
/* ------------------------------------------------------------------------------------------ */
/* ------------------------------------------------------------------------------------------ */

PetscErrorCode DerivativeOperatorsPos::sigmoid (Vec temp, Vec input) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    ierr = VecCopy (input, temp);                                  CHKERRQ (ierr);

    ierr = VecScale (temp, -1.0);                                  CHKERRQ (ierr);
    ierr = VecShift (temp, n_misc_->exp_shift_);                   CHKERRQ (ierr);
    ierr = VecExp (temp);                                          CHKERRQ (ierr);
    ierr = VecShift (temp, 1.0);                                   CHKERRQ (ierr);
    ierr = VecReciprocal (temp);                                   CHKERRQ (ierr);

    PetscFunctionReturn (0);
}


PetscErrorCode DerivativeOperatorsPos::evaluateObjective (PetscReal *J, Vec x, Vec data) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    ierr = tumor_->phi_->apply (temp_phip_, x);                     CHKERRQ (ierr);
    ierr = sigmoid (tumor_->c_0_, temp_phip_);  
    ierr = pde_operators_->solveState (0);
    ierr = tumor_->obs_->apply (temp_, tumor_->c_t_);               CHKERRQ (ierr); 
    ierr = VecAXPY (temp_, -1.0, data);                             CHKERRQ (ierr);
    ierr = VecDot (temp_, temp_, J);                                CHKERRQ (ierr);

    PetscReal reg;
    ierr = VecDot (tumor_->c_0_, tumor_->c_0_, &reg);               CHKERRQ (ierr);
    reg *= 0.5 * n_misc_->beta_;

    (*J) *= 0.5;
    (*J) += reg;

    ierr = VecDot (x, x, &reg);                                     CHKERRQ (ierr);
    reg *= 0.5 * n_misc_->penalty_;

    (*J) += reg;

    PetscFunctionReturn(0);
}

PetscErrorCode DerivativeOperatorsPos::evaluateGradient (Vec dJ, Vec x, Vec data){
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    ierr = tumor_->phi_->apply (temp_phip_, x);                     CHKERRQ (ierr);
    ierr = sigmoid (tumor_->c_0_, temp_phip_);  
    ierr = pde_operators_->solveState (0);

    ierr = tumor_->obs_->apply (temp_, tumor_->c_t_);               CHKERRQ (ierr);
    ierr = VecAXPY (temp_, -1.0, data);                             CHKERRQ (ierr);

    ierr = tumor_->obs_->apply (tumor_->p_t_, temp_);               CHKERRQ (ierr);
    ierr = VecScale (tumor_->p_t_, -1.0);                           CHKERRQ (ierr);

    ierr = pde_operators_->solveAdjoint (1);

    ierr = VecCopy (temp_phip_, temp_);                             CHKERRQ (ierr);
    double *temp_ptr, *p_ptr;
    ierr = VecGetArray (temp_, &temp_ptr);                          CHKERRQ (ierr);
    for (int i = 0; i < n_misc_->n_local_; i++) {
        temp_ptr[i] = (1 / (1 + exp(-temp_ptr[i] + n_misc_->exp_shift_))) *
                        (1 / (1 + exp(-temp_ptr[i] + n_misc_->exp_shift_))) *
                        (1 / (1 + exp(-temp_ptr[i] + n_misc_->exp_shift_))) *
                        exp(-temp_ptr[i] + n_misc_->exp_shift_);
    }
    ierr = VecRestoreArray (temp_, &temp_ptr);                      CHKERRQ (ierr);
    ierr = tumor_->phi_->applyTranspose (ptemp_, temp_);
    ierr = VecScale (ptemp_, n_misc_->beta_);                       CHKERRQ (ierr);

    ierr = VecCopy (ptemp_, dJ);                                    CHKERRQ (ierr);

    ierr = VecCopy (temp_phip_, temp_);                             CHKERRQ (ierr);
    ierr = VecGetArray (temp_, &temp_ptr);                          CHKERRQ (ierr);
    ierr = VecGetArray (tumor_->p_0_, &p_ptr);                      CHKERRQ (ierr);
    for (int i = 0; i < n_misc_->n_local_; i++) {
        temp_ptr[i] = p_ptr[i] * 
                        (1 / (1 + exp(-temp_ptr[i] + n_misc_->exp_shift_))) *
                        (1 / (1 + exp(-temp_ptr[i] + n_misc_->exp_shift_))) *
                        exp(-temp_ptr[i] + n_misc_->exp_shift_);
    }
    ierr = VecRestoreArray (temp_, &temp_ptr);                      CHKERRQ (ierr);
    ierr = VecRestoreArray (tumor_->p_0_, &p_ptr);                  CHKERRQ (ierr);

    ierr = tumor_->phi_->applyTranspose (ptemp_, temp_);
    ierr = VecAXPY (dJ, -1.0, ptemp_);                              CHKERRQ (ierr);
    ierr = VecAXPY (dJ, n_misc_->penalty_, x);                      CHKERRQ (ierr);


    PetscFunctionReturn(0);
}

PetscErrorCode DerivativeOperatorsPos::evaluateHessian (Vec y, Vec x){   
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    ierr = tumor_->phi_->apply (temp_phip_, p_current_);            CHKERRQ (ierr);
    ierr = VecCopy (temp_phip_, tumor_->c_0_);                      CHKERRQ (ierr);
    double *temp_ptr;
    ierr = VecGetArray (tumor_->c_0_, &temp_ptr);                   CHKERRQ (ierr);
    for (int i = 0; i < n_misc_->n_local_; i++) {
        temp_ptr[i] = (1 / (1 + exp(-temp_ptr[i] + n_misc_->exp_shift_))) *
                        (1 / (1 + exp(-temp_ptr[i] + n_misc_->exp_shift_))) *
                        exp(-temp_ptr[i] + n_misc_->exp_shift_);
    }
    ierr = VecRestoreArray (tumor_->c_0_, &temp_ptr);               CHKERRQ (ierr);
    ierr = tumor_->phi_->apply (temp_phiptilde_, x);                CHKERRQ (ierr);
    ierr = VecPointwiseMult (tumor_->c_0_, tumor_->c_0_, temp_phiptilde_);    CHKERRQ (ierr);
    ierr = pde_operators_->solveState (1);

    ierr = tumor_->obs_->apply (temp_, tumor_->c_t_);               CHKERRQ (ierr);
    ierr = tumor_->obs_->apply (tumor_->p_t_, temp_);               CHKERRQ (ierr);
    ierr = VecScale (tumor_->p_t_, -1.0);                           CHKERRQ (ierr);

    ierr = pde_operators_->solveAdjoint (2);

    double *phip_ptr, *phiptilde_ptr, *p_ptr;
    ierr = VecGetArray (temp_phip_, &phip_ptr);                     CHKERRQ (ierr);
    ierr = VecGetArray (temp_phiptilde_, &phiptilde_ptr);           CHKERRQ (ierr);
    ierr = VecGetArray (temp_, &temp_ptr);                          CHKERRQ (ierr);
    ierr = VecGetArray (tumor_->p_0_, &p_ptr);                      CHKERRQ (ierr);
    for (int i = 0; i < n_misc_->n_local_; i++) {
        temp_ptr[i] = -n_misc_->beta_ * 
                        (1 / (1 + exp(-phip_ptr[i] + n_misc_->exp_shift_))) *
                        (1 / (1 + exp(-phip_ptr[i] + n_misc_->exp_shift_))) *
                        (1 / (1 + exp(-phip_ptr[i] + n_misc_->exp_shift_))) *
                        exp(-phip_ptr[i] + n_misc_->exp_shift_) *
                        phiptilde_ptr[i];

        temp_ptr[i] += 3.0 * n_misc_->beta_ * 
                        (1 / (1 + exp(-phip_ptr[i] + n_misc_->exp_shift_))) *
                        (1 / (1 + exp(-phip_ptr[i] + n_misc_->exp_shift_))) *
                        (1 / (1 + exp(-phip_ptr[i] + n_misc_->exp_shift_))) *
                        (1 / (1 + exp(-phip_ptr[i] + n_misc_->exp_shift_))) *
                        exp(-phip_ptr[i] + n_misc_->exp_shift_) *
                        exp(-phip_ptr[i] + n_misc_->exp_shift_) *
                        phiptilde_ptr[i];

        temp_ptr[i] += -p_ptr[i] * 
                        (1 / (1 + exp(-phip_ptr[i] + n_misc_->exp_shift_))) *
                        (1 / (1 + exp(-phip_ptr[i] + n_misc_->exp_shift_))) *
                        exp(-phip_ptr[i] + n_misc_->exp_shift_);
    }
    ierr = VecRestoreArray (temp_phip_, &phip_ptr);                  CHKERRQ (ierr);
    ierr = VecRestoreArray (temp_phiptilde_, &phiptilde_ptr);        CHKERRQ (ierr);
    ierr = VecRestoreArray (temp_, &temp_ptr);                       CHKERRQ (ierr);
    ierr = VecRestoreArray (tumor_->p_0_, &p_ptr);                   CHKERRQ (ierr);

    
    ierr = tumor_->phi_->applyTranspose (ptemp_, temp_);
    ierr = VecCopy (x, y);                                          CHKERRQ (ierr);
    ierr = VecScale (y, n_misc_->penalty_);                         CHKERRQ (ierr);
    ierr = VecAXPY (y, 1.0, ptemp_);                                CHKERRQ (ierr);

    PetscFunctionReturn(0);
}

