#include "DerivativeOperators.h"
#include "Utils.h"

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

PetscErrorCode DerivativeOperatorsPos::evaluateGradient (Vec dJ, Vec x, Vec data) {
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

        if (std::isnan(temp_ptr[i])) {
            temp_ptr[i] = 0.0;
        }  
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
        if (std::isnan(temp_ptr[i])) {
            temp_ptr[i] = 0.0;
        }                
    }


    ierr = VecRestoreArray (temp_, &temp_ptr);                      CHKERRQ (ierr);
    ierr = VecRestoreArray (tumor_->p_0_, &p_ptr);                  CHKERRQ (ierr);

    ierr = tumor_->phi_->applyTranspose (ptemp_, temp_);
    ierr = VecAXPY (dJ, -1.0, ptemp_);                              CHKERRQ (ierr);
    ierr = VecAXPY (dJ, n_misc_->penalty_, x);                      CHKERRQ (ierr);
}

PetscErrorCode DerivativeOperatorsPos::evaluateHessian (Vec y, Vec x) {   
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
        if (std::isnan(temp_ptr[i])) {
            temp_ptr[i] = 1.0;
        }
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

        if (std::isnan(temp_ptr[i])) {
            temp_ptr[i] = 0.0;
        }   

        temp_ptr[i] += 3.0 * n_misc_->beta_ * 
                        (1 / (1 + exp(-phip_ptr[i] + n_misc_->exp_shift_))) *
                        (1 / (1 + exp(-phip_ptr[i] + n_misc_->exp_shift_))) *
                        (1 / (1 + exp(-phip_ptr[i] + n_misc_->exp_shift_))) *
                        (1 / (1 + exp(-phip_ptr[i] + n_misc_->exp_shift_))) *
                        exp(-phip_ptr[i] + n_misc_->exp_shift_) *
                        exp(-phip_ptr[i] + n_misc_->exp_shift_) *
                        phiptilde_ptr[i];

        if (std::isnan(temp_ptr[i])) {
            temp_ptr[i] = 0.0;
        }   

        temp_ptr[i] += -p_ptr[i] * 
                        (1 / (1 + exp(-phip_ptr[i] + n_misc_->exp_shift_))) *
                        (1 / (1 + exp(-phip_ptr[i] + n_misc_->exp_shift_))) *
                        exp(-phip_ptr[i] + n_misc_->exp_shift_);

        if (std::isnan(temp_ptr[i])) {
            temp_ptr[i] = 0.0;
        }   
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

// =============================================================================
// =============================================================================
// =============================================================================

PetscErrorCode DerivativeOperatorsRDObj::evaluateObjective (PetscReal *J, Vec x, Vec data) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    TU_assert (data != nullptr, "DerivativeOperatorsRDObj::evaluateObjective: requires non-null input data.");
    PetscScalar misfit_tu = 0, misfit_brain = 0;
    PetscReal reg;

    //compute c0
    ierr = tumor_->phi_->apply (tumor_->c_0_, x);                CHKERRQ (ierr);
    // compute c1
    ierr = pde_operators_->solveState (0);                       CHKERRQ (ierr);
    ierr = tumor_->obs_->apply (temp_, tumor_->c_t_);            CHKERRQ (ierr);
    // geometric coupling, update probability maps
    ierr = geometricCoupling(
      xi_wm_, xi_gm_, xi_csf_, xi_glm_, xi_bg_,
      mR_wm_, mR_gm_, mR_csf_, mR_glm_,  mR_bg_,
      tumor_->c_t_, n_misc_);                                    CHKERRQ (ierr);
    // evaluate tumor distance meassure || c(1) - d ||
    ierr = VecAXPY (temp_, -1.0, data);                          CHKERRQ (ierr);
    ierr = VecDot (temp_, temp_, &misfit_tu);                    CHKERRQ (ierr);
    // evaluate brain tissue distance meassure || mR - mT ||, mR = mA0(1-c), mT = patient
    geometricCouplingAdjoint(&misfit_brain,
      xi_wm_, xi_gm_, xi_csf_, xi_glm_,  xi_bg_,
      mR_wm_, mR_gm_, mR_csf_, mR_glm_,  mR_bg_,
      mT_wm_, mT_gm_, mT_csf_, mT_glm_,  mT_bg_);                CHKERRQ (ierr);
    // compute regularization
    ierr = VecDot (tumor_->c_0_, tumor_->c_0_, &reg);            CHKERRQ (ierr);
    reg *= 0.5 * n_misc_->beta_;
    // compute objective function value
    misfit_brain *= 0.5;
    misfit_tu  *= 0.5;
    (*J) = misfit_tu + misfit_brain;
    (*J) *= 1./nc_;
    (*J) += reg;

    std::stringstream s;
    s << "  J(v) = Dm(v,c) + Dc(c) + S(c0) = "<< std::setprecision(12) << (*J) <<" = " << std::setprecision(12) <<misfit_brain * 1./nc_ <<" + "<< std::setprecision(12)<< misfit_tu * 1./nc_ <<" + "<< std::setprecision(12) <<reg<<"";  ierr = tuMSGstd(s.str()); CHKERRQ(ierr); s.str(""); s.clear();
    PetscFunctionReturn(0);
}

PetscErrorCode DerivativeOperatorsRDObj::evaluateGradient (Vec dJ, Vec x, Vec data) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    PetscScalar misfit_brain;

    ierr = tumor_->phi_->apply (tumor_->c_0_, x);                CHKERRQ (ierr);
    ierr = pde_operators_->solveState (0);                       CHKERRQ (ierr);

    ierr = tumor_->obs_->apply (temp_, tumor_->c_t_);            CHKERRQ (ierr);
    ierr = VecAXPY (temp_, -1.0, data);                          CHKERRQ (ierr);
    ierr = tumor_->obs_->apply (tumor_->p_t_, temp_);            CHKERRQ (ierr);
    ierr = VecScale (tumor_->p_t_, -1.0);                        CHKERRQ (ierr);
    ierr = geometricCoupling(
      xi_wm_, xi_gm_, xi_csf_, xi_glm_, xi_bg_,
      mR_wm_, mR_gm_, mR_csf_, mR_glm_,  mR_bg_,
      tumor_->c_t_, n_misc_);                                    CHKERRQ (ierr);
    // evaluate brain tissue distance meassure || mR - mT ||, mR = mA0(1-c), mT = patient
    geometricCouplingAdjoint(&misfit_brain,
      xi_wm_, xi_gm_, xi_csf_, xi_glm_,  xi_bg_,
      mR_wm_, mR_gm_, mR_csf_, mR_glm_,  mR_bg_,
      mT_wm_, mT_gm_, mT_csf_, mT_glm_,  mT_bg_);                CHKERRQ (ierr);
    // compute xi * mA0, add    -\xi * mA0 to adjoint final cond.
    if(mR_wm_ != nullptr) {
  		ierr = VecPointwiseMult (temp_, xi_wm_, mR_wm_);         CHKERRQ (ierr);
      ierr = VecAXPY (tumor_->p_t_, -1.0, temp_);                CHKERRQ (ierr);
  	}
  	if(mR_gm_ != nullptr) {
      ierr = VecPointwiseMult (temp_, xi_gm_, mR_gm_);           CHKERRQ (ierr);
      ierr = VecAXPY (tumor_->p_t_, -1.0, temp_);                CHKERRQ (ierr);
  	}
  	if(mR_csf_ != nullptr) {
      ierr = VecPointwiseMult (temp_, xi_csf_, mR_csf_);         CHKERRQ (ierr);
      ierr = VecAXPY (tumor_->p_t_, -1.0, temp_);                CHKERRQ (ierr);
  	}
  	if(mR_glm_ != nullptr) {
      ierr = VecPointwiseMult (temp_, xi_glm_, mR_glm_);         CHKERRQ (ierr);
      ierr = VecAXPY (tumor_->p_t_, -1.0, temp_);                CHKERRQ (ierr);
  	}
    ierr = VecScale (tumor_->p_t_, 1.0/nc_);                     CHKERRQ (ierr);
    // solve adjoint equation with specified final condition
    ierr = pde_operators_->solveAdjoint (1);
    // evaluate gradient
    ierr = tumor_->phi_->applyTranspose (ptemp_, tumor_->p_0_);  CHKERRQ (ierr);
    ierr = tumor_->phi_->applyTranspose (dJ, tumor_->c_0_);      CHKERRQ (ierr);
    ierr = VecScale (dJ, n_misc_->beta_);                        CHKERRQ (ierr);
    ierr = VecAXPY (dJ, -1.0, ptemp_);                           CHKERRQ (ierr);
    PetscFunctionReturn(0);
}


PetscErrorCode DerivativeOperatorsRDObj::evaluateHessian (Vec y, Vec x){
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    ierr = tumor_->phi_->apply (tumor_->c_0_, x);                CHKERRQ (ierr);
    ierr = pde_operators_->solveState (1);                       CHKERRQ (ierr);
    // alpha(1) = - O^TO \tilde{c(1)}
    ierr = tumor_->obs_->apply (temp_, tumor_->c_t_);            CHKERRQ (ierr);
    ierr = tumor_->obs_->apply (tumor_->p_t_, temp_);            CHKERRQ (ierr);
    ierr = VecScale (tumor_->p_t_, -1.0);                        CHKERRQ (ierr);
    // alpha(1) = - O^TO \tilde{c(1)} - mA0 mA0 \tilde{c(1)}
    if(mR_wm_ != nullptr) {
  		ierr = VecPointwiseMult (temp_, mR_wm_, mR_wm_);         CHKERRQ (ierr);
      ierr = VecPointwiseMult (temp_, temp_, tumor_->c_t_);      CHKERRQ (ierr);
      ierr = VecAXPY (tumor_->p_t_, -1.0, temp_);                CHKERRQ (ierr);
  	}
  	if(mR_gm_ != nullptr) {
      ierr = VecPointwiseMult (temp_, mR_gm_, mR_gm_);           CHKERRQ (ierr);
      ierr = VecPointwiseMult (temp_, temp_, tumor_->c_t_);      CHKERRQ (ierr);
      ierr = VecAXPY (tumor_->p_t_, -1.0, temp_);                CHKERRQ (ierr);
  	}
  	if(mR_csf_ != nullptr) {
      ierr = VecPointwiseMult (temp_, mR_csf_, mR_csf_);         CHKERRQ (ierr);
      ierr = VecPointwiseMult (temp_, temp_, tumor_->c_t_);      CHKERRQ (ierr);
      ierr = VecAXPY (tumor_->p_t_, -1.0, temp_);                CHKERRQ (ierr);
  	}
  	if(mR_glm_ != nullptr) {
      ierr = VecPointwiseMult (temp_, mR_glm_, mR_glm_);         CHKERRQ (ierr);
      ierr = VecPointwiseMult (temp_, temp_, tumor_->c_t_);      CHKERRQ (ierr);
      ierr = VecAXPY (tumor_->p_t_, -1.0, temp_);                CHKERRQ (ierr);
  	}

    ierr = VecScale (tumor_->p_t_, 1.0/nc_);                     CHKERRQ (ierr);
    ierr = pde_operators_->solveAdjoint (2);                     CHKERRQ (ierr);
    ierr = tumor_->phi_->applyTranspose (ptemp_, tumor_->p_0_);  CHKERRQ (ierr);
    ierr = tumor_->phi_->applyTranspose (y, tumor_->c_0_);       CHKERRQ (ierr);
    ierr = VecScale (y, n_misc_->beta_);                         CHKERRQ (ierr);
    ierr = VecAXPY (y, -1.0, ptemp_);                            CHKERRQ (ierr);

    PetscFunctionReturn(0);
}

