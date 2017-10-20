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

// =============================================================================

PetscErrorCode DerivativeOperatorsRDObj::evaluateObjective (PetscReal *J, Vec x, Vec data) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    TU_assert (data != nullptr, "DerivativeOperatorsRDObj::evaluateObjective: requires non-null input data.");
    PetscScalar misfit_tu = 0, misfit_brain = 0, mis_wm = 0, mis_gm = 0, mis_csf = 0, mis_glm = 0;
    PetscReal reg;

    //compute c0
    ierr = tumor_->phi_->apply (tumor_->c_0_, x);                   CHKERRQ (ierr);
    // compute c1
    ierr = pde_operators_->solveState (0);
    ierr = tumor_->obs_->apply (temp_, tumor_->c_t_);               CHKERRQ (ierr);
    // geometric coupling, update probability maps
    ierr = geometricCoupling(mR_wm_, mR_gm_, mR_csf_, mR_bg_, tumor_->c_t_, n_misc_);  CHKERRQ (ierr);
    // evaluate tumor distance meassure || c(1) - d ||
    ierr = VecAXPY (temp_, -1.0, data);                             CHKERRQ (ierr);
    ierr = VecDot (temp_, temp_, &misfit_tu);                       CHKERRQ (ierr);
    // evaluate brain tissue distance meassure || mR - mT ||, mR = mA0(1-c), mT = patient
    if(mR_wm_ != nullptr) {
      ierr = VecAXPY (mR_wm_, -1.0, mT_wm_);                        CHKERRQ (ierr);
      ierr = VecDot (mR_wm_, mR_wm_, &mis_wm);                      CHKERRQ (ierr);
    }
    if(mR_gm_ != nullptr) {
      ierr = VecAXPY (mR_gm_, -1.0, mT_gm_);                        CHKERRQ (ierr);
      ierr = VecDot (mR_gm_, mR_gm_, &mis_gm);                      CHKERRQ (ierr);
    }
    if(mR_csf_ != nullptr) {
      ierr = VecAXPY (mR_csf_, -1.0, mT_csf_);                      CHKERRQ (ierr);
      ierr = VecDot (mR_csf_, mR_csf_, &mis_csf);                   CHKERRQ (ierr);
    }
    if(mR_glm_ != nullptr) {
      ierr = VecAXPY (mR_glm_, -1.0, mT_glm_);                      CHKERRQ (ierr);
      ierr = VecDot (mR_glm_, mR_glm_, &mis_glm);                   CHKERRQ (ierr);
    }
    // compute regularization
    ierr = VecDot (tumor_->c_0_, tumor_->c_0_, &reg);               CHKERRQ (ierr);
    reg *= 0.5 * n_misc_->beta_;

    misfit_reg  = mis_wm + mis_gm + mis_csf + mis_glm;
    misfit_reg *= 0.5;
    misfit_tu  *= 0.5;
    (*J) = misfit_tu + misfit_reg;
    (*J) += reg;

    // re-set reference and template pointer to nullptr
    mR_wm_ = nullptr; mR_gm_ = nullptr; mR_csf_ = nullptr; mR_glm_ = nullptr; mR_bg_ = nullptr;
    mT_wm_ = nullptr; mT_gm_ = nullptr; mT_csf_ = nullptr; mT_glm_ = nullptr; mT_bg_ = nullptr;

    PetscFunctionReturn(0);
}
