#include "DerivativeOperators.h"
#include <petsc/private/vecimpl.h>
/* #### ------------------------------------------------------------------- #### */
/* #### ========    REACTION DIFFUSION W/ MODIFIED OBJECTIVE (MP)  ======== #### */
/* #### ========    REACTION DIFFUSION FOR MOVING ATLAS (MA)       ======== #### */
/* #### ------------------------------------------------------------------- #### */
PetscErrorCode DerivativeOperatorsRDObj::evaluateObjective(PetscReal *J, Vec x, std::shared_ptr<Data> data_inv) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  TU_assert(data_inv != nullptr, "DerivativeOperatorsRDObj::evaluateObjective: requires non-null input data.");
  ScalarType misfit_tu = 0, misfit_brain = 0;
  PetscReal reg = 0;
  params_->tu_->statistics_.nb_obj_evals++;

  Vec data = data_inv->dt1();

  // compute c0
  ierr = tumor_->phi_->apply(tumor_->c_0_, x); CHKERRQ(ierr);
  // compute c1
  ierr = pde_operators_->solveState(0); CHKERRQ(ierr);
  ierr = tumor_->obs_->apply(temp_, tumor_->c_t_, 1); CHKERRQ(ierr);
  // geometric coupling, update probability maps
  ierr = geometricCoupling(xi_wm_, xi_gm_, xi_csf_, xi_glm_, xi_bg_, m_geo_wm_, m_geo_gm_, m_geo_csf_, m_geo_glm_, m_geo_bg_, tumor_->c_t_, params_); CHKERRQ(ierr);
  // evaluate tumor distance meassure || c(1) - d ||
  ierr = VecAXPY(temp_, -1.0, data); CHKERRQ(ierr);
  ierr = VecDot(temp_, temp_, &misfit_tu); CHKERRQ(ierr);
  // evaluate brain tissue distance meassure || mR - mT ||, mR = mA0(1-c), mT = patient
  geometricCouplingAdjoint(&misfit_brain, xi_wm_, xi_gm_, xi_csf_, xi_glm_, xi_bg_, m_geo_wm_, m_geo_gm_, m_geo_csf_, m_geo_glm_, m_geo_bg_, m_data_wm_, m_data_gm_, m_data_csf_, m_data_glm_,
                           m_data_bg_); CHKERRQ(ierr);

  /*Regularization term*/
  if (params_->opt_->regularization_norm_ == L2) {
    ierr = VecDot(tumor_->c_0_, tumor_->c_0_, &reg); CHKERRQ(ierr);
    reg *= 0.5 * params_->opt_->beta_ * params_->grid_->lebesgue_measure_;
  } else if (params_->opt_->regularization_norm_ == L2b) {
    ierr = VecDot(x, x, &reg); CHKERRQ(ierr);
    reg *= 0.5 * params_->opt_->beta_;
  }

  // compute objective function value
  misfit_brain *= 0.5 * params_->grid_->lebesgue_measure_;
  misfit_tu *= 0.5 * params_->grid_->lebesgue_measure_;
  (*J) = misfit_tu + misfit_brain;
  (*J) *= 1. / nc_;
  (*J) += reg;

  std::stringstream s;
  s << "  J(p,m) = Dm(v,c) + Dc(c) + S(c0) = " << std::setprecision(12) << (*J) << " = " << std::setprecision(12) << misfit_brain * 1. / nc_ << " + " << std::setprecision(12) << misfit_tu * 1. / nc_
    << " + " << std::setprecision(12) << reg << "";
  ierr = tuMSGstd(s.str()); CHKERRQ(ierr);
  s.str("");
  s.clear();
  PetscFunctionReturn(ierr);
}

PetscErrorCode DerivativeOperatorsRDObj::evaluateGradient(Vec dJ, Vec x, std::shared_ptr<Data> data_inv) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  ScalarType misfit_brain;
  params_->tu_->statistics_.nb_grad_evals++;

  Vec data = data_inv->dt1();

  ScalarType dJ_val = 0, norm_alpha = 0, norm_phiTalpha = 0, norm_phiTphic0 = 0;
  ScalarType norm_adjfinal1 = 0., norm_adjfinal2 = 0., norm_c0 = 0., norm_c1 = 0., norm_d = 0.;
  std::stringstream s;

  if (params_->tu_->verbosity_ >= 2) {
    ierr = VecNorm(data, NORM_2, &norm_d); CHKERRQ(ierr);
  }

  ierr = tumor_->phi_->apply(tumor_->c_0_, x); CHKERRQ(ierr);
  if (params_->tu_->verbosity_ >= 2) {
    ierr = VecNorm(tumor_->c_0_, NORM_2, &norm_c0); CHKERRQ(ierr);
  }
  ierr = pde_operators_->solveState(0); CHKERRQ(ierr);

  ierr = tumor_->obs_->apply(temp_, tumor_->c_t_, 1); CHKERRQ(ierr);
  if (params_->tu_->verbosity_ >= 2) {
    ierr = VecNorm(tumor_->c_t_, NORM_2, &norm_c1); CHKERRQ(ierr);
  }
  ierr = VecAXPY(temp_, -1.0, data); CHKERRQ(ierr);
  ierr = tumor_->obs_->apply(tumor_->p_t_, temp_, 1); CHKERRQ(ierr);
  ierr = VecScale(tumor_->p_t_, -1.0); CHKERRQ(ierr);
  ierr = geometricCoupling(xi_wm_, xi_gm_, xi_csf_, xi_glm_, xi_bg_, m_geo_wm_, m_geo_gm_, m_geo_csf_, m_geo_glm_, m_geo_bg_, tumor_->c_t_, params_); CHKERRQ(ierr);
  // evaluate brain tissue distance meassure || mR - mT ||, mR = mA0(1-c), mT = patient
  geometricCouplingAdjoint(&misfit_brain, xi_wm_, xi_gm_, xi_csf_, xi_glm_, xi_bg_, m_geo_wm_, m_geo_gm_, m_geo_csf_, m_geo_glm_, m_geo_bg_, m_data_wm_, m_data_gm_, m_data_csf_, m_data_glm_,
                           m_data_bg_); CHKERRQ(ierr);
  // compute xi * mA0, add    -\xi * mA0 to adjoint final cond.
  if (m_geo_wm_ != nullptr) {
    ierr = VecPointwiseMult(temp_, xi_wm_, m_geo_wm_); CHKERRQ(ierr);
    ierr = VecAXPY(tumor_->p_t_, -1.0, temp_); CHKERRQ(ierr);
  }
  if (m_geo_gm_ != nullptr) {
    ierr = VecPointwiseMult(temp_, xi_gm_, m_geo_gm_); CHKERRQ(ierr);
    ierr = VecAXPY(tumor_->p_t_, -1.0, temp_); CHKERRQ(ierr);
  }
  if (m_geo_csf_ != nullptr) {
    ierr = VecPointwiseMult(temp_, xi_csf_, m_geo_csf_); CHKERRQ(ierr);
    ierr = VecAXPY(tumor_->p_t_, -1.0, temp_); CHKERRQ(ierr);
  }
  if (m_geo_glm_ != nullptr) {
    ierr = VecPointwiseMult(temp_, xi_glm_, m_geo_glm_); CHKERRQ(ierr);
    ierr = VecAXPY(tumor_->p_t_, -1.0, temp_); CHKERRQ(ierr);
  }
  if (params_->tu_->verbosity_ >= 2) {
    ierr = VecNorm(tumor_->p_t_, NORM_2, &norm_adjfinal1); CHKERRQ(ierr);
  }
  ierr = VecScale(tumor_->p_t_, 1.0 / nc_); CHKERRQ(ierr);
  if (params_->tu_->verbosity_ >= 2) {
    ierr = VecNorm(tumor_->p_t_, NORM_2, &norm_adjfinal2); CHKERRQ(ierr);
  }

  // solve adjoint equation with specified final condition
  ierr = pde_operators_->solveAdjoint(1);
  // evaluate gradient
  ierr = tumor_->phi_->applyTranspose(ptemp_, tumor_->p_0_); CHKERRQ(ierr);

  ierr = VecScale(ptemp_, params_->grid_->lebesgue_measure_); CHKERRQ(ierr);

  if (params_->tu_->verbosity_ >= 2) {
    ierr = VecNorm(ptemp_, NORM_2, &norm_phiTalpha); CHKERRQ(ierr);
  }

  // gradient according to reg paramater
  if (params_->opt_->regularization_norm_ == L2) {
    ierr = tumor_->phi_->applyTranspose(dJ, tumor_->c_0_);
    ierr = VecScale(dJ, params_->opt_->beta_ * params_->grid_->lebesgue_measure_); CHKERRQ(ierr);
    if (params_->tu_->verbosity_ >= 2) {
      ierr = VecNorm(dJ, NORM_2, &norm_phiTphic0); CHKERRQ(ierr);
    }
    ierr = VecAXPY(dJ, -1.0, ptemp_); CHKERRQ(ierr);
  } else if (params_->opt_->regularization_norm_ == L2b) {
    ierr = VecCopy(x, dJ); CHKERRQ(ierr);
    ierr = VecScale(dJ, params_->opt_->beta_); CHKERRQ(ierr);
    ierr = VecAXPY(dJ, -1.0, ptemp_); CHKERRQ(ierr);
  }

  // TODO: add inversion for diffusivity

  // additional information
  ierr = VecNorm(dJ, NORM_2, &dJ_val); CHKERRQ(ierr);
  ierr = VecNorm(tumor_->p_0_, NORM_2, &norm_alpha); CHKERRQ(ierr);

  if (params_->tu_->verbosity_ >= 2) {
    s << "||phiTc0|| = " << std::scientific << std::setprecision(6) << norm_phiTphic0 << " ||phiTa(0)|| = " << std::scientific << std::setprecision(6) << norm_phiTalpha;
    ierr = tuMSGstd(s.str()); CHKERRQ(ierr);
    s.str("");
    s.clear();
    s << "||a(1)|| = " << std::scientific << std::setprecision(6) << norm_adjfinal1 << " ||a(1)||s = " << std::scientific << std::setprecision(6) << norm_adjfinal2 << " ||c(1)|| = " << std::scientific
      << std::setprecision(6) << norm_c1 << " ||c(0)|| = " << std::scientific << std::setprecision(6) << norm_c0 << " ||d|| = " << std::scientific << std::setprecision(6) << norm_d;
    ierr = tuMSGstd(s.str()); CHKERRQ(ierr);
    s.str("");
    s.clear();
  }
  s << "dJ(p,m) = " << std::scientific << std::setprecision(6) << dJ_val << " ||a(0)|| = " << std::scientific << std::setprecision(6) << norm_alpha;
  ierr = tuMSGstd(s.str()); CHKERRQ(ierr);
  s.str("");
  s.clear();
  PetscFunctionReturn(ierr);
}

// TODO: implement optimized version
PetscErrorCode DerivativeOperatorsRDObj::evaluateObjectiveAndGradient(PetscReal *J, Vec dJ, Vec x, std::shared_ptr<Data> data_inv) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  params_->tu_->statistics_.nb_obj_evals++;
  params_->tu_->statistics_.nb_grad_evals++;
  ierr = evaluateObjective(J, x, data_inv); CHKERRQ(ierr);
  ierr = evaluateGradient(dJ, x, data_inv); CHKERRQ(ierr);
  PetscFunctionReturn(ierr);
}

PetscErrorCode DerivativeOperatorsRDObj::evaluateHessian(Vec y, Vec x) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  params_->tu_->statistics_.nb_hessian_evals++;

  ierr = tumor_->phi_->apply(tumor_->c_0_, x); CHKERRQ(ierr);
  ierr = pde_operators_->solveState(1); CHKERRQ(ierr);
  // alpha(1) = - O^TO \tilde{c(1)}
  ierr = tumor_->obs_->apply(temp_, tumor_->c_t_, 1); CHKERRQ(ierr);
  ierr = tumor_->obs_->apply(tumor_->p_t_, temp_, 1); CHKERRQ(ierr);
  ierr = VecScale(tumor_->p_t_, -1.0); CHKERRQ(ierr);
  // alpha(1) = - O^TO \tilde{c(1)} - mA0 mA0 \tilde{c(1)}
  if (m_geo_wm_ != nullptr) {
    ierr = VecPointwiseMult(temp_, m_geo_wm_, m_geo_wm_); CHKERRQ(ierr);
    ierr = VecPointwiseMult(temp_, temp_, tumor_->c_t_); CHKERRQ(ierr);
    ierr = VecAXPY(tumor_->p_t_, -1.0, temp_); CHKERRQ(ierr);
  }
  if (m_geo_gm_ != nullptr) {
    ierr = VecPointwiseMult(temp_, m_geo_gm_, m_geo_gm_); CHKERRQ(ierr);
    ierr = VecPointwiseMult(temp_, temp_, tumor_->c_t_); CHKERRQ(ierr);
    ierr = VecAXPY(tumor_->p_t_, -1.0, temp_); CHKERRQ(ierr);
  }
  if (m_geo_csf_ != nullptr) {
    ierr = VecPointwiseMult(temp_, m_geo_csf_, m_geo_csf_); CHKERRQ(ierr);
    ierr = VecPointwiseMult(temp_, temp_, tumor_->c_t_); CHKERRQ(ierr);
    ierr = VecAXPY(tumor_->p_t_, -1.0, temp_); CHKERRQ(ierr);
  }
  if (m_geo_glm_ != nullptr) {
    ierr = VecPointwiseMult(temp_, m_geo_glm_, m_geo_glm_); CHKERRQ(ierr);
    ierr = VecPointwiseMult(temp_, temp_, tumor_->c_t_); CHKERRQ(ierr);
    ierr = VecAXPY(tumor_->p_t_, -1.0, temp_); CHKERRQ(ierr);
  }

  ierr = VecScale(tumor_->p_t_, 1.0 / nc_); CHKERRQ(ierr);
  ierr = pde_operators_->solveAdjoint(2); CHKERRQ(ierr);
  ierr = tumor_->phi_->applyTranspose(ptemp_, tumor_->p_0_); CHKERRQ(ierr);
  ierr = VecScale(ptemp_, params_->grid_->lebesgue_measure_); CHKERRQ(ierr);

  // No hessian info for L1 for now
  if (params_->opt_->regularization_norm_ == L2b) {
    ierr = VecCopy(x, y); CHKERRQ(ierr);
    ierr = VecScale(y, params_->opt_->beta_); CHKERRQ(ierr);
    ierr = VecAXPY(y, -1.0, ptemp_); CHKERRQ(ierr);
  } else {
    ierr = tumor_->phi_->applyTranspose(y, tumor_->c_0_);
    ierr = VecScale(y, params_->opt_->beta_ * params_->grid_->lebesgue_measure_); CHKERRQ(ierr);
    ierr = VecAXPY(y, -1.0, ptemp_); CHKERRQ(ierr);
  }

  PetscFunctionReturn(ierr);
}