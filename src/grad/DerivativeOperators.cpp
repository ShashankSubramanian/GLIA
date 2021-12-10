#include "DerivativeOperators.h"
#include <petsc/private/vecimpl.h>

/* obj helpers */

/// @brief computes difference diff = x - y
PetscErrorCode computeDifference(ScalarType *sqrdl2norm, Vec diff_wm, Vec diff_gm, Vec diff_csf, Vec diff_glm, Vec diff_bg, Vec x_wm, Vec x_gm, Vec x_csf, Vec x_glm, Vec x_bg, Vec y_wm, Vec y_gm, Vec y_csf, Vec y_glm, Vec y_bg) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ScalarType mis_wm = 0, mis_gm = 0, mis_csf = 0, mis_glm = 0;
  // diff = x - y
  if (x_wm != nullptr) {
    ierr = VecWAXPY(diff_wm, -1.0, y_wm, x_wm); CHKERRQ(ierr);
    ierr = VecDot(diff_wm, diff_wm, &mis_wm); CHKERRQ(ierr);
  }
  if (x_gm != nullptr) {
    ierr = VecWAXPY(diff_gm, -1.0, y_gm, x_gm); CHKERRQ(ierr);
    ierr = VecDot(diff_gm, diff_gm, &mis_gm); CHKERRQ(ierr);
  }
  if (x_csf != nullptr) {
    ierr = VecWAXPY(diff_csf, -1.0, y_csf, x_csf); CHKERRQ(ierr);
    ierr = VecDot(diff_csf, diff_csf, &mis_csf); CHKERRQ(ierr);
  }
  if (x_glm != nullptr) {
    ierr = VecWAXPY(diff_glm, -1.0, y_glm, x_glm); CHKERRQ(ierr);
    ierr = VecDot(diff_glm, diff_glm, &mis_glm); CHKERRQ(ierr);
  }
  *sqrdl2norm = mis_wm + mis_gm + mis_csf + mis_glm;
  // PetscPrintf(PETSC_COMM_WORLD," geometricCouplingAdjoint mis(WM): %1.6e, mis(GM): %1.6e, mis(CSF): %1.6e, mis(GLM): %1.6e, \n", 0.5*mis_wm, 0.5*mis_gm, 0.5* mis_csf, 0.5*mis_glm);
  PetscFunctionReturn(ierr);
}


/** @brief computes difference xi = m_data - m_geo
 *  - function assumes that on input, xi = m_geo * (1-c(1))   */
PetscErrorCode geometricCouplingAdjoint(ScalarType *sqrdl2norm, Vec xi_wm, Vec xi_gm, Vec xi_csf, Vec xi_glm, Vec xi_bg, Vec m_geo_wm, Vec m_geo_gm, Vec m_geo_csf, Vec m_geo_glm, Vec m_geo_bg, Vec m_data_wm, Vec m_data_gm, Vec m_data_csf, Vec m_data_glm, Vec m_data_bg) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ScalarType mis_wm = 0, mis_gm = 0, mis_csf = 0, mis_glm = 0;
  if (m_geo_wm != nullptr) {
    ierr = VecAXPY(xi_wm, -1.0, m_data_wm); CHKERRQ(ierr);
    ierr = VecScale(xi_wm, -1.0); CHKERRQ(ierr);
    ierr = VecDot(xi_wm, xi_wm, &mis_wm); CHKERRQ(ierr);
  }
  if (m_geo_gm != nullptr) {
    ierr = VecAXPY(xi_gm, -1.0, m_data_gm); CHKERRQ(ierr);
    ierr = VecScale(xi_gm, -1.0); CHKERRQ(ierr);
    ierr = VecDot(xi_gm, xi_gm, &mis_gm); CHKERRQ(ierr);
  }
  if (m_geo_csf != nullptr) {
    ierr = VecAXPY(xi_csf, -1.0, m_data_csf); CHKERRQ(ierr);
    ierr = VecScale(xi_csf, -1.0); CHKERRQ(ierr);
    ierr = VecDot(xi_csf, xi_csf, &mis_csf); CHKERRQ(ierr);
  }
  if (m_geo_glm != nullptr) {
    ierr = VecAXPY(xi_glm, -1.0, m_data_glm); CHKERRQ(ierr);
    ierr = VecScale(xi_glm, -1.0); CHKERRQ(ierr);
    ierr = VecDot(xi_glm, xi_glm, &mis_glm); CHKERRQ(ierr);
  }
  *sqrdl2norm = mis_wm + mis_gm + mis_csf + mis_glm;
  // PetscPrintf(PETSC_COMM_WORLD," geometricCouplingAdjoint mis(WM): %1.6e, mis(GM): %1.6e, mis(CSF): %1.6e, mis(GLM): %1.6e, \n", 0.5*mis_wm, 0.5*mis_gm, 0.5* mis_csf, 0.5*mis_glm);
  PetscFunctionReturn(ierr);
}

/// @brief computes geometric tumor coupling m1 = m0(1-c(1))
PetscErrorCode geometricCoupling(Vec m1_wm, Vec m1_gm, Vec m1_csf, Vec m1_glm, Vec m1_bg, Vec m0_wm, Vec m0_gm, Vec m0_csf, Vec m0_glm, Vec m0_bg, Vec c1, std::shared_ptr<Parameters> nmisc) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ScalarType *ptr_wm, *ptr_gm, *ptr_csf, *ptr_glm, *ptr_bg, *ptr_tu;
  ScalarType *ptr_m1_wm, *ptr_m1_gm, *ptr_m1_csf, *ptr_m1_glm, *ptr_m1_bg;
  ScalarType sum = 0;
  if (m0_wm != nullptr) {
    ierr = VecGetArray(m0_wm, &ptr_wm); CHKERRQ(ierr);
  }
  if (m0_gm != nullptr) {
    ierr = VecGetArray(m0_gm, &ptr_gm); CHKERRQ(ierr);
  }
  if (m0_csf != nullptr) {
    ierr = VecGetArray(m0_csf, &ptr_csf); CHKERRQ(ierr);
  }
  if (m0_glm != nullptr) {
    ierr = VecGetArray(m0_glm, &ptr_glm); CHKERRQ(ierr);
  }
  if (m0_bg != nullptr) {
    ierr = VecGetArray(m0_bg, &ptr_bg); CHKERRQ(ierr);
  }
  if (m1_wm != nullptr) {
    ierr = VecGetArray(m1_wm, &ptr_m1_wm); CHKERRQ(ierr);
  }
  if (m1_gm != nullptr) {
    ierr = VecGetArray(m1_gm, &ptr_m1_gm); CHKERRQ(ierr);
  }
  if (m1_csf != nullptr) {
    ierr = VecGetArray(m1_csf, &ptr_m1_csf); CHKERRQ(ierr);
  }
  if (m1_glm != nullptr) {
    ierr = VecGetArray(m1_glm, &ptr_m1_glm); CHKERRQ(ierr);
  }
  if (m1_bg != nullptr) {
    ierr = VecGetArray(m1_bg, &ptr_m1_bg); CHKERRQ(ierr);
  }
  if (c1 != nullptr) {
    ierr = VecGetArray(c1, &ptr_tu); CHKERRQ(ierr);
  }
  // m = m0(1-c(1))
  for (PetscInt j = 0; j < nmisc->grid_->nl_; j++) {
    sum = 0;
    if (m0_gm != nullptr) {
      ptr_m1_gm[j] = ptr_gm[j] * (1 - ptr_tu[j]);
      sum += ptr_m1_gm[j];
    }
    if (m0_csf != nullptr) {
      ptr_m1_csf[j] = ptr_csf[j] * (1 - ptr_tu[j]);
      sum += ptr_m1_csf[j];
    }
    if (m0_glm != nullptr) {
      ptr_m1_glm[j] = ptr_glm[j] * (1 - ptr_tu[j]);
      sum += ptr_m1_glm[j];
    }
    if (m0_bg != nullptr) {
      ptr_m1_bg[j] = ptr_bg[j] * (1 - ptr_tu[j]);
      sum += ptr_m1_bg[j];
    }
    if (m0_wm != nullptr) {
      ptr_m1_wm[j] = 1. - (sum + ptr_tu[j]);
    }
  }
  if (m0_wm != nullptr) {
    ierr = VecRestoreArray(m0_wm, &ptr_wm); CHKERRQ(ierr);
  }
  if (m0_gm != nullptr) {
    ierr = VecRestoreArray(m0_gm, &ptr_gm); CHKERRQ(ierr);
  }
  if (m0_csf != nullptr) {
    ierr = VecRestoreArray(m0_csf, &ptr_csf); CHKERRQ(ierr);
  }
  if (m0_glm != nullptr) {
    ierr = VecRestoreArray(m0_glm, &ptr_glm); CHKERRQ(ierr);
  }
  if (m0_bg != nullptr) {
    ierr = VecRestoreArray(m0_bg, &ptr_bg); CHKERRQ(ierr);
  }
  if (m1_wm != nullptr) {
    ierr = VecRestoreArray(m1_wm, &ptr_m1_wm); CHKERRQ(ierr);
  }
  if (m1_gm != nullptr) {
    ierr = VecRestoreArray(m1_gm, &ptr_m1_gm); CHKERRQ(ierr);
  }
  if (m1_csf != nullptr) {
    ierr = VecRestoreArray(m1_csf, &ptr_m1_csf); CHKERRQ(ierr);
  }
  if (m1_glm != nullptr) {
    ierr = VecRestoreArray(m1_glm, &ptr_m1_glm); CHKERRQ(ierr);
  }
  if (m1_bg != nullptr) {
    ierr = VecRestoreArray(m1_bg, &ptr_m1_bg); CHKERRQ(ierr);
  }
  if (c1 != nullptr) {
    ierr = VecRestoreArray(c1, &ptr_tu); CHKERRQ(ierr);
  }
  // go home
  PetscFunctionReturn(ierr);
}

/* #### ------------------------------------------------------------------- #### */
/* #### ========         RESET (CHANGE SIZE OF WORK VECTORS)       ======== #### */
/* #### ------------------------------------------------------------------- #### */
PetscErrorCode DerivativeOperators::reset(Vec p, std::shared_ptr<PdeOperators> pde_operators, std::shared_ptr<Parameters> params, std::shared_ptr<Tumor> tumor) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;

  // delete and re-create p vectors
  if (ptemp_ != nullptr) {
    ierr = VecDestroy(&ptemp_); CHKERRQ(ierr);
    ptemp_ = nullptr;
  }
  ierr = VecDuplicate(p, &ptemp_); CHKERRQ(ierr);
  if (temp_ != nullptr) {
    ierr = VecSet(temp_, 0.0); CHKERRQ(ierr);
  }

  pde_operators_ = pde_operators;
  tumor_ = tumor;
  params_ = params;
  PetscFunctionReturn(ierr);
}


PetscErrorCode DerivativeOperatorsMultiSpecies::reset(Vec p, std::shared_ptr<PdeOperators> pde_operators, std::shared_ptr<Parameters> params, std::shared_ptr<Tumor> tumor) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;

  // delete and re-create p vectors
  if (ptemp_ != nullptr) {
    ierr = VecDestroy(&ptemp_); CHKERRQ(ierr);
    ptemp_ = nullptr;
  }
  ierr = VecDuplicate(p, &ptemp_); CHKERRQ(ierr);
  if (temp_ != nullptr) {
    ierr = VecSet(temp_, 0.0); CHKERRQ(ierr);
  }

  pde_operators_ = pde_operators;
  tumor_ = tumor;
  params_ = params;
  PetscFunctionReturn(ierr);
}


/* model helpers */
PetscErrorCode DerivativeOperators::gradDiffusion(Vec dJ) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;

  std::bitset<3> XYZ;
  XYZ[0] = 1;
  XYZ[1] = 1;
  XYZ[2] = 1;
  std::array<double, 7> t = {0};

  ScalarType *dj_ptr;
  ScalarType temp_scalar;
  /* (2) compute grad_k   int_T int_Omega { m_i * (grad c)^T grad alpha } dx dt */
  ScalarType integration_weight = 1.0;
  if (params_->opt_->diffusivity_inversion_ || params_->opt_->flag_reaction_inv_) {
    ierr = VecSet(temp_, 0.0); CHKERRQ(ierr);
    // compute numerical time integration using trapezoidal rule
    for (int i = 0; i < params_->tu_->nt_ + 1; i++) {
      // integration weight for chain trapezoidal rule
      if (i == 0 || i == params_->tu_->nt_)
        integration_weight = 0.5;
      else
        integration_weight = 1.0;

      // compute x = (grad c)^T grad \alpha
      // compute gradient of state variable c(t)
      pde_operators_->spec_ops_->computeGradient(tumor_->work_[1], tumor_->work_[2], tumor_->work_[3], pde_operators_->c_[i], &XYZ, t.data());
      // compute gradient of adjoint variable p(t)
      pde_operators_->spec_ops_->computeGradient(tumor_->work_[4], tumor_->work_[5], tumor_->work_[6], pde_operators_->p_[i], &XYZ, t.data());
      // scalar product (grad c)^T grad \alpha
      ierr = VecPointwiseMult(tumor_->work_[0], tumor_->work_[1], tumor_->work_[4]); CHKERRQ(ierr);
      ierr = VecPointwiseMult(tumor_->work_[1], tumor_->work_[2], tumor_->work_[5]); CHKERRQ(ierr);
      ierr = VecAXPY(tumor_->work_[0], 1.0, tumor_->work_[1]); CHKERRQ(ierr);
      ierr = VecPointwiseMult(tumor_->work_[1], tumor_->work_[3], tumor_->work_[6]); CHKERRQ(ierr);
      ierr = VecAXPY(tumor_->work_[0], 1.0, tumor_->work_[1]); CHKERRQ(ierr);

      // numerical time integration using trapezoidal rule
      ierr = VecAXPY(temp_, params_->tu_->dt_ * integration_weight, tumor_->work_[0]); CHKERRQ(ierr);
    }
    // time integration of [ int_0 (grad c)^T grad alpha dt ] done, result in temp_
    // integration over omega (i.e., inner product, as periodic boundary and no lebesque measure in tumor code)
    ierr = VecGetArray(dJ, &dj_ptr); CHKERRQ(ierr);
    ierr = VecDot(tumor_->mat_prop_->wm_, temp_, &dj_ptr[params_->tu_->np_]); CHKERRQ(ierr);
    dj_ptr[params_->tu_->np_] *= params_->grid_->lebesgue_measure_;

    if (params_->tu_->nk_ == 1) {
      // Inverting for only one parameters a.k.a diffusivity in WM. Provide user with the option of setting a diffusivity for
      // other tissue types using params - Hence, the gradient will change accordingly.
      // Implicitly assuming there's no glm. TODO: remove glm from all subsequent iterations of the solver.
      ierr = VecDot(tumor_->mat_prop_->gm_, temp_, &temp_scalar); CHKERRQ(ierr);
      temp_scalar *= params_->grid_->lebesgue_measure_;
      temp_scalar *= params_->tu_->k_gm_wm_ratio_;  // this ratio will control the diffusivity in gm
      dj_ptr[params_->tu_->np_] += temp_scalar;
    }

    if (params_->tu_->nk_ > 1) {
      ierr = VecDot(tumor_->mat_prop_->gm_, temp_, &dj_ptr[params_->tu_->np_ + 1]); CHKERRQ(ierr);
      dj_ptr[params_->tu_->np_ + 1] *= params_->grid_->lebesgue_measure_;
    }
    if (params_->tu_->nk_ > 2) {
      ierr = VecDot(tumor_->mat_prop_->csf_, temp_, &dj_ptr[params_->tu_->np_ + 2]); CHKERRQ(ierr);
      dj_ptr[params_->tu_->np_ + 2] *= params_->grid_->lebesgue_measure_;
    }
    ierr = VecRestoreArray(dJ, &dj_ptr); CHKERRQ(ierr);
  }

  PetscFunctionReturn(ierr);
}

PetscErrorCode DerivativeOperators::gradReaction(Vec dJ) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;

  std::bitset<3> XYZ;
  XYZ[0] = 1;
  XYZ[1] = 1;
  XYZ[2] = 1;
  std::array<double, 7> t = {0};

  ScalarType *dj_ptr;
  ScalarType temp_scalar;
  /* INVERSION FOR REACTION COEFFICIENT */
  ScalarType integration_weight = 1.0;
  if (params_->opt_->flag_reaction_inv_) {
    ierr = VecSet(temp_, 0.0); CHKERRQ(ierr);
    // compute numerical time integration using trapezoidal rule
    for (int i = 0; i < params_->tu_->nt_ + 1; i++) {
      // integration weight for chain trapezoidal rule
      if (i == 0 || i == params_->tu_->nt_)
        integration_weight = 0.5;
      else
        integration_weight = 1.0;

      ierr = VecPointwiseMult(tumor_->work_[0], pde_operators_->c_[i], pde_operators_->c_[i]); CHKERRQ(ierr);
      ierr = VecAXPY(tumor_->work_[0], -1.0, pde_operators_->c_[i]); CHKERRQ(ierr);
      ierr = VecPointwiseMult(tumor_->work_[0], pde_operators_->p_[i], tumor_->work_[0]); CHKERRQ(ierr);

      // numerical time integration using trapezoidal rule
      ierr = VecAXPY(temp_, params_->tu_->dt_ * integration_weight, tumor_->work_[0]); CHKERRQ(ierr);
    }
    // time integration of [ int_0 (grad c)^T grad alpha dt ] done, result in temp_
    // integration over omega (i.e., inner product, as periodic boundary)

    ierr = VecGetArray(dJ, &dj_ptr); CHKERRQ(ierr);
    ierr = VecDot(tumor_->mat_prop_->wm_, temp_, &dj_ptr[params_->tu_->np_ + params_->tu_->nk_]); CHKERRQ(ierr);
    dj_ptr[params_->tu_->np_ + params_->tu_->nk_] *= params_->grid_->lebesgue_measure_;

    if (params_->tu_->nr_ == 1) {
      // Inverting for only one parameters a.k.a reaction in WM. Provide user with the option of setting a reaction for
      // other tissue types using params - Hence, the gradient will change accordingly.
      // Implicitly assuming there's no glm. TODO: remove glm from all subsequent iterations of the solver.
      ierr = VecDot(tumor_->mat_prop_->gm_, temp_, &temp_scalar); CHKERRQ(ierr);
      temp_scalar *= params_->grid_->lebesgue_measure_;
      temp_scalar *= params_->tu_->r_gm_wm_ratio_;  // this ratio will control the reaction coefficient in gm
      dj_ptr[params_->tu_->np_ + params_->tu_->nk_] += temp_scalar;
    }

    if (params_->tu_->nr_ > 1) {
      ierr = VecDot(tumor_->mat_prop_->gm_, temp_, &dj_ptr[params_->tu_->np_ + params_->tu_->nk_ + 1]); CHKERRQ(ierr);
      dj_ptr[params_->tu_->np_ + params_->tu_->nk_ + 1] *= params_->grid_->lebesgue_measure_;
    }

    if (params_->tu_->nr_ > 2) {
      ierr = VecDot(tumor_->mat_prop_->csf_, temp_, &dj_ptr[params_->tu_->np_ + params_->tu_->nk_ + 2]); CHKERRQ(ierr);
      dj_ptr[params_->tu_->np_ + params_->tu_->nk_ + 2] *= params_->grid_->lebesgue_measure_;
    }

    ierr = VecRestoreArray(dJ, &dj_ptr); CHKERRQ(ierr);
  }


  PetscFunctionReturn(ierr);
}

PetscErrorCode DerivativeOperators::updateReactionAndDiffusion(Vec x) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;

  ScalarType r1, r2, r3, k1, k2, k3;
  const ScalarType *x_ptr;

  ScalarType scale_rho = params_->opt_->rho_scale_;
  ScalarType scale_kap = params_->opt_->k_scale_;

  ierr = VecGetArrayRead(x, &x_ptr); CHKERRQ(ierr);
  if (params_->opt_->diffusivity_inversion_ || params_->opt_->flag_reaction_inv_) {
    k1 = x_ptr[params_->tu_->np_] * scale_kap;
    k2 = (params_->tu_->nk_ > 1) ? x_ptr[params_->tu_->np_ + 1] * scale_kap : 0;
    k3 = (params_->tu_->nk_ > 2) ? x_ptr[params_->tu_->np_ + 2] * scale_kap : 0;
    ierr = tumor_->k_->updateIsotropicCoefficients(k1, k2, k3, tumor_->mat_prop_, params_); CHKERRQ(ierr);
    // need to update prefactors for diffusion KSP preconditioner, as k changed
    pde_operators_->diff_solver_->precFactor();
  }
  if (params_->opt_->flag_reaction_inv_) {
    r1 = x_ptr[params_->tu_->np_ + params_->tu_->nk_] * scale_rho;
    r2 = (params_->tu_->nr_ > 1) ? x_ptr[params_->tu_->np_ + params_->tu_->nk_ + 1] * scale_rho : 0;
    r3 = (params_->tu_->nr_ > 2) ? x_ptr[params_->tu_->np_ + params_->tu_->nk_ + 2] * scale_rho : 0;
    ierr = tumor_->rho_->updateIsotropicCoefficients(r1, r2, r3, tumor_->mat_prop_, params_);
  }
  ierr = VecRestoreArrayRead(x, &x_ptr); CHKERRQ(ierr);

  std::stringstream s;
  if (params_->tu_->verbosity_ >= 3) {
    if (params_->opt_->diffusivity_inversion_ || params_->opt_->flag_reaction_inv_) {
      s << " Diffusivity guess = (" << k1 << ", " << k2 << ", " << k3 << ")";
      ierr = tuMSGstd(s.str()); CHKERRQ(ierr);
      s.str("");
      s.clear();
    }
    if (params_->opt_->flag_reaction_inv_) {
      s << " Reaction  guess   = (" << r1 << ", " << r2 << ", " << r3 << ")";
      ierr = tuMSGstd(s.str()); CHKERRQ(ierr);
      s.str("");
      s.clear();
    }
  }

  PetscFunctionReturn(ierr);
}




PetscErrorCode DerivativeOperators::checkGradient(Vec p, std::shared_ptr<Data> data_inv) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  std::stringstream s;

  Vec data = data_inv->dt1();

  s << " ----- Gradient check with taylor expansion ----- ";
  ierr = tuMSGwarn(s.str()); CHKERRQ(ierr);
  s.str("");
  s.clear();

  ScalarType norm;
  ierr = VecNorm(p, NORM_2, &norm); CHKERRQ(ierr);

  s << "Gradient check performed at x with norm: " << norm;
  ierr = tuMSGwarn(s.str()); CHKERRQ(ierr);
  s.str("");
  s.clear();
  ScalarType *x_ptr, k1, k2, k3;
  if (params_->opt_->diffusivity_inversion_) {
    ierr = VecGetArray(p, &x_ptr); CHKERRQ(ierr);
    k1 = x_ptr[params_->tu_->np_];
    k2 = (params_->tu_->nk_ > 1) ? x_ptr[params_->tu_->np_ + 1] : 0;
    k3 = (params_->tu_->nk_ > 2) ? x_ptr[params_->tu_->np_ + 2] : 0;
    s << "k1: " << k1 << " k2: " << k2 << " k3: " << k3;
    ierr = tuMSGwarn(s.str()); CHKERRQ(ierr);
    s.str("");
    s.clear();
    ierr = VecRestoreArray(p, &x_ptr); CHKERRQ(ierr);
  }

  if (params_->opt_->flag_reaction_inv_) {
    ierr = VecGetArray(p, &x_ptr); CHKERRQ(ierr);
    k1 = x_ptr[params_->tu_->np_ + params_->tu_->nk_];
    k2 = (params_->tu_->nr_ > 1) ? x_ptr[params_->tu_->np_ + params_->tu_->nk_ + 1] : 0;
    k3 = (params_->tu_->nr_ > 2) ? x_ptr[params_->tu_->np_ + params_->tu_->nk_ + 2] : 0;
    s << "r1: " << k1 << " r2: " << k2 << " r3: " << k3;
    ierr = tuMSGwarn(s.str()); CHKERRQ(ierr);
    s.str("");
    s.clear();
    ierr = VecRestoreArray(p, &x_ptr); CHKERRQ(ierr);
  }

  ScalarType h[6];
  ScalarType J, J_taylor, J_p, diff;

  Vec dJ;
  Vec p_tilde;
  Vec p_new;
  ierr = VecDuplicate(p, &dJ); CHKERRQ(ierr);
  ierr = VecDuplicate(p, &p_tilde); CHKERRQ(ierr);
  ierr = VecDuplicate(p, &p_new); CHKERRQ(ierr);

  ierr = evaluateObjectiveAndGradient(&J_p, dJ, p, data_inv); CHKERRQ(ierr);

  PetscRandom rctx;
  ierr = PetscRandomCreate(PETSC_COMM_SELF, &rctx); CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rctx); CHKERRQ(ierr);
  ierr = VecSetRandom(p_tilde, rctx); CHKERRQ(ierr);

  ScalarType xg_dot;
  for (int i = 0; i < 6; i++) {
    h[i] = std::pow(10, -i);
    ierr = VecWAXPY(p_new, h[i], p_tilde, p); CHKERRQ(ierr);
    ierr = evaluateObjective(&J, p_new, data_inv);
    ierr = VecDot(dJ, p_tilde, &xg_dot); CHKERRQ(ierr);
    J_taylor = J_p + xg_dot * h[i];
    diff = std::abs(J - J_taylor);
    s << "h[i]: " << h[i] << " |J - J_taylor|: " << diff << "  log10(diff) : " << log10(diff) << " g_fd - xg_dot: " << (J - J_p) / h[i] - xg_dot;
    ierr = tuMSGwarn(s.str()); CHKERRQ(ierr);
    s.str("");
    s.clear();
  }

  ierr = VecDestroy(&dJ); CHKERRQ(ierr);
  ierr = VecDestroy(&p_tilde); CHKERRQ(ierr);
  ierr = VecDestroy(&p_new); CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rctx); CHKERRQ(ierr);

  PetscFunctionReturn(ierr);
}

PetscErrorCode DerivativeOperators::checkHessian(Vec p, std::shared_ptr<Data> data_inv) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  std::stringstream s;
  s << "----- Hessian check with taylor expansion ----- ";
  ierr = tuMSGwarn(s.str()); CHKERRQ(ierr);
  s.str("");
  s.clear();

  Vec data = data_inv->dt1();
  ScalarType norm;
  ierr = VecNorm(p, NORM_2, &norm); CHKERRQ(ierr);

  s << "Hessian check performed at x with norm: " << norm;
  ierr = tuMSGwarn(s.str()); CHKERRQ(ierr);
  s.str("");
  s.clear();
  ScalarType *x_ptr, k1, k2, k3;
  if (params_->opt_->diffusivity_inversion_) {
    ierr = VecGetArray(p, &x_ptr); CHKERRQ(ierr);
    k1 = x_ptr[params_->tu_->np_];
    k2 = (params_->tu_->nk_ > 1) ? x_ptr[params_->tu_->np_ + 1] : 0;
    k3 = (params_->tu_->nk_ > 2) ? x_ptr[params_->tu_->np_ + 2] : 0;
    s << "k1: " << k1 << " k2: " << k2 << " k3: " << k3;
    ierr = tuMSGwarn(s.str()); CHKERRQ(ierr);
    s.str("");
    s.clear();
    ierr = VecRestoreArray(p, &x_ptr); CHKERRQ(ierr);
  }

  if (params_->opt_->flag_reaction_inv_) {
    ierr = VecGetArray(p, &x_ptr); CHKERRQ(ierr);
    k1 = x_ptr[params_->tu_->np_ + params_->tu_->nk_];
    k2 = (params_->tu_->nr_ > 1) ? x_ptr[params_->tu_->np_ + params_->tu_->nk_ + 1] : 0;
    k3 = (params_->tu_->nr_ > 2) ? x_ptr[params_->tu_->np_ + params_->tu_->nk_ + 2] : 0;
    s << "r1: " << k1 << " r2: " << k2 << " r3: " << k3;
    ierr = tuMSGwarn(s.str()); CHKERRQ(ierr);
    s.str("");
    s.clear();
    ierr = VecRestoreArray(p, &x_ptr); CHKERRQ(ierr);
  }

  ScalarType h[6] = {0, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10};
  ScalarType J, J_taylor, J_p, diff;

  Vec dJ, Hx, temp;
  Vec p_tilde;
  Vec p_new;
  ierr = VecDuplicate(p, &dJ); CHKERRQ(ierr);
  ierr = VecDuplicate(p, &temp); CHKERRQ(ierr);
  ierr = VecDuplicate(p, &Hx); CHKERRQ(ierr);
  ierr = VecDuplicate(p, &p_tilde); CHKERRQ(ierr);
  ierr = VecDuplicate(p, &p_new); CHKERRQ(ierr);

  ierr = evaluateObjectiveAndGradient(&J_p, dJ, p, data_inv); CHKERRQ(ierr);

  PetscRandom rctx;
  ierr = PetscRandomCreate(PETSC_COMM_SELF, &rctx); CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rctx); CHKERRQ(ierr);
  ierr = VecSetRandom(p_tilde, rctx); CHKERRQ(ierr);
  ierr = VecCopy(p_tilde, temp); CHKERRQ(ierr);
  ScalarType hess_term = 0.;
  for (int i = 0; i < 6; i++) {
    ierr = VecWAXPY(p_new, h[i], p_tilde, p); CHKERRQ(ierr);
    ierr = evaluateObjective(&J, p_new, data_inv);
    ierr = VecDot(dJ, p_tilde, &J_taylor); CHKERRQ(ierr);
    J_taylor *= h[i];
    J_taylor += J_p;
    // H(p)*p_tilde
    ierr = VecCopy(p_tilde, temp); CHKERRQ(ierr);
    ierr = VecScale(temp, h[i]); CHKERRQ(ierr);
    ierr = evaluateHessian(Hx, p_tilde); CHKERRQ(ierr);
    ierr = VecDot(p_tilde, Hx, &hess_term); CHKERRQ(ierr);
    hess_term *= 0.5 * h[i] * h[i];
    J_taylor += hess_term;
    diff = std::abs(J - J_taylor);
    s << "|J - J_taylor|: " << diff << "  log10(diff) : " << log10(diff) << std::endl;
    ierr = tuMSGwarn(s.str()); CHKERRQ(ierr);
    s.str("");
    s.clear();
  }

  ierr = VecDestroy(&dJ); CHKERRQ(ierr);
  ierr = VecDestroy(&temp); CHKERRQ(ierr);
  ierr = VecDestroy(&Hx); CHKERRQ(ierr);
  ierr = VecDestroy(&p_tilde); CHKERRQ(ierr);
  ierr = VecDestroy(&p_new); CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rctx); CHKERRQ(ierr);

  PetscFunctionReturn(ierr);
}


// computes a FD approximation to the hessian of the objective function
PetscErrorCode DerivativeOperators::computeFDHessian(Vec x, std::shared_ptr<Data> data, std::string ss_str) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;

  int procid, nprocs;
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &procid);

  PetscInt sz;
  ierr = VecGetSize(x, &sz); CHKERRQ(ierr);
  std::vector<ScalarType> hessian(sz*sz);

  // since gradient might be inaccurate, use the func eval directly
  ScalarType small = std::pow(PETSC_MACHINE_EPSILON, (1.0/3.0));
  PetscReal J, Jij;
  ierr = evaluateObjective(&J, x, data); CHKERRQ(ierr);

  Vec dx;
  ierr = VecDuplicate(x, &dx); CHKERRQ(ierr);
  ierr = VecSet(dx, 0); CHKERRQ(ierr);
  std::vector<ScalarType> fplus(sz);
  ScalarType *x_ptr, *dx_ptr;
  ScalarType h, hi, hj;
  for (int i = 0; i < sz; i++) {
    ierr = VecCopy(x, dx); CHKERRQ(ierr);
    ierr = VecGetArray(dx, &dx_ptr); CHKERRQ(ierr);
    h = (dx_ptr[i] == 0) ? small : small * dx_ptr[i];
    dx_ptr[i] += h;
    ierr = VecRestoreArray(dx, &dx_ptr); CHKERRQ(ierr);
    ierr = evaluateObjective(&fplus[i], dx, data); CHKERRQ(ierr);
  }

  ScalarType temp = 0;
  for (int i = 0; i < sz; i++) {
    for (int j = 0; j < sz; j++) {
      // hessian is symm by construction; only compute lower triangle and diagonal
      if (i < j) {
        // compute f(x + h*e_i + h*e_j)
        ierr = VecCopy(x, dx); CHKERRQ(ierr);
        ierr = VecGetArray(dx, &dx_ptr); CHKERRQ(ierr);
        hi = (dx_ptr[i] == 0) ? small : small * dx_ptr[i];
        dx_ptr[i] += hi;
        hj = (dx_ptr[j] == 0) ? small : small * dx_ptr[j];
        dx_ptr[j] += hj;
        ierr = VecRestoreArray(dx, &dx_ptr); CHKERRQ(ierr);
        ierr = evaluateObjective(&Jij, dx, data); CHKERRQ(ierr);
      } else if (i == j) {
        ierr = VecCopy(x, dx); CHKERRQ(ierr);
        ierr = VecGetArray(dx, &dx_ptr); CHKERRQ(ierr);
        hi = (dx_ptr[i] == 0) ? small : small * dx_ptr[i];
        hj = hi;
        dx_ptr[i] += (hi + hj);
        ierr = VecRestoreArray(dx, &dx_ptr); CHKERRQ(ierr);
        ierr = evaluateObjective(&Jij, dx, data); CHKERRQ(ierr);
      }
      if (i <= j) hessian[sz*i + j] = (1.0/(hi*hj)) * (Jij - fplus[i] - fplus[j] + J);
    }
  }

  // write hessian to a file
  if (procid == 0) {
    std::ofstream f;
    f.open(params_->tu_->writepath_ + "hessian_" + ss_str + ".txt");
    for (int i = 0; i < sz; i++) {
      for (int j = 0; j < sz; j++) {
        if (i > j) hessian[sz*i + j] = hessian[sz*j + i]; // symmetric
        f << hessian[sz*i + j] << " ";
      }
      f << "\n";
    }
    f.close();
  }
  bool verbose = false;
  // print the hessian
  std::stringstream s;
  if (verbose) {
    for (int i = 0; i < sz; i++) {
      for (int j = 0; j < sz; j++) s << hessian[sz*i + j] << " ";
      ierr = tuMSGwarn(s.str()); CHKERRQ(ierr);
      s.str("");
      s.clear();
    }
  }

  PetscFunctionReturn(ierr);
}

PetscErrorCode DerivativeOperators::visLossLandscape(Vec start, Vec d1, Vec d2, std::shared_ptr<Data> data, std::string fname) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;

  int procid, nprocs;
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &procid);

  int n = 10;
  std::vector<ScalarType> alpha(n);
  ScalarType end = 1.2, beg = 0;
  ScalarType del = (end - beg) / (n - 1);
  std::vector<ScalarType>::iterator x;
  ScalarType val;
  for (x = alpha.begin(), val = beg; x != alpha.end(); ++x, val += del) *x = val;

  ScalarType a1, a2;
  PetscInt sz;
  PetscReal J = 0;
  Vec sol;
  ierr = VecDuplicate(start, &sol); CHKERRQ(ierr);
  ierr = VecGetSize(start, &sz); CHKERRQ(ierr);
  ScalarType *sol_ptr, *start_ptr, *d1_ptr, *d2_ptr;
  ierr = VecGetArray(start, &start_ptr); CHKERRQ(ierr);
  ierr = VecGetArray(d1, &d1_ptr); CHKERRQ(ierr);
  if (d2 != nullptr) {
    ierr = VecGetArray(d2, &d2_ptr); CHKERRQ(ierr);
  }
  std::ofstream f;
  if (procid == 0) {
    f.open(params_->tu_->writepath_ + fname + ".txt");
  }
  if (d2 == nullptr) {
    for (int i = 0; i < n; i++) {
      a1 = alpha[i];
      ierr = VecGetArray(sol, &sol_ptr); CHKERRQ(ierr);
      for (int k = 0; k < sz; k++) sol_ptr[k] = start_ptr[k] + a1 * d1_ptr[k];
      if (procid == 0) {
        f << a1 << ",";
        for (int k = 0; k < sz; k++) f << sol_ptr[k] << ",";
      }
      ierr = VecRestoreArray(sol, &sol_ptr); CHKERRQ(ierr);
      ierr = evaluateObjective(&J, sol, data); CHKERRQ(ierr);
      if(procid == 0) f << J << "\n";
    }
  } else {
    for (int i = 0; i < n; i++) {
      a1 = alpha[i];
      for (int j = 0; j < n; j++) {
        a2 = alpha[j];
        ierr = VecGetArray(sol, &sol_ptr); CHKERRQ(ierr);
        for (int k = 0; k < sz; k++) sol_ptr[k] = start_ptr[k] + a1 * d1_ptr[k] + a2 * d2_ptr[k];
        if (procid == 0) {
          f << a1 << "," << a2 << ",";
          for (int k = 0; k < sz; k++) f << sol_ptr[k] << ",";
        }
        ierr = VecRestoreArray(sol, &sol_ptr); CHKERRQ(ierr);
        ierr = evaluateObjective(&J, sol, data); CHKERRQ(ierr);
        if(procid == 0) f << J << "\n";
      }
    }
  }
  if (procid == 0) f.close();
  ierr = VecRestoreArray(start, &start_ptr); CHKERRQ(ierr);
  ierr = VecRestoreArray(d1, &d1_ptr); CHKERRQ(ierr);
  if (d2 != nullptr) {
    ierr = VecRestoreArray(d2, &d2_ptr); CHKERRQ(ierr);
  }
  ierr = VecDestroy(&sol); CHKERRQ(ierr);

  PetscFunctionReturn(ierr);
}
