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

PetscErrorCode DerivativeOperatorsMassEffect::reset(Vec p, std::shared_ptr<PdeOperators> pde_operators, std::shared_ptr<Parameters> params, std::shared_ptr<Tumor> tumor) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;

  // delete and re-create p vectors
  if (ptemp_ != nullptr) {
    ierr = VecDestroy(&ptemp_); CHKERRQ(ierr);
    ptemp_ = nullptr;
  }
  ierr = VecDuplicate(delta_, &ptemp_); CHKERRQ(ierr);
  if (temp_ != nullptr) {
    ierr = VecSet(temp_, 0.0); CHKERRQ(ierr);
  }

  pde_operators_ = pde_operators;
  tumor_ = tumor;
  params_ = params;
  PetscFunctionReturn(ierr);
}

/* #### ------------------------------------------------------------------- #### */
/* #### ========  Deriv. Ops.: Adjoints {rho,kappa,p} for RD Model ======== #### */
/* #### ------------------------------------------------------------------- #### */
PetscErrorCode DerivativeOperatorsRD::evaluateObjective(PetscReal *J, Vec x, std::shared_ptr<Data> data) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  params_->tu_->statistics_.nb_obj_evals++;
  ScalarType *x_ptr, k1, k2, k3;

  int x_sz;
  PetscReal m1 = 0, m0 = 0, reg = 0;

#if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 9)
  int lock_state;
  ierr = VecLockGet(x, &lock_state); CHKERRQ(ierr);
  if (lock_state != 0) {
    x->lock = 0;
  }
#endif

  int procid, nprocs;
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &procid);

  if (params_->opt_->diffusivity_inversion_ || params_->opt_->flag_reaction_inv_) {
    ierr = VecGetArray(x, &x_ptr); CHKERRQ(ierr);
    k1 = x_ptr[params_->tu_->np_];
    k2 = (params_->tu_->nk_ > 1) ? x_ptr[params_->tu_->np_ + 1] : 0;
    k3 = (params_->tu_->nk_ > 2) ? x_ptr[params_->tu_->np_ + 2] : 0;
    ierr = VecRestoreArray(x, &x_ptr); CHKERRQ(ierr);
    ierr = tumor_->k_->updateIsotropicCoefficients(k1, k2, k3, tumor_->mat_prop_, params_); CHKERRQ(ierr);
    // need to update prefactors for diffusion KSP preconditioner, as k changed
    pde_operators_->diff_solver_->precFactor();
  }

  ScalarType r1, r2, r3;
  if (params_->opt_->flag_reaction_inv_) {
    ierr = VecGetArray(x, &x_ptr); CHKERRQ(ierr);
    r1 = x_ptr[params_->tu_->np_ + params_->tu_->nk_];
    r2 = (params_->tu_->nr_ > 1) ? x_ptr[params_->tu_->np_ + params_->tu_->nk_ + 1] : 0;
    r3 = (params_->tu_->nr_ > 2) ? x_ptr[params_->tu_->np_ + params_->tu_->nk_ + 2] : 0;
    ierr = tumor_->rho_->updateIsotropicCoefficients(r1, r2, r3, tumor_->mat_prop_, params_);
    ierr = VecRestoreArray(x, &x_ptr); CHKERRQ(ierr);
  }

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

  // compute mismatch ||Oc(1) - d1||
  // p and rho cannot be simultaneosly inverted: apply phi only when reaction inversion is off
  if (!params_->opt_->flag_reaction_inv_) {
    ierr = tumor_->phi_->apply(tumor_->c_0_, x); CHKERRQ(ierr);       // c(0)
  }
  ierr = pde_operators_->solveState(0);                               // c(1)
  ierr = tumor_->obs_->apply(temp_, tumor_->c_t_, 1); CHKERRQ(ierr);  // Oc(1)
  ierr = VecAXPY(temp_, -1.0, data->dt1()); CHKERRQ(ierr);            // Oc(1) - d1
  ierr = VecDot(temp_, temp_, &m1); CHKERRQ(ierr);                    // ||.||^2

  // compute mismatch ||Oc(0) - d0||
  if (params_->tu_->two_time_points_) {
    ierr = tumor_->obs_->apply (temp_, tumor_->c_0_, 0); CHKERRQ (ierr); // Oc(0)
    ierr = VecAXPY (temp_, -1.0, data->dt0()); CHKERRQ (ierr);           // Oc(0) - d0
    ierr = VecDot (temp_, temp_, &m0); CHKERRQ (ierr);                   // ||.||^2
  }


  /*Regularization term*/
  PetscReal reg = 0;
  if (params_->opt_->regularization_norm_ == L2) {  // In tumor space, so scale norm by lebesque measure
    ierr = VecDot(tumor_->c_0_, tumor_->c_0_, &reg); CHKERRQ(ierr);
    reg *= 0.5 * params_->opt_->beta_;
    reg *= params_->grid_->lebesgue_measure_;
  } else if (params_->opt_->regularization_norm_ == L2b) {
    // Reg term only on the initial condition. Leave out the diffusivity.
    ierr = VecGetArray(x, &x_ptr); CHKERRQ(ierr);
    for (int i = 0; i < params_->tu_->np_; i++) {
      reg += x_ptr[i] * x_ptr[i];
    }
    ierr = VecRestoreArray(x, &x_ptr); CHKERRQ(ierr);
    reg *= 0.5 * params_->opt_->beta_;
  }

  // objective function value
  (*J) = params_->grid_->lebesgue_measure_ * 0.5 *(m1 + m0) + reg;

  if (params_->tu_->two_time_points_) {
    s << "  obj: J(p) = D(c1) + D(c0) + S(c0) = "<< std::setprecision(12) << (*J)<<" = " << std::setprecision(12)<< params_->grid_->lebesgue_measure_ * 0.5 * m1 <<" + " << std::setprecision(12)<< params_->grid_->lebesgue_measure_ * 0.5 * m0 <<" + "<< std::setprecision(12) <<reg<<"";  ierr = tuMSGstd(s.str()); CHKERRQ(ierr); s.str(""); s.clear();
  } else {
    s << "  obj: J(p) = D(c1) + S(c0) = "<< std::setprecision(12) << (*J)<<" = " << std::setprecision(12)<< params_->grid_->lebesgue_measure_ * 0.5 * m1 <<" + "<< std::setprecision(12) <<reg<<"";  ierr = tuMSGstd(s.str()); CHKERRQ(ierr); s.str(""); s.clear();
  }

#if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 9)
  if (lock_state != 0) {
    x->lock = lock_state;
  }
#endif

  PetscFunctionReturn(ierr);
}

PetscErrorCode DerivativeOperatorsRD::evaluateGradient(Vec dJ, Vec x, std::shared_ptr<Data> data) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  ScalarType *x_ptr, *p_ptr;
  std::bitset<3> XYZ;
  XYZ[0] = 1;
  XYZ[1] = 1;
  XYZ[2] = 1;
  params_->tu_->statistics_.nb_grad_evals++;
  Event e("tumor-eval-grad");
  std::array<double, 7> t = {0};
  double self_exec_time = -MPI_Wtime();
  ScalarType k1, k2, k3;

  int x_sz;

#if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 9)
  int lock_state;
  ierr = VecLockGet(x, &lock_state); CHKERRQ(ierr);
  if (lock_state != 0) {
    x->lock = 0;
  }
#endif

  ierr = VecSet(dJ, 0); CHKERRQ(ierr);

  int procid, nprocs;
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &procid);

  if (params_->opt_->diffusivity_inversion_ || params_->opt_->flag_reaction_inv_) {
    ierr = VecGetArray(x, &x_ptr); CHKERRQ(ierr);
    k1 = x_ptr[params_->tu_->np_];
    k2 = (params_->tu_->nk_ > 1) ? x_ptr[params_->tu_->np_ + 1] : 0;
    k3 = (params_->tu_->nk_ > 2) ? x_ptr[params_->tu_->np_ + 2] : 0;
    ierr = VecRestoreArray(x, &x_ptr); CHKERRQ(ierr);
    ierr = tumor_->k_->updateIsotropicCoefficients(k1, k2, k3, tumor_->mat_prop_, params_); CHKERRQ(ierr);
    // need to update prefactors for diffusion KSP preconditioner, as k changed
    pde_operators_->diff_solver_->precFactor();
  }

  ScalarType r1, r2, r3;
  if (params_->opt_->flag_reaction_inv_) {
    ierr = VecGetArray(x, &x_ptr); CHKERRQ(ierr);
    r1 = x_ptr[params_->tu_->np_ + params_->tu_->nk_];
    r2 = (params_->tu_->nr_ > 1) ? x_ptr[params_->tu_->np_ + params_->tu_->nk_ + 1] : 0;
    r3 = (params_->tu_->nr_ > 2) ? x_ptr[params_->tu_->np_ + params_->tu_->nk_ + 2] : 0;
    ierr = tumor_->rho_->updateIsotropicCoefficients(r1, r2, r3, tumor_->mat_prop_, params_);
    ierr = VecRestoreArray(x, &x_ptr); CHKERRQ(ierr);
  }

  /* ------------------ */
  /* (1) compute grad_p */
  // c = Phi(p), solve state
  // p and rho cannot be simultaneosly inverted ~ apply phi only when reaction inversion is off
  if (!params_->opt_->flag_reaction_inv_) {
    ierr = tumor_->phi_->apply(tumor_->c_0_, x); CHKERRQ(ierr);        // c(0)
  }
  ierr = pde_operators_->solveState(0);                                // c(1)
  // final cond adjoint
  ierr = tumor_->obs_->apply(temp_, tumor_->c_t_, 1); CHKERRQ(ierr);   // Oc(1)
  ierr = VecAXPY(temp_, -1.0, data->dt1()); CHKERRQ(ierr);             // Oc(1) - d1
  ierr = tumor_->obs_->applyT(tumor_->p_t_, temp_, 1); CHKERRQ(ierr);  // O^T(O(c1) - d1)
  ierr = VecScale(tumor_->p_t_, -1.0); CHKERRQ(ierr);                  // -O^T(O(c1) - d1)
  // solve adjoint
  ierr = pde_operators_->solveAdjoint(1); CHKERRQ(ierr);               // a(0)
  // compute gradient
  if (!params_->opt_->flag_reaction_inv_) { // g_p does not exist if reaction is inverted for
    if (!params_->tu_->phi_store_) {
      // restructure phi compute because it is now expensive
      // assume that reg norm is L2 for now
      // TODO: change to normal if reg norm is not L2
      // p0 = p0 - beta * phi * p
      ierr = VecAXPY(tumor_->p_0_, -params_->opt_->beta_, tumor_->c_0_); CHKERRQ(ierr);
      // dJ is phiT p0 - beta * phiT * phi * p
      ierr = tumor_->phi_->applyTranspose(dJ, tumor_->p_0_); CHKERRQ(ierr);
      // dJ is beta * phiT * phi * p - phiT * p0
      ierr = VecScale(dJ, -params_->grid_->lebesgue_measure_); CHKERRQ(ierr);
    } else {
      ierr = tumor_->phi_->applyTranspose(ptemp_, tumor_->p_0_); CHKERRQ(ierr);  // Phi^T a(0)
      ierr = VecScale(ptemp_, params_->grid_->lebesgue_measure_); CHKERRQ(ierr); // lebesgue

      if (params_->opt_->regularization_norm_ == L2) {
        ierr = tumor_->phi_->applyTranspose(dJ, tumor_->c_0_); CHKERRQ(ierr);
        ierr = VecScale(dJ, params_->opt_->beta_ * params_->grid_->lebesgue_measure_); CHKERRQ(ierr);
        ierr = VecAXPY(dJ, -1.0, ptemp_); CHKERRQ(ierr);
      } else if (params_->opt_->regularization_norm_ == L2b) {
        ierr = VecCopy(x, dJ); CHKERRQ(ierr);
        ierr = VecScale(dJ, params_->opt_->beta_); CHKERRQ(ierr);
        ierr = VecAXPY(dJ, -1.0, ptemp_); CHKERRQ(ierr);
      }
    }
  }

  // compute gradient part Phi^T [ O^T(Oc(0)-d0) ] originating from ||O(c0)-d0||
  if (params_->tu_->two_time_points_) {
      ierr = tumor_->obs_->apply(temp_, tumor_->c_0_, 0); CHKERRQ (ierr);     // O(c0)
      ierr = VecAXPY(temp_, -1.0, data->dt0()); CHKERRQ (ierr);               // O(c0) - d0
      ierr = tumor_->obs_->applyT(temp_, temp_, 0); CHKERRQ (ierr);           // O^T(O(c0) - d0)
      ierr = tumor_->phi_->applyTranspose(ptemp_, temp_); CHKERRQ (ierr);     // Phi^T [O^T(O(c0) - d0)]    // TODO: IS THIS REQUIRED, also LEBESGUE
      ierr = VecAXPY(dJ, params_->grid_->lebesgue_measure_, ptemp_); CHKERRQ (ierr); // add to dJ, lebesgue
  }

  ScalarType temp_scalar;
  /* ------------------------- */
  /* INVERSION FOR DIFFUSIVITY */
  /* ------------------------- */
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
    ierr = VecGetArray(dJ, &x_ptr); CHKERRQ(ierr);
    ierr = VecDot(tumor_->mat_prop_->wm_, temp_, &x_ptr[params_->tu_->np_]); CHKERRQ(ierr);
    x_ptr[params_->tu_->np_] *= params_->grid_->lebesgue_measure_;

    if (params_->tu_->nk_ == 1) {
      // Inverting for only one parameters a.k.a diffusivity in WM. Provide user with the option of setting a diffusivity for
      // other tissue types using params - Hence, the gradient will change accordingly.
      // Implicitly assuming there's no glm. TODO: remove glm from all subsequent iterations of the solver.
      ierr = VecDot(tumor_->mat_prop_->gm_, temp_, &temp_scalar); CHKERRQ(ierr);
      temp_scalar *= params_->grid_->lebesgue_measure_;
      temp_scalar *= params_->tu_->k_gm_wm_ratio_;  // this ratio will control the diffusivity in gm
      x_ptr[params_->tu_->np_] += temp_scalar;
    }

    if (params_->tu_->nk_ > 1) {
      ierr = VecDot(tumor_->mat_prop_->gm_, temp_, &x_ptr[params_->tu_->np_ + 1]); CHKERRQ(ierr);
      x_ptr[params_->tu_->np_ + 1] *= params_->grid_->lebesgue_measure_;
    }
    if (params_->tu_->nk_ > 2) {
      ierr = VecDot(tumor_->mat_prop_->csf_, temp_, &x_ptr[params_->tu_->np_ + 2]); CHKERRQ(ierr);
      x_ptr[params_->tu_->np_ + 2] *= params_->grid_->lebesgue_measure_;
    }
    ierr = VecRestoreArray(dJ, &x_ptr); CHKERRQ(ierr);
  }

  /* INVERSION FOR REACTION COEFFICIENT */
  integration_weight = 1.0;
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

    ierr = VecGetArray(dJ, &x_ptr); CHKERRQ(ierr);
    ierr = VecDot(tumor_->mat_prop_->wm_, temp_, &x_ptr[params_->tu_->np_ + params_->tu_->nk_]); CHKERRQ(ierr);
    x_ptr[params_->tu_->np_ + params_->tu_->nk_] *= params_->grid_->lebesgue_measure_;

    if (params_->tu_->nr_ == 1) {
      // Inverting for only one parameters a.k.a reaction in WM. Provide user with the option of setting a reaction for
      // other tissue types using params - Hence, the gradient will change accordingly.
      // Implicitly assuming there's no glm. TODO: remove glm from all subsequent iterations of the solver.
      ierr = VecDot(tumor_->mat_prop_->gm_, temp_, &temp_scalar); CHKERRQ(ierr);
      temp_scalar *= params_->grid_->lebesgue_measure_;
      temp_scalar *= params_->tu_->r_gm_wm_ratio_;  // this ratio will control the reaction coefficient in gm
      x_ptr[params_->tu_->np_ + params_->tu_->nk_] += temp_scalar;
    }

    if (params_->tu_->nr_ > 1) {
      ierr = VecDot(tumor_->mat_prop_->gm_, temp_, &x_ptr[params_->tu_->np_ + params_->tu_->nk_ + 1]); CHKERRQ(ierr);
      x_ptr[params_->tu_->np_ + params_->tu_->nk_ + 1] *= params_->grid_->lebesgue_measure_;
    }

    if (params_->tu_->nr_ > 2) {
      ierr = VecDot(tumor_->mat_prop_->csf_, temp_, &x_ptr[params_->tu_->np_ + params_->tu_->nk_ + 2]); CHKERRQ(ierr);
      x_ptr[params_->tu_->np_ + params_->tu_->nk_ + 2] *= params_->grid_->lebesgue_measure_;
    }

    ierr = VecRestoreArray(dJ, &x_ptr); CHKERRQ(ierr);
  }

#if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 9)
  if (lock_state != 0) {
    x->lock = lock_state;
  }
#endif

  // timing
  self_exec_time += MPI_Wtime();
  t[5] = self_exec_time;
  e.addTimings(t);
  e.stop();
  PetscFunctionReturn(ierr);
}

// saves on forward solve
PetscErrorCode DerivativeOperatorsRD::evaluateObjectiveAndGradient(PetscReal *J, Vec dJ, Vec x, std::shared_ptr<Data> data) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  params_->tu_->statistics_.nb_obj_evals++;
  params_->tu_->statistics_.nb_grad_evals++;
  std::bitset<3> XYZ;
  XYZ[0] = 1;
  XYZ[1] = 1;
  XYZ[2] = 1;
  Event e("tumor-eval-objandgrad");
  std::array<double, 7> t = {0};
  double self_exec_time = -MPI_Wtime();
  ScalarType *x_ptr, k1, k2, k3;

  int x_sz;
  PetscReal m1 = 0, m0 = 0;

#if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 9)
  int lock_state;
  ierr = VecLockGet(x, &lock_state); CHKERRQ(ierr);
  if (lock_state != 0) {
    x->lock = 0;
  }
#endif

  ierr = VecSet(dJ, 0); CHKERRQ(ierr);

  int procid, nprocs;
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &procid);

  if (params_->opt_->diffusivity_inversion_ || params_->opt_->flag_reaction_inv_) {  // if solveForParameters is happening always invert for diffusivity
    ierr = VecGetArray(x, &x_ptr); CHKERRQ(ierr);
    k1 = x_ptr[params_->tu_->np_];
    k2 = (params_->tu_->nk_ > 1) ? x_ptr[params_->tu_->np_ + 1] : 0;
    k3 = (params_->tu_->nk_ > 2) ? x_ptr[params_->tu_->np_ + 2] : 0;
    ierr = VecRestoreArray(x, &x_ptr); CHKERRQ(ierr);
    ierr = tumor_->k_->updateIsotropicCoefficients(k1, k2, k3, tumor_->mat_prop_, params_); CHKERRQ(ierr);
    // need to update prefactors for diffusion KSP preconditioner, as k changed
    pde_operators_->diff_solver_->precFactor();
  }

  ScalarType r1, r2, r3;
  if (params_->opt_->flag_reaction_inv_) {
    ierr = VecGetArray(x, &x_ptr); CHKERRQ(ierr);
    r1 = x_ptr[params_->tu_->np_ + params_->tu_->nk_];
    r2 = (params_->tu_->nr_ > 1) ? x_ptr[params_->tu_->np_ + params_->tu_->nk_ + 1] : 0;
    r3 = (params_->tu_->nr_ > 2) ? x_ptr[params_->tu_->np_ + params_->tu_->nk_ + 2] : 0;
    ierr = tumor_->rho_->updateIsotropicCoefficients(r1, r2, r3, tumor_->mat_prop_, params_);
    ierr = VecRestoreArray(x, &x_ptr); CHKERRQ(ierr);
  }

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

  // c(0) = Phi(p)
  if (!params_->opt_->flag_reaction_inv_) {
    ierr = tumor_->phi_->apply(tumor_->c_0_, x); CHKERRQ(ierr);         // c(0)
  }
  // compute mismatch ||Oc(0) - d0||
  if (params_->tu_->two_time_points_) {
    ierr = tumor_->obs_->apply(temp_, tumor_->c_0_, 0); CHKERRQ (ierr); // Oc(0)
    ierr = VecAXPY (temp_, -1.0, data->dt0()); CHKERRQ (ierr);          // Oc(0) - d0
    ierr = VecDot (temp_, temp_, &m0); CHKERRQ (ierr);                  // ||.||^2
  }
  // compute mismatch ||Oc(1) - d1||
  ierr = pde_operators_->solveState(0); CHKERRQ(ierr);                  // c(1)
  ierr = tumor_->obs_->apply(temp_, tumor_->c_t_, 1); CHKERRQ(ierr);    // Oc(1)
  ierr = VecAXPY(temp_, -1.0, data->dt1()); CHKERRQ(ierr);              // Oc(1) - d1
  ierr = VecDot(temp_, temp_, &m1); CHKERRQ(ierr);                      // ||.||^2
  // solve adjoint
  ierr = tumor_->obs_->applyT(tumor_->p_t_, temp_, 1); CHKERRQ (ierr);  // O^T(Oc(1) - d1)
  ierr = VecScale(tumor_->p_t_, -1.0); CHKERRQ (ierr);                  // - O^T(O(c1) - d1)
  ierr = pde_operators_->solveAdjoint(1); CHKERRQ (ierr);               // a(0)

  if (!params_->opt_->flag_reaction_inv_) {
    if (!params_->tu_->phi_store_) {
      // restructure phi compute because it is now expensive
      // assume that reg norm is L2 for now
      // TODO: change to normal if reg norm is not L2
      ierr = VecAXPY(tumor_->p_0_, -params_->opt_->beta_, tumor_->c_0_); CHKERRQ (ierr); // a(0) - bata Phi p
      ierr = tumor_->phi_->applyTranspose(dJ, tumor_->p_0_); CHKERRQ (ierr);     // Phi^T (a(0) - bata Phi p)
      ierr = VecScale(dJ, -params_->grid_->lebesgue_measure_); CHKERRQ (ierr);   // lebesgue
    } else {
      ierr = tumor_->phi_->applyTranspose(ptemp_, tumor_->p_0_); CHKERRQ(ierr);  // Phi^T a(0)
      ierr = VecScale(ptemp_, params_->grid_->lebesgue_measure_); CHKERRQ(ierr); // lebesgue

      // Gradient according to reg parameter chosen
      if (params_->opt_->regularization_norm_ == L2) {
        ierr = tumor_->phi_->applyTranspose(dJ, tumor_->c_0_);
        ierr = VecScale(dJ, params_->opt_->beta_ * params_->grid_->lebesgue_measure_); CHKERRQ(ierr);
        ierr = VecAXPY(dJ, -1.0, ptemp_); CHKERRQ(ierr);
      } else if (params_->opt_->regularization_norm_ == L2b) {
        ierr = VecCopy(x, dJ); CHKERRQ(ierr);
        ierr = VecScale(dJ, params_->opt_->beta_); CHKERRQ(ierr);
        ierr = VecAXPY(dJ, -1.0, ptemp_); CHKERRQ(ierr);
      }
    }
  }

  // compute gradient part Phi^T [ O^T(Oc(0)-d0) ] originating from ||O(c0)-d0||
  if (params_->tu_->two_time_points_) {
    ierr = tumor_->obs_->apply(temp_, tumor_->c_0_, 0); CHKERRQ (ierr);     // O(c0)
    ierr = VecAXPY(temp_, -1.0, data->dt0()); CHKERRQ (ierr);               // O(c0) - d0
    ierr = tumor_->obs_->applyT(temp_, temp_, 0); CHKERRQ (ierr);           // O^T(O(c0) - d0)
    ierr = tumor_->phi_->applyTranspose(ptemp_, temp_); CHKERRQ (ierr);     // Phi^T [O^T(O(c0) - d0)]    // TODO: IS THIS REQUIRED, also LEBESGUE
    ierr = VecAXPY(dJ, params_->grid_->lebesgue_measure_, ptemp_); CHKERRQ (ierr); // add to dJ, lebesgue
  }

  // compute regularization
  PetscReal reg = 0;
  if (params_->opt_->regularization_norm_ == L2) {
    ierr = VecDot(tumor_->c_0_, tumor_->c_0_, &reg); CHKERRQ(ierr);
    reg *= 0.5 * params_->opt_->beta_;
    reg *= params_->grid_->lebesgue_measure_;
  } else if (params_->opt_->regularization_norm_ == L2b) {
    // Reg term only on the initial condition. Leave out the diffusivity.
    ierr = VecGetArray(x, &x_ptr); CHKERRQ(ierr);
    for (int i = 0; i < params_->tu_->np_; i++) {
      reg += x_ptr[i] * x_ptr[i];
    }
    ierr = VecRestoreArray(x, &x_ptr); CHKERRQ(ierr);
    reg *= 0.5 * params_->opt_->beta_;
  }

  // objective function value
  (*J) = params_->grid_->lebesgue_measure_ * 0.5 *(m1 + m0) + reg;
  if (params_->tu_->two_time_points_) {
    s << "  obj: J(p) = D(c1) + D(c0) + S(c0) = "<< std::setprecision(12) << (*J)<<" = " << std::setprecision(12)<< params_->grid_->lebesgue_measure_ * 0.5 * m1 <<" + " << std::setprecision(12)<< params_->grid_->lebesgue_measure_  * 0.5 * m0 <<" + "<< std::setprecision(12) <<reg<<"";  ierr = tuMSGstd(s.str()); CHKERRQ(ierr); s.str(""); s.clear();
  } else {
    s << "  obj: J(p) = D(c1) + S(c0) = "<< std::setprecision(12) << (*J)<<" = " << std::setprecision(12)<< params_->grid_->lebesgue_measure_  * 0.5 * m1 <<" + "<< std::setprecision(12) <<reg<<"";  ierr = tuMSGstd(s.str()); CHKERRQ(ierr); s.str(""); s.clear();
  }

  ScalarType temp_scalar;
  /* ------------------------- */
  /* INVERSION FOR DIFFUSIVITY */
  /* ------------------------- */
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
    ierr = VecGetArray(dJ, &x_ptr); CHKERRQ(ierr);
    ierr = VecDot(tumor_->mat_prop_->wm_, temp_, &x_ptr[params_->tu_->np_]); CHKERRQ(ierr);
    x_ptr[params_->tu_->np_] *= params_->grid_->lebesgue_measure_;

    if (params_->tu_->nk_ == 1) {
      // Inverting for only one parameters a.k.a diffusivity in WM. Provide user with the option of setting a diffusivity for
      // other tissue types using params - Hence, the gradient will change accordingly.
      // Implicitly assuming there's no glm. TODO: remove glm from all subsequent iterations of the solver.
      ierr = VecDot(tumor_->mat_prop_->gm_, temp_, &temp_scalar); CHKERRQ(ierr);
      temp_scalar *= params_->grid_->lebesgue_measure_;
      temp_scalar *= params_->tu_->k_gm_wm_ratio_;  // this ratio will control the diffusivity in gm
      x_ptr[params_->tu_->np_] += temp_scalar;
    }

    if (params_->tu_->nk_ > 1) {
      ierr = VecDot(tumor_->mat_prop_->gm_, temp_, &x_ptr[params_->tu_->np_ + 1]); CHKERRQ(ierr);
      x_ptr[params_->tu_->np_ + 1] *= params_->grid_->lebesgue_measure_;
    }
    if (params_->tu_->nk_ > 2) {
      ierr = VecDot(tumor_->mat_prop_->csf_, temp_, &x_ptr[params_->tu_->np_ + 2]); CHKERRQ(ierr);
      x_ptr[params_->tu_->np_ + 2] *= params_->grid_->lebesgue_measure_;
    }
    ierr = VecRestoreArray(dJ, &x_ptr); CHKERRQ(ierr);
  }

  /* INVERSION FOR REACTION COEFFICIENT */
  integration_weight = 1.0;
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

    ierr = VecGetArray(dJ, &x_ptr); CHKERRQ(ierr);
    ierr = VecDot(tumor_->mat_prop_->wm_, temp_, &x_ptr[params_->tu_->np_ + params_->tu_->nk_]); CHKERRQ(ierr);
    x_ptr[params_->tu_->np_ + params_->tu_->nk_] *= params_->grid_->lebesgue_measure_;

    if (params_->tu_->nr_ == 1) {
      // Inverting for only one parameters a.k.a reaction in WM. Provide user with the option of setting a reaction for
      // other tissue types using params - Hence, the gradient will change accordingly.
      // Implicitly assuming there's no glm. TODO: remove glm from all subsequent iterations of the solver.
      ierr = VecDot(tumor_->mat_prop_->gm_, temp_, &temp_scalar); CHKERRQ(ierr);
      temp_scalar *= params_->grid_->lebesgue_measure_;
      temp_scalar *= params_->tu_->r_gm_wm_ratio_;  // this ratio will control the reaction coefficient in gm
      x_ptr[params_->tu_->np_ + params_->tu_->nk_] += temp_scalar;
    }

    if (params_->tu_->nr_ > 1) {
      ierr = VecDot(tumor_->mat_prop_->gm_, temp_, &x_ptr[params_->tu_->np_ + params_->tu_->nk_ + 1]); CHKERRQ(ierr);
      x_ptr[params_->tu_->np_ + params_->tu_->nk_ + 1] *= params_->grid_->lebesgue_measure_;
    }

    if (params_->tu_->nr_ > 2) {
      ierr = VecDot(tumor_->mat_prop_->csf_, temp_, &x_ptr[params_->tu_->np_ + params_->tu_->nk_ + 2]); CHKERRQ(ierr);
      x_ptr[params_->tu_->np_ + params_->tu_->nk_ + 2] *= params_->grid_->lebesgue_measure_;
    }

    ierr = VecRestoreArray(dJ, &x_ptr); CHKERRQ(ierr);
  }

#if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 9)
  if (lock_state != 0) {
    x->lock = lock_state;
  }
#endif

  // timing
  self_exec_time += MPI_Wtime();
  t[5] = self_exec_time;
  e.addTimings(t);
  e.stop();

  PetscFunctionReturn(ierr);
}

PetscErrorCode DerivativeOperatorsRD::evaluateHessian(Vec y, Vec x) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  params_->tu_->statistics_.nb_hessian_evals++;

  if (params_->tu_->two_time_points_) {ierr = tuMSGwarn("Error: Hessian currently not implemented for two-snapshot scenario. Exiting..."); CHKERRQ(ierr); PetscFunctionReturn(1);}

  std::bitset<3> XYZ;
  XYZ[0] = 1;
  XYZ[1] = 1;
  XYZ[2] = 1;
  Event e("tumor-eval-hessian");
  std::array<double, 7> t = {0};
  double self_exec_time = -MPI_Wtime();
  ScalarType *y_ptr;

  if (params_->opt_->diffusivity_inversion_) {
    /* HESSIAN WITH DIFFUSIVITY INVERSION
      Hx = [Hpp p_tilde + Hpk k_tilde; Hkp p_tiilde + Hkk k_tilde]
      Each Matvec is computed separately by eliminating the
      incremental forward and adjoint equations and the result is added into y = Hx
    */
    //  -------------------  -------------------  -------------------  -------------------  -------------------  -------------------
    // --------------- Compute Hpp * p_tilde -------------------
    // Solve incr fwd with k_tilde = 0 and c0_tilde = \phi * p_tilde
    ierr = tumor_->phi_->apply(tumor_->c_0_, x); CHKERRQ(ierr);
    ierr = pde_operators_->solveState(1);
    // Solve incr adj with alpha1_tilde = -OT * O * c1_tilde
    ierr = tumor_->obs_->apply(temp_, tumor_->c_t_, 1); CHKERRQ(ierr);
    ierr = tumor_->obs_->applyT(tumor_->p_t_, temp_, 1); CHKERRQ(ierr);
    ierr = VecScale(tumor_->p_t_, -1.0); CHKERRQ(ierr);
    ierr = pde_operators_->solveAdjoint(2);
    // Matvec is \beta\phiT\phi p_tilde - \phiT \alpha0_tilde
    ierr = tumor_->phi_->applyTranspose(ptemp_, tumor_->p_0_);
    ierr = tumor_->phi_->applyTranspose(y, tumor_->c_0_); CHKERRQ(ierr);
    ierr = VecScale(y, params_->opt_->beta_); CHKERRQ(ierr);
    ierr = VecAXPY(y, -1.0, ptemp_); CHKERRQ(ierr);
    ierr = VecScale(y, params_->grid_->lebesgue_measure_); CHKERRQ(ierr);

    //  -------------------  -------------------  -------------------  -------------------  -------------------  -------------------
    // --------------- Compute Hkp * p_tilde -- \int \int m_i \grad c . \grad \alpha_tilde -------------------
    ScalarType integration_weight = 1.0;
    ScalarType temp_scalar = 0.;
    ierr = VecSet(temp_, 0.0); CHKERRQ(ierr);
    // compute numerical time integration using trapezoidal rule
    for (int i = 0; i < params_->tu_->nt_ + 1; i++) {
      // integration weight for chain trapezoidal rule
      if (i == 0 || i == params_->tu_->nt_)
        integration_weight = 0.5;
      else
        integration_weight = 1.0;

      // compute x = (grad c)^T grad \alpha_tilde
      // compute gradient of c(t)
      pde_operators_->spec_ops_->computeGradient(tumor_->work_[1], tumor_->work_[2], tumor_->work_[3], pde_operators_->c_[i], &XYZ, t.data());
      // compute gradient of \alpha_tilde(t)
      pde_operators_->spec_ops_->computeGradient(tumor_->work_[4], tumor_->work_[5], tumor_->work_[6], pde_operators_->p_[i], &XYZ, t.data());
      // scalar product (grad c)^T grad \alpha_tilde
      ierr = VecPointwiseMult(tumor_->work_[0], tumor_->work_[1], tumor_->work_[4]); CHKERRQ(ierr);
      ierr = VecPointwiseMult(tumor_->work_[1], tumor_->work_[2], tumor_->work_[5]); CHKERRQ(ierr);
      ierr = VecAXPY(tumor_->work_[0], 1.0, tumor_->work_[1]); CHKERRQ(ierr);
      ierr = VecPointwiseMult(tumor_->work_[1], tumor_->work_[3], tumor_->work_[6]); CHKERRQ(ierr);
      ierr = VecAXPY(tumor_->work_[0], 1.0, tumor_->work_[1]); CHKERRQ(ierr);

      // numerical time integration using trapezoidal rule
      ierr = VecAXPY(temp_, params_->tu_->dt_ * integration_weight, tumor_->work_[0]); CHKERRQ(ierr);
    }
    // time integration of [ int_0 (grad c)^T grad alpha_tilde dt ] done, result in temp_
    // integration over omega (i.e., inner product, as periodic boundary and no lebesque measure in tumor code)
    ierr = VecGetArray(y, &y_ptr); CHKERRQ(ierr);
    ierr = VecDot(tumor_->mat_prop_->wm_, temp_, &y_ptr[params_->tu_->np_]); CHKERRQ(ierr);
    y_ptr[params_->tu_->np_] *= params_->grid_->lebesgue_measure_;

    if (params_->tu_->nk_ == 1) {
      // Inverting for only one parameters a.k.a diffusivity in WM. Provide user with the option of setting a diffusivity for
      // other tissue types using params - Hence, the gradient will change accordingly.
      // Implicitly assuming there's no glm. TODO: remove glm from all subsequent iterations of the solver.
      ierr = VecDot(tumor_->mat_prop_->gm_, temp_, &temp_scalar); CHKERRQ(ierr);
      temp_scalar *= params_->grid_->lebesgue_measure_;
      temp_scalar *= params_->tu_->k_gm_wm_ratio_;  // this ratio will control the diffusivity in gm
      y_ptr[params_->tu_->np_] += temp_scalar;
    }

    if (params_->tu_->nk_ > 1) {
      ierr = VecDot(tumor_->mat_prop_->gm_, temp_, &y_ptr[params_->tu_->np_ + 1]); CHKERRQ(ierr);
      y_ptr[params_->tu_->np_ + 1] *= params_->grid_->lebesgue_measure_;
    }
    if (params_->tu_->nk_ > 2) {
      ierr = VecDot(tumor_->mat_prop_->csf_, temp_, &y_ptr[params_->tu_->np_ + 2]); CHKERRQ(ierr);
      y_ptr[params_->tu_->np_ + 2] *= params_->grid_->lebesgue_measure_;
    }
    ierr = VecRestoreArray(y, &y_ptr); CHKERRQ(ierr);

    //  -------------------  -------------------  -------------------  -------------------  -------------------  -------------------
    // --------------- Compute Hpk * k_tilde -- -\phiT \alpha0_tilde -------------------
    // Set c0_tilde to zero
    ierr = VecSet(tumor_->c_0_, 0.); CHKERRQ(ierr);
    // solve tumor incr fwd with k_tilde
    // get the update on kappa -- this is used in tandem with the actual kappa in
    // the incr fwd solves and hence we cannot re-use the diffusivity vectors
    // TODO: here, it is assumed that the update is isotropic updates- this has
    // to be modified later is anisotropy is included
    ScalarType k1, k2, k3;
    ierr = VecGetArray(x, &y_ptr);
    k1 = y_ptr[params_->tu_->np_];
    k2 = (params_->tu_->nk_ > 1) ? y_ptr[params_->tu_->np_ + 1] : 0.;
    k3 = (params_->tu_->nk_ > 2) ? y_ptr[params_->tu_->np_ + 2] : 0.;
    ierr = tumor_->k_->setSecondaryCoefficients(k1, k2, k3, tumor_->mat_prop_, params_); CHKERRQ(ierr);
    ierr = VecRestoreArray(x, &y_ptr);

    ierr = pde_operators_->solveState(2); CHKERRQ(ierr);
    // Solve incr adj with alpha1_tilde = -OT * O * c1_tilde
    ierr = tumor_->obs_->apply(temp_, tumor_->c_t_, 1); CHKERRQ(ierr);
    ierr = tumor_->obs_->applyT(tumor_->p_t_, temp_, 1); CHKERRQ(ierr);
    ierr = VecScale(tumor_->p_t_, -1.0); CHKERRQ(ierr);
    ierr = pde_operators_->solveAdjoint(2);
    // Matvec is  - \phiT \alpha0_tilde
    ierr = VecSet(ptemp_, 0.); CHKERRQ(ierr);
    ierr = tumor_->phi_->applyTranspose(ptemp_, tumor_->p_0_); CHKERRQ(ierr);
    ierr = VecScale(ptemp_, -params_->grid_->lebesgue_measure_); CHKERRQ(ierr);
    // Add Hpk k_tilde to Hpp p_tilde:  Note the kappa/rho components are zero
    // so are unchanged in y
    ierr = VecAXPY(y, 1.0, ptemp_); CHKERRQ(ierr);

    //  -------------------  -------------------  -------------------  -------------------  -------------------  -------------------
    // --------------- Compute Hkk * k_tilde -- \int \int mi \grad c \grad \alpha_tilde -------------------
    integration_weight = 1.0;
    ierr = VecSet(temp_, 0.0); CHKERRQ(ierr);
    // compute numerical time integration using trapezoidal rule
    for (int i = 0; i < params_->tu_->nt_ + 1; i++) {
      // integration weight for chain trapezoidal rule
      if (i == 0 || i == params_->tu_->nt_)
        integration_weight = 0.5;
      else
        integration_weight = 1.0;

      // compute x = (grad c)^T grad \alpha_tilde
      // compute gradient of c(t)
      pde_operators_->spec_ops_->computeGradient(tumor_->work_[1], tumor_->work_[2], tumor_->work_[3], pde_operators_->c_[i], &XYZ, t.data());
      // compute gradient of \alpha_tilde(t)
      pde_operators_->spec_ops_->computeGradient(tumor_->work_[4], tumor_->work_[5], tumor_->work_[6], pde_operators_->p_[i], &XYZ, t.data());
      // scalar product (grad c)^T grad \alpha_tilde
      ierr = VecPointwiseMult(tumor_->work_[0], tumor_->work_[1], tumor_->work_[4]); CHKERRQ(ierr);
      ierr = VecPointwiseMult(tumor_->work_[1], tumor_->work_[2], tumor_->work_[5]); CHKERRQ(ierr);
      ierr = VecAXPY(tumor_->work_[0], 1.0, tumor_->work_[1]); CHKERRQ(ierr);
      ierr = VecPointwiseMult(tumor_->work_[1], tumor_->work_[3], tumor_->work_[6]); CHKERRQ(ierr);
      ierr = VecAXPY(tumor_->work_[0], 1.0, tumor_->work_[1]); CHKERRQ(ierr);

      // numerical time integration using trapezoidal rule
      ierr = VecAXPY(temp_, params_->tu_->dt_ * integration_weight, tumor_->work_[0]); CHKERRQ(ierr);
    }
    // time integration of [ int_0 (grad c)^T grad alpha_tilde dt ] done, result in temp_
    // integration over omega (i.e., inner product, as periodic boundary and no lebesque measure in tumor code)
    ierr = VecGetArray(y, &y_ptr); CHKERRQ(ierr);
    ierr = VecDot(tumor_->mat_prop_->wm_, temp_, &temp_scalar); CHKERRQ(ierr);
    temp_scalar *= params_->grid_->lebesgue_measure_;
    y_ptr[params_->tu_->np_] += temp_scalar;

    if (params_->tu_->nk_ == 1) {
      // Inverting for only one parameters a.k.a diffusivity in WM. Provide user with the option of setting a diffusivity for
      // other tissue types using params - Hence, the gradient will change accordingly.
      // Implicitly assuming there's no glm. TODO: remove glm from all subsequent iterations of the solver.
      ierr = VecDot(tumor_->mat_prop_->gm_, temp_, &temp_scalar); CHKERRQ(ierr);
      temp_scalar *= params_->grid_->lebesgue_measure_;
      temp_scalar *= params_->tu_->k_gm_wm_ratio_;  // this ratio will control the diffusivity in gm
      y_ptr[params_->tu_->np_] += temp_scalar;
    }

    if (params_->tu_->nk_ > 1) {
      ierr = VecDot(tumor_->mat_prop_->gm_, temp_, &temp_scalar); CHKERRQ(ierr);
      temp_scalar *= params_->grid_->lebesgue_measure_;
      y_ptr[params_->tu_->np_ + 1] += temp_scalar;
    }
    if (params_->tu_->nk_ > 2) {
      ierr = VecDot(tumor_->mat_prop_->csf_, temp_, &temp_scalar); CHKERRQ(ierr);
      temp_scalar *= params_->grid_->lebesgue_measure_;
      y_ptr[params_->tu_->np_ + 2] += temp_scalar;
    }
    ierr = VecRestoreArray(y, &y_ptr); CHKERRQ(ierr);
  } else {
    ierr = tumor_->phi_->apply(tumor_->c_0_, x); CHKERRQ(ierr);
    ierr = pde_operators_->solveState(1);

    ierr = tumor_->obs_->apply(temp_, tumor_->c_t_, 1); CHKERRQ(ierr);
    ierr = tumor_->obs_->applyT(tumor_->p_t_, temp_, 1); CHKERRQ(ierr);
    ierr = VecScale(tumor_->p_t_, -1.0); CHKERRQ(ierr);

    ierr = pde_operators_->solveAdjoint(2);

    ierr = tumor_->phi_->applyTranspose(ptemp_, tumor_->p_0_);
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
  }
  self_exec_time += MPI_Wtime();
  t[5] = self_exec_time;
  e.addTimings(t);
  e.stop();

  PetscFunctionReturn(ierr);
}

PetscErrorCode DerivativeOperatorsRD::evaluateConstantHessianApproximation(Vec y, Vec x) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;

  ierr = tumor_->phi_->apply(tumor_->c_0_, x); CHKERRQ(ierr);
  ierr = tumor_->phi_->applyTranspose(y, tumor_->c_0_); CHKERRQ(ierr);
  ierr = VecScale(y, params_->opt_->beta_); CHKERRQ(ierr);
  PetscFunctionReturn(ierr);
}


/* #### --------------------------------------------------------------------- #### */
/* #### ========  Deriv. Ops.: Finite Diff. {rho,kappa} for RD Model ======== #### */
/* #### --------------------------------------------------------------------- #### */
PetscErrorCode DerivativeOperatorsRDOnlyFD::evaluateObjective (PetscReal *J, Vec x, std::shared_ptr<Data> data) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  params_->tu_->statistics_.nb_obj_evals++;
  int procid, nprocs;
  MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank (MPI_COMM_WORLD, &procid);

  ScalarType *x_ptr, k1, k2, k3;
  PetscReal m1 = 0, m0 = 0, reg = 0;

  ScalarType scale_rho = params_->opt_->rho_scale_;
  ScalarType scale_kap = params_->opt_->k_scale_;

  // Reset mat-props and diffusion and reaction operators, tumor IC does not change
  ierr = tumor_->mat_prop_->resetValues (); CHKERRQ (ierr);

  #if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 9)
  int lock_state;
  ierr = VecLockGet (x, &lock_state); CHKERRQ (ierr);
  if (lock_state != 0) {
    x->lock = 0;
  }
  #endif

  // update diffusion coefficient based on new k_i
  if (params_->opt_->diffusivity_inversion_ || params_->opt_->flag_reaction_inv_) {
    ierr = VecGetArray(x, &x_ptr); CHKERRQ (ierr);
    #ifdef POSITIVITY_DIFF_COEF
      //Positivity clipping in diffusio coefficient
      for(int i=0; i<params_->tu_->nk_; i++)
          x_ptr[i] = x_ptr[i] > 0 ? x_ptr[i] : 0;
    #endif
    k1 = scale_kap * x_ptr[0];
    k2 = (params_->tu_->nk_ > 1) ? scale_kap * x_ptr[1] : 0;
    k3 = (params_->tu_->nk_ > 2) ? scale_kap * x_ptr[2] : 0;
    ierr = VecRestoreArray(x, &x_ptr); CHKERRQ (ierr);
    ierr = tumor_->k_->updateIsotropicCoefficients(k1, k2, k3, tumor_->mat_prop_, params_); CHKERRQ(ierr);
    // need to update prefactors for diffusion KSP preconditioner, as k changed
    pde_operators_->diff_solver_->precFactor();
  }
  // update reaction coefficient based on new rho
  ScalarType r1, r2, r3;
  if (params_->opt_->flag_reaction_inv_) {
    ierr = VecGetArray(x, &x_ptr); CHKERRQ(ierr);
    r1 = scale_rho * x_ptr[params_->tu_->nk_];
    r2 = (params_->tu_->nr_ > 1) ? scale_rho * x_ptr[params_->tu_->nk_ + 1] : 0;
    r3 = (params_->tu_->nr_ > 2) ? scale_rho * x_ptr[params_->tu_->nk_ + 2] : 0;
    ierr = tumor_->rho_->updateIsotropicCoefficients(r1, r2, r3, tumor_->mat_prop_, params_);
    ierr = VecRestoreArray(x, &x_ptr); CHKERRQ(ierr);
  }
  std::stringstream s;
  if (params_->tu_->verbosity_ >= 3) {
    if (params_->opt_->diffusivity_inversion_ || params_->opt_->flag_reaction_inv_) {
      s << " Diffusivity guess = (" << k1 << ", " << k2 << ", " << k3 << ")"; ierr = tuMSGstd(s.str()); CHKERRQ(ierr); s.str(""); s.clear();
    }
    if (params_->opt_->flag_reaction_inv_) {
      s << " Reaction  guess   = (" << r1 << ", " << r2 << ", " << r3 << ")"; ierr = tuMSGstd(s.str()); CHKERRQ(ierr); s.str(""); s.clear();
    }
  }

  // compute mismatch ||Oc(1) - d1||
  ierr = pde_operators_->solveState(0); CHKERRQ (ierr);               // c(1)
  ierr = tumor_->obs_->apply(temp_, tumor_->c_t_, 1); CHKERRQ (ierr); // Oc(1)
  ierr = VecAXPY(temp_, -1.0, data->dt1()); CHKERRQ (ierr);           // Oc(1) - d1
  ierr = VecDot(temp_, temp_, &m1); CHKERRQ (ierr);                   // ||.||^2

  // compute mismatch ||Oc(0) - d0||
  if (params_->tu_->two_time_points_) {
      ierr = tumor_->obs_->apply (temp_, tumor_->c_0_, 0); CHKERRQ (ierr); // Oc(0)
      ierr = VecAXPY (temp_, -1.0, data->dt0()); CHKERRQ (ierr);           // Oc(0) - d0
      ierr = VecDot (temp_, temp_, &m0); CHKERRQ (ierr);                   // ||.||^2
      // compute regularization (modified observation operator)
      //ierr = tumor_->obs_->apply (temp_, tumor_->c_t_, 1, true);  CHKERRQ (ierr); // I-Oc(1)
      //ierr = VecDot (temp_, temp_, &reg);                          CHKERRQ (ierr); // ||.||^2
      //reg *= 0.5 * params_->opt_->beta_;
  }

  // objective function value
  (*J) = params_->grid_->lebesgue_measure_ * 0.5 *(m1 + m0) + reg;

  if (params_->tu_->two_time_points_) {
    s << "  obj: J(p) = D(c1) + D(c0) + S(c0) = "<< std::setprecision(12) << (*J)<<" = " << std::setprecision(12)<<  params_->grid_->lebesgue_measure_ *0.5*m1 <<" + " << std::setprecision(12)<<  params_->grid_->lebesgue_measure_ * 0.5 * m0 <<" + "<< std::setprecision(12) <<reg<<"";  ierr = tuMSGstd(s.str()); CHKERRQ(ierr); s.str(""); s.clear();
  } else {
    s << "  obj: J(p) = D(c1) + S(c0) = "<< std::setprecision(12) << (*J)<<" = " << std::setprecision(12)<<  params_->grid_->lebesgue_measure_ * 0.5 * m1 <<" + "<< std::setprecision(12) <<reg<<"";  ierr = tuMSGstd(s.str()); CHKERRQ(ierr); s.str(""); s.clear();
  }

  #if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 9)
  if (lock_state != 0) {
    x->lock = lock_state;
  }
  #endif
  PetscFunctionReturn (ierr);
}

PetscErrorCode DerivativeOperatorsRDOnlyFD::evaluateGradient (Vec dJ, Vec x, std::shared_ptr<Data> data){
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  const ScalarType *x_ptr;
  std::bitset<3> XYZ; XYZ[0] = 1; XYZ[1] = 1; XYZ[2] = 1;
  params_->tu_->statistics_.nb_grad_evals++;
  Event e ("tumor-eval-grad");
  std::array<double, 7> t = {0};
  double self_exec_time = -MPI_Wtime ();
  int procid, nprocs;
  MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank (MPI_COMM_WORLD, &procid);

  //ScalarType scale_rho = 1;
  //ScalarType scale_kap = 1E-2;

  #if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 9)
  int lock_state;
  ierr = VecLockGet (x, &lock_state); CHKERRQ (ierr);
  if (lock_state != 0) {
    x->lock = 0;
  }
  #endif

  // Finite difference gradient (forward differences)
  ScalarType h, dx;
  ScalarType volatile xph;
  PetscReal J_f, J_b;
  Vec delta_;

  ierr = VecDuplicate(x, &delta_); CHKERRQ (ierr);
  ierr = evaluateObjective (&J_b, x, data); CHKERRQ (ierr);
  int sz;
  ScalarType *delta_ptr, *dj_ptr;
  ierr = VecGetSize (x, &sz); CHKERRQ (ierr);
  ierr = VecGetArray (dJ, &dj_ptr); CHKERRQ (ierr);
  std::array<ScalarType, 3> characteristic_scale = {1, 1, 1};
  //std::array<ScalarType, 3> characteristic_scale = {scale_kap * 1, scale_rho * 1, 1};
  #ifdef SINGLE
  ScalarType small = 3.45266983e-04F;
  #else
  ScalarType small = 3.45266983e-04;
  #endif
  for (int i = 0; i < sz; i++) {
    ierr = VecCopy (x, delta_); CHKERRQ (ierr);
    ierr = VecGetArray (delta_, &delta_ptr);  CHKERRQ (ierr);
    ierr = VecGetArrayRead (x, &x_ptr); CHKERRQ (ierr);
    h = (x_ptr[i] == 0) ? small * characteristic_scale[i] : small * x_ptr[i] * characteristic_scale[i];
    xph = x_ptr[i] + h;
    dx = xph - x_ptr[i];
    delta_ptr[i] = xph;
    ierr = VecRestoreArray (delta_, &delta_ptr); CHKERRQ (ierr);
    ierr = evaluateObjective (&J_f, delta_, data); CHKERRQ (ierr);
    dj_ptr[i] = (J_f - J_b) / dx;
    ierr = VecRestoreArrayRead (x, &x_ptr); CHKERRQ (ierr);
  }
  ierr = VecRestoreArray (dJ, &dj_ptr); CHKERRQ (ierr);

  #if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 9)
  if (lock_state != 0) {
    x->lock = lock_state;
  }
  #endif

  if(procid == 0) { ierr = VecView(dJ, PETSC_VIEWER_STDOUT_SELF); CHKERRQ (ierr);}
  if(delta_ != nullptr) {ierr = VecDestroy(&delta_); CHKERRQ(ierr);}

  // timing
  self_exec_time += MPI_Wtime(); t[5] = self_exec_time; e.addTimings (t); e.stop ();
  PetscFunctionReturn (ierr);
}

PetscErrorCode DerivativeOperatorsRDOnlyFD::evaluateObjectiveAndGradient (PetscReal *J, Vec dJ, Vec x, std::shared_ptr<Data> data) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  params_->tu_->statistics_.nb_obj_evals++;
  params_->tu_->statistics_.nb_grad_evals++;
  std::bitset<3> XYZ; XYZ[0] = 1; XYZ[1] = 1; XYZ[2] = 1;
  Event e ("tumor-eval-objandgrad");
  std::array<double, 7> t = {0};
  double self_exec_time = -MPI_Wtime ();

  int procid, nprocs;
  MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank (MPI_COMM_WORLD, &procid);

  //ScalarType scale_rho = 1;
  //ScalarType scale_kap = 1E-02

  #if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 9)
  int lock_state;
  ierr = VecLockGet (x, &lock_state); CHKERRQ (ierr);
  if (lock_state != 0) {
    x->lock = 0;
  }
  #endif

  ierr = evaluateObjective (J, x, data); CHKERRQ(ierr);

  // Finite difference gradient (forward differences)
  ScalarType h, dx;
  ScalarType volatile xph;
  PetscReal J_f;
  Vec delta_;
  ierr = VecDuplicate(x, &delta_); CHKERRQ (ierr);

  int sz;
  ScalarType *delta_ptr, *dj_ptr;
  ScalarType const *x_ptr;
  ierr = VecGetSize (x, &sz); CHKERRQ (ierr);
  ierr = VecGetArray (dJ, &dj_ptr); CHKERRQ (ierr);

  ScalarType scale = 1;
  std::array<ScalarType, 3> characteristic_scale = {1, 1, 1};
  //std::array<ScalarType, 3> characteristic_scale = {scale_kap * 1, scale_rho * 1, 1};
  #ifdef SINGLE
  ScalarType small = 3.45266983e-04F;
  #else
  ScalarType small = 3.45266983e-04;
  #endif
  ScalarType J_b = (*J);

  for (int i = 0; i < sz; i++) {
    ierr = VecCopy (x, delta_); CHKERRQ (ierr);
    ierr = VecGetArray (delta_, &delta_ptr); CHKERRQ (ierr);
    ierr = VecGetArrayRead (x, &x_ptr); CHKERRQ (ierr);
    h = (x_ptr[i] == 0) ? small * characteristic_scale[i] : small * x_ptr[i] * characteristic_scale[i];
    xph = x_ptr[i] + h;
    dx = xph - x_ptr[i];
    delta_ptr[i] = xph;
    ierr = VecRestoreArray (delta_, &delta_ptr); CHKERRQ (ierr);
    ierr = evaluateObjective (&J_f, delta_, data); CHKERRQ (ierr);
    dj_ptr[i] = (J_f - J_b) / dx;
    ierr = VecRestoreArrayRead (x, &x_ptr); CHKERRQ (ierr);
  }

  ierr = VecRestoreArray (dJ, &dj_ptr); CHKERRQ (ierr);

  #if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 9)
  if (lock_state != 0) {
    x->lock = lock_state;
  }
  #endif

  if(procid == 0) { ierr = VecView(dJ, PETSC_VIEWER_STDOUT_SELF); CHKERRQ (ierr);}
  if(delta_ != nullptr) {ierr = VecDestroy(&delta_); CHKERRQ(ierr);}
  // timing
  self_exec_time += MPI_Wtime(); t[5] = self_exec_time; e.addTimings (t); e.stop ();
  PetscFunctionReturn (ierr);
}

PetscErrorCode DerivativeOperatorsRDOnlyFD::evaluateHessian (Vec y, Vec x){
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    params_->tu_->statistics_.nb_hessian_evals++;

    std::bitset<3> XYZ; XYZ[0] = 1; XYZ[1] = 1; XYZ[2] = 1;
    Event e ("tumor-eval-hessian");
    std::array<double, 7> t = {0};
    double self_exec_time = -MPI_Wtime ();

    // no-op, i.e., gradient descent
    ierr = VecCopy(x, y); CHKERRQ (ierr);

    self_exec_time += MPI_Wtime(); t[5] = self_exec_time; e.addTimings (t); e.stop ();
    PetscFunctionReturn (ierr);
}




/* #### -------------------------------------------------------------------------------------- #### */
/* #### ========  Deriv. Ops.: Adjoints {rho,kappa,p} for RD Model with KL-Divergence ======== #### */
/* #### -------------------------------------------------------------------------------------- #### */
PetscErrorCode DerivativeOperatorsKL::evaluateObjective(PetscReal *J, Vec x, std::shared_ptr<Data> data_inv) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  params_->tu_->statistics_.nb_obj_evals++;
  ScalarType *x_ptr, k1, k2, k3;

  int x_sz;
#if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 9)
  int lock_state;
  ierr = VecLockGet(x, &lock_state); CHKERRQ(ierr);
  if (lock_state != 0) {
    x->lock = 0;
  }
#endif
  Vec data = data_inv->dt1();

  if (params_->opt_->diffusivity_inversion_ || params_->opt_->flag_reaction_inv_) {
    ierr = VecGetArray(x, &x_ptr); CHKERRQ(ierr);
    k1 = x_ptr[params_->tu_->np_];
    k2 = (params_->tu_->nk_ > 1) ? x_ptr[params_->tu_->np_ + 1] : 0;
    k3 = (params_->tu_->nk_ > 2) ? x_ptr[params_->tu_->np_ + 2] : 0;
    ierr = VecRestoreArray(x, &x_ptr); CHKERRQ(ierr);
    ierr = tumor_->k_->updateIsotropicCoefficients(k1, k2, k3, tumor_->mat_prop_, params_); CHKERRQ(ierr);
    // need to update prefactors for diffusion KSP preconditioner, as k changed
    pde_operators_->diff_solver_->precFactor();
  }

  ScalarType r1, r2, r3;
  if (params_->opt_->flag_reaction_inv_) {
    ierr = VecGetArray(x, &x_ptr); CHKERRQ(ierr);
    r1 = x_ptr[params_->tu_->np_ + params_->tu_->nk_];
    r2 = (params_->tu_->nr_ > 1) ? x_ptr[params_->tu_->np_ + params_->tu_->nk_ + 1] : 0;
    r3 = (params_->tu_->nr_ > 2) ? x_ptr[params_->tu_->np_ + params_->tu_->nk_ + 2] : 0;
    ierr = tumor_->rho_->updateIsotropicCoefficients(r1, r2, r3, tumor_->mat_prop_, params_);
    ierr = VecRestoreArray(x, &x_ptr); CHKERRQ(ierr);
  }

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

  if (!params_->opt_->flag_reaction_inv_) {
    ierr = tumor_->phi_->apply(tumor_->c_0_, x); CHKERRQ(ierr);
  }
  ierr = pde_operators_->solveState(0);
  // cross entropy obj is -(dlog(c) + (1-d)*log(1-c))
  ScalarType eps = eps_;
  *J = 0;
  ScalarType *c_ptr, *d_ptr, *ce_ptr;
  ierr = vecGetArray(tumor_->c_t_, &c_ptr); CHKERRQ(ierr);
  ierr = vecGetArray(data, &d_ptr); CHKERRQ(ierr);
  ierr = vecGetArray(temp_, &ce_ptr); CHKERRQ(ierr);
#ifdef CUDA
  computeCrossEntropyCuda(ce_ptr, d_ptr, c_ptr, eps, params_->grid_->nl_);
  // vecSumCuda(ce_ptr, J, params_->grid_->nl_);
  cublasStatus_t status;
  cublasHandle_t handle;
  PetscCUBLASGetHandle(&handle);
  status = cublasSum(handle, params_->grid_->nl_, ce_ptr, 1, J);
  cublasCheckError(status);
#else
  for (int i = 0; i < params_->grid_->nl_; i++) {
    c_ptr[i] = (c_ptr[i] < eps) ? eps : c_ptr[i];
    c_ptr[i] = (c_ptr[i] > 1 - eps) ? 1 - eps : c_ptr[i];
    (*J) += -(d_ptr[i] * log(c_ptr[i]) + (1 - d_ptr[i]) * log(1 - c_ptr[i]));
  }
#endif
  ierr = vecRestoreArray(tumor_->c_t_, &c_ptr); CHKERRQ(ierr);
  ierr = vecRestoreArray(data, &d_ptr); CHKERRQ(ierr);
  ierr = vecRestoreArray(temp_, &ce_ptr); CHKERRQ(ierr);

  /*Regularization term*/
  PetscReal reg = 0;
  if (params_->opt_->regularization_norm_ == L2) {  // In tumor space, so scale norm by lebesque measure
    ierr = VecDot(tumor_->c_0_, tumor_->c_0_, &reg); CHKERRQ(ierr);
    reg *= 0.5 * params_->opt_->beta_;
    reg *= params_->grid_->lebesgue_measure_;
  } else if (params_->opt_->regularization_norm_ == L2b) {
    // Reg term only on the initial condition. Leave out the diffusivity.
    ierr = VecGetArray(x, &x_ptr); CHKERRQ(ierr);
    for (int i = 0; i < params_->tu_->np_; i++) {
      reg += x_ptr[i] * x_ptr[i];
    }
    ierr = VecRestoreArray(x, &x_ptr); CHKERRQ(ierr);
    reg *= 0.5 * params_->opt_->beta_;
  }

  (*J) *= params_->grid_->lebesgue_measure_;

  s << "  obj: J(p) = Dc(c) + S(c0) = " << std::setprecision(12) << (*J) + reg << " = " << std::setprecision(12) << (*J) << " + " << std::setprecision(12) << reg << "";
  ierr = tuMSGstd(s.str()); CHKERRQ(ierr);
  s.str("");
  s.clear();

  (*J) += reg;

#if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 9)
  if (lock_state != 0) {
    x->lock = lock_state;
  }
#endif
  PetscFunctionReturn(ierr);
}

PetscErrorCode DerivativeOperatorsKL::evaluateGradient(Vec dJ, Vec x, std::shared_ptr<Data> data_inv) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  ScalarType *x_ptr, *p_ptr;
  std::bitset<3> XYZ;
  XYZ[0] = 1;
  XYZ[1] = 1;
  XYZ[2] = 1;
  params_->tu_->statistics_.nb_grad_evals++;
  Event e("tumor-eval-grad");
  std::array<double, 7> t = {0};
  double self_exec_time = -MPI_Wtime();
  ScalarType k1, k2, k3;

  int x_sz;
#if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 9)
  int lock_state;
  ierr = VecLockGet(x, &lock_state); CHKERRQ(ierr);
  if (lock_state != 0) {
    x->lock = 0;
  }
#endif

  Vec data = data_inv->dt1();

  if (params_->opt_->diffusivity_inversion_ || params_->opt_->flag_reaction_inv_) {
    ierr = VecGetArray(x, &x_ptr); CHKERRQ(ierr);
    k1 = x_ptr[params_->tu_->np_];
    k2 = (params_->tu_->nk_ > 1) ? x_ptr[params_->tu_->np_ + 1] : 0;
    k3 = (params_->tu_->nk_ > 2) ? x_ptr[params_->tu_->np_ + 2] : 0;
    ierr = VecRestoreArray(x, &x_ptr); CHKERRQ(ierr);
    ierr = tumor_->k_->updateIsotropicCoefficients(k1, k2, k3, tumor_->mat_prop_, params_); CHKERRQ(ierr);
    // need to update prefactors for diffusion KSP preconditioner, as k changed
    pde_operators_->diff_solver_->precFactor();
  }

  ScalarType r1, r2, r3;
  if (params_->opt_->flag_reaction_inv_) {
    ierr = VecGetArray(x, &x_ptr); CHKERRQ(ierr);
    r1 = x_ptr[params_->tu_->np_ + params_->tu_->nk_];
    r2 = (params_->tu_->nr_ > 1) ? x_ptr[params_->tu_->np_ + params_->tu_->nk_ + 1] : 0;
    r3 = (params_->tu_->nr_ > 2) ? x_ptr[params_->tu_->np_ + params_->tu_->nk_ + 2] : 0;
    ierr = tumor_->rho_->updateIsotropicCoefficients(r1, r2, r3, tumor_->mat_prop_, params_);
    ierr = VecRestoreArray(x, &x_ptr); CHKERRQ(ierr);
  }

  /* ------------------ */
  /* (1) compute grad_p */
  // c = Phi(p), solve state
  if (!params_->opt_->flag_reaction_inv_) {
    ierr = tumor_->phi_->apply(tumor_->c_0_, x); CHKERRQ(ierr);
  }
  ierr = pde_operators_->solveState(0);
  // final cond adjoint
  ScalarType eps = eps_;
  ScalarType *c_ptr, *d_ptr, *a_ptr;
  ierr = vecGetArray(tumor_->c_t_, &c_ptr); CHKERRQ(ierr);
  ierr = vecGetArray(data, &d_ptr); CHKERRQ(ierr);
  ierr = vecGetArray(tumor_->p_t_, &a_ptr); CHKERRQ(ierr);
#ifdef CUDA
  computeCrossEntropyAdjointICCuda(a_ptr, d_ptr, c_ptr, eps, params_->grid_->nl_);
#else
  for (int i = 0; i < params_->grid_->nl_; i++) {
    c_ptr[i] = (c_ptr[i] < eps) ? eps : c_ptr[i];
    c_ptr[i] = (c_ptr[i] > 1 - eps) ? 1 - eps : c_ptr[i];
    a_ptr[i] = (d_ptr[i] / (c_ptr[i]) - (1 - d_ptr[i]) / (1 - c_ptr[i]));
  }
#endif
  ierr = vecRestoreArray(tumor_->c_t_, &c_ptr); CHKERRQ(ierr);
  ierr = vecRestoreArray(data, &d_ptr); CHKERRQ(ierr);
  ierr = vecRestoreArray(tumor_->p_t_, &a_ptr); CHKERRQ(ierr);
  // solve adjoint
  ierr = pde_operators_->solveAdjoint(1);
  // compute gradient
  if (!params_->opt_->flag_reaction_inv_) {
    if (!params_->tu_->phi_store_) {
      // restructure phi compute because it is now expensive
      // assume that reg norm is L2 for now
      // TODO: change to normal if reg norm is not L2

      // p0 = p0 - beta * phi * p
      ierr = VecAXPY(tumor_->p_0_, -params_->opt_->beta_, tumor_->c_0_); CHKERRQ(ierr);
      // dJ is phiT p0 - beta * phiT * phi * p
      ierr = tumor_->phi_->applyTranspose(dJ, tumor_->p_0_); CHKERRQ(ierr);
      // dJ is beta * phiT * phi * p - phiT * p0
      ierr = VecScale(dJ, -params_->grid_->lebesgue_measure_); CHKERRQ(ierr);

    } else {
      ierr = tumor_->phi_->applyTranspose(ptemp_, tumor_->p_0_);
      ierr = VecScale(ptemp_, params_->grid_->lebesgue_measure_); CHKERRQ(ierr);

      // Gradient according to reg parameter chosen
      if (params_->opt_->regularization_norm_ == L2) {
        ierr = tumor_->phi_->applyTranspose(dJ, tumor_->c_0_);
        ierr = VecScale(dJ, params_->opt_->beta_ * params_->grid_->lebesgue_measure_); CHKERRQ(ierr);
        ierr = VecAXPY(dJ, -1.0, ptemp_); CHKERRQ(ierr);
      } else if (params_->opt_->regularization_norm_ == L2b) {
        ierr = VecCopy(x, dJ); CHKERRQ(ierr);
        ierr = VecScale(dJ, params_->opt_->beta_); CHKERRQ(ierr);
        ierr = VecAXPY(dJ, -1.0, ptemp_); CHKERRQ(ierr);
      }
    }
  }

  ScalarType temp_scalar;
  /* ------------------------- */
  /* INVERSION FOR DIFFUSIVITY */
  /* ------------------------- */
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
    ierr = VecGetArray(dJ, &x_ptr); CHKERRQ(ierr);
    ierr = VecDot(tumor_->mat_prop_->wm_, temp_, &x_ptr[params_->tu_->np_]); CHKERRQ(ierr);
    x_ptr[params_->tu_->np_] *= params_->grid_->lebesgue_measure_;

    if (params_->tu_->nk_ == 1) {
      // Inverting for only one parameters a.k.a diffusivity in WM. Provide user with the option of setting a diffusivity for
      // other tissue types using params - Hence, the gradient will change accordingly.
      // Implicitly assuming there's no glm. TODO: remove glm from all subsequent iterations of the solver.
      ierr = VecDot(tumor_->mat_prop_->gm_, temp_, &temp_scalar); CHKERRQ(ierr);
      temp_scalar *= params_->grid_->lebesgue_measure_;
      temp_scalar *= params_->tu_->k_gm_wm_ratio_;  // this ratio will control the diffusivity in gm
      x_ptr[params_->tu_->np_] += temp_scalar;
    }

    if (params_->tu_->nk_ > 1) {
      ierr = VecDot(tumor_->mat_prop_->gm_, temp_, &x_ptr[params_->tu_->np_ + 1]); CHKERRQ(ierr);
      x_ptr[params_->tu_->np_ + 1] *= params_->grid_->lebesgue_measure_;
    }
    if (params_->tu_->nk_ > 2) {
      ierr = VecDot(tumor_->mat_prop_->csf_, temp_, &x_ptr[params_->tu_->np_ + 2]); CHKERRQ(ierr);
      x_ptr[params_->tu_->np_ + 2] *= params_->grid_->lebesgue_measure_;
    }
    ierr = VecRestoreArray(dJ, &x_ptr); CHKERRQ(ierr);
  }

  /* INVERSION FOR REACTION COEFFICIENT */
  integration_weight = 1.0;
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

    ierr = VecGetArray(dJ, &x_ptr); CHKERRQ(ierr);
    ierr = VecDot(tumor_->mat_prop_->wm_, temp_, &x_ptr[params_->tu_->np_ + params_->tu_->nk_]); CHKERRQ(ierr);
    x_ptr[params_->tu_->np_ + params_->tu_->nk_] *= params_->grid_->lebesgue_measure_;

    if (params_->tu_->nr_ == 1) {
      // Inverting for only one parameters a.k.a reaction in WM. Provide user with the option of setting a reaction for
      // other tissue types using params - Hence, the gradient will change accordingly.
      // Implicitly assuming there's no glm. TODO: remove glm from all subsequent iterations of the solver.
      ierr = VecDot(tumor_->mat_prop_->gm_, temp_, &temp_scalar); CHKERRQ(ierr);
      temp_scalar *= params_->grid_->lebesgue_measure_;
      temp_scalar *= params_->tu_->r_gm_wm_ratio_;  // this ratio will control the reaction coefficient in gm
      x_ptr[params_->tu_->np_ + params_->tu_->nk_] += temp_scalar;
    }

    if (params_->tu_->nr_ > 1) {
      ierr = VecDot(tumor_->mat_prop_->gm_, temp_, &x_ptr[params_->tu_->np_ + params_->tu_->nk_ + 1]); CHKERRQ(ierr);
      x_ptr[params_->tu_->np_ + params_->tu_->nk_ + 1] *= params_->grid_->lebesgue_measure_;
    }

    if (params_->tu_->nr_ > 2) {
      ierr = VecDot(tumor_->mat_prop_->csf_, temp_, &x_ptr[params_->tu_->np_ + params_->tu_->nk_ + 2]); CHKERRQ(ierr);
      x_ptr[params_->tu_->np_ + params_->tu_->nk_ + 2] *= params_->grid_->lebesgue_measure_;
    }

    ierr = VecRestoreArray(dJ, &x_ptr); CHKERRQ(ierr);
  }

#if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 9)
  if (lock_state != 0) {
    x->lock = lock_state;
  }
#endif
  // timing
  self_exec_time += MPI_Wtime();
  t[5] = self_exec_time;
  e.addTimings(t);
  e.stop();
  PetscFunctionReturn(ierr);
}

// saves on forward solve
PetscErrorCode DerivativeOperatorsKL::evaluateObjectiveAndGradient(PetscReal *J, Vec dJ, Vec x, std::shared_ptr<Data> data_inv) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  params_->tu_->statistics_.nb_obj_evals++;
  params_->tu_->statistics_.nb_grad_evals++;
  std::bitset<3> XYZ;
  XYZ[0] = 1;
  XYZ[1] = 1;
  XYZ[2] = 1;
  Event e("tumor-eval-objandgrad");
  std::array<double, 7> t = {0};
  double self_exec_time = -MPI_Wtime();
  ScalarType *x_ptr, k1, k2, k3;

  int x_sz;
#if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 9)
  int lock_state;
  ierr = VecLockGet(x, &lock_state); CHKERRQ(ierr);
  if (lock_state != 0) {
    x->lock = 0;
  }
#endif

  Vec data = data_inv->dt1();

  if (params_->opt_->diffusivity_inversion_ || params_->opt_->flag_reaction_inv_) {  // if solveForParameters is happening always invert for diffusivity
    ierr = VecGetArray(x, &x_ptr); CHKERRQ(ierr);
    k1 = x_ptr[params_->tu_->np_];
    k2 = (params_->tu_->nk_ > 1) ? x_ptr[params_->tu_->np_ + 1] : 0;
    k3 = (params_->tu_->nk_ > 2) ? x_ptr[params_->tu_->np_ + 2] : 0;
    ierr = VecRestoreArray(x, &x_ptr); CHKERRQ(ierr);
    ierr = tumor_->k_->updateIsotropicCoefficients(k1, k2, k3, tumor_->mat_prop_, params_); CHKERRQ(ierr);
    // need to update prefactors for diffusion KSP preconditioner, as k changed
    pde_operators_->diff_solver_->precFactor();
  }

  ScalarType r1, r2, r3;
  if (params_->opt_->flag_reaction_inv_) {
    ierr = VecGetArray(x, &x_ptr); CHKERRQ(ierr);
    r1 = x_ptr[params_->tu_->np_ + params_->tu_->nk_];
    r2 = (params_->tu_->nr_ > 1) ? x_ptr[params_->tu_->np_ + params_->tu_->nk_ + 1] : 0;
    r3 = (params_->tu_->nr_ > 2) ? x_ptr[params_->tu_->np_ + params_->tu_->nk_ + 2] : 0;
    ierr = tumor_->rho_->updateIsotropicCoefficients(r1, r2, r3, tumor_->mat_prop_, params_);
    ierr = VecRestoreArray(x, &x_ptr); CHKERRQ(ierr);
  }

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

  // solve state
  if (!params_->opt_->flag_reaction_inv_) {
    ierr = tumor_->phi_->apply(tumor_->c_0_, x); CHKERRQ(ierr);
  }
  ierr = pde_operators_->solveState(0);
  // cross entropy obj is -(dlog(c) + (1-d)*log(1-c))
  ScalarType eps = eps_;
  *J = 0;
  ScalarType *c_ptr, *d_ptr, *a_ptr, *ce_ptr;
  ierr = vecGetArray(tumor_->c_t_, &c_ptr); CHKERRQ(ierr);
  ierr = vecGetArray(tumor_->p_t_, &a_ptr); CHKERRQ(ierr);
  ierr = vecGetArray(data, &d_ptr); CHKERRQ(ierr);
  ierr = vecGetArray(temp_, &ce_ptr); CHKERRQ(ierr);
#ifdef CUDA
  computeCrossEntropyCuda(ce_ptr, d_ptr, c_ptr, eps, params_->grid_->nl_);
  // vecSumCuda(ce_ptr, J, params_->grid_->nl_);
  cublasStatus_t status;
  cublasHandle_t handle;
  PetscCUBLASGetHandle(&handle);
  status = cublasSum(handle, params_->grid_->nl_, ce_ptr, 1, J);
  cublasCheckError(status);
  computeCrossEntropyAdjointICCuda(a_ptr, d_ptr, c_ptr, eps, params_->grid_->nl_);
#else
  for (int i = 0; i < params_->grid_->nl_; i++) {
    c_ptr[i] = (c_ptr[i] < eps) ? eps : c_ptr[i];
    c_ptr[i] = (c_ptr[i] > 1 - eps) ? 1 - eps : c_ptr[i];
    (*J) += -(d_ptr[i] * log(c_ptr[i]) + (1 - d_ptr[i]) * log(1 - c_ptr[i]));
    a_ptr[i] = (d_ptr[i] / (c_ptr[i]) - (1 - d_ptr[i]) / (1 - c_ptr[i]));
  }
#endif
  ierr = vecRestoreArray(temp_, &ce_ptr); CHKERRQ(ierr);
  ierr = vecRestoreArray(tumor_->c_t_, &c_ptr); CHKERRQ(ierr);
  ierr = vecRestoreArray(tumor_->p_t_, &a_ptr); CHKERRQ(ierr);
  ierr = vecRestoreArray(data, &d_ptr); CHKERRQ(ierr);
  ierr = pde_operators_->solveAdjoint(1);

  if (!params_->opt_->flag_reaction_inv_) {
    if (!params_->tu_->phi_store_) {
      // restructure phi compute because it is now expensive
      // assume that reg norm is L2 for now
      // TODO: change to normal if reg norm is not L2

      // p0 = p0 - beta * phi * p
      ierr = VecAXPY(tumor_->p_0_, -params_->opt_->beta_, tumor_->c_0_); CHKERRQ(ierr);
      // dJ is phiT p0 - beta * phiT * phi * p
      ierr = tumor_->phi_->applyTranspose(dJ, tumor_->p_0_); CHKERRQ(ierr);
      // dJ is beta * phiT * phi * p - phiT * p0
      ierr = VecScale(dJ, -params_->grid_->lebesgue_measure_); CHKERRQ(ierr);

    } else {
      ierr = tumor_->phi_->applyTranspose(ptemp_, tumor_->p_0_);
      ierr = VecScale(ptemp_, params_->grid_->lebesgue_measure_); CHKERRQ(ierr);

      // Gradient according to reg parameter chosen
      if (params_->opt_->regularization_norm_ == L2) {
        ierr = tumor_->phi_->applyTranspose(dJ, tumor_->c_0_);
        ierr = VecScale(dJ, params_->opt_->beta_ * params_->grid_->lebesgue_measure_); CHKERRQ(ierr);
        ierr = VecAXPY(dJ, -1.0, ptemp_); CHKERRQ(ierr);
      } else if (params_->opt_->regularization_norm_ == L2b) {
        ierr = VecCopy(x, dJ); CHKERRQ(ierr);
        ierr = VecScale(dJ, params_->opt_->beta_); CHKERRQ(ierr);
        ierr = VecAXPY(dJ, -1.0, ptemp_); CHKERRQ(ierr);
      }
    }
  }

  // regularization
  PetscReal reg = 0;
  if (params_->opt_->regularization_norm_ == L2) {
    ierr = VecDot(tumor_->c_0_, tumor_->c_0_, &reg); CHKERRQ(ierr);
    reg *= 0.5 * params_->opt_->beta_;
    reg *= params_->grid_->lebesgue_measure_;
  } else if (params_->opt_->regularization_norm_ == L2b) {
    // Reg term only on the initial condition. Leave out the diffusivity.
    ierr = VecGetArray(x, &x_ptr); CHKERRQ(ierr);
    for (int i = 0; i < params_->tu_->np_; i++) {
      reg += x_ptr[i] * x_ptr[i];
    }
    ierr = VecRestoreArray(x, &x_ptr); CHKERRQ(ierr);
    reg *= 0.5 * params_->opt_->beta_;
  }

  (*J) *= params_->grid_->lebesgue_measure_;

  s << "  obj: J(p) = Dc(c) + S(c0) = " << std::setprecision(12) << (*J) + reg << " = " << std::setprecision(12) << (*J) << " + " << std::setprecision(12) << reg << "";
  ierr = tuMSGstd(s.str()); CHKERRQ(ierr);
  s.str("");
  s.clear();
  // objective function value
  (*J) += reg;

  ScalarType temp_scalar;
  /* ------------------------- */
  /* INVERSION FOR DIFFUSIVITY */
  /* ------------------------- */
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
    ierr = VecGetArray(dJ, &x_ptr); CHKERRQ(ierr);
    ierr = VecDot(tumor_->mat_prop_->wm_, temp_, &x_ptr[params_->tu_->np_]); CHKERRQ(ierr);
    x_ptr[params_->tu_->np_] *= params_->grid_->lebesgue_measure_;

    if (params_->tu_->nk_ == 1) {
      // Inverting for only one parameters a.k.a diffusivity in WM. Provide user with the option of setting a diffusivity for
      // other tissue types using params - Hence, the gradient will change accordingly.
      // Implicitly assuming there's no glm. TODO: remove glm from all subsequent iterations of the solver.
      ierr = VecDot(tumor_->mat_prop_->gm_, temp_, &temp_scalar); CHKERRQ(ierr);
      temp_scalar *= params_->grid_->lebesgue_measure_;
      temp_scalar *= params_->tu_->k_gm_wm_ratio_;  // this ratio will control the diffusivity in gm
      x_ptr[params_->tu_->np_] += temp_scalar;
    }

    if (params_->tu_->nk_ > 1) {
      ierr = VecDot(tumor_->mat_prop_->gm_, temp_, &x_ptr[params_->tu_->np_ + 1]); CHKERRQ(ierr);
      x_ptr[params_->tu_->np_ + 1] *= params_->grid_->lebesgue_measure_;
    }
    if (params_->tu_->nk_ > 2) {
      ierr = VecDot(tumor_->mat_prop_->csf_, temp_, &x_ptr[params_->tu_->np_ + 2]); CHKERRQ(ierr);
      x_ptr[params_->tu_->np_ + 2] *= params_->grid_->lebesgue_measure_;
    }
    ierr = VecRestoreArray(dJ, &x_ptr); CHKERRQ(ierr);
  }

  /* INVERSION FOR REACTION COEFFICIENT */
  integration_weight = 1.0;
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

    ierr = VecGetArray(dJ, &x_ptr); CHKERRQ(ierr);
    ierr = VecDot(tumor_->mat_prop_->wm_, temp_, &x_ptr[params_->tu_->np_ + params_->tu_->nk_]); CHKERRQ(ierr);
    x_ptr[params_->tu_->np_ + params_->tu_->nk_] *= params_->grid_->lebesgue_measure_;

    if (params_->tu_->nr_ == 1) {
      // Inverting for only one parameters a.k.a reaction in WM. Provide user with the option of setting a reaction for
      // other tissue types using params - Hence, the gradient will change accordingly.
      // Implicitly assuming there's no glm. TODO: remove glm from all subsequent iterations of the solver.
      ierr = VecDot(tumor_->mat_prop_->gm_, temp_, &temp_scalar); CHKERRQ(ierr);
      temp_scalar *= params_->grid_->lebesgue_measure_;
      temp_scalar *= params_->tu_->r_gm_wm_ratio_;  // this ratio will control the reaction coefficient in gm
      x_ptr[params_->tu_->np_ + params_->tu_->nk_] += temp_scalar;
    }

    if (params_->tu_->nr_ > 1) {
      ierr = VecDot(tumor_->mat_prop_->gm_, temp_, &x_ptr[params_->tu_->np_ + params_->tu_->nk_ + 1]); CHKERRQ(ierr);
      x_ptr[params_->tu_->np_ + params_->tu_->nk_ + 1] *= params_->grid_->lebesgue_measure_;
    }

    if (params_->tu_->nr_ > 2) {
      ierr = VecDot(tumor_->mat_prop_->csf_, temp_, &x_ptr[params_->tu_->np_ + params_->tu_->nk_ + 2]); CHKERRQ(ierr);
      x_ptr[params_->tu_->np_ + params_->tu_->nk_ + 2] *= params_->grid_->lebesgue_measure_;
    }

    ierr = VecRestoreArray(dJ, &x_ptr); CHKERRQ(ierr);
  }
#if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 9)
  if (lock_state != 0) {
    x->lock = lock_state;
  }
#endif
  // timing
  self_exec_time += MPI_Wtime();
  t[5] = self_exec_time;
  e.addTimings(t);
  e.stop();

  PetscFunctionReturn(ierr);

  // PetscFunctionBegin;
  // PetscErrorCode ierr = 0;
  // params_->tu_->statistics_.nb_obj_evals++;
  // params_->tu_->statistics_.nb_grad_evals++; CHKERRQ(ierr);
  // ierr = evaluateObjective (J, x, data);                        CHKERRQ(ierr); CHKERRQ(ierr);
  // PetscFunctionReturn (ierr);
}

PetscErrorCode DerivativeOperatorsKL::evaluateHessian(Vec y, Vec x) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  params_->tu_->statistics_.nb_hessian_evals++;
  // TODO: hessian is implemented using L2 objective. Needs to be changed to cross
  // entropy
  std::bitset<3> XYZ;
  XYZ[0] = 1;
  XYZ[1] = 1;
  XYZ[2] = 1;
  Event e("tumor-eval-hessian");
  std::array<double, 7> t = {0};
  double self_exec_time = -MPI_Wtime();
  ScalarType *y_ptr;

  if (params_->opt_->diffusivity_inversion_) {
    /* HESSIAN WITH DIFFUSIVITY INVERSION
      Hx = [Hpp p_tilde + Hpk k_tilde; Hkp p_tiilde + Hkk k_tilde]
      Each Matvec is computed separately by eliminating the
      incremental forward and adjoint equations and the result is added into y = Hx
    */
    //  -------------------  -------------------  -------------------  -------------------  -------------------  -------------------
    // --------------- Compute Hpp * p_tilde -------------------
    // Solve incr fwd with k_tilde = 0 and c0_tilde = \phi * p_tilde
    ierr = tumor_->phi_->apply(tumor_->c_0_, x); CHKERRQ(ierr);
    ierr = pde_operators_->solveState(1);
    // Solve incr adj with alpha1_tilde = -OT * O * c1_tilde
    ierr = tumor_->obs_->apply(temp_, tumor_->c_t_, 1); CHKERRQ(ierr);
    ierr = tumor_->obs_->apply(tumor_->p_t_, temp_, 1); CHKERRQ(ierr);
    ierr = VecScale(tumor_->p_t_, -1.0); CHKERRQ(ierr);
    ierr = pde_operators_->solveAdjoint(2);
    // Matvec is \beta\phiT\phi p_tilde - \phiT \alpha0_tilde
    ierr = tumor_->phi_->applyTranspose(ptemp_, tumor_->p_0_);
    ierr = tumor_->phi_->applyTranspose(y, tumor_->c_0_); CHKERRQ(ierr);
    ierr = VecScale(y, params_->opt_->beta_); CHKERRQ(ierr);
    ierr = VecAXPY(y, -1.0, ptemp_); CHKERRQ(ierr);
    ierr = VecScale(y, params_->grid_->lebesgue_measure_); CHKERRQ(ierr);

    //  -------------------  -------------------  -------------------  -------------------  -------------------  -------------------
    // --------------- Compute Hkp * p_tilde -- \int \int m_i \grad c . \grad \alpha_tilde -------------------
    ScalarType integration_weight = 1.0;
    ScalarType temp_scalar = 0.;
    ierr = VecSet(temp_, 0.0); CHKERRQ(ierr);
    // compute numerical time integration using trapezoidal rule
    for (int i = 0; i < params_->tu_->nt_ + 1; i++) {
      // integration weight for chain trapezoidal rule
      if (i == 0 || i == params_->tu_->nt_)
        integration_weight = 0.5;
      else
        integration_weight = 1.0;

      // compute x = (grad c)^T grad \alpha_tilde
      // compute gradient of c(t)
      pde_operators_->spec_ops_->computeGradient(tumor_->work_[1], tumor_->work_[2], tumor_->work_[3], pde_operators_->c_[i], &XYZ, t.data());
      // compute gradient of \alpha_tilde(t)
      pde_operators_->spec_ops_->computeGradient(tumor_->work_[4], tumor_->work_[5], tumor_->work_[6], pde_operators_->p_[i], &XYZ, t.data());
      // scalar product (grad c)^T grad \alpha_tilde
      ierr = VecPointwiseMult(tumor_->work_[0], tumor_->work_[1], tumor_->work_[4]); CHKERRQ(ierr);
      ierr = VecPointwiseMult(tumor_->work_[1], tumor_->work_[2], tumor_->work_[5]); CHKERRQ(ierr);
      ierr = VecAXPY(tumor_->work_[0], 1.0, tumor_->work_[1]); CHKERRQ(ierr);
      ierr = VecPointwiseMult(tumor_->work_[1], tumor_->work_[3], tumor_->work_[6]); CHKERRQ(ierr);
      ierr = VecAXPY(tumor_->work_[0], 1.0, tumor_->work_[1]); CHKERRQ(ierr);

      // numerical time integration using trapezoidal rule
      ierr = VecAXPY(temp_, params_->tu_->dt_ * integration_weight, tumor_->work_[0]); CHKERRQ(ierr);
    }
    // time integration of [ int_0 (grad c)^T grad alpha_tilde dt ] done, result in temp_
    // integration over omega (i.e., inner product, as periodic boundary and no lebesque measure in tumor code)
    ierr = VecGetArray(y, &y_ptr); CHKERRQ(ierr);
    ierr = VecDot(tumor_->mat_prop_->wm_, temp_, &y_ptr[params_->tu_->np_]); CHKERRQ(ierr);
    y_ptr[params_->tu_->np_] *= params_->grid_->lebesgue_measure_;

    if (params_->tu_->nk_ == 1) {
      // Inverting for only one parameters a.k.a diffusivity in WM. Provide user with the option of setting a diffusivity for
      // other tissue types using params - Hence, the gradient will change accordingly.
      // Implicitly assuming there's no glm. TODO: remove glm from all subsequent iterations of the solver.
      ierr = VecDot(tumor_->mat_prop_->gm_, temp_, &temp_scalar); CHKERRQ(ierr);
      temp_scalar *= params_->grid_->lebesgue_measure_;
      temp_scalar *= params_->tu_->k_gm_wm_ratio_;  // this ratio will control the diffusivity in gm
      y_ptr[params_->tu_->np_] += temp_scalar;
    }

    if (params_->tu_->nk_ > 1) {
      ierr = VecDot(tumor_->mat_prop_->gm_, temp_, &y_ptr[params_->tu_->np_ + 1]); CHKERRQ(ierr);
      y_ptr[params_->tu_->np_ + 1] *= params_->grid_->lebesgue_measure_;
    }
    if (params_->tu_->nk_ > 2) {
      ierr = VecDot(tumor_->mat_prop_->csf_, temp_, &y_ptr[params_->tu_->np_ + 2]); CHKERRQ(ierr);
      y_ptr[params_->tu_->np_ + 2] *= params_->grid_->lebesgue_measure_;
    }
    ierr = VecRestoreArray(y, &y_ptr); CHKERRQ(ierr);

    //  -------------------  -------------------  -------------------  -------------------  -------------------  -------------------
    // --------------- Compute Hpk * k_tilde -- -\phiT \alpha0_tilde -------------------
    // Set c0_tilde to zero
    ierr = VecSet(tumor_->c_0_, 0.); CHKERRQ(ierr);
    // solve tumor incr fwd with k_tilde
    // get the update on kappa -- this is used in tandem with the actual kappa in
    // the incr fwd solves and hence we cannot re-use the diffusivity vectors
    // TODO: here, it is assumed that the update is isotropic updates- this has
    // to be modified later is anisotropy is included
    ScalarType k1, k2, k3;
    ierr = VecGetArray(x, &y_ptr);
    k1 = y_ptr[params_->tu_->np_];
    k2 = (params_->tu_->nk_ > 1) ? y_ptr[params_->tu_->np_ + 1] : 0.;
    k3 = (params_->tu_->nk_ > 2) ? y_ptr[params_->tu_->np_ + 2] : 0.;
    ierr = tumor_->k_->setSecondaryCoefficients(k1, k2, k3, tumor_->mat_prop_, params_); CHKERRQ(ierr);
    ierr = VecRestoreArray(x, &y_ptr);

    ierr = pde_operators_->solveState(2); CHKERRQ(ierr);
    // Solve incr adj with alpha1_tilde = -OT * O * c1_tilde
    ierr = tumor_->obs_->apply(temp_, tumor_->c_t_, 1); CHKERRQ(ierr);
    ierr = tumor_->obs_->apply(tumor_->p_t_, temp_, 1); CHKERRQ(ierr);
    ierr = VecScale(tumor_->p_t_, -1.0); CHKERRQ(ierr);
    ierr = pde_operators_->solveAdjoint(2);
    // Matvec is  - \phiT \alpha0_tilde
    ierr = VecSet(ptemp_, 0.); CHKERRQ(ierr);
    ierr = tumor_->phi_->applyTranspose(ptemp_, tumor_->p_0_); CHKERRQ(ierr);
    ierr = VecScale(ptemp_, -params_->grid_->lebesgue_measure_); CHKERRQ(ierr);
    // Add Hpk k_tilde to Hpp p_tilde:  Note the kappa/rho components are zero
    // so are unchanged in y
    ierr = VecAXPY(y, 1.0, ptemp_); CHKERRQ(ierr);

    //  -------------------  -------------------  -------------------  -------------------  -------------------  -------------------
    // --------------- Compute Hkk * k_tilde -- \int \int mi \grad c \grad \alpha_tilde -------------------
    integration_weight = 1.0;
    ierr = VecSet(temp_, 0.0); CHKERRQ(ierr);
    // compute numerical time integration using trapezoidal rule
    for (int i = 0; i < params_->tu_->nt_ + 1; i++) {
      // integration weight for chain trapezoidal rule
      if (i == 0 || i == params_->tu_->nt_)
        integration_weight = 0.5;
      else
        integration_weight = 1.0;

      // compute x = (grad c)^T grad \alpha_tilde
      // compute gradient of c(t)
      pde_operators_->spec_ops_->computeGradient(tumor_->work_[1], tumor_->work_[2], tumor_->work_[3], pde_operators_->c_[i], &XYZ, t.data());
      // compute gradient of \alpha_tilde(t)
      pde_operators_->spec_ops_->computeGradient(tumor_->work_[4], tumor_->work_[5], tumor_->work_[6], pde_operators_->p_[i], &XYZ, t.data());
      // scalar product (grad c)^T grad \alpha_tilde
      ierr = VecPointwiseMult(tumor_->work_[0], tumor_->work_[1], tumor_->work_[4]); CHKERRQ(ierr);
      ierr = VecPointwiseMult(tumor_->work_[1], tumor_->work_[2], tumor_->work_[5]); CHKERRQ(ierr);
      ierr = VecAXPY(tumor_->work_[0], 1.0, tumor_->work_[1]); CHKERRQ(ierr);
      ierr = VecPointwiseMult(tumor_->work_[1], tumor_->work_[3], tumor_->work_[6]); CHKERRQ(ierr);
      ierr = VecAXPY(tumor_->work_[0], 1.0, tumor_->work_[1]); CHKERRQ(ierr);

      // numerical time integration using trapezoidal rule
      ierr = VecAXPY(temp_, params_->tu_->dt_ * integration_weight, tumor_->work_[0]); CHKERRQ(ierr);
    }
    // time integration of [ int_0 (grad c)^T grad alpha_tilde dt ] done, result in temp_
    // integration over omega (i.e., inner product, as periodic boundary and no lebesque measure in tumor code)
    ierr = VecGetArray(y, &y_ptr); CHKERRQ(ierr);
    ierr = VecDot(tumor_->mat_prop_->wm_, temp_, &temp_scalar); CHKERRQ(ierr);
    temp_scalar *= params_->grid_->lebesgue_measure_;
    y_ptr[params_->tu_->np_] += temp_scalar;

    if (params_->tu_->nk_ == 1) {
      // Inverting for only one parameters a.k.a diffusivity in WM. Provide user with the option of setting a diffusivity for
      // other tissue types using params - Hence, the gradient will change accordingly.
      // Implicitly assuming there's no glm. TODO: remove glm from all subsequent iterations of the solver.
      ierr = VecDot(tumor_->mat_prop_->gm_, temp_, &temp_scalar); CHKERRQ(ierr);
      temp_scalar *= params_->grid_->lebesgue_measure_;
      temp_scalar *= params_->tu_->k_gm_wm_ratio_;  // this ratio will control the diffusivity in gm
      y_ptr[params_->tu_->np_] += temp_scalar;
    }

    if (params_->tu_->nk_ > 1) {
      ierr = VecDot(tumor_->mat_prop_->gm_, temp_, &temp_scalar); CHKERRQ(ierr);
      temp_scalar *= params_->grid_->lebesgue_measure_;
      y_ptr[params_->tu_->np_ + 1] += temp_scalar;
    }
    if (params_->tu_->nk_ > 2) {
      ierr = VecDot(tumor_->mat_prop_->csf_, temp_, &temp_scalar); CHKERRQ(ierr);
      temp_scalar *= params_->grid_->lebesgue_measure_;
      y_ptr[params_->tu_->np_ + 2] += temp_scalar;
    }
    ierr = VecRestoreArray(y, &y_ptr); CHKERRQ(ierr);
  } else {
    ierr = tumor_->phi_->apply(tumor_->c_0_, x); CHKERRQ(ierr);
    ierr = pde_operators_->solveState(1);

    ierr = tumor_->obs_->apply(temp_, tumor_->c_t_, 1); CHKERRQ(ierr);
    ierr = tumor_->obs_->apply(tumor_->p_t_, temp_, 1); CHKERRQ(ierr);
    ierr = VecScale(tumor_->p_t_, -1.0); CHKERRQ(ierr);

    ierr = pde_operators_->solveAdjoint(2);

    ierr = tumor_->phi_->applyTranspose(ptemp_, tumor_->p_0_);
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
  }
  self_exec_time += MPI_Wtime();
  t[5] = self_exec_time;
  e.addTimings(t);
  e.stop();

  PetscFunctionReturn(ierr);
}

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

/* #### --------------------------------------------------------------------------- #### */
/* #### ========  Deriv. Ops.: Finite Diff. {gamma,rho,kappa} for ME Model ======== #### */
/* #### --------------------------------------------------------------------------- #### */
PetscErrorCode DerivativeOperatorsMassEffect::computeMisfitBrain(PetscReal *J) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;

  PetscReal ms = 0;
  *J = 0;
  ierr = VecCopy(tumor_->mat_prop_->gm_, temp_); CHKERRQ(ierr);
  ierr = VecAXPY(temp_, -1.0, gm_); CHKERRQ(ierr);
  ierr = VecDot(temp_, temp_, &ms); CHKERRQ(ierr);
  *J += ms;

  ierr = VecCopy(tumor_->mat_prop_->wm_, temp_); CHKERRQ(ierr);
  ierr = VecAXPY(temp_, -1.0, wm_); CHKERRQ(ierr);
  ierr = VecDot(temp_, temp_, &ms); CHKERRQ(ierr);
  *J += ms;

  ierr = VecCopy(tumor_->mat_prop_->vt_, temp_); CHKERRQ(ierr);
  ierr = VecAXPY(temp_, -1.0, vt_); CHKERRQ(ierr);
  ierr = VecDot(temp_, temp_, &ms); CHKERRQ(ierr);
  *J += ms;

  ierr = VecCopy(tumor_->mat_prop_->csf_, temp_); CHKERRQ(ierr);
  ierr = VecAXPY(temp_, -1.0, csf_); CHKERRQ(ierr);
  ierr = VecDot(temp_, temp_, &ms); CHKERRQ(ierr);
  *J += ms;

  PetscFunctionReturn(ierr);
}

PetscErrorCode DerivativeOperatorsMassEffect::evaluateObjective(PetscReal *J, Vec x, std::shared_ptr<Data> data_inv) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  params_->tu_->statistics_.nb_obj_evals++;
  ScalarType *x_ptr;

#if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 9)
  int lock_state;
  ierr = VecLockGet(x, &lock_state); CHKERRQ(ierr);
  if (lock_state != 0) {
    x->lock = 0;
  }
#endif

  Vec data = data_inv->dt1();

  std::stringstream s;
  ierr = VecGetArray(x, &x_ptr); CHKERRQ(ierr);
  params_->tu_->forcing_factor_ = params_->opt_->gamma_scale_ * x_ptr[0];  // re-scaling parameter scales
  params_->tu_->rho_ = params_->opt_->rho_scale_ * x_ptr[1];               // rho
  params_->tu_->k_ = params_->opt_->k_scale_ * x_ptr[2];              // kappa
  ierr = VecRestoreArray(x, &x_ptr); CHKERRQ(ierr);

  if (!disable_verbose_) {
    s << " Forcing factor at current guess = " << params_->tu_->forcing_factor_;
    ierr = tuMSGstd(s.str()); CHKERRQ(ierr);
    s.str("");
    s.clear();
    s << " Reaction at current guess       = " << params_->tu_->rho_;
    ierr = tuMSGstd(s.str()); CHKERRQ(ierr);
    s.str("");
    s.clear();
    s << " Diffusivity at current guess    = " << params_->tu_->k_;
    ierr = tuMSGstd(s.str()); CHKERRQ(ierr);
    s.str("");
    s.clear();
  }

  // Reset mat-props and diffusion and reaction operators, tumor IC does not change
  ierr = tumor_->mat_prop_->resetValues(); CHKERRQ(ierr);
  ierr = tumor_->rho_->setValues(params_->tu_->rho_, params_->tu_->r_gm_wm_ratio_, params_->tu_->r_glm_wm_ratio_, tumor_->mat_prop_, params_);
  ierr = tumor_->k_->setValues(params_->tu_->k_, params_->tu_->k_gm_wm_ratio_, params_->tu_->k_glm_wm_ratio_, tumor_->mat_prop_, params_);
  ierr = tumor_->velocity_->set(0);
  ierr = tumor_->displacement_->set(0);

  ierr = pde_operators_->solveState(0);
  ierr = tumor_->obs_->apply(temp_, tumor_->c_t_, 1); CHKERRQ(ierr);
  ierr = VecAXPY(temp_, -1.0, data); CHKERRQ(ierr);
  ierr = VecDot(temp_, temp_, J); CHKERRQ(ierr);
  (*J) *= 0.5 * params_->grid_->lebesgue_measure_;
  PetscReal misfit_brain = 0.;
  // ierr = computeMisfitBrain (&misfit_brain);                      CHKERRQ (ierr);
  misfit_brain *= 0.5 * params_->grid_->lebesgue_measure_;
  if (!disable_verbose_) {
    s << "J = misfit_tu + misfit_brain = " << std::setprecision(12) << *J << " + " << misfit_brain << " = " << (*J) + misfit_brain;
    ierr = tuMSGstd(s.str()); CHKERRQ(ierr);
    s.str("");
    s.clear();
  }

  (*J) += misfit_brain;

#if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 9)
  if (lock_state != 0) {
    x->lock = lock_state;
  }
#endif

  PetscFunctionReturn(ierr);
}

PetscErrorCode DerivativeOperatorsMassEffect::evaluateGradient(Vec dJ, Vec x, std::shared_ptr<Data> data_inv) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  params_->tu_->statistics_.nb_grad_evals++;

  Vec data = data_inv->dt1();
  disable_verbose_ = true;
  // Finite difference gradient -- forward for now
  ScalarType h, dx;
  ScalarType volatile xph;
  PetscReal J_f, J_b;

  ierr = evaluateObjective(&J_b, x, data_inv); CHKERRQ(ierr);
  int sz;
  ScalarType *delta_ptr, *dj_ptr;
  ScalarType const *x_ptr;
  ierr = VecGetSize(x, &sz); CHKERRQ(ierr);
  ierr = VecGetArray(dJ, &dj_ptr); CHKERRQ(ierr);
  std::array<ScalarType, 3> characteristic_scale = {1, 1, 1};
  //    #ifdef SINGLE
  //    ScalarType small = 3.45266983e-04F;
  //    #else
  //    ScalarType small = 3.45266983e-04;
  //    #endif
  ScalarType small = 1E-5;
  for (int i = 0; i < sz; i++) {
    ierr = VecCopy(x, delta_); CHKERRQ(ierr);
    ierr = VecGetArray(delta_, &delta_ptr); CHKERRQ(ierr);
    ierr = VecGetArrayRead(x, &x_ptr); CHKERRQ(ierr);
    h = (x_ptr[i] == 0) ? small * characteristic_scale[i] : small * x_ptr[i] * characteristic_scale[i];
    xph = x_ptr[i] + h;
    dx = xph - x_ptr[i];
    delta_ptr[i] = xph;
    ierr = VecRestoreArray(delta_, &delta_ptr); CHKERRQ(ierr);
    ierr = evaluateObjective(&J_f, delta_, data_inv); CHKERRQ(ierr);
    dj_ptr[i] = (J_f - J_b) / dx;
    ierr = VecRestoreArrayRead(x, &x_ptr); CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(dJ, &dj_ptr); CHKERRQ(ierr);

  disable_verbose_ = false;

  PetscFunctionReturn(ierr);
}

PetscErrorCode DerivativeOperatorsMassEffect::evaluateObjectiveAndGradient(PetscReal *J, Vec dJ, Vec x, std::shared_ptr<Data> data_inv) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  params_->tu_->statistics_.nb_obj_evals++;
  params_->tu_->statistics_.nb_grad_evals++;

  Event e("tumor-eval-objandgrad");
  std::array<double, 7> t = {0};
  double self_exec_time = -MPI_Wtime();

  Vec data = data_inv->dt1();
  ierr = evaluateObjective(J, x, data_inv); CHKERRQ(ierr);
  // Finite difference gradient -- forward for now
  ScalarType h, dx;
  ScalarType volatile xph;
  PetscReal J_f;

  disable_verbose_ = true;
  int sz;
  ScalarType *delta_ptr, *dj_ptr;
  ScalarType const *x_ptr;
  ierr = VecGetSize(x, &sz); CHKERRQ(ierr);
  ierr = VecGetArray(dJ, &dj_ptr); CHKERRQ(ierr);

  ScalarType scale = 1;
  std::array<ScalarType, 3> characteristic_scale = {1, 1, 1};
  //    #ifdef SINGLE
  //    ScalarType small = 3.45266983e-04F;
  //    #else
  //    ScalarType small = 3.45266983e-04;
  //    #endif
  ScalarType J_b = (*J);
  ScalarType small = 1E-5;
  for (int i = 0; i < sz; i++) {
    ierr = VecCopy(x, delta_); CHKERRQ(ierr);
    ierr = VecGetArray(delta_, &delta_ptr); CHKERRQ(ierr);
    ierr = VecGetArrayRead(x, &x_ptr); CHKERRQ(ierr);
    h = (x_ptr[i] == 0) ? small * characteristic_scale[i] : small * x_ptr[i] * characteristic_scale[i];
    xph = x_ptr[i] + h;
    dx = xph - x_ptr[i];
    delta_ptr[i] = xph;
    ierr = VecRestoreArray(delta_, &delta_ptr); CHKERRQ(ierr);
    ierr = evaluateObjective(&J_f, delta_, data_inv); CHKERRQ(ierr);
    dj_ptr[i] = (J_f - J_b) / dx;
    ierr = VecRestoreArrayRead(x, &x_ptr); CHKERRQ(ierr);
  }

  ierr = VecRestoreArray(dJ, &dj_ptr); CHKERRQ(ierr);

  disable_verbose_ = false;

  // // timing
  // self_exec_time += MPI_Wtime(); t[5] = self_exec_time; e.addTimings (t); e.stop ();
  PetscFunctionReturn(ierr);
}

PetscErrorCode DerivativeOperatorsMassEffect::evaluateHessian(Vec y, Vec x) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  params_->tu_->statistics_.nb_hessian_evals++;

  std::bitset<3> XYZ;
  XYZ[0] = 1;
  XYZ[1] = 1;
  XYZ[2] = 1;
  Event e("tumor-eval-hessian");
  std::array<double, 7> t = {0};
  double self_exec_time = -MPI_Wtime();

  // gradient descent
  ierr = VecCopy(x, y); CHKERRQ(ierr);

  self_exec_time += MPI_Wtime();
  t[5] = self_exec_time;
  e.addTimings(t);
  e.stop();
  PetscFunctionReturn(ierr);
}

PetscErrorCode DerivativeOperatorsMassEffect::checkGradient(Vec x, std::shared_ptr<Data> data_inv) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;

  Vec data = data_inv->dt1();
  std::stringstream s;
  s << " ----- Gradient check with taylor expansion ----- ";
  ierr = tuMSGwarn(s.str()); CHKERRQ(ierr);
  s.str("");
  s.clear();

  ScalarType h[10];
  ScalarType J, J_taylor, J_p, diff;

  Vec dJ, x_tilde, x_new;
  ierr = VecDuplicate(x, &dJ); CHKERRQ(ierr);
  ierr = VecDuplicate(x, &x_tilde); CHKERRQ(ierr);
  ierr = VecDuplicate(x, &x_new); CHKERRQ(ierr);

  ierr = evaluateObjectiveAndGradient(&J_p, dJ, x, data_inv); CHKERRQ(ierr);

  PetscRandom rctx;
  ierr = PetscRandomCreate(PETSC_COMM_SELF, &rctx); CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rctx); CHKERRQ(ierr);
  ierr = VecSetRandom(x_tilde, rctx); CHKERRQ(ierr);

  ScalarType xg_dot, sum;
  ierr = VecSum(x_tilde, &sum); CHKERRQ(ierr);
  ScalarType start = std::pow(2, -1);
  for (int i = 0; i < 10; i++) {
    h[i] = start * std::pow(2, -i);
    ierr = VecWAXPY(x_new, h[i], x_tilde, x); CHKERRQ(ierr);
    ierr = evaluateObjective(&J, x_new, data_inv);
    ierr = VecDot(dJ, x_tilde, &xg_dot); CHKERRQ(ierr);
    J_taylor = J_p + xg_dot * h[i];
    diff = std::abs(J - J_taylor);
    // s << "h[i]: " << h[i] << " |J - J_taylor|: " << diff << "  log2(diff) : " << log2(diff) << " g_fd - xg_dot: " << ((J - J_p)/h[i] - xg_dot) / sum;
    s << "h: " << h[i] << " |J - J*|: " << std::abs(J - J_p) << " |J - J_taylor|: " << diff;
    ierr = tuMSGwarn(s.str()); CHKERRQ(ierr);
    s.str("");
    s.clear();
  }

  ierr = VecDestroy(&dJ); CHKERRQ(ierr);
  ierr = VecDestroy(&x_tilde); CHKERRQ(ierr);
  ierr = VecDestroy(&x_new); CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rctx); CHKERRQ(ierr);

  PetscFunctionReturn(ierr);
}

/* #### ------------------------------------------------------------------- #### */
/* #### ========                  BASE CLASS                       ======== #### */
/* #### ------------------------------------------------------------------- #### */
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
  int procid, nprocs;
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &procid);
  PCOUT << "\n\n----- Hessian check with taylor expansion ----- " << std::endl;

  Vec data = data_inv->dt1();
  ScalarType norm;
  ierr = VecNorm(p, NORM_2, &norm); CHKERRQ(ierr);

  PCOUT << "Hessian check performed at x with norm: " << norm << std::endl;
  ScalarType *x_ptr, k1, k2, k3;
  if (params_->opt_->diffusivity_inversion_) {
    ierr = VecGetArray(p, &x_ptr); CHKERRQ(ierr);
    k1 = x_ptr[params_->tu_->np_];
    k2 = (params_->tu_->nk_ > 1) ? x_ptr[params_->tu_->np_ + 1] : 0;
    k3 = (params_->tu_->nk_ > 2) ? x_ptr[params_->tu_->np_ + 2] : 0;
    PCOUT << "k1: " << k1 << " k2: " << k2 << " k3: " << k3 << std::endl;
    ierr = VecRestoreArray(p, &x_ptr); CHKERRQ(ierr);
  }

  if (params_->opt_->flag_reaction_inv_) {
    ierr = VecGetArray(p, &x_ptr); CHKERRQ(ierr);
    k1 = x_ptr[params_->tu_->np_ + params_->tu_->nk_];
    k2 = (params_->tu_->nr_ > 1) ? x_ptr[params_->tu_->np_ + params_->tu_->nk_ + 1] : 0;
    k3 = (params_->tu_->nr_ > 2) ? x_ptr[params_->tu_->np_ + params_->tu_->nk_ + 2] : 0;
    PCOUT << "r1: " << k1 << " r2: " << k2 << " r3: " << k3 << std::endl;
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
    PCOUT << "|J - J_taylor|: " << diff << "  log10(diff) : " << log10(diff) << std::endl;
  }
  PCOUT << "\n\n";

  ierr = VecDestroy(&dJ); CHKERRQ(ierr);
  ierr = VecDestroy(&temp); CHKERRQ(ierr);
  ierr = VecDestroy(&Hx); CHKERRQ(ierr);
  ierr = VecDestroy(&p_tilde); CHKERRQ(ierr);
  ierr = VecDestroy(&p_new); CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rctx); CHKERRQ(ierr);

  PetscFunctionReturn(ierr);
}
