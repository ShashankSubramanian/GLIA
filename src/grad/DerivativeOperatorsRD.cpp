#include "DerivativeOperators.h"
#include <petsc/private/vecimpl.h>

/* #### ------------------------------------------------------------------- #### */
/* #### ========  Deriv. Ops.: Adjoints {rho,kappa,p} for RD Model ======== #### */
/* #### ------------------------------------------------------------------- #### */
PetscErrorCode DerivativeOperatorsRD::evaluateObjective(PetscReal *J, Vec x, std::shared_ptr<Data> data) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  params_->tu_->statistics_.nb_obj_evals++;
  const ScalarType *x_ptr;

  int x_sz;
  PetscReal m1 = 0, m0 = 0, reg = 0;
  std::stringstream s;

  ierr = updateReactionAndDiffusion(x); CHKERRQ(ierr);

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
  if (params_->opt_->regularization_norm_ == L2) {  // In tumor space, so scale norm by lebesque measure
    ierr = VecDot(tumor_->c_0_, tumor_->c_0_, &reg); CHKERRQ(ierr);
    reg *= 0.5 * params_->opt_->beta_;
    reg *= params_->grid_->lebesgue_measure_;
  } else if (params_->opt_->regularization_norm_ == L2b) {
    // Reg term only on the initial condition. Leave out the diffusivity.
    ierr = VecGetArrayRead(x, &x_ptr); CHKERRQ(ierr);
    for (int i = 0; i < params_->tu_->np_; i++) {
      reg += x_ptr[i] * x_ptr[i];
    }
    ierr = VecRestoreArrayRead(x, &x_ptr); CHKERRQ(ierr);
    reg *= 0.5 * params_->opt_->beta_;
  }

  // objective function value
  (*J) = params_->grid_->lebesgue_measure_ * 0.5 *(m1 + m0) + reg;

  if (params_->tu_->two_time_points_) {
    s << "  obj: J(p) = D(c1) + D(c0) + S(c0) = "<< std::setprecision(12) << (*J)<<" = " << std::setprecision(12)<< params_->grid_->lebesgue_measure_ * 0.5 * m1 <<" + " << std::setprecision(12)<< params_->grid_->lebesgue_measure_ * 0.5 * m0 <<" + "<< std::setprecision(12) <<reg<<"";  ierr = tuMSGstd(s.str()); CHKERRQ(ierr); s.str(""); s.clear();
  } else {
    s << "  obj: J(p) = D(c1) + S(c0) = "<< std::setprecision(12) << (*J)<<" = " << std::setprecision(12)<< params_->grid_->lebesgue_measure_ * 0.5 * m1 <<" + "<< std::setprecision(12) <<reg<<"";  ierr = tuMSGstd(s.str()); CHKERRQ(ierr); s.str(""); s.clear();
  }

  PetscFunctionReturn(ierr);
}

PetscErrorCode DerivativeOperatorsRD::evaluateGradient(Vec dJ, Vec x, std::shared_ptr<Data> data) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;

  ScalarType *dj_ptr;
  params_->tu_->statistics_.nb_grad_evals++;

  ierr = VecSet(dJ, 0); CHKERRQ(ierr);
  ierr = updateReactionAndDiffusion(x); CHKERRQ(ierr);

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

  ierr = gradDiffusion(dJ); CHKERRQ(ierr);
  ierr = gradReaction(dJ); CHKERRQ(ierr);

  PetscFunctionReturn(ierr);
}

// saves on forward solve
PetscErrorCode DerivativeOperatorsRD::evaluateObjectiveAndGradient(PetscReal *J, Vec dJ, Vec x, std::shared_ptr<Data> data) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  params_->tu_->statistics_.nb_obj_evals++;
  params_->tu_->statistics_.nb_grad_evals++;
 
  std::stringstream s;
  const ScalarType *x_ptr;
  ScalarType *dj_ptr;
  PetscReal m1 = 0, m0 = 0;

  ierr = VecSet(dJ, 0); CHKERRQ(ierr);
  ierr = updateReactionAndDiffusion(x); CHKERRQ(ierr);

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
    ierr = VecGetArrayRead(x, &x_ptr); CHKERRQ(ierr);
    for (int i = 0; i < params_->tu_->np_; i++) {
      reg += x_ptr[i] * x_ptr[i];
    }
    ierr = VecRestoreArrayRead(x, &x_ptr); CHKERRQ(ierr);
    reg *= 0.5 * params_->opt_->beta_;
  }

  // objective function value
  (*J) = params_->grid_->lebesgue_measure_ * 0.5 * (m1 + m0) + reg;
  if (params_->tu_->two_time_points_) {
    s << "  obj: J(p) = D(c1) + D(c0) + S(c0) = " << std::setprecision(12) << (*J)<<" = " << std::setprecision(12)<< params_->grid_->lebesgue_measure_ * 0.5 * m1 <<" + " << std::setprecision(12)<< params_->grid_->lebesgue_measure_  * 0.5 * m0 <<" + "<< std::setprecision(12) <<reg<<"";  ierr = tuMSGstd(s.str()); CHKERRQ(ierr); s.str(""); s.clear();
  } else {
    s << "  obj: J(p) = D(c1) + S(c0) = " << std::setprecision(12) << (*J)<<" = " << std::setprecision(12)<< params_->grid_->lebesgue_measure_  * 0.5 * m1 <<" + "<< std::setprecision(12) <<reg<<"";  ierr = tuMSGstd(s.str()); CHKERRQ(ierr); s.str(""); s.clear();
  }

  ierr = gradDiffusion(dJ); CHKERRQ(ierr);
  ierr = gradReaction(dJ); CHKERRQ(ierr);

  PetscFunctionReturn(ierr);
}

// TODO: untested in cleanup 
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