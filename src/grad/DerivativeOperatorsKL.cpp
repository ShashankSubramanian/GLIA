#include "DerivativeOperators.h"
#include <petsc/private/vecimpl.h>

/* #### -------------------------------------------------------------------------------------- #### */
/* #### ========  Deriv. Ops.: Adjoints {rho,kappa,p} for RD Model with KL-Divergence ======== #### */
/* #### -------------------------------------------------------------------------------------- #### */
PetscErrorCode DerivativeOperatorsKL::evaluateObjective(PetscReal *J, Vec x, std::shared_ptr<Data> data_inv) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  params_->tu_->statistics_.nb_obj_evals++;

  const ScalarType *x_ptr;
  Vec data = data_inv->dt1();
  std::stringstream s;
  ierr = updateReactionAndDiffusion(x); CHKERRQ(ierr);

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
#ifdef CUDA
  ierr = vecGetArray(temp_, &ce_ptr); CHKERRQ(ierr);
  computeCrossEntropyCuda(ce_ptr, d_ptr, c_ptr, eps, params_->grid_->nl_);
  // vecSumCuda(ce_ptr, J, params_->grid_->nl_);
  cublasStatus_t status;
  cublasHandle_t handle;
  PetscCUBLASGetHandle(&handle);
  status = cublasSum(handle, params_->grid_->nl_, ce_ptr, 1, J);
  cublasCheckError(status);
  ierr = vecRestoreArray(temp_, &ce_ptr); CHKERRQ(ierr);
#else
  ierr = vecGetArray(temp_, &ce_ptr); CHKERRQ(ierr);
  for (int i = 0; i < params_->grid_->nl_; i++) {
    c_ptr[i] = (c_ptr[i] < eps) ? eps : c_ptr[i];
    c_ptr[i] = (c_ptr[i] > 1 - eps) ? 1 - eps : c_ptr[i];
    ce_ptr[i] += -(d_ptr[i] * log(c_ptr[i]) + (1 - d_ptr[i]) * log(1 - c_ptr[i]));
  }
  ierr = vecRestoreArray(temp_, &ce_ptr); CHKERRQ(ierr);
  ierr = VecSum(temp_, J); CHKERRQ(ierr);
#endif
  ierr = vecRestoreArray(tumor_->c_t_, &c_ptr); CHKERRQ(ierr);
  ierr = vecRestoreArray(data, &d_ptr); CHKERRQ(ierr);

  /*Regularization term*/
  PetscReal reg = 0;
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

  (*J) *= params_->grid_->lebesgue_measure_;

  s << "  obj: J(p) = Dc(c) + S(c0) = " << std::setprecision(12) << (*J) + reg << " = " << std::setprecision(12) << (*J) << " + " << std::setprecision(12) << reg << "";
  ierr = tuMSGstd(s.str()); CHKERRQ(ierr);
  s.str("");
  s.clear();

  (*J) += reg;

  PetscFunctionReturn(ierr);
}

PetscErrorCode DerivativeOperatorsKL::evaluateGradient(Vec dJ, Vec x, std::shared_ptr<Data> data_inv) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  params_->tu_->statistics_.nb_grad_evals++;
  
  Vec data = data_inv->dt1();
  ierr = updateReactionAndDiffusion(x); CHKERRQ(ierr);

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

  ierr = gradDiffusion(dJ); CHKERRQ(ierr);
  ierr = gradReaction(dJ); CHKERRQ(ierr);

  PetscFunctionReturn(ierr);
}

// saves on forward solve
PetscErrorCode DerivativeOperatorsKL::evaluateObjectiveAndGradient(PetscReal *J, Vec dJ, Vec x, std::shared_ptr<Data> data_inv) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  params_->tu_->statistics_.nb_obj_evals++;
  params_->tu_->statistics_.nb_grad_evals++;

  const ScalarType *x_ptr;
  Vec data = data_inv->dt1();
  std::stringstream s;

  ierr = updateReactionAndDiffusion(x); CHKERRQ(ierr);

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
#ifdef CUDA
  ierr = vecGetArray(temp_, &ce_ptr); CHKERRQ(ierr);
  computeCrossEntropyCuda(ce_ptr, d_ptr, c_ptr, eps, params_->grid_->nl_);
  // vecSumCuda(ce_ptr, J, params_->grid_->nl_);
  cublasStatus_t status;
  cublasHandle_t handle;
  PetscCUBLASGetHandle(&handle);
  status = cublasSum(handle, params_->grid_->nl_, ce_ptr, 1, J);
  cublasCheckError(status);
  computeCrossEntropyAdjointICCuda(a_ptr, d_ptr, c_ptr, eps, params_->grid_->nl_);
  ierr = vecRestoreArray(temp_, &ce_ptr); CHKERRQ(ierr);
#else
  ierr = vecGetArray(temp_, &ce_ptr); CHKERRQ(ierr);
  for (int i = 0; i < params_->grid_->nl_; i++) {
    c_ptr[i] = (c_ptr[i] < eps) ? eps : c_ptr[i];
    c_ptr[i] = (c_ptr[i] > 1 - eps) ? 1 - eps : c_ptr[i];
    ce_ptr[i] += -(d_ptr[i] * log(c_ptr[i]) + (1 - d_ptr[i]) * log(1 - c_ptr[i]));
    a_ptr[i] = (d_ptr[i] / (c_ptr[i]) - (1 - d_ptr[i]) / (1 - c_ptr[i]));
  }
  ierr = vecRestoreArray(temp_, &ce_ptr); CHKERRQ(ierr);
  ierr = VecSum(temp_, J); CHKERRQ(ierr);
#endif
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
    ierr = VecGetArrayRead(x, &x_ptr); CHKERRQ(ierr);
    for (int i = 0; i < params_->tu_->np_; i++) {
      reg += x_ptr[i] * x_ptr[i];
    }
    ierr = VecRestoreArrayRead(x, &x_ptr); CHKERRQ(ierr);
    reg *= 0.5 * params_->opt_->beta_;
  }

  (*J) *= params_->grid_->lebesgue_measure_;

  s << "  obj: J(p) = Dc(c) + S(c0) = " << std::setprecision(12) << (*J) + reg << " = " << std::setprecision(12) << (*J) << " + " << std::setprecision(12) << reg << "";
  ierr = tuMSGstd(s.str()); CHKERRQ(ierr);
  s.str("");
  s.clear();
  // objective function value
  (*J) += reg;

  ierr = gradDiffusion(dJ); CHKERRQ(ierr);
  ierr = gradReaction(dJ); CHKERRQ(ierr);

  PetscFunctionReturn(ierr);
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
