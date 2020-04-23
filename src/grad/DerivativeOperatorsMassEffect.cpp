#include "DerivativeOperators.h"
#include <petsc/private/vecimpl.h>
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
  const ScalarType *x_ptr;

  Vec data = data_inv->dt1();

  std::stringstream s;
  ierr = VecGetArrayRead(x, &x_ptr); CHKERRQ(ierr);
  params_->tu_->forcing_factor_ = params_->opt_->gamma_scale_ * x_ptr[0];  // re-scaling parameter scales
  params_->tu_->rho_ = params_->opt_->rho_scale_ * x_ptr[1];               // rho
  params_->tu_->k_ = params_->opt_->k_scale_ * x_ptr[2];                   // kappa
  ierr = VecRestoreArrayRead(x, &x_ptr); CHKERRQ(ierr);

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
