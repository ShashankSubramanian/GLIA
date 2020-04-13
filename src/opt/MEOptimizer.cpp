#include "MEOptimizer.h"

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
MEOptimizer::initialize(
  std::shared_ptr<DerivativeOperators> derivative_operators,
  std::shared_ptr <PdeOperators> pde_operators,
  std::shared_ptr <Parameters> params,
  std::shared_ptr <Tumor> tumor)) {

    PetscErrorCode ierr = 0;
    PetscFunctionBegin;
    std::stringstream ss;
    // number of dofs = {rho, kappa, gamma}
    n_inv_ = params_->get_nr() +  params_->get_nk() + 1;
    ss << " Initializing mass-effect optimizer with = " << n_inv_ << " = " <<  params_->get_nr() << " + " <<  params_->get_nk() << " + 1 dofs.";
    ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
    // initialize super class
    ierr = Optimizer::initialize(derivative_operators, pde_operators, params, tumor); CHKERRQ(ierr);

    PetscFunctionReturn(ierr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode MEOptimizer::allocateTaoObjects() {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;

  ierr = Optimizer::allocateTaoObjects(); CHKERRQ(ierr);
  // set initial guess TODO(K): move to solve()
  ScalarType *ptr;
  ierr = VecGetArray(xrec_, &ptr); CHKERRQ (ierr);
  ptr[0] = 1; ptr[1] = 6; ptr[2] = 0.5;
  ierr = VecRestoreArray(xrec_, &ptr); CHKERRQ (ierr);
  PetscFunctionReturn (ierr);
}


// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode MEOptimizer::setVariableBounds() {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  ierr = tuMSGstd(" .. setting variable bounds for {gamma, rho, kappa}."); CHKERRQ(ierr);
  ScalarType *ptr;
  Vec lower_bound, upper_bound;
  ierr = VecDuplicate (xrec_, &lower_bound); CHKERRQ(ierr);
  ierr = VecSet (lower_bound, 0.); CHKERRQ(ierr);
  ierr = VecDuplicate (xrec_, &upper_bound); CHKERRQ(ierr);
  ierr = VecSet (upper_bound, PETSC_INFINITY); CHKERRQ(ierr);
  // upper bound
  ierr = VecGetArray(upper_bound, &ptr);CHKERRQ (ierr);
  ptr[0] = ctx_->params_->opt_->gamma_ub_;
  ptr[1] = ctx_->params_->opt_->rho_ub_;
  ptr[2] = ctx_->params_->opt_->k_ub_;
  ctx_->params_->opt_->bounds_array_[0] = ptr[0];
  ctx_->params_->opt_->bounds_array_[1] = ptr[1];
  ctx_->params_->opt_->bounds_array_[2] = ptr[2];
  ierr = VecRestoreArray(upper_bound, &ptr); CHKERRQ (ierr);
  // lower bound
  ierr = VecGetArray(lower_bound, &ptr);CHKERRQ (ierr);
  ptr[0] = ctx_->params_->opt_->gamma_lb_;
  ptr[1] = ctx_->params_->opt_->rho_lb_;
  ptr[2] = ctx_->params_->opt_->k_lb_;
  ierr = VecRestoreArray(lower_bound, &ptr); CHKERRQ (ierr);
  // set
  ierr = TaoSetVariableBounds(tao_, lower_bound, upper_bound); CHKERRQ (ierr);
  ierr = VecDestroy(&lower_bound); CHKERRQ(ierr);
  ierr = VecDestroy(&upper_bound); CHKERRQ(ierr);
  PetscFunctionReturn(ierr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode MEOptimizer::setTaoOptions() {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;

  ierr = Optimizer::setTaoOptions(tao_, ctx_); CHKERRQ(ierr);

  // set monitor fro mass-effect inversion
  ierr = TaoSetMonitor (tao_, optimizationMonitorMassEffect, (void *) ctx_, NULL); CHKERRQ(ierr);
  // set convergence test routine
  ierr = TaoSetConvergenceTest (tao_, checkConvergenceGradMassEffect, (void *) ctx_); CHKERRQ(ierr);

  PetscFunctionReturn(ierr);
}
