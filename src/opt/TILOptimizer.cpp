#include "TILOptimizer.h"

TILOptimizer::initialize(
  std::shared_ptr<DerivativeOperators> derivative_operators,
  std::shared_ptr <PdeOperators> pde_operators,
  std::shared_ptr <Parameters> params,
  std::shared_ptr <Tumor> tumor)) {

  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  std::stringstream ss;

  // number of dofs = {p, kappa}
  n_inv_ = params_->tu_->np_ +  params_->get_nk();
  ss << " Initializing TIL optimizer with = " << n_inv_ << " = " << params_->tu_->np_ << " + " << params_->get_nk() << " dofs.";
  ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
  // initialize super class
  ierr = Optimizer::initialize(derivative_operators, pde_operators, params, tumor); CHKERRQ(ierr);

  PetscFunctionReturn(ierr);
}


// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode TILOptimizer::setVariableBounds() {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  ierr = tuMSGstd(" .. setting variable bounds for {p, kappa}."); CHKERRQ(ierr);
  ScalarType *ptr;
  Vec lower_bound, upper_bound;
  ierr = VecDuplicate (xrec_, &lower_bound); CHKERRQ(ierr);
  ierr = VecSet (lower_bound, 0.); CHKERRQ(ierr);
  ierr = VecDuplicate (xrec_, &upper_bound); CHKERRQ(ierr);
  ierr = VecSet (upper_bound, PETSC_INFINITY); CHKERRQ(ierr);
  // upper bound
  if (ctx_->params_->opt_->diffusivity_inversion_) {
    ierr = VecGetArray (upper_bound, &ptr); CHKERRQ (ierr);
    ptr[ctx_->params_->tu_->np_] = ctx_->params_->opt_->k_ub_;
    if (ctx_->params_->tu_->nk_ > 1) ptr[ctx_->params_->tu_->np_ + 1] = ctx_->params_->opt_->k_ub_;
    if (ctx_->params_->tu_->nk_ > 2) ptr[ctx_->params_->tu_->np_ + 2] = ctx_->params_->opt_->k_ub_;
    ierr = VecRestoreArray (upper_bound, &ptr); CHKERRQ (ierr);

    ierr = VecGetArray (lower_bound, &ptr); CHKERRQ (ierr);
    ptr[ctx_->params_->tu_->np_] = ctx_->params_->opt_->k_lb_;
    if (ctx_->params_->tu_->nk_ > 1) ptr[ctx_->params_->tu_->np_ + 1] = ctx_->params_->opt_->k_lb_;
    if (ctx_->params_->tu_->nk_ > 2) ptr[ctx_->params_->tu_->np_ + 2] = ctx_->params_->opt_->k_lb_;
    ierr = VecRestoreArray (lower_bound, &ptr); CHKERRQ (ierr);
  }
  // set
  ierr = TaoSetVariableBounds(tao_, lower_bound, upper_bound); CHKERRQ (ierr);
  ierr = VecDestroy(&lower_bound); CHKERRQ(ierr);
  ierr = VecDestroy(&upper_bound); CHKERRQ(ierr);
  PetscFunctionReturn(ierr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode TILOptimizer::setTaoOptions() {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;

  ierr = Optimizer::setTaoOptions(tao_, ctx_); CHKERRQ(ierr);


  ierr = TaoSetConvergenceTest (tao_, checkConvergenceGrad, (void *) ctx_); CHKERRQ(ierr);


  PetscFunctionReturn(ierr);
}
