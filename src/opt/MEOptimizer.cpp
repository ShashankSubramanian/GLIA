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
PetscErrorCode MEOptimizer::solve() {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  int procid, nprocs;
  MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank (MPI_COMM_WORLD, &procid);
  TU_assert (initialized_, "MEOptimizer needs to be initialized.")
  TU_assert (data_->dt1() != nullptr, "MEOptimizer requires non-null input data for inversion.");
  TU_assert (xrec_ != nullptr, "MEOptimizer requires non-null xrec_ vector to be set.");
  TU_assert (xin_ != nullptr, "MEOptimizer requires non-null xin_ vector to be set.");

  std::stringstream ss;
  ierr = tuMSGstd (""); CHKERRQ (ierr);
  ierr = tuMSG("### ----------------------------------------------------------------------------------------------------- ###");CHKERRQ (ierr);
  ierr = tuMSG("###                                 Mass Effect Inversion                                                 ###");CHKERRQ (ierr);
  ierr = tuMSG("### ----------------------------------------------------------------------------------------------------- ###");CHKERRQ (ierr);

  // DOFs
  int nk = ctx_->params_->tu_->nk_;
  int nr = ctx_->params_->tu_->nr_;
  TU_assert(n_inv_ == nr + nk + 1, "MEOptimizer: n_inv is inconsistent.");

  // initial guess
  ierr = VecCopy(xin_, xrec_); CHKERRQ(ierr);

  if (ctx_->x_old != nullptr) {
    ierr = VecDestroy (&ctx_->x_old);  CHKERRQ (ierr);
    ctx_->x_old = nullptr;
  }
  ierr = VecDuplicate(xrec_, &ctx_->x_old); CHKERRQ(ierr);
  ierr = VecCopy(xrec_, ctx_->x_old); CHKERRQ(ierr);

  // reset feedback variables, reset data
  ctx_->update_reference_gradient_hessian_ksp = true;
  ctx_->params_->tu_->statistics_.reset();
  ctx_->params_->optf_->reset();
  ctx_->data = data_;

  // reset tao, if we want virgin TAO for every inverse solve
  if (ctx_->params_->opt_->reset_tao_) {
      ierr = resetTao(ctx_->params_); CHKERRQ(ierr);
  }

  // set tao options
  if (tao_reset_) {
    tuMSGstd(" Setting tao options for TIL optimizer."); CHKERRQ(ierr);
    ierr = setTaoOptions(); CHKERRQ(ierr);
  }

  ss << " using tumor regularization = "<< ctx_->params_->opt_->beta_ << " type: " << ctx_->params_->opt_->regularization_norm_;  ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
  if (ctx_->params_->tu_->verbosity_ >= 2) { ctx_->params_->tu_->outfile_sol_  << "\n ## ----- ##" << std::endl << std::flush; ctx_->params_->tu_->outfile_grad_ << "\n ## ----- ## "<< std::endl << std::flush; }
  //Gradient check begin
  //    ierr = ctx_->derivative_operators_->checkGradient (ctx_->tumor_->p_, ctx_->data);
  //Gradient check end

  /* === solve === */
  double self_exec_time_tuninv = -MPI_Wtime(); double invtime = 0;
  ierr = TaoSolve (tao_); CHKERRQ(ierr);
  self_exec_time_tuninv += MPI_Wtime();
  MPI_Reduce(&self_exec_time_tuninv, &invtime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  /* === get solution === */
  Vec sol; ierr = TaoGetSolutionVector (tao_, &sol); CHKERRQ(ierr);
  ierr = VecCopy (sol, xrec_); CHKERRQ(ierr);
  ierr = VecCopy (sol, xout_); CHKERRQ(ierr);

  PetscScalar *x_ptr;
  ierr = VecGetArray (xrec_, &x_ptr); CHKERRQ (ierr);
  ctx_->params_->tu_->forcing_factor_ = 1E4 * x_ptr[0]; // re-scaling parameter scales
  ctx_->params_->tu_->rho_ = 1 * x_ptr[1];              // rho
  ctx_->params_->tu_->k_ = 1E-2 * x_ptr[2];             // kappa
  ierr = VecRestoreArray (xrec_, &x_ptr); CHKERRQ (ierr);

  ss << " Forcing factor at final guess = " << ctx_->params_->tu_->forcing_factor_; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
  ss << " Reaction at final guess       = " << ctx_->params_->tu_->rho_; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
  ss << " Diffusivity at final guess    = " << ctx_->params_->tu_->k_; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();

  /* === get termination info === */
  TaoConvergedReason reason;
  ierr = TaoGetConvergedReason (tao_, &reason); CHKERRQ(ierr);

  /* === get solution status === */
  PetscScalar xdiff;
  ierr = TaoGetSolutionStatus (tao_, NULL, &ctx_->params_->optf_->jval_, &ctx_->params_->optf_->gradnorm_, NULL, &xdiff, NULL); CHKERRQ(ierr);
  /* display convergence reason: */
  ierr = dispTaoConvReason (reason, ctx_->params_->optf_->solverstatus_); CHKERRQ(ierr);
  ctx_->params_->optf_->nb_newton_it_--;
  ss << " optimization done: #N-it: " << ctx_->params_->optf_->nb_newton_it_    << ", #K-it: " << ctx_->params_->optf_->nb_krylov_it_
     << ", #matvec: " << ctx_->params_->optf_->nb_matvecs_    << ", #evalJ: " << ctx_->params_->optf_->nb_objevals_
     << ", #evaldJ: " << ctx_->params_->optf_->nb_gradevals_  << ", exec time: " << invtime;
  ierr = tuMSGstd ("------------------------------------------------------------------------------------------------"); CHKERRQ(ierr);
  ierr = tuMSGstd (ss.str()); CHKERRQ(ierr);  ss.str(""); ss.clear();
  ierr = tuMSGstd ("------------------------------------------------------------------------------------------------"); CHKERRQ(ierr);
  ctx_->params_->tu_->statistics_.print();
  ctx_->params_->tu_->statistics_.reset();
  // only update if triggered from outside, i.e., if new information to the ITP solver is present
  ctx_->update_reference_gradient = false;
  tao_reset_ = false;
  // cleanup
  if (ctx_->x_old != nullptr) {ierr = VecDestroy (&ctx_->x_old);  CHKERRQ (ierr); ctx_->x_old = nullptr;}
  PetscFunctionReturn(ierr);
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

  // TODO(K): Do we need separate methods, double check.
  // set monitor fro mass-effect inversion
  ierr = TaoSetMonitor (tao_, optimizationMonitorMassEffect, (void *) ctx_, NULL); CHKERRQ(ierr);
  // set convergence test routine
  ierr = TaoSetConvergenceTest (tao_, checkConvergenceGradMassEffect, (void *) ctx_); CHKERRQ(ierr);
  // hessian routine
  ierr = TaoSetHessianRoutine (tao_, H_, H_, matfreeHessian, (void *) ctx_.get()); CHKERRQ(ierr);

  PetscFunctionReturn(ierr);
}
