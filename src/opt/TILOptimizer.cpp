#include <iostream>
#include <limits>

#include "petsctao.h"
// #include <petsc/private/vecimpl.h>

#include "Optimizer.h"
#include "TaoInterface.h"
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
PetscErrorCode TILOptimizer::setInitialGuess(Vec x_init) {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  if (!initialized_) {
    ierr = tuMSGwarn("Error: Optimizer not initialized."); CHKERRQ(ierr);
    PetscFunctionReturn(ierr);
  }
  std::stringstream ss;
  int nk = ctx_->params_->tu_->nk_;
  int np = ctx_->params_->tu_->np_;
  int in_size;
  ierr = VecGetSize(x_init, &in_size); CHKERRQ(ierr);
  ScalarType *init_ptr, *x_ptr;

  if(xin_ != nullptr) {ierr = VecDestroy(&xin_); CHKERRQ(ierr);}
  if(x_out != nullptr) {ierr = VecDestroy(&x_out); CHKERRQ(ierr);}

  ss << " Setting initial guess: ";
  int off = 0; off_in = 0;
  PetscReal ic_max = 0.;
  // 1. TIL not given as parametrization
  if(ctx_->params_->tu_->use_c0_) {
    ierr = tuMSGwarn(" Error: cannot invert for TIL if c(0) parametrization is disabled (set use_c0 = false)."); CHKERRQ(ierr);
    exit(0);
  // 2. TIL given as parametrization
  } else {
    ierr = VecCreateSeq (PETSC_COMM_SELF, np + nk, &xin_); CHKERRQ (ierr);
    ierr = setupVec (xin_, SEQ); CHKERRQ (ierr);
    ierr = VecSet (xin_, 0.0); CHKERRQ (ierr);
    ierr = VecCopy(x_init, xin_); CHKERRQ(ierr);
    // === init TIL
    ierr = itctx_->tumor_->phi_->apply (itctx_->tumor_->c_0_, x_init); CHKERRQ(ierr);
    ierr = VecMax(ctx_->tumor_->c_0_, NULL, &ic_max); CHKERRQ (ierr);
    ss << "TIL as Phi(p) (max=" << ic_max<<"); ";
    if(ctx_->params_->opt_->rescale_init_cond_) {
      ScalarType scale = (1.0 / ic_max);
      if(ctx_->params_->opt_->multilevel_) scale = (1.0/4.0 * ctx_->params_->grid_->n_[0]/64.  / ic_max);
      ierr = VecGetArray (xin_, &x_ptr); CHKERRQ (ierr);
      for (int i = 0; i < np ; i++) x_ptr[i] *= scale;
      ierr = VecRestoreArray (xin_, &x_ptr); CHKERRQ (ierr);
      ss << "rescaled; ";
    }
    if (ctx_->params_->tu_->write_p_checkpoint_) {writeCheckpoint(xin_, ctx_->tumor_->phi_, ctx_->params_->tu_->writepath_ .str(), std::string("til-init"));}
    off = off_in = np;
  }
  // === kappa init guess
  if (in_size > np) {
    ierr = VecGetArray(x_init, &init_ptr); CHKERRQ (ierr);
    ierr = VecGetArray(xin_, &x_ptr); CHKERRQ (ierr);
    k_init_   = x_ptr[off_in + 0] = x_init[off];               // k1
    if (nk > 1) x_ptr[off_in + 1] = x_init[off + 1];           // k2
    if (nk > 2) x_ptr[off_in + 2] = x_init[off + 2];           // k3
    ierr = VecRestoreArray(xin_, &x_ptr); CHKERRQ (ierr);
    ierr = VecRestoreArray(x_init, &init_ptr); CHKERRQ (ierr);
  }

  ss << "k_init="<<k_init_<<;
  ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
  // create xout_ vec
  ierr = VecDuplicate(xin_, &xout_); CHKERRQ (ierr);
  ierr = VecSet(xout_, 0.0); CHKERRQ (ierr);

  PetscFunctionReturn(ierr);
}


// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode TILOptimizer::solve() {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  int procid, nprocs;
  MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank (MPI_COMM_WORLD, &procid);
  TU_assert (initialized_, "TILOptimizer needs to be initialized.")
  TU_assert (data_->dt1() != nullptr, "TILOptimizer requires non-null input data for inversion.");
  TU_assert (xrec_ != nullptr, "TILOptimizer requires non-null xrec_ vector to be set.");
  TU_assert (xin_ != nullptr, "TILOptimizer requires non-null xin_ vector to be set.");

  std::stringstream ss;
  ierr = tuMSGstd (""); CHKERRQ (ierr);
  ierr = tuMSG("### ----------------------------------------------------------------------------------------------------- ###");CHKERRQ (ierr);
  ierr = tuMSG("###                                         TIL inversion                                                 ###");CHKERRQ (ierr);
  ierr = tuMSG("### ----------------------------------------------------------------------------------------------------- ###");CHKERRQ (ierr);

  // DOFs
  int nk = ctx_->params_->tu_->nk_;
  int np = ctx_->params_->tu_->np_;
  TU_assert(n_inv_ == np + nk, "TILOptimizer: n_inv is inconsistent.");

  // === reset tao, (if we want virgin tao for every inverse solve)
  if (ctx_->params_->opt_->reset_tao_) {
      ierr = resetTao(ctx_->params_); CHKERRQ(ierr);
  }
  // === set tao options
  if (tao_reset_) {
    tuMSGstd(" Setting tao options for TIL optimizer."); CHKERRQ(ierr);
    ierr = setTaoOptions(); CHKERRQ(ierr);
    // ctx_->update_reference_gradient = true;   // TODO: K: I commented this; for CoSaMp_RS we don't want to re-compute reference gradient between inexact blocks (if coupled with sibia, the data will change)
    // ctx_->update_reference_objective = true;  // TODO: K: I commented this; for CoSaMp_RS we don't want to re-compute reference gradient between inexact blocks (if coupled with sibia, the data will change)
  }

  // === initial guess
  ierr = VecCopy(xin_, xrec_); CHKERRQ(ierr);
  ierr = TaoSetInitialVector(tao_, xrec_); CHKERRQ (ierr);

  // === initialize inverse tumor context TODO(K) check if all of those are needed
  if (ctx_->c0_old == nullptr) {
    ierr = VecDuplicate(data_->dt1(), &ctx_->c0_old); CHKERRQ(ierr);
  }
  ierr = VecSet (ctx_->c0_old, 0.0); CHKERRQ(ierr);
  if (ctx_->tmp == nullptr) {
    ierr = VecDuplicate(data_->dt1(), &ctx_->tmp); CHKERRQ(ierr);
    ierr = VecSet(ctx_->tmp, 0.0); CHKERRQ(ierr);
  }
  if (ctx_->x_old != nullptr) { // TODO(K) I've changed this from tumor_->p_ to xrec_
    ierr = VecDestroy(&ctx_->x_old); CHKERRQ (ierr);
    ctx_->x_old = nullptr;
  }
  ierr = VecDuplicate(xrec_, &ctx_->x_old); CHKERRQ(ierr);
  ierr = VecCopy(xrec_, ctx_->x_old); CHKERRQ(ierr);
  // if (ctx_->x_old == nullptr)  {
    // ierr = VecDuplicate (xrec_, &ctx_->x_old); CHKERRQ (ierr);
    // ierr = VecCopy (xrec_, ctx_->x_old); CHKERRQ (ierr);
  // }

  // reset feedback variables, reset data
  ctx_->update_reference_gradient_hessian_ksp = true;
  ctx_->params_->tu_->statistics_.reset();
  ctx_->params_->optf_->reset();
  ctx_->data = data_;
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

  /* === get termination info === */
  TaoConvergedReason reason;
  ierr = TaoGetConvergedReason (tao_, &reason); CHKERRQ(ierr);

  /* === get solution status === */
  PetscScalar xdiff;
  ierr = TaoGetSolutionStatus (tao_, NULL, &ctx_->params_->optf_->jval_, &ctx_->params_->optf_->gradnorm_, NULL, &xdiff, NULL); CHKERRQ(ierr);
  // display convergence reason:
  ierr = dispTaoConvReason (reason, ctx_->params_->optf_->solverstatus_); CHKERRQ(ierr);
  ctx_->params_->optf_->nb_newton_it_--;
  ss << " optimization done: #N-it: " << ctx_->params_->optf_->nb_newton_it_
    << ", #K-it: " << ctx_->params_->optf_->nb_krylov_it_
    << ", #matvec: " << ctx_->params_->optf_->nb_matvecs_
    << ", #evalJ: " << ctx_->params_->optf_->nb_objevals_
    << ", #evaldJ: " << ctx_->params_->optf_->nb_gradevals_
    << ", exec time: " << invtime;
  ierr = tuMSGstd ("------------------------------------------------------------------------------------------------"); CHKERRQ(ierr);
  ierr = tuMSGstd (ss.str());                                                                                            CHKERRQ(ierr);  ss.str(""); ss.clear();
  ierr = tuMSGstd ("------------------------------------------------------------------------------------------------"); CHKERRQ(ierr);
  ctx_->params_->tu_->statistics_.print();
  ctx_->params_->tu_->statistics_.reset();
  ctx_->update_reference_gradient = false;
  tao_reset_ = false;

  // === populate solution to xout_
  // * {p, gamma, rho, kappa}, if c(0) is given as parametrization
  // * {gamma, rho, kappa} otherwise
  ierr = VecCopy(xin_, xout_); CHKERRQ(ierr);
  ierr = VecGetArray (xrec_, &x_ptr); CHKERRQ (ierr);
  ierr = VecGetArray (xout_, &xout_ptr); CHKERRQ (ierr);
  for(int i = 0; i < n_inv_; ++i)
    xout_ptr[off + i] = x_ptr[i];
  ierr = VecRestoreArray(xrec_, &x_ptr); CHKERRQ (ierr);
  ierr = VecRestoreArray(xout_, &xout_ptr); CHKERRQ (ierr);
  if (ctx_->params_->tu_->write_p_checkpoint_) {writeCheckpoint(xout_, ctx_->tumor_->phi_, ctx_->params_->tu_->writepath_ .str(), std::string("me-out"));}

  // === update diffusivity and reaction in coefficients
  ierr = VecGetArray (xrec_, &x_ptr); CHKERRQ (ierr);
  ctx_->params_->tu_->forcing_factor_ = x_ptr[0]; // gamma
  ctx_->params_->tu_->rho_ = x_ptr[1];            // rho
  ctx_->params_->tu_->k_   = x_ptr[1 + nr];       // kappa
  ierr = VecRestoreArray (xrec_, &x_ptr); CHKERRQ (ierr);
  PetscReal r1, r2, r3, k1, k2, k3;
  r1 = ctx_->params_->tu_->rho_;
  r2 = (nr > 1) ? ctx_->params_->tu_->rho_ * ctx_->params_->tu_->r_gm_wm_ratio_  : 0;
  r3 = (nr > 2) ? ctx_->params_->tu_->rho_ * ctx_->params_->tu_->r_glm_wm_ratio_ : 0;
  k1 = ctx_->params_->tu_->k_;
  k2 = (nk > 1) ? ctx_->params_->tu_->k_   * ctx_->params_->tu_->k_gm_wm_ratio_  : 0;
  k3 = (nk > 2) ? ctx_->params_->tu_->k_   * ctx_->params_->tu_->k_glm_wm_ratio_ : 0;
  ierr = ctx_->tumor_->k_->updateIsotropicCoefficients (k1, k2, k3, ctx_->tumor_->mat_prop_, ctx_->params_); CHKERRQ (ierr);
  ierr = ctx_->tumor_->rho_->updateIsotropicCoefficients (r1, r2, r3, ctx_->tumor_->mat_prop_, ctx_->params_); CHKERRQ (ierr);

  ierr = tuMSG("### ------------------------------------- gamma/rho/kappa solver end ------------------------------------ ###");CHKERRQ (ierr);
  ierr = tuMSGstd (""); CHKERRQ (ierr);
  ierr = tuMSGstd ("");                                                     CHKERRQ (ierr);
  ierr = tuMSGstd ("### ------------------------------------------------- ###"); CHKERRQ (ierr);
  ierr = tuMSG    ("### estimated forcing factor:                         ###"); CHKERRQ (ierr);
  ss << "    gamma: "<< ctx_->params_->tu_->forcing_factor_;
  ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
  ierr = tuMSG    ("### estimated reaction coefficients:                  ###"); CHKERRQ (ierr);
  ss << "    r1: "<< r1 << ", r2: " << r2 << ", r3: "<< r3;
  ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
  ierr = tuMSG    ("### estimated diffusion coefficients:                 ###"); CHKERRQ (ierr);
  ss << "    k1: "<< k1 << ", k2: " << k2 << ", k3: "<< k3;
  ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
  ierr = tuMSGstd ("### ------------------------------------------------- ###"); CHKERRQ (ierr);

  // cleanup
  if (ctx_->x_old != nullptr) {ierr = VecDestroy (&ctx_->x_old);  CHKERRQ (ierr); ctx_->x_old = nullptr;}
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
