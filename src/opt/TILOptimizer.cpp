#include <iostream>
#include <limits>

#include <petsc/private/vecimpl.h>
#include "petsctao.h"
#include "petsc/private/taoimpl.h"
#include "petsc/private/taolinesearchimpl.h"
// #include <petsc/private/vecimpl.h>

#include "Optimizer.h"
#include "TaoInterface.h"
#include "TILOptimizer.h"

PetscErrorCode TILOptimizer::initialize(
  std::shared_ptr<DerivativeOperators> derivative_operators,
  std::shared_ptr <PdeOperators> pde_operators,
  std::shared_ptr <Parameters> params,
  std::shared_ptr <Tumor> tumor) {

  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  std::stringstream ss;

  // number of dofs = {p, kappa}
  n_inv_ = params->tu_->np_ +  params->get_nk();
  ss << " Initializing TIL optimizer with n_inv = " << n_inv_ << " = " << params->tu_->np_ << " + " << params->get_nk() << " dofs.";
  ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
  // initialize super class
  ierr = Optimizer::initialize(derivative_operators, pde_operators, params, tumor); CHKERRQ(ierr);

  PetscFunctionReturn(ierr);
}


// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode TILOptimizer::allocateTaoObjects() {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  std::stringstream ss;

  // number of dofs = {p, kappa}
  n_inv_ = ctx_->params_->tu_->np_ +  ctx_->params_->get_nk();
  ss << " Re-itializing TIL optimizer with changed n_inv = " << n_inv_ << " = " << ctx_->params_->tu_->np_ << " + " << ctx_->params_->get_nk() << " dofs.";
  ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
  ierr = Optimizer::allocateTaoObjects(); CHKERRQ(ierr);
  PetscFunctionReturn (ierr);
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
  if(xout_ != nullptr) {ierr = VecDestroy(&xout_); CHKERRQ(ierr);}

  ss << " Setting initial guess: ";
  int off = 0, off_in = 0;
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
    ierr = ctx_->tumor_->phi_->apply (ctx_->tumor_->c_0_, x_init); CHKERRQ(ierr);
    ierr = VecMax(ctx_->tumor_->c_0_, NULL, &ic_max); CHKERRQ (ierr);
    ss << "TIL as Phi(p) (max=" << ic_max<<"); ";
    if (ctx_->params_->tu_->write_p_checkpoint_) {writeCheckpoint(xin_, ctx_->tumor_->phi_, ctx_->params_->tu_->writepath_, std::string("til-init"));}
    off = off_in = np;
  }
  // === kappa init guess
  if (in_size > np) {
    ierr = VecGetArray(x_init, &init_ptr); CHKERRQ (ierr);
    ierr = VecGetArray(xin_, &x_ptr); CHKERRQ (ierr);
    x_ptr[off_in + 0] = init_ptr[off];                           // k1
    k_init_ = x_ptr[off_in + 0];
    if (nk > 1) x_ptr[off_in + 1] = init_ptr[off + 1];           // k2
    if (nk > 2) x_ptr[off_in + 2] = init_ptr[off + 2];           // k3
    ierr = VecRestoreArray(xin_, &x_ptr); CHKERRQ (ierr);
    ierr = VecRestoreArray(x_init, &init_ptr); CHKERRQ (ierr);
  }

  ss << "k_init="<<k_init_;
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
      ierr = resetTao(); CHKERRQ(ierr);
  }
  // === set tao options
  if (tao_reset_) {
    tuMSGstd(" Setting tao options for TIL optimizer."); CHKERRQ(ierr);
    ierr = setTaoOptions(); CHKERRQ(ierr);
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
  ss << " .. using tumor regularization beta: "<< ctx_->params_->opt_->beta_ << " and type: " << ctx_->params_->opt_->regularization_norm_;  ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
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

  /* === get termination info === */
  TaoConvergedReason reason;
  ierr = TaoGetConvergedReason (tao_, &reason); CHKERRQ(ierr);

  /* === get solution status === */
  PetscScalar xdiff;
  ierr = TaoGetSolutionStatus(tao_, NULL, &ctx_->params_->optf_->jval_, &ctx_->params_->optf_->gradnorm_, NULL, &xdiff, NULL); CHKERRQ(ierr);
  // display convergence reason:
  ierr = dispTaoConvReason(reason, ctx_->params_->optf_->solverstatus_); CHKERRQ(ierr);
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
  ierr = VecCopy(xrec_, xout_); CHKERRQ(ierr);

  // === update diffusivity coefficient
  ScalarType *x_ptr;
  ierr = VecGetArray(xrec_, &x_ptr); CHKERRQ (ierr);
  ctx_->params_->tu_->k_ =  x_ptr[np];
  ierr = VecRestoreArray(xrec_, &x_ptr); CHKERRQ (ierr);
  PetscReal  k1, k2, k3;
  k1 = ctx_->params_->tu_->k_;
  k2 = (nk > 1) ? ctx_->params_->tu_->kf_ : 0;
  k3 = (nk > 2) ? ctx_->params_->tu_->k_   * ctx_->params_->tu_->k_glm_wm_ratio_ : 0;
  ierr = ctx_->tumor_->k_->updateIsotropicCoefficients (k1, k2, k3, ctx_->tumor_->mat_prop_, ctx_->params_); CHKERRQ (ierr);
  ierr = tuMSGstd ("");                                                     CHKERRQ (ierr);
  ierr = tuMSGstd ("### ------------------------------------------------- ###"); CHKERRQ (ierr);
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
    if (ctx_->params_->tu_->nk_ > 1) ptr[ctx_->params_->tu_->np_ + 1] = ctx_->params_->opt_->kf_ub_;
    if (ctx_->params_->tu_->nk_ > 2) ptr[ctx_->params_->tu_->np_ + 2] = ctx_->params_->opt_->k_ub_;
    ierr = VecRestoreArray (upper_bound, &ptr); CHKERRQ (ierr);

    ierr = VecGetArray (lower_bound, &ptr); CHKERRQ (ierr);
    ptr[ctx_->params_->tu_->np_] = ctx_->params_->opt_->k_lb_;
    if (ctx_->params_->tu_->nk_ > 1) ptr[ctx_->params_->tu_->np_ + 1] = ctx_->params_->opt_->kf_lb_;
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

  ierr = Optimizer::setTaoOptions(); CHKERRQ(ierr);
  ierr = TaoSetConvergenceTest(tao_, checkConvergenceGrad, (void *) ctx_.get()); CHKERRQ(ierr);

  PetscFunctionReturn(ierr);
}
