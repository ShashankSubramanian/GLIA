#include <iostream>
#include <limits>

#include "petsctao.h"

#include "Optimizer.h"
#include "Parameters.h"
#include "DerivativeOperators.h"
#include "PdeOperators.h"

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
Optimizer::~Optimizer() {
  if(tao_  != nullptr) TaoDestroy (&tao_);
  if(H_    != nullptr) MatDestroy (&H_);
  if(xrec_ != nullptr) VecDestroy (&xrec_);
  if(xin_  != nullptr) VecDestroy(&xin_);
  if(xout_ != nullptr) VecDestroy(&xout_);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
Optimizer::initialize(
  std::shared_ptr<DerivativeOperators> derivative_operators,
  std::shared_ptr <PdeOperators> pde_operators,
  std::shared_ptr <Parameters> params,
  std::shared_ptr <Tumor> tumor)) {

  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  if (initialized_) PetscFunctionReturn(ierr);

  ctx_ = std::make_shared<CtxInv> ();
  ctx_->derivative_operators_ = derivative_operators;
  ctx_->pde_operators_ = pde_operators;
  ctx_->params_ = params;
  ctx_->tumor_ = tumor;

  ierr = allocateTaoObjects(); CHKERRQ(ierr);
  initialized_ = true;
  PetscFunctionReturn (ierr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode Optimizer::allocateTaoObjects() {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  if (!initialized_) {
    ierr = tuMSGwarn("Error: Optimizer not initialized."); CHKERRQ(ierr);
    PetscFunctionReturn(ierr);
  }

  // allocate memory for xrec_ (n_inv_ is already set in the specialized function)
  ierr = VecCreateSeq (PETSC_COMM_SELF, n_inv_, &xrec_); CHKERRQ (ierr);
  ierr = setupVec (xrec_, SEQ); CHKERRQ (ierr);
  ierr = VecSet (xrec_, 0.0); CHKERRQ (ierr);

  // set up routine to compute the hessian matrix vector product
  if (H_ == nullptr) { // TODO(K): is this also needed for QN? (ctx_->params_->opt_->newton_solver_ == GAUSSNEWTON)
    ierr = MatCreateShell (PETSC_COMM_SELF, n_inv_, n_inv_, n_inv_, n_inv_, (void*) ctx_.get(), &H_); CHKERRQ(ierr);
  }
  ierr = MatShellSetOperation (H_, MATOP_MULT, (void (*)(void))hessianMatVec); CHKERRQ(ierr);
  #if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 10)
      ierr = MatShellSetOperation (H_, MATOP_CREATE_VECS, (void(*)(void)) operatorCreateVecs);
  #endif
  ierr = MatSetOption (H_, MAT_SYMMETRIC, PETSC_TRUE); CHKERRQ(ierr);

  // create tao object
  if (tao_ == nullptr) {
    ierr = TaoCreate (PETSC_COMM_SELF, &tao_);
    tao_reset_ = true;  // triggers setTaoOptions TODO(K): check if we need this
  }
  PetscFunctionReturn (ierr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode Optimizer::resetTao () {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  if (tao_  != nullptr) {ierr = TaoDestroy (&tao_);  CHKERRQ(ierr); tao_  = nullptr;}
  if (H_    != nullptr) {ierr = MatDestroy (&H_);    CHKERRQ(ierr); H_    = nullptr;}
  if (xrec_ != nullptr) {ierr = VecDestroy (&xrec_); CHKERRQ(ierr); xrec_ = nullptr;}
  // allocate memory for H, x_rec and TAO
  ierr = allocateTaoObjects (); CHKERRQ(ierr);
  tao_reset_ = true; // triggers setTaoOptions
  PetscFunctionReturn (ierr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode Optimizer::reset(Vec p) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  // reset tumor_ object, re-size solution vector and copy p into tumor_->p_
  ierr = ctx_->tumor_->setParams (p, ctx_->params_, true); CHKERRQ (ierr);
  // reset derivative operators, re-size vectors
  ierr = ctx_->derivative_operators_->reset(p, ctx_->pde_operators_, ctx_->params_, ctx_->tumor_); CHKERRQ(ierr);
  if (ctx_->x_old != nullptr) {ierr = VecDestroy(&ctx_->x_old); CHKERRQ(ierr); ctx_->x_old = nullptr;}
  // ctx_->x_old = nullptr; // re-allocate memory
  ierr = resetTao(); CHKERRQ(ierr);
  PetscFunctionReturn (ierr);
}


// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode Optimizer::setTaoOptions() {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  std::Stringstream ss;
  std::string msg;

  int procid, nprocs;
  MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank (MPI_COMM_WORLD, &procid);

  // set tao solver type
  if (ctx_->params_->opt_->newton_solver_ == QUASINEWTON) {
    ierr = TaoSetType(tao_, "blmvm"); CHKERRQ (ierr);
  } else {
    ierr = TaoSetType(tao_, "bnls"); CHKERRQ(ierr);
  }

  #if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 9)
    TaoType taotype = NULL;
    ierr = TaoGetType(tao_, &taotype); CHKERRQ(ierr);
  #else
    const TaoType taotype;
    ierr = TaoGetType (tao_, &taotype); CHKERRQ(ierr);
  #endif
  if (strcmp(taotype, "nls") == 0) {ierr = tuMSGstd(" Newton's method (line search; unconstrained) selected"); CHKERRQ(ierr);}
  else if (strcmp(taotype, "bnls") == 0) {ierr = tuMSGstd(" Newton's method (line search; bound constraints) selected"); CHKERRQ(ierr);}
  else if (strcmp(taotype, "bqnls") == 0) {ierr = tuMSGstd(" Quasi-Newton's method BGNLS (line search; bound constraints) selected"); CHKERRQ(ierr);}
  else if (strcmp(taotype, "blmvm") == 0) {ierr = tuMSGstd(" Quasi-Newton's method BLMVM (line search; bound constraints) selected"); CHKERRQ(ierr);}
  else if (strcmp(taotype, "lmvm") == 0) {ierr = tuMSGstd(" Quasi-Newton's method LMVM (line search; unconstrained) selected"); CHKERRQ(ierr);}
  else if (strcmp(taotype, "tao_blmvm_m") == 0) {ierr = tuMSGstd(" User modified quasi-Newton's method BLMVM (line search; bound constraints) selected"); CHKERRQ(ierr);}
  else    (strcmp(taotype, "fd_test") == 0) {ierr = tuMSGstd(" Gradient test selected"); CHKERRQ(ierr);}

  ierr = tuMSGstd(" parameters (optimizer):"); CHKERRQ(ierr);

  // set line-search method and minstep
  TaoLineSearch linesearch;
  ierr = TaoGetLineSearch (tao_, &linesearch); CHKERRQ(ierr);
  linesearch->stepmin = ctx_->params_->opt_->ls_minstep_;
  if (ctx_->params_->opt_->linesearch_ == ARMIJO) {
    ierr = TaoLineSearchSetType (linesearch, "armijo"); CHKERRQ(ierr);
    ierr = tuMSGstd(" .. using line-search type: armijo"); CHKERRQ(ierr);
  } else {
    ierr = TaoLineSearchSetType (linesearch, "mt"); CHKERRQ(ierr);
    ierr = tuMSGstd("  .. using line-search type: more-thuene"); CHKERRQ(ierr);
  }
  ierr = TaoLineSearchSetInitialStepLength (linesearch, 1.0); CHKERRQ(ierr);
  ierr = TaoLineSearchSetOptionsPrefix (linesearch,"tumor_"); CHKERRQ(ierr);

  // manually set petsc options
  PetscBool flag = PETSC_FALSE;
  #if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 7)
  // disable hessian preconditioner
  ierr = PetscOptionsHasName (NULL, NULL, "-tao_nls_pc_type", &flag); CHKERRQ(ierr);
  if (flag == PETSC_FALSE) {ierr = PetscOptionsSetValue (NULL, "-tao_nls_pc_type", "none"); CHKERRQ(ierr);}
  // set krylov solver type to conjugate gradient
  ierr = PetscOptionsHasName (NULL, NULL, "-tao_nls_ksp_type", &flag); CHKERRQ(ierr);
  if (flag == PETSC_FALSE) {ierr = PetscOptionsSetValue (NULL, "-tao_nls_ksp_type", "cg"); CHKERRQ(ierr);}
  // disable preconditioner
  ierr = PetscOptionsHasName (NULL, NULL, "-tao_ntr_pc_type", &flag); CHKERRQ(ierr);
  if (flag == PETSC_FALSE) {ierr = PetscOptionsSetValue (NULL, "-tao_ntr_pc_type", "none"); CHKERRQ(ierr);}
  // set mat lmvm number fo vectors for quasi-Newton update
  ierr = PetscOptionsHasName (NULL, NULL, "-tao_blmvm_mat_lmvm_num_vecs", &flag); CHKERRQ(ierr);
  if (flag == PETSC_FALSE) {
    ierr = PetscOptionsSetValue (NULL, "-tao_blmvm_mat_lmvm_num_vecs", ctx_->params_->opt_->lbfgs_vectors_); CHKERRQ(ierr);
    ss << " .. using " << ctx_->params_->opt_->lbfgs_vectors_ << " vectors for inverse Hessian update of quasi-Newton method."; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
  }
  // set mat lmvm type of inverse Hessian initialization
  ierr = PetscOptionsHasName (NULL, NULL, "-tao_blmvm_mat_lmvm_scale_type", &flag); CHKERRQ(ierr);
  if (flag == PETSC_FALSE) {
    ierr = PetscOptionsSetValue (NULL, "-tao_blmvm_mat_lmvm_scale_type", ctx_->params_->opt_->lbfgs_scale_type_); CHKERRQ(ierr);
    ss << " .. setting inverse Hessian initial guess type to: " << ctx_->params_->opt_->lbfgs_scale_type_; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
  }
  // set max number of line-search function evaluations per Newton step
  ierr = PetscOptionsHasName (NULL, NULL, "-tumor_tao_ls_max_funcs", &flag); CHKERRQ(ierr);
  if (flag == PETSC_FALSE) {
    ierr = PetscOptionsSetValue (NULL, "-tumor_tao_ls_max_funcs", ctx_->params_->opt_->ls_max_func_evals); CHKERRQ(ierr);
    ierr = PetscOptionsSetValue (NULL, "-tao_ls_max_funcs", ctx_->params_->opt_->ls_max_func_evals); CHKERRQ(ierr);
    ss << " .. setting max number of line-search function evaluations to: " << ctx_->params_->opt_->ls_max_func_evals; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
  }
  // set tolerances
  ierr = TaoSetTolerances (tao_, ctx_->params_->opt_->gatol_, ctx_->params_->opt_->grtol_, ctx_->params_->opt_->opttolgrad_); CHKERRQ(ierr);
  #else
  // disable hessian preconditioner
  ierr = PetscOptionsHasName (NULL, "-tao_nls_pc_type", &flag); CHKERRQ(ierr);
  if (flag == PETSC_FALSE) {ierr = PetscOptionsSetValue ("-tao_nls_pc_type", "none"); CHKERRQ(ierr);}
  // set krylov solver type to conjugate gradient
  ierr = PetscOptionsHasName (NULL, "-tao_nls_ksp_type", &flag); CHKERRQ(ierr);
  if (flag == PETSC_FALSE) {ierr = PetscOptionsSetValue ("-tao_nls_ksp_type", "cg"); CHKERRQ(ierr);}
  // disable preconditioner
  ierr = PetscOptionsHasName (NULL, "-tao_ntr_pc_type", &flag); CHKERRQ(ierr);
  if (flag == PETSC_FALSE) {ierr = PetscOptionsSetValue ("-tao_ntr_pc_type", "none"); CHKERRQ(ierr);}
  // set mat lmvm number fo vectors for quasi-Newton update
  ierr = PetscOptionsHasName (NULL, "-tao_lmm_vectors", &flag); CHKERRQ(ierr);
  if (flag == PETSC_FALSE) {
    ierr = PetscOptionsSetValue ("-tao_lmm_vectors", ctx_->params_->opt_->lbfgs_vectors_); CHKERRQ(ierr);
    ss << " .. using " << ctx_->params_->opt_->lbfgs_vectors_ << " vectors for inverse Hessian update of quasi-Newton method."; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
  }
  // set mat lmvm type of inverse Hessian initialization
  ierr = PetscOptionsHasName (NULL, "-tao_lmm_scale_type", &flag); CHKERRQ(ierr);
  if (flag == PETSC_FALSE) {
    ierr = PetscOptionsSetValue ("-tao_lmm_scale_type", "broyden"); CHKERRQ(ierr);
    ierr = PetscOptionsSetValue ("-tao_lmm_scalar_history", 5); CHKERRQ(ierr);
    ierr = PetscOptionsSetValue ("-tao_lmm_rescale_type", "scalar"); CHKERRQ(ierr);
    ierr = PetscOptionsSetValue ("-tao_lmm_rescale_history", 5); CHKERRQ(ierr);
    ss << " .. setting inverse Hessian initial guess type to: " << ctx_->params_->opt_->lbfgs_scale_type_; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
  }
  // set max number of line-search function evaluations per Newton step
  ierr = PetscOptionsHasName ("-tumor_tao_ls_max_funcs", &flag); CHKERRQ(ierr);
  if (flag == PETSC_FALSE) {
    ierr = PetscOptionsSetValue ("-tumor_tao_ls_max_funcs", ctx_->params_->opt_->ls_max_func_evals); CHKERRQ(ierr);
    ierr = PetscOptionsSetValue ("-tao_ls_max_funcs", ctx_->params_->opt_->ls_max_func_evals); CHKERRQ(ierr);
    ss << " .. setting max number of line-search function evaluations to: " << ctx_->params_->opt_->ls_max_func_evals; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
  }
  //set tolerances
  ierr = TaoSetTolerances (tao_, 1E-12, 1E-12, ctx_->params_->opt_->gatol_, ctx_->params_->opt_->grtol_, ctx_->params_->opt_->opttolgrad_); CHKERRQ(ierr);
  #endif

  // parse options user has set
  ierr = TaoSetFromOptions (tao_); CHKERRQ(ierr);
  ierr = TaoSetInitialVector (tao_, xrec_); CHKERRQ(ierr); // TODO(K) I've changed this from tumor->p_ to xrec_
  ierr = setVariableBounds(tao_, ctx_); CHKERRQ(ierr);
  ierr = TaoSetMaximumIterations (tao_, ctx_->params_->opt_->newton_maxit_); CHKERRQ(ierr);

  ierr = TaoSetObjectiveRoutine (tao_, evaluateObjectiveFunction, (void*) ctx_); CHKERRQ(ierr);
  ierr = TaoSetGradientRoutine (tao_, evaluateGradient, (void*) ctx_); CHKERRQ(ierr);
  ierr = TaoSetObjectiveAndGradientRoutine (tao_, evaluateObjectiveFunctionAndGradient, (void*) ctx_); CHKERRQ(ierr);
  ierr = TaoSetMonitor (tao_, optimizationMonitor, (void *) ctx_, NULL); CHKERRQ(ierr);
  ierr = TaoSetConvergenceTest (tao_, checkConvergenceGrad, ctx_); CHKERRQ(ierr);
  // ierr = TaoSetConvergenceTest(tao, checkConvergenceGradObj, ctx_); CHKERRQ(ierr);

  ierr = tuMSGstd(" tolerances (stopping conditions):"); CHKERRQ(ierr);
  ss << "   gatol: "<< ctx_->params_->opt_->gatol_; tuMSGstd(ss.str()); ss.str(""); ss.clear();
  ss << "   grtol: "<< ctx_->params_->opt_->grtol_; tuMSGstd(ss.str()); ss.str(""); ss.clear();
  ss << "   gttol: "<< ctx_->params_->opt_->opttolgrad_; tuMSGstd(ss.str()); ss.str(""); ss.clear();

  // ksp solver settings for Guass-Newton
  if (ctx_->params_->opt_->newton_solver_ == GAUSSNEWTON) {
    KSP ksp = PETSC_NULL; PC pc = PETSC_NULL;
    ierr = TaoGetKSP(tao_, &ksp); CHKERRQ(ierr);
    if (ksp != PETSC_NULL) {
        ierr = KSPSetOptionsPrefix(ksp, "hessian_"); CHKERRQ(ierr);
        ierr = KSPSetTolerances(ksp, 1E-6, PETSC_DEFAULT, PETSC_DEFAULT, ctx_->params_->opt_->krylov_maxit_); CHKERRQ(ierr);
        // use Eisenstat/Walker convergence crit.
        KSPSetPreSolve (ksp, preKrylovSolve, ctx_); CHKERRQ(ierr);
        ierr = KSPMonitorSet(ksp, hessianKSPMonitor,ctx_, 0); CHKERRQ(ierr);
        // ierr = KSPSetComputeSingularValues(ksp, PETSC_TRUE); CHKERRQ (ierr);  // To compute the condition number
        ierr = KSPSetFromOptions(ksp); CHKERRQ (ierr);
        // set preconditioner
        ierr = KSPGetPC(ksp, &pc); CHKERRQ(ierr);
        ierr = PCSetType (pc, PCSHELL); CHKERRQ(ierr);
        ierr = PCShellSetApply(pc, preconditionerMatVec); CHKERRQ(ierr);
        ierr = PCShellSetContext(pc, ctx_); CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn (ierr);
}
