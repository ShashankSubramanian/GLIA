#include <iostream>
#include <limits>

#include <petsc/private/vecimpl.h>
#include "petsctao.h"
#include "petsc/private/taoimpl.h"
#include "petsc/private/taolinesearchimpl.h"

#include "Parameters.h"
#include "DerivativeOperators.h"
#include "PdeOperators.h"
#include "Optimizer.h"
#include "SparseTILOptimizer.h"
#include "TaoInterface.h"


// ### ________________________________________________________________________________________ ___
// ### //////////////////////////////////////////////////////////////////////////////////////// ###
// ### ======== //////////////////////  TAO INTERFACE METHODS  /////////////////////// ======== ###
// ### //////////////////////////////////////////////////////////////////////////////////////// ###


// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode evaluateObjectiveFunction(Tao tao, Vec x, PetscReal *J, void *ptr) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  Event e ("tao-eval-obj-tumor");
  std::array<double, 7> t = {0};
  double self_exec_time = -MPI_Wtime ();
  CtxInv *itctx = reinterpret_cast<CtxInv*>(ptr);
  itctx->params_->optf_->nb_objevals_++;
  ierr = itctx->derivative_operators_->evaluateObjective (J, x, itctx->data);
  self_exec_time += MPI_Wtime ();
  accumulateTimers (itctx->params_->tu_->timers_, t, self_exec_time);
  e.addTimings (t);
  e.stop ();
  PetscFunctionReturn (ierr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode evaluateGradient(Tao tao, Vec x, Vec dJ, void *ptr) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  Event e ("tao-eval-grad-tumor");
  std::array<double, 7> t = {0};
  double self_exec_time = -MPI_Wtime ();
  CtxInv *itctx = reinterpret_cast<CtxInv*>(ptr);
  itctx->params_->optf_->nb_gradevals_++;
  ierr = itctx->derivative_operators_->evaluateGradient (dJ, x, itctx->data);

  // use petsc default fd gradient
  // itctx->derivative_operators_->disable_verbose_ = true;
  // ierr = TaoDefaultComputeGradient(tao, x, dJ, ptr); CHKERRQ(ierr);
  // itctx->derivative_operators_->disable_verbose_ = false;
  if (itctx->params_->tu_->verbosity_ > 1) {
    std::stringstream s;
    ScalarType gnorm;
    ierr = VecNorm (dJ, NORM_2, &gnorm); CHKERRQ(ierr);
    s << " norm of gradient ||g||_2 = " << std::scientific << gnorm; ierr = tuMSGstd(s.str()); CHKERRQ(ierr); s.str(""); s.clear();
  }
  self_exec_time += MPI_Wtime ();
  accumulateTimers (itctx->params_->tu_->timers_, t, self_exec_time);
  e.addTimings (t);
  e.stop ();
  PetscFunctionReturn (ierr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode evaluateObjectiveFunctionAndGradient(Tao tao, Vec x, PetscReal *J, Vec dJ, void *ptr){
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  Event e ("tao-eval-obj/grad-tumor");
  std::array<double, 7> t = {0};
  double self_exec_time = -MPI_Wtime ();
  CtxInv *itctx = reinterpret_cast<CtxInv*>(ptr);
  itctx->params_->optf_->nb_objevals_++;
  itctx->params_->optf_->nb_gradevals_++;
  ierr = itctx->derivative_operators_->evaluateObjectiveAndGradient (J, dJ, x, itctx->data);

  if (itctx->params_->tu_->verbosity_ > 1) {
    std::stringstream s;
    ScalarType gnorm;
    ierr = VecNorm (dJ, NORM_2, &gnorm); CHKERRQ(ierr);
    s << " norm of gradient ||g||_2 = " << std::scientific << gnorm; ierr = tuMSGstd(s.str()); CHKERRQ(ierr); s.str(""); s.clear();
  }
  self_exec_time += MPI_Wtime ();
  accumulateTimers (itctx->params_->tu_->timers_, t, self_exec_time);
  e.addTimings (t); e.stop ();
  PetscFunctionReturn (ierr);
}


/* #### ------------------------------------------------------------------- #### */
// #### hessian

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode hessianMatVec(Mat A, Vec x, Vec y) {    //y = Ax
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  Event e ("tao-hess-matvec-tumor");
  std::array<double, 7> t = {0}; double self_exec_time = -MPI_Wtime ();
  void *ptr;
  ierr = MatShellGetContext(A, &ptr); CHKERRQ (ierr);
  CtxInv *itctx = reinterpret_cast<CtxInv*>(ptr);
  itctx->params_->optf_->nb_matvecs_++;
  ierr = itctx->derivative_operators_->evaluateHessian(y, x); CHKERRQ(ierr);
  if (itctx->params_->tu_->verbosity_ > 1) {
      PetscPrintf(MPI_COMM_WORLD, " applying hessian done!\n");
      ScalarType xnorm;
      ierr = VecNorm (x, NORM_2, &xnorm); CHKERRQ(ierr);
      PetscPrintf (MPI_COMM_WORLD, " norm of search direction ||x||_2 = %e\n", xnorm);
  }
  self_exec_time += MPI_Wtime ();
  accumulateTimers (itctx->params_->tu_->timers_, t, self_exec_time);
  e.addTimings (t); e.stop ();
  PetscFunctionReturn (ierr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode matfreeHessian(Tao tao, Vec x, Mat H, Mat precH, void *ptr) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  // NO-OP
  PetscFunctionReturn (ierr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode preconditionerMatVec(PC pinv, Vec x, Vec pinvx) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  void *ptr;
  ierr = PCShellGetContext(pinv, &ptr); CHKERRQ(ierr);
  CtxInv *itctx = reinterpret_cast<CtxInv*> (ptr);
  // === PRECONDITIONER CURRENTLY DISABLED ===
  //ierr = applyPreconditioner(ptr, x, pinvx); CHKERRQ(ierr);
  //ierr = itctx->derivative_operators_->evaluateHessian(pinvx, x);
  PetscFunctionReturn (ierr);
}

/* ------------------------------------------------------------------- */
/*
 Preprocess right hand side and initial condition before entering the
 krylov subspace method; in the context of numerical optimization this
 means we preprocess the gradient and the incremental control variable

 input parameters:
  . KSP ksp       KSP solver object
  . Vec b         right hand side
  . Vec x         solution vector
  . void *ptr     optional user defined context
 */
// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode preKrylovSolve(KSP ksp, Vec b, Vec x, void *ptr) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  PetscReal gnorm = 0., g0norm = 1., reltol, abstol = 0., divtol = 0.;
  PetscReal uppergradbound, lowergradbound;
  PetscInt maxit;
  int nprocs, procid;
  MPI_Comm_rank (PETSC_COMM_WORLD, &procid);
  MPI_Comm_size (PETSC_COMM_WORLD, &nprocs);

  CtxInv *itctx = reinterpret_cast<CtxInv*> (ptr);
  ierr = VecNorm (b, NORM_2, &gnorm); CHKERRQ(ierr);
  if(itctx->update_reference_gradient_hessian_ksp) { // set initial gradient norm
    itctx->ksp_gradnorm0 = gnorm;                  // for KSP Hessian solver
    itctx->update_reference_gradient_hessian_ksp = false;
  }
  g0norm = itctx->ksp_gradnorm0; // get reference gradient
  gnorm /= g0norm;
  ierr = KSPGetTolerances (ksp, &reltol, &abstol, &divtol, &maxit); CHKERRQ(ierr);
  uppergradbound = 0.5; // assuming quadratic convergence
  lowergradbound = 1E-10;
  // user forcing sequence to estimate adequate tolerance for solution of KKT system (Eisenstat-Walker)
  if (itctx->params_->opt_->fseqtype_ == QDFS) {
    // assuming quadratic convergence (we do not solver more accurately than 12 digits)
    reltol = PetscMax(lowergradbound, PetscMin(uppergradbound, gnorm));
  }
  else {
    // assuming superlinear convergence
    reltol = PetscMax (lowergradbound, PetscMin (uppergradbound, std::sqrt(gnorm)));
  }
  // overwrite tolerances with estimate
  ierr = KSPSetTolerances (ksp, reltol, abstol, divtol, maxit); CHKERRQ(ierr);
  //if (procid == 0){
  //    std::cout << " ksp rel-tol (Eisenstat/Walker): " << reltol << ", grad0norm: " << g0norm<<", gnorm/grad0norm: " << gnorm << std::endl;
  //}
  PetscFunctionReturn (ierr);
}

/* #### ------------------------------------------------------------------- #### */
// #### monitors

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode hessianKSPMonitor (KSP ksp, PetscInt its, PetscReal rnorm, void *ptr) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  std::stringstream s;
  Vec x; int maxit; PetscScalar divtol, abstol, reltol;
  ierr = KSPBuildSolution (ksp,NULL,&x);
  ierr = KSPGetTolerances (ksp, &reltol, &abstol, &divtol, &maxit); CHKERRQ(ierr);
  CtxInv *itctx = reinterpret_cast<CtxInv*>(ptr); // get user context
  itctx->params_->optf_->nb_krylov_it_++;         // accumulate number of krylov iterations
  if (its == 0) {
   s << std::setw(3)  << " PCG:" << " computing solution of hessian system (tol="
     << std::scientific << std::setprecision(5) << reltol << ")";
   ierr = tuMSGstd (s.str()); CHKERRQ(ierr); s.str(""); s.clear();
  }
  s << std::setw(3)  << " PCG:" << std::setw(15) << " " << std::setfill('0') << std::setw(3)<< its
  << "   ||r||_2 = " << std::scientific << std::setprecision(5) << rnorm;
  ierr = tuMSGstd (s.str()); CHKERRQ(ierr); s.str(""); s.clear();

  // compute extreme singular values
  // int ksp_itr;
  // ierr = KSPGetIterationNumber (ksp, &ksp_itr); CHKERRQ (ierr);
  // ScalarType e_max, e_min;
  // if (ksp_itr % 10 == 0 || ksp_itr == maxit) {
  //   ierr = KSPComputeExtremeSingularValues (ksp, &e_max, &e_min); CHKERRQ (ierr);
  //   s << "Condition number of hessian is: " << e_max / e_min << " | largest singular values is: " << e_max << ", smallest singular values is: " << e_min << std::endl;
  //   ierr = tuMSGstd (s.str()); CHKERRQ(ierr); s.str (""); s.clear ();
  // }
  PetscFunctionReturn (ierr);
}

 // ### ______________________________________________________________________ ___
 // ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode optimizationMonitor (Tao tao, void *ptr) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  int procid, nprocs;
  MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank (MPI_COMM_WORLD, &procid);
  PetscInt its;
  ScalarType J = 0, gnorm = 0, cnorm = 0 , step = 0, D = 0, J0 = 0, D0 = 0, gnorm0 = 0;
  Vec x = nullptr;
  char msg[256];
  std::string statusmsg;
  std::stringstream s;
  TaoConvergedReason flag;
  CtxInv *itctx = reinterpret_cast<CtxInv*> (ptr);

  itctx->params_->optf_->nb_newton_it_++;

  // get current iteration, objective value, norm of gradient, norm of constraint,
  // step length / trust region radius of iteratore and termination reason
  Vec tao_x, tao_grad;
  ierr = TaoGetSolutionStatus (tao, &its, &J, &gnorm, &cnorm, &step, &flag); CHKERRQ(ierr);
  ierr = TaoGetSolutionVector(tao, &tao_x); CHKERRQ(ierr);
  // get gradient vector norm for bqnls since gnorm is a different residual in this algorithm
  ierr = TaoGetGradientVector(tao, &tao_grad); CHKERRQ(ierr);
  ierr = VecNorm (tao_grad, NORM_2, &gnorm); CHKERRQ (ierr);

  if (itctx->params_->tu_->verbosity_ >= 2) {
    ScalarType *grad_ptr, *sol_ptr;
    ierr = VecGetArray(tao_x, &sol_ptr); CHKERRQ(ierr);
    ierr = VecGetArray(tao_grad, &grad_ptr); CHKERRQ(ierr);
    for (int i = 0; i < itctx->params_->tu_->np_; i++){
      if(procid == 0){
        itctx->params_->tu_->outfile_sol_  << sol_ptr[i]  << ", ";
        itctx->params_->tu_->outfile_grad_ << grad_ptr[i] << ", ";
      }
    }
    if(procid == 0){
      itctx->params_->tu_->outfile_sol_  << sol_ptr[itctx->params_->tu_->np_]  << ";" <<std::endl;
      itctx->params_->tu_->outfile_grad_ << grad_ptr[itctx->params_->tu_->np_] << ";" <<std::endl;
    }
    ierr = VecRestoreArray(tao_x, &sol_ptr); CHKERRQ(ierr);
    ierr = VecRestoreArray(tao_grad, &grad_ptr); CHKERRQ(ierr);
  }

  // === update reference gradient
  // (for older petsc the orrder of monitor and check convergence is switched)
#if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 9)
  if (itctx->update_reference_gradient) {
    Vec dJ, p0;
    ScalarType norm_gref = 0.;
    ierr = VecDuplicate (itctx->tumor_->p_, &dJ); CHKERRQ(ierr);
    ierr = VecDuplicate (itctx->tumor_->p_, &p0); CHKERRQ(ierr);
    ierr = VecSet (dJ, 0.); CHKERRQ(ierr);
    ierr = VecSet (p0, 0.); CHKERRQ(ierr);

    if (itctx->params_->opt_->flag_reaction_inv_) {
      norm_gref = gnorm;
    } else {
      ierr = evaluateGradient(tao, p0, dJ, (void*) itctx);
      ierr = VecNorm (dJ, NORM_2, &norm_gref); CHKERRQ(ierr);
    }
    itctx->params_->optf_->gradnorm0_ = norm_gref;
    itctx->params_->optf_->j0_ = J;
    //ctx->gradnorm0 = gnorm;
    itctx->update_reference_gradient = false;
    s <<" .. updating reference gradient; new norm(g0) = " << itctx->params_->optf_->gradnorm0_
      << " and reference objective: "  << itctx->params_->optf_->j0_;
    ierr = tuMSGstd(s.str()); CHKERRQ(ierr);s.str ("");s.clear ();
    if (dJ  != nullptr) {VecDestroy (&dJ);  CHKERRQ(ierr);  dJ  = nullptr;}
    if (p0  != nullptr) {VecDestroy (&p0);  CHKERRQ(ierr);  p0  = nullptr;}
  }
#endif

  // warn if tumor ic is clipped
  ierr = printVecBounds(itctx->tumor_->c_0_, "c(0)"); CHKERRQ(ierr);
  ierr = printVecBounds(itctx->tumor_->c_t_, "c(1)"); CHKERRQ(ierr);
  ierr = tuMSGstd(s.str()); CHKERRQ(ierr);s.str ("");s.clear ();

  if (its == 0) {
    s << std::setw(4)  << " iter"              << "   " << std::setw(18) << "objective (abs)" << "   "
      << std::setw(18) << "||gradient||_2,rel" << "   " << std::setw(18) << "||gradient||_2"  << "   "
      << std::setw(18) << "step" << "   ";
    ierr = tuMSGstd ("---------------------------------------------------------------------------------------------------------------"); CHKERRQ(ierr);
    ierr = tuMSGwarn (s.str()); CHKERRQ(ierr); s.str (""); s.clear ();
    ierr = tuMSGstd ("---------------------------------------------------------------------------------------------------------------"); CHKERRQ(ierr);
  }
  s << " "   << std::scientific << std::setprecision(5)  << std::setfill('0') << std::setw(4) << its << std::setfill(' ')
    << "   " << std::scientific << std::setprecision(12) << std::setw(18) << J
    << "   " << std::scientific << std::setprecision(12) << std::setw(18) << gnorm/itctx->params_->optf_->gradnorm0_
    << "   " << std::scientific << std::setprecision(12) << std::setw(18) << gnorm
    << "   " << std::scientific << std::setprecision(12) << std::setw(18) << step;
  ierr = tuMSGwarn (s.str()); CHKERRQ(ierr); s.str ("");s.clear ();

  // print the model coefficients
  ScalarType r, k, g;
  ScalarType *x_ptr;
  ierr = VecGetArray(tao_x, &x_ptr); CHKERRQ(ierr);
  if (itctx->params_->tu_->model_ == 4) {
    g = x_ptr[0];
    r = x_ptr[1];
    k = x_ptr[1 + itctx->params_->tu_->nr_];
    s << "  Scalar parameters: (rho, kappa, gamma) = (" << r << ", " <<  k << ", " << g << ")";
  } else {
    r = (itctx->params_->opt_->flag_reaction_inv_ == true) ? x_ptr[itctx->params_->get_nk()] : itctx->params_->tu_->rho_;
    k = (itctx->params_->opt_->diffusivity_inversion_ == true) ? x_ptr[itctx->params_->tu_->np_] : itctx->params_->tu_->k_;
    g = 0;
    s << "  Scalar parameters: (rho, kappa) = (" << r << ", " <<  k << ")";
  }
  ierr = tuMSGwarn (s.str()); CHKERRQ(ierr); s.str ("");s.clear ();
  ierr = VecRestoreArray(tao_x, &x_ptr); CHKERRQ(ierr);

  PetscFunctionReturn (ierr);
}

/* #### ------------------------------------------------------------------- #### */
// #### convergence

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode checkConvergenceGrad (Tao tao, void *ptr) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;

  std::stringstream ss;
  PetscInt its, nl, ng;
  PetscInt iter, maxiter;
  ScalarType J, gnorm, step, gatol, grtol, gttol, g0norm;
  Vec x = nullptr, g = nullptr, tao_grad = nullptr;
  bool stop[3];
  CtxInv *ctx = reinterpret_cast<CtxInv*>(ptr); // user context
  PetscInt verbosity = ctx->params_->tu_->verbosity_;
  ScalarType minstep = ctx->params_->opt_->ls_minstep_;
  PetscInt miniter = ctx->params_->opt_->newton_minit_;
  // get tolerances
  #if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 7)
    ierr = TaoGetTolerances (tao, &gatol, &grtol, &gttol); CHKERRQ(ierr);
  #else
    ierr = TaoGetTolerances( tao, NULL, NULL, &gatol, &grtol, &gttol); CHKERRQ(ierr);
  #endif

  // get line-search status
  TaoLineSearch ls = nullptr;
  TaoLineSearchConvergedReason ls_flag;
  nl = ctx->params_->grid_->nl_;
  ng = ctx->params_->grid_->ng_;
  ierr = TaoGetSolutionVector(tao, &x); CHKERRQ(ierr);
  ierr = TaoGetLineSearch(tao, &ls); CHKERRQ (ierr);
  ierr = VecDuplicate(ctx->tumor_->p_, &g); CHKERRQ(ierr);
  ierr = TaoLineSearchGetSolution(ls, x, &J, g, &step, &ls_flag); CHKERRQ (ierr);
  // display line-search convergence reason
  ierr = dispLineSearchStatus(tao, ctx, ls_flag); CHKERRQ(ierr);
  ierr = TaoGetMaximumIterations(tao, &maxiter); CHKERRQ(ierr);
  ierr = TaoGetSolutionStatus(tao, &iter, &J, &gnorm, NULL, &step, NULL); CHKERRQ(ierr);
  // get gradient vector norm for bqnls since gnorm is a different residual in this algorithm
  ierr = TaoGetGradientVector(tao, &tao_grad); CHKERRQ(ierr);
  ierr = VecNorm(tao_grad, NORM_2, &gnorm); CHKERRQ (ierr);

  // update/set reference gradient (with p = zeros)
  #if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR < 9)
  if (ctx->update_reference_gradient) {
    Vec dJ = nullptr, p0 = nullptr;
    ScalarType norm_gref = 0.;
    ierr = VecDuplicate (ctx->tumor_->p_, &dJ); CHKERRQ(ierr);
    ierr = VecDuplicate (ctx->tumor_->p_, &p0); CHKERRQ(ierr);
    ierr = VecSet (dJ, 0.); CHKERRQ(ierr);
    ierr = VecSet (p0, 0.); CHKERRQ(ierr);
    if (ctx->params_->opt_->flag_reaction_inv_) {
      norm_gref = gnorm;
    } else {
      ierr = evaluateGradient(tao, p0, dJ, (void*) ctx);
      ierr = VecNorm (dJ, NORM_2, &norm_gref); CHKERRQ(ierr);
    }
    ctx->params_->optf_->gradnorm0_ = norm_gref;
    ctx->params_->optf_->j0_ = J;
    //ctx->gradnorm0 = gnorm;
    ctx->update_reference_gradient = false;
    s <<" .. updating reference gradient; new norm(g0) = " << itctx->params_->optf_->gradnorm0_
      << " and reference objective: "  << ctx->params_->optf_->j0_;
    ierr = tuMSGstd(s.str()); CHKERRQ(ierr);
    if (dJ != nullptr) {ierr = VecDestroy(&dJ); CHKERRQ(ierr); dJ = nullptr;}
    if (p0 != nullptr) {ierr = VecDestroy(&p0); CHKERRQ(ierr); p0 = nullptr;}
  }
  #endif
  // get initial gradient
  g0norm = ctx->params_->optf_->gradnorm0_;
  g0norm = (g0norm > 0.0) ? g0norm : 1.0;
  ctx->convergence_message.clear();

  // check for NaN value
  if (PetscIsInfOrNanReal(J)) {
    ierr = tuMSGwarn ("objective is NaN"); CHKERRQ(ierr);
    ierr = TaoSetConvergedReason (tao, TAO_DIVERGED_NAN); CHKERRQ(ierr);
    if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}
    PetscFunctionReturn (ierr);
  }
  // check for NaN value
  if (PetscIsInfOrNanReal(gnorm)) {
    ierr = tuMSGwarn("||g|| is NaN"); CHKERRQ(ierr);
    ierr = TaoSetConvergedReason(tao, TAO_DIVERGED_NAN); CHKERRQ(ierr);
    if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}
    PetscFunctionReturn(ierr);
  }
  // == check convergence
  stop[0] = false; stop[1] = false; stop[2] = false;
  ctx->params_->optf_->converged_ = false;
  ctx->cosamp_->converged_l2 = false;
  ctx->cosamp_->converged_error_l2 = false;
  if (iter >= miniter) {
    if (verbosity > 1) {
      ss << "  step size in linesearch: " << std::scientific << step;
      ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(std::string()); ss.clear();
    }
    if (step < minstep) {
      ss << "step  = " << std::scientific << step << " < " << minstep << " = " << "bound";
      ierr = tuMSGwarn(ss.str()); CHKERRQ(ierr); ss.str(std::string()); ss.clear();
      ierr = TaoSetConvergedReason(tao, TAO_CONVERGED_STEPTOL); CHKERRQ(ierr);
      if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}
      ctx->cosamp_->converged_error_l2 = true;
      PetscFunctionReturn(ierr);
    }
    if (ls_flag != 1 && ls_flag != 0 && ls_flag != 2) {
      ss << "step  = " << std::scientific << step << ". ls failed with status " << ls_flag;
      ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(std::string()); ss.clear();
      ierr = TaoSetConvergedReason(tao, TAO_DIVERGED_LS_FAILURE);
      if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}
      ctx->cosamp_->converged_error_l2 = true;
      PetscFunctionReturn(ierr);
    }
    // ||g_k||_2 < tol*||g_0||
    if (gnorm < gttol*g0norm) {
      ierr = TaoSetConvergedReason(tao, TAO_CONVERGED_GTTOL); CHKERRQ(ierr);
      stop[0] = true;
    }
    ss << "  " << stop[0] << "    ||g|| = " << std::setw(14)
       << std::right << std::scientific << gnorm << "    <    "
       << std::left << std::setw(14) << gttol*g0norm << " = " << "tol";
    ctx->convergence_message.push_back(ss.str());
    if(verbosity >= 3) { ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); } ss.str(std::string()); ss.clear();

    // ||g_k||_2 < tol
    if (gnorm < gatol) {
      ierr = TaoSetConvergedReason(tao, TAO_CONVERGED_GATOL); CHKERRQ(ierr);
      stop[1] = true;
    }
    ss  << "  " << stop[1] << "    ||g|| = " << std::setw(14)
        << std::right << std::scientific << gnorm << "    <    "
        << std::left << std::setw(14) << gatol << " = " << "tol";
    ctx->convergence_message.push_back(ss.str());
    if(verbosity >= 3) { ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); } ss.str(std::string()); ss.clear();

    // iteration number exceeds limit
    if (iter > maxiter) {
      ierr = TaoSetConvergedReason(tao, TAO_DIVERGED_MAXITS); CHKERRQ(ierr);
      stop[2] = true;
    }
    ss  << "  " << stop[2] << "     iter = " << std::setw(14)
        << std::right << iter  << "    >    "
        << std::left << std::setw(14) << maxiter << " = " << "maxiter";
    ctx->convergence_message.push_back(ss.str());
    if(verbosity >= 3) { ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); } ss.str(std::string()); ss.clear();

    // store objective function value
    ctx->jvalold = J;
    if (stop[0] || stop[1]) {ctx->cosamp_->converged_l2 = true;} // for CoSaMpRS to split up L2 solve
    if (stop[0] || stop[1] || stop[2]) {
      ctx->params_->optf_->converged_ = true;
      if (g != nullptr) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = nullptr;}
      PetscFunctionReturn(ierr);
    }
  // iter < miniter
  } else {
    // if the gradient is less than abstol, we should terminate immediately
    if (gnorm < gatol) {
      ss << "||g|| = " << std::scientific << gatol  << " = " << "bound";
      ierr = tuMSGwarn(ss.str()); CHKERRQ(ierr); ss.str(std::string()); ss.clear();
      ierr = TaoSetConvergedReason(tao, TAO_CONVERGED_GATOL); CHKERRQ(ierr);
      if (g != nullptr) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = nullptr;}
      ctx->cosamp_->converged_l2 = true;
      PetscFunctionReturn(ierr);
    }
  }
  // if we're here, we're good to go
  ierr = TaoSetConvergedReason (tao, TAO_CONTINUE_ITERATING); CHKERRQ(ierr);
  if (g != nullptr) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = nullptr;}
  PetscFunctionReturn (ierr);
}

PetscErrorCode checkConvergenceGradObj (Tao tao, void *ptr) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;

  std::stringstream ss;
  PetscInt its, nl, ng;
  PetscInt iter, maxiter, miniter;
  PetscReal gnorm, step, gatol, grtol, gttol, g0norm, minstep;
  PetscReal jx, jxold, gtolbound, theta, normx, normdx, tolj, tolx, tolg;
  const int nstop = 7;
  bool stop[nstop];
  int verbosity;
  Vec x = nullptr, g = nullptr, tao_grad = nullptr;
  TaoLineSearch ls = nullptr;
  TaoLineSearchConvergedReason ls_flag;

  CtxInv *ctx = reinterpret_cast<CtxInv*> (ptr);     // get user context
  verbosity = ctx->params_->tu_->verbosity_;
  minstep = ctx->params_->opt_->ls_minstep_;
  miniter = ctx->params_->opt_->newton_minit_;
  // get tolerances
#if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 7)
  ierr = TaoGetTolerances (tao, &gatol, &grtol, &gttol); CHKERRQ(ierr);
#else
  ierr = TaoGetTolerances( tao, NULL, NULL, &gatol, &grtol, &gttol); CHKERRQ(ierr);
#endif

  // get line-search status
  nl = ctx->params_->grid_->nl_;
  ng = ctx->params_->grid_->ng_;
  ierr = TaoGetSolutionVector(tao, &x); CHKERRQ(ierr);
  ierr = TaoGetLineSearch(tao, &ls); CHKERRQ (ierr);
  ierr = VecDuplicate (x, &g); CHKERRQ (ierr);
  ierr = TaoLineSearchGetSolution(ls, x, &jx, g, &step, &ls_flag); CHKERRQ (ierr);
  // display line-search convergence reason
  ierr = dispLineSearchStatus(tao, ctx, ls_flag); CHKERRQ(ierr);
  ierr = TaoGetMaximumIterations(tao, &maxiter); CHKERRQ(ierr);
  ierr = TaoGetSolutionStatus(tao, &iter, &jx, &gnorm, NULL, &step, NULL); CHKERRQ(ierr);
  // get gradient vector norm for bqnls since gnorm is a different residual in this algorithm
  ierr = TaoGetGradientVector(tao, &tao_grad); CHKERRQ(ierr);
  ierr = VecNorm(tao_grad, NORM_2, &gnorm); CHKERRQ (ierr);

  // update/set reference gradient
  #if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR < 9)
  if (ctx->update_reference_gradient) {
    ctx->params_->optf_->gradnorm0_ = gnorm;
    ctx->params_->optf_->j0_ = jx;
    ctx->update_reference_gradient = false;
    ss <<" .. updating reference gradient; new norm(g0) = " << itctx->params_->optf_->gradnorm0_
      << " and reference objective: "  << ctx->params_->optf_->j0_;
    ierr = tuMSGstd(ss.str());  CHKERRQ(ierr);
  }
  #endif

  g0norm = ctx->params_->optf_->gradnorm0_;
  g0norm = (g0norm > 0.0) ? g0norm : 1.0;
  // compute tolerances for stopping conditions
  tolj = grtol;
  tolx = std::sqrt(grtol);
#if __cplusplus > 199711L
  tolg = std::cbrt(grtol);
#else
  tolg = std::pow(grtol, 1.0/3.0);
#endif
  // compute theta
  theta = 1.0 + std::abs(ctx->params_->optf_->j0_);
  ierr = VecNorm (x, NORM_2, &normx); CHKERRQ(ierr);  // comp norm x
  ierr = VecAXPY (ctx->x_old, -1.0, x); CHKERRQ(ierr);  // comp dx
  ierr = VecNorm (ctx->x_old, NORM_2, &normdx); CHKERRQ(ierr);  // comp norm dx
  ierr = VecCopy (x, ctx->x_old); CHKERRQ(ierr);  // save old x
  // get old objective function value
  jxold = ctx->jvalold;
  ctx->convergence_message.clear();

  // check for NaN value
  if (PetscIsInfOrNanReal(jx)) {
    ierr = tuMSGwarn ("objective is NaN"); CHKERRQ(ierr);
    ierr = TaoSetConvergedReason (tao, TAO_DIVERGED_NAN); CHKERRQ(ierr);
    if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}
    PetscFunctionReturn (ierr);
  }
  // check for NaN value
  if (PetscIsInfOrNanReal(gnorm)) {
    ierr = tuMSGwarn("||g|| is NaN"); CHKERRQ(ierr);
    ierr = TaoSetConvergedReason(tao, TAO_DIVERGED_NAN); CHKERRQ(ierr);
    if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}
    PetscFunctionReturn(ierr);
  }

  // only check convergence criteria after a certain number of iterations
  // initialize flags for stopping conditions
  for (int i = 0; i < nstop; i++) {
    stop[i] = false;
  }
  ctx->params_->optf_->converged_     = false;
  ctx->cosamp_->converged_l2       = false;
  ctx->cosamp_->converged_error_l2 = false;
  if (iter >= miniter && iter > 1) {
    if (step < minstep) {
      ss << "step  = " << std::scientific << step << " < " << minstep << " = " << "bound";
      ierr = tuMSGwarn(ss.str()); CHKERRQ(ierr); ss.str(std::string()); ss.clear();
      ierr = TaoSetConvergedReason(tao, TAO_CONVERGED_STEPTOL); CHKERRQ(ierr);
      if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}
      PetscFunctionReturn (ierr);
    }
    if (ls_flag != 1 && ls_flag != 0 && ls_flag != 2) {
      ss << "step  = " << std::scientific << step << ". ls failed with status " << ls_flag;
      ierr = tuMSGwarn(ss.str()); CHKERRQ(ierr); ss.str(std::string()); ss.clear();
      ierr = TaoSetConvergedReason(tao, TAO_DIVERGED_LS_FAILURE);
      if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}
      PetscFunctionReturn (ierr);
    }
    // |j(x_{k-1}) - j(x_k)| < tolj*abs(1+J)
    if (std::abs(jxold-jx) < tolj*theta) {
      stop[0] = true;
    }
    ss << "  " << stop[0] << "    |dJ|  = " << std::setw(18)
    << std::right << std::scientific << std::abs(jxold-jx) << "    <    "
    << std::left << std::setw(18) << tolj*theta << " = " << "tol*|1+J|";
    ctx->convergence_message.push_back(ss.str());
    ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
    ss.str(std::string());
    ss.clear();
    // ||dx|| < sqrt(tolj)*(1+||x||)
    if (normdx < tolx*(1+normx)) {
      stop[1] = true;
    }
    ss << "  " << stop[1] << "    |dx|  = " << std::setw(18)
    << std::right << std::scientific << normdx << "    <    "
    << std::left << std::setw(18) << tolx*(1+normx) << " = " << "sqrt(tol)*(1+||x||)";
    ctx->convergence_message.push_back(ss.str());
    ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
    ss.str(std::string());
    ss.clear();
    // ||g_k||_2 < cbrt(tolj)*abs(1+Jc)
    if (gnorm < tolg*theta) {
      stop[2] = true;
    }
    ss  << "  " << stop[2] << "    ||g|| = " << std::setw(18)
    << std::right << std::scientific << gnorm << "    <    "
    << std::left << std::setw(18) << tolg*theta << " = " << "cbrt(tol)*|1+J|";
    ctx->convergence_message.push_back(ss.str());
    ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
    ss.str(std::string());
    ss.clear();
    // ||g_k||_2 < tol
    if (gnorm < gatol || std::abs(jxold-jx) <= PETSC_MACHINE_EPSILON)  {
      stop[3] = true;
    }
    ss  << "  " << stop[3] << "    ||g|| = " << std::setw(18)
    << std::right << std::scientific << gnorm << "    <    "
    << std::left << std::setw(18) << gatol << " = " << "tol";
    ctx->convergence_message.push_back(ss.str());
    ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
    ss.str(std::string());
    ss.clear();

    if (gnorm < gttol*g0norm) {
      stop[4] = true;
    }
    ss << "  " << stop[4] << "    ||g|| = " << std::setw(18)
     << std::right << std::scientific << gnorm << "    <    "
     << std::left << std::setw(18) << gttol*g0norm << " = " << "tol";
    ctx->convergence_message.push_back(ss.str());
    ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
    ss.str(std::string());
    ss.clear();

    if (iter > maxiter) {
      stop[5] = true;
    }
    ss  << "  " << stop[5] << "    iter  = " << std::setw(18)
    << std::right << iter  << "    >    "
    << std::left << std::setw(18) << maxiter << " = " << "maxiter";
    ctx->convergence_message.push_back(ss.str());
    ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
    ss.str(std::string());
    ss.clear();

    // ||dJ||_2 < eps
    if (std::abs(jxold-jx) <= PETSC_MACHINE_EPSILON)  {
      stop[6] = true;
    }
    ss  << "  " << stop[6] << "    ||dJ|| = " << std::setw(18)
    << std::right << std::scientific << std::abs(jxold-jx) << "    <    "
    << std::left << std::setw(18) << PETSC_MACHINE_EPSILON << " = " << "tol";
    ctx->convergence_message.push_back(ss.str());
    ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
    ss.str(std::string());
    ss.clear();

    // store objective function value
    ctx->jvalold = jx;

    if (stop[0] && stop[1] && stop[2]) {
      ierr = TaoSetConvergedReason(tao, TAO_CONVERGED_USER); CHKERRQ(ierr);
      ctx->params_->optf_->converged_ = true;
      if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}
      PetscFunctionReturn (ierr);
    } else if (stop[3]) {
      ierr = TaoSetConvergedReason(tao, TAO_CONVERGED_GATOL); CHKERRQ(ierr);
      ctx->params_->optf_->converged_ = true;
      if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}
      PetscFunctionReturn (ierr);
    } else if (stop[4]) {
      ierr = TaoSetConvergedReason(tao, TAO_CONVERGED_GTTOL); CHKERRQ(ierr);
      if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}
      ctx->params_->optf_->converged_ = true;
      PetscFunctionReturn (ierr);
    } else if (stop[5]) {
      ierr = TaoSetConvergedReason(tao, TAO_DIVERGED_MAXITS); CHKERRQ(ierr);
      ctx->params_->optf_->converged_ = true;
      if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}
      PetscFunctionReturn (ierr);
    } else if (stop[6]) {
      ierr = TaoSetConvergedReason(tao, TAO_CONVERGED_USER); CHKERRQ(ierr);
      ctx->params_->optf_->converged_ = true;
      if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}
      PetscFunctionReturn (ierr);
    }
  }
  else {
    // if the gradient is zero, we should terminate immediately
    if (gnorm < gatol) {
      ss << "||g|| = " << std::scientific << " < " << gatol;
      ierr = tuMSGwarn(ss.str()); CHKERRQ(ierr);ss.str(std::string()); ss.clear();
      ierr = TaoSetConvergedReason(tao, TAO_CONVERGED_GATOL); CHKERRQ(ierr);
      if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}
      PetscFunctionReturn (ierr);
    }
    // store objective function value
    ctx->jvalold = jx;
  }
  // if we're here, we're good to go
  ierr = TaoSetConvergedReason (tao, TAO_CONTINUE_ITERATING); CHKERRQ(ierr);
  if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}
  PetscFunctionReturn (ierr);
}


// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode dispTaoConvReason (TaoConvergedReason flag, std::string &msg) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  switch (flag) {
    case TAO_CONVERGED_GATOL : {
      msg = "solver converged: ||g(x)|| <= tol";
      ierr = tuMSGwarn(msg); CHKERRQ(ierr);
      break;
    }
    case TAO_CONVERGED_GRTOL : {
      msg = "solver converged: ||g(x)||/J(x) <= tol";
      ierr = tuMSGwarn(msg); CHKERRQ(ierr);
      break;
    }
    case TAO_CONVERGED_GTTOL : {
      msg = "solver converged: ||g(x)||/||g(x0)|| <= tol";
      ierr = tuMSGwarn(msg); CHKERRQ(ierr);
      break;
    }
    case TAO_CONVERGED_STEPTOL : {
      msg = "step size too small";
      ierr = tuMSGwarn(msg); CHKERRQ(ierr);
      break;
    }
    case TAO_CONVERGED_MINF : {
      msg = "objective value to small";
      ierr = tuMSGwarn(msg);  CHKERRQ(ierr);
      break;
    }
    case TAO_CONVERGED_USER : {
      msg = "solver converged user";
      ierr = tuMSGwarn(msg); CHKERRQ(ierr);
      break;
    }
    case TAO_DIVERGED_MAXITS : {
      msg = "maximum number of iterations reached";
      ierr = tuMSGwarn(msg); CHKERRQ(ierr);
      break;
    }
    case TAO_DIVERGED_NAN : {
      msg = "numerical problems (NAN detected)";
      ierr = tuMSGwarn(msg); CHKERRQ(ierr);
      break;
    }
    case TAO_DIVERGED_MAXFCN : {
      msg = "maximal number of function evaluations reached";
      ierr = tuMSGwarn(msg); CHKERRQ(ierr);
      break;
    }
    case TAO_DIVERGED_LS_FAILURE : {
      msg = "line search failed";
      ierr = tuMSGwarn(msg); CHKERRQ(ierr);
      break;
    }
    case TAO_DIVERGED_TR_REDUCTION : {
      msg = "trust region failed";
      ierr = tuMSGwarn(msg); CHKERRQ(ierr);
      break;
    }
    case TAO_DIVERGED_USER : {
      msg = "user defined divergence criterion met";
      ierr = tuMSGwarn(msg); CHKERRQ(ierr);
      break;
    }
    case TAO_CONTINUE_ITERATING : {
      // display nothing
      break;
    }
    default : {
      msg = "convergence reason not defined";
      ierr = tuMSGwarn(msg); CHKERRQ(ierr);
      break;
    }
  }
  PetscFunctionReturn(ierr);
}


// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode dispLineSearchStatus(Tao tao, void* ptr, TaoLineSearchConvergedReason flag) {
  PetscErrorCode ierr = 0;
  std::string msg;
  PetscFunctionBegin;

  switch(flag) {
    case TAOLINESEARCH_FAILED_INFORNAN: {
      msg = "  linesearch: function evaluation gave INF or NaN";
      ierr = tuMSGwarn(msg); CHKERRQ(ierr);
      break;
    }
    case TAOLINESEARCH_FAILED_BADPARAMETER: {
      msg = "  linesearch: bad parameter detected";
      ierr = tuMSGwarn(msg); CHKERRQ(ierr);
      break;
    }
    case TAOLINESEARCH_FAILED_ASCENT: {
      msg = "  linesearch: search direction is not a descent direction";
      ierr = tuMSGwarn(msg); CHKERRQ(ierr);
      break;
    }
    case TAOLINESEARCH_HALTED_MAXFCN: {
      msg = "  linesearch: maximum number of function evaluations reached";
      ierr = tuMSGwarn(msg); CHKERRQ(ierr);
      break;
    }
    case TAOLINESEARCH_HALTED_UPPERBOUND: {
      msg = "  linesearch: step size reached upper bound";
      ierr = tuMSGwarn(msg); CHKERRQ(ierr);
      break;
    }
    case TAOLINESEARCH_HALTED_LOWERBOUND: {
      msg = "  linesearch: step size reached lower bound";
      ierr = tuMSGwarn(msg); CHKERRQ(ierr);
      break;
    }
    case TAOLINESEARCH_HALTED_RTOL: {
      msg = "  linesearch: range of uncertainty is smaller than given tolerance";
      ierr = tuMSGwarn(msg); CHKERRQ(ierr);
      break;
    }
    case TAOLINESEARCH_HALTED_OTHER: {
      msg = "  linesearch: stopped (other)";
      ierr = tuMSGwarn(msg); CHKERRQ(ierr);
      break;
    }
    case TAOLINESEARCH_CONTINUE_ITERATING: {
      // do nothing; everything's fine
      break;
    }
    case TAOLINESEARCH_SUCCESS: {
      msg = "  linesearch: successful";
      ierr = tuMSGstd(msg); CHKERRQ(ierr);
      break;
    }
    default: {
      msg = "  linesearch: status not defined";
      ierr = tuMSGwarn(msg); CHKERRQ(ierr);
      break;
    }
  }
  PetscFunctionReturn (ierr);
}


/* #### ------------------------------------------------------------------- #### */
// #### misc

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode operatorCreateVecsOptimizer(Mat A, Vec *left, Vec *right) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  CtxInv *ctx;
  ierr = MatShellGetContext(A, &ctx); CHKERRQ (ierr);
  if (right) {
    ierr = VecDuplicate(ctx->x_old, right); CHKERRQ(ierr); // TODO: correct for TIL, RD optimizer, check others
  }
  if (left) {
    ierr = VecDuplicate(ctx->x_old, left); CHKERRQ(ierr);
  }
  PetscFunctionReturn (ierr);
}
