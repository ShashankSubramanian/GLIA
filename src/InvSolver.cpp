#include "InvSolver.h"
#include "petsctao.h"
#include <iostream>
#include <limits>
#include "DerivativeOperators.h"
#include "PdeOperators.h"
#include "Utils.h"
#include "TaoL1Solver.h"


InvSolver::InvSolver (std::shared_ptr <DerivativeOperators> derivative_operators, std::shared_ptr <NMisc> n_misc, std::shared_ptr <Tumor> tumor) :
initialized_(false),
tao_is_reset_(true),
data_(),
data_gradeval_(),
optsettings_(),
optfeedback_(),
itctx_() {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    tao_  = nullptr;
    H_    = nullptr;
    xrec_ = nullptr;
    if( derivative_operators != nullptr && n_misc != nullptr && tumor != nullptr) {
        initialize (derivative_operators, n_misc, tumor);
    }
}

PetscErrorCode InvSolver::initialize (std::shared_ptr<DerivativeOperators> derivative_operators, std::shared_ptr<NMisc> n_misc, std::shared_ptr<Tumor> tumor) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    if (initialized_)
        PetscFunctionReturn (0);
    optsettings_ = std::make_shared<OptimizerSettings> ();
    optfeedback_ = std::make_shared<OptimizerFeedback> ();
    itctx_ = std::make_shared<CtxInv> ();
    itctx_->derivative_operators_ = derivative_operators;
    itctx_->n_misc_ = n_misc;
    itctx_->tumor_ = tumor;
    itctx_->optsettings_ = this->optsettings_;
    itctx_->optfeedback_ = this->optfeedback_;

    // allocate memory for H, x_rec and TAO
    ierr = allocateTaoObjects(); CHKERRQ(ierr);

    initialized_ = true;
    PetscFunctionReturn (0);
}

PetscErrorCode InvSolver::allocateTaoObjects (bool initialize_tao) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;

  int np = itctx_->n_misc_->np_;
  int nk = (itctx_->n_misc_->diffusivity_inversion_) ?  itctx_->n_misc_->nk_ : 0;

  if (itctx_->n_misc_->regularization_norm_ == L1) {//Register new Tao solver and initialize variables for parameter continuation
    ierr = TaoRegister ("tao_L1", TaoCreate_ISTA);                              CHKERRQ (ierr);
    itctx_->lam_right = itctx_->n_misc_->lambda_;
    itctx_->lam_left = 0;
  }

  if (itctx_->n_misc_->regularization_norm_ == wL2) {
    if (itctx_->weights != nullptr) {
      ierr = VecDestroy (&itctx_->weights);                                     CHKERRQ (ierr);
    }
    ierr = VecDuplicate (itctx_->tumor_->p_, &itctx_->weights);                 CHKERRQ (ierr);
  }

  #ifdef SERIAL
    // allocate memory for xrec_
      ierr = VecDuplicate (itctx_->tumor_->p_, &xrec_);                         CHKERRQ(ierr);
    // set up routine to compute the hessian matrix vector product
    if (H_ == nullptr) {
      ierr = MatCreateShell (PETSC_COMM_SELF, np + nk, np + nk, np + nk, np + nk, (void*) itctx_.get(), &H_); CHKERRQ(ierr);
    }
    // create TAO solver object
    if ( tao_ == nullptr && initialize_tao) {
      ierr = TaoCreate (PETSC_COMM_SELF, &tao_); tao_is_reset_ = true;  // triggers setTaoOptions
    }
  #else
    TU_assert (!itctx_->n_misc_->diffusivity_inversion_, "Inversion for diffusifity is only implemented for SERIAL p");
    // allocate memory for xrec_
    ierr = VecDuplicate (itctx_->tumor_->p_, &xrec_);                           CHKERRQ(ierr);
    // set up routine to compute the hessian matrix vector product
    if (H_ == nullptr) {
      ierr = MatCreateShell (MPI_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, np + nk, np + nk, (void*) itctx_.get(), &H_); CHKERRQ(ierr);
    }
    // create TAO solver object
    if( tao_ == nullptr && initialize_tao) {
      ierr = TaoCreate (MPI_COMM_WORLD, &tao_); tao_is_reset_ = true;  // triggers setTaoOptions
    }
  #endif
  ierr = VecSet (xrec_, 0.0);                                                   CHKERRQ(ierr);

  // if tao's lmvm (l-bfgs) method is used and the initial hessian approximation is explicitly set
  if ((itctx_->optsettings_->newtonsolver == QUASINEWTON) && itctx_->optsettings_->lmvm_set_hessian) {
    ierr = MatShellSetOperation (H_, MATOP_MULT, (void (*)(void))constApxHessianMatVec); CHKERRQ(ierr);
    ierr = MatSetOption (H_, MAT_SYMMETRIC, PETSC_TRUE);                                 CHKERRQ(ierr);
    // if tao's nls (gauss-newton) method is used, define hessian matvec
  }
  else {
    ierr = MatShellSetOperation (H_, MATOP_MULT, (void (*)(void))hessianMatVec);         CHKERRQ(ierr);
    ierr = MatSetOption (H_, MAT_SYMMETRIC, PETSC_TRUE);                                 CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

PetscErrorCode InvSolver::resetTao (std::shared_ptr<NMisc> n_misc) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  if (tao_  != nullptr) {ierr = TaoDestroy (&tao_);  CHKERRQ(ierr); tao_  = nullptr;}
  if (H_    != nullptr) {ierr = MatDestroy (&H_);    CHKERRQ(ierr); H_    = nullptr;}
  if (xrec_ != nullptr) {ierr = VecDestroy (&xrec_); CHKERRQ(ierr); xrec_ = nullptr;}

  // allocate memory for H, x_rec and TAO
  ierr = allocateTaoObjects (); CHKERRQ(ierr);
  PetscFunctionReturn (0);
}

PetscErrorCode InvSolver::setParams (std::shared_ptr<DerivativeOperators> derivative_operators, std::shared_ptr<NMisc> n_misc, std::shared_ptr<Tumor> tumor, bool npchanged) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    itctx_->derivative_operators_ = derivative_operators;
    itctx_->n_misc_ = n_misc;
    itctx_->tumor_ = tumor;
    // re-allocate memory
    if (npchanged){                              // re-allocate memory for xrec_
      // allocate memory for H, x_rec and TAO
      itctx_->x_old = nullptr; // Will be set accordingly in the solver
      if (H_    != nullptr) {ierr = MatDestroy (&H_);    CHKERRQ(ierr); H_    = nullptr;}
      if (xrec_ != nullptr) {ierr = VecDestroy (&xrec_); CHKERRQ(ierr); xrec_ = nullptr;}
      ierr = allocateTaoObjects (false); CHKERRQ(ierr);
    }
    tao_is_reset_ = true;                        // triggers setTaoOptions
    PetscFunctionReturn (0);
}

PetscErrorCode checkConvergenceGradForParameters (Tao tao, void *ptr) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    PetscInt its, nl, ng;
    PetscInt iter, maxiter, miniter;
    PetscReal J, gnorm, step, gatol, grtol, gttol, g0norm, minstep;
    bool stop[3];
    int verbosity;
    std::stringstream ss, sc;
    Vec x = nullptr, g = nullptr;
    ierr = TaoGetSolutionVector(tao, &x);                                     CHKERRQ(ierr);
    TaoLineSearch ls = nullptr;
    TaoLineSearchConvergedReason ls_flag;

    CtxInv *ctx = reinterpret_cast<CtxInv*> (ptr);     // get user context
    verbosity = ctx->optsettings_->verbosity;
    minstep = ctx->optsettings_->ls_minstep;
    miniter = ctx->optsettings_->newton_minit;
    // get tolerances
    #if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 7)
        ierr = TaoGetTolerances (tao, &gatol, &grtol, &gttol);                  CHKERRQ(ierr);
    #else
        ierr = TaoGetTolerances( tao, NULL, NULL, &gatol, &grtol, &gttol);      CHKERRQ(ierr);
    #endif

    // get line-search status
    nl = ctx->n_misc_->n_local_;
    ng = ctx->n_misc_->n_global_;
    ierr = TaoGetLineSearch(tao, &ls);                                          CHKERRQ (ierr);
    ierr = VecDuplicate (x, &g);                                  CHKERRQ(ierr);
    ierr = TaoLineSearchGetSolution(ls, x, &J, g, &step, &ls_flag);             CHKERRQ (ierr);
    // display line-search convergence reason
    ierr = dispLineSearchStatus(tao, ctx, ls_flag);                             CHKERRQ(ierr);
    ierr = TaoGetMaximumIterations(tao, &maxiter);                              CHKERRQ(ierr);
    ierr = TaoGetSolutionStatus(tao, &iter, &J, &gnorm, NULL, &step, NULL);     CHKERRQ(ierr);

    ScalarType norm_gref = 0.;
    // update/set reference gradient 
    #if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR < 9)
    if (ctx->update_reference_gradient) {
      norm_gref = gnorm;
      ctx->optfeedback_->gradnorm0 = norm_gref;
      ctx->update_reference_gradient = false;
      std::stringstream s; s <<"updated reference gradient for relative convergence criterion, Quasi-Newton solver: " << ctx->optfeedback_->gradnorm0;
      ierr = tuMSGstd(s.str());                                                 CHKERRQ(ierr);
    }
    #endif
    // get initial gradient
    g0norm = ctx->optfeedback_->gradnorm0;
    g0norm = (g0norm > 0.0) ? g0norm : 1.0;
    ctx->convergence_message.clear();
    // check for NaN value
    if (PetscIsInfOrNanReal(J)) {
      ierr = tuMSGwarn ("objective is NaN");                                      CHKERRQ(ierr);
      ierr = TaoSetConvergedReason (tao, TAO_DIVERGED_NAN);                       CHKERRQ(ierr);
      if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}
      PetscFunctionReturn (ierr);
    }
    // check for NaN value
    if (PetscIsInfOrNanReal(gnorm)) {
      ierr = tuMSGwarn("||g|| is NaN");                                           CHKERRQ(ierr);
      ierr = TaoSetConvergedReason(tao, TAO_DIVERGED_NAN);                        CHKERRQ(ierr);
      if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}
      PetscFunctionReturn(ierr);
    }
    //if(verbosity >= 1) {
    //  ierr = PetscPrintf (MPI_COMM_WORLD, "||g(x)|| / ||g(x0)|| = %6E, ||g(x0)|| = %6E \n", gnorm/g0norm, g0norm);
    //}
    // only check convergence criteria after a certain number of iterations
    stop[0] = false; stop[1] = false; stop[2] = false;
    ctx->optfeedback_->converged = false;
    if (iter >= miniter) {
      if (verbosity > 1) {
          ss << "step size in linesearch: " << std::scientific << step;
          ierr = tuMSGstd(ss.str());                                            CHKERRQ(ierr);
          ss.str(std::string());
                ss.clear();
      }
      if (step < minstep) {
          ss << "step  = " << std::scientific << step << " < " << minstep << " = " << "bound";
          ierr = tuMSGwarn(ss.str()); CHKERRQ(ierr);
          ss.str(std::string());
                ss.clear();
          ierr = TaoSetConvergedReason(tao, TAO_CONVERGED_STEPTOL);             CHKERRQ(ierr);
          if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}
          PetscFunctionReturn(ierr);
      }
      if (ls_flag != 1 && ls_flag != 0 && ls_flag != 2) {
        ss << "step  = " << std::scientific << step << " < " << minstep << " = " << "bound";
        ierr = tuMSGwarn(ss.str()); CHKERRQ(ierr);
        ss.str(std::string());
              ss.clear();
        ierr = TaoSetConvergedReason(tao, TAO_DIVERGED_LS_FAILURE);
        if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}
        PetscFunctionReturn(ierr);
      }
      // ||g_k||_2 < tol*||g_0||
      if (gnorm < gttol*g0norm) {
          ierr = TaoSetConvergedReason(tao, TAO_CONVERGED_GTTOL);               CHKERRQ(ierr);
          stop[0] = true;
      }
      ss << "  " << stop[0] << "    ||g|| = " << std::setw(14)
         << std::right << std::scientific << gnorm << "    <    "
         << std::left << std::setw(14) << gttol*g0norm << " = " << "tol";
      ctx->convergence_message.push_back(ss.str());
      if(verbosity >= 3) {
        ierr = tuMSGstd(ss.str());                                              CHKERRQ(ierr);
      }
      ss.str(std::string());
        ss.clear();
      // ||g_k||_2 < tol
      if (gnorm < gatol) {
      ierr = TaoSetConvergedReason(tao, TAO_CONVERGED_GATOL);                   CHKERRQ(ierr);
      stop[1] = true;
      }
      ss  << "  " << stop[1] << "    ||g|| = " << std::setw(14)
          << std::right << std::scientific << gnorm << "    <    "
          << std::left << std::setw(14) << gatol << " = " << "tol";
      ctx->convergence_message.push_back(ss.str());
      if(verbosity >= 3) {
        ierr = tuMSGstd(ss.str());                                              CHKERRQ(ierr);
      }
      ss.str(std::string());
        ss.clear();
      // iteration number exceeds limit
      if (iter > maxiter) {
      ierr = TaoSetConvergedReason(tao, TAO_DIVERGED_MAXITS);                    CHKERRQ(ierr);
      stop[2] = true;
      }
      ss  << "  " << stop[2] << "     iter = " << std::setw(14)
          << std::right << iter  << "    >    "
          << std::left << std::setw(14) << maxiter << " = " << "maxiter";
      ctx->convergence_message.push_back(ss.str());
      if(verbosity >= 3) {
        ierr = tuMSGstd(ss.str());                                              CHKERRQ(ierr);
      }
      ss.str(std::string());
        ss.clear();
      // store objective function value
      ctx->jvalold = J;
      if (stop[0] || stop[1] || stop[2]) {
        ctx->optfeedback_->converged = true;
        if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}
        PetscFunctionReturn(ierr);
      }

    }
    else {
    // if the gradient is zero, we should terminate immediately
    if (gnorm < gatol) {
      ss << "||g|| = " << std::scientific << 0.0 << " < " << gatol  << " = " << "bound";
      ierr = tuMSGwarn(ss.str());                                               CHKERRQ(ierr);
      ss.str(std::string());
            ss.clear();
      ierr = TaoSetConvergedReason(tao, TAO_CONVERGED_GATOL);                   CHKERRQ(ierr);
      if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}
      PetscFunctionReturn(ierr);
    }
    }
    // if we're here, we're good to go
    ierr = TaoSetConvergedReason (tao, TAO_CONTINUE_ITERATING);                 CHKERRQ(ierr);

    if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}

    PetscFunctionReturn (0);
}


PetscErrorCode InvSolver::solveForParameters (Vec x_in) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;

  TU_assert (initialized_, "InvSolver::solve (): InvSolver needs to be initialized.")
  TU_assert (data_ != nullptr, "InvSolver:solve (): requires non-null input data for inversion.");
  TU_assert (data_gradeval_ != nullptr, "InvSolver:solve (): requires non-null input data for gradient evaluation.");
  TU_assert (xrec_ != nullptr, "InvSolver:solve (): requires non-null p_rec vector to be set");
  TU_assert (optsettings_ != nullptr, "InvSolver:solve (): requires non-null optimizer settings to be passed.");

  int procid, nprocs;
  MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank (MPI_COMM_WORLD, &procid);


  ScalarType *d_ptr;
  ierr = VecGetArray (data_, &d_ptr);                                                 CHKERRQ(ierr);

  /* === Add Noise === */
  Vec noise; ScalarType *noise_ptr;
  ierr = VecCreate (PETSC_COMM_WORLD, &noise);                                        CHKERRQ(ierr);
  ierr = VecSetSizes(noise, itctx_->n_misc_->n_local_, itctx_->n_misc_->n_global_);   CHKERRQ(ierr);
  ierr = setupVec(noise);                                                    CHKERRQ(ierr);
  ierr = VecSetRandom(noise, NULL);                                                   CHKERRQ(ierr);
  ierr = VecGetArray (noise, &noise_ptr);                                             CHKERRQ(ierr);
  for (int i = 0; i < itctx_->n_misc_->n_local_; i++) {
      d_ptr[i] += noise_ptr[i] * itctx_->n_misc_->noise_scale_;
      noise_ptr[i] = d_ptr[i];                                                        //just to measure d norm
  }
  ierr = VecRestoreArray (noise, &noise_ptr);                                         CHKERRQ(ierr);
  ierr = VecRestoreArray (data_, &d_ptr);                                             CHKERRQ(ierr);
  ScalarType max, min;                                                // compute d-norm
  ScalarType d_norm = 0., d_errorl2norm = 0., d_errorInfnorm = 0.;
  #ifdef POSITIVITY
    ierr = enforcePositivity (data_, itctx_->n_misc_);
    ierr = enforcePositivity (noise, itctx_->n_misc_);
  #endif
  ierr = VecNorm (noise, NORM_2, &d_norm);                                            CHKERRQ(ierr);
  ierr = VecMax (noise, NULL, &max);                                                  CHKERRQ(ierr);
  ierr = VecMin (noise, NULL, &min);                                                  CHKERRQ(ierr);
  ierr = VecAXPY (noise, -1.0, data_);                                                CHKERRQ(ierr);
  ierr = VecNorm (noise, NORM_2, &d_errorl2norm);                                     CHKERRQ(ierr);
  ierr = VecNorm (noise, NORM_INFINITY, &d_errorInfnorm);                             CHKERRQ(ierr);
  std::stringstream s;
  s << "data (ITP), with noise: l2norm = "<< d_norm <<" [max: "<<max<<", min: "<<min<<"]";  ierr = tuMSGstd(s.str()); CHKERRQ(ierr); s.str(""); s.clear();
  s << "IT data error due to thresholding and smoothing: l2norm = "<< d_errorl2norm <<", inf-norm = " <<d_errorInfnorm;  ierr = tuMSGstd(s.str()); CHKERRQ(ierr); s.str(""); s.clear();
  
  // Reset tao
  if (tao_  != nullptr) {ierr = TaoDestroy (&tao_);  CHKERRQ(ierr); tao_  = nullptr;}
  if (H_    != nullptr) {ierr = MatDestroy (&H_);    CHKERRQ(ierr); H_    = nullptr;}
  if (xrec_ != nullptr) {ierr = VecDestroy (&xrec_); CHKERRQ(ierr); xrec_ = nullptr;}


  // TODO: x_old here is used to store the full solution vector and not old guess. (Maybe change to new vector to avoid confusion?)
  if (itctx_->x_old == nullptr)  {
      ierr = VecDuplicate (itctx_->tumor_->p_, &itctx_->x_old);                        CHKERRQ (ierr);
      ierr = VecCopy (x_in, itctx_->x_old);                                            CHKERRQ (ierr);
  }

  ierr = VecDuplicate (itctx_->tumor_->p_, &xrec_);                                    CHKERRQ(ierr);

  CtxInv *ctx = itctx_.get();

  ierr = TaoCreate (PETSC_COMM_SELF, &tao_);                                          CHKERRQ (ierr);
  ierr = TaoSetType (tao_, "blmvm");                                                   CHKERRQ (ierr);

  int x_sz;
  int nk = (itctx_->n_misc_->diffusivity_inversion_) ? itctx_->n_misc_->nk_ : 0;
  int nr = (itctx_->n_misc_->reaction_inversion_) ? itctx_->n_misc_->nr_ : 0;
  x_sz = nk + nr;
  Vec x;
  ierr = VecCreateSeq (PETSC_COMM_SELF, x_sz, &x);                                    CHKERRQ (ierr);  // Inversion for rho and k
  ierr = setupVec (x, SEQ);                                                                CHKERRQ (ierr);
  ierr = VecSet (x, 0.);                                                              CHKERRQ (ierr);

  // set initial guess to current state
  ScalarType *x_in_ptr, *x_ptr;
  ierr = VecGetArray (x_in, &x_in_ptr);                                               CHKERRQ (ierr);
  ierr = VecGetArray (x, &x_ptr);                                                     CHKERRQ (ierr);
  x_ptr[0] = x_in_ptr[itctx_->n_misc_->np_];   // k1
  if (nk > 1) x_ptr[1] = x_in_ptr[itctx_->n_misc_->np_ + 1];  // k2
  if (nk > 2) x_ptr[2] = x_in_ptr[itctx_->n_misc_->np_ + 2];  // k3


  // Guess the reaction coefficient and use as IC.
  std::array<ScalarType, 7> rho_guess = {0, 3, 6, 9, 10, 12, 15}; // guess values --  these span 0 to 15 so estimate 
                                                              // roughly where to start, else we could get stuck in 
                                                              // a bad local minimum

  ScalarType min_norm = 1E15, norm = 0.;
  int idx = 0;
  for (int i = 0; i < rho_guess.size(); i++) {
    // update the tumor with this rho    
    ierr = itctx_->tumor_->rho_->updateIsotropicCoefficients (rho_guess[i], 0., 0., itctx_->tumor_->mat_prop_, itctx_->n_misc_);
    ierr = itctx_->tumor_->phi_->apply (itctx_->tumor_->c_0_, x_in);                   CHKERRQ (ierr);   // apply scaled p to IC
    ierr = itctx_->derivative_operators_->pde_operators_->solveState (0);    // solve state with guess reaction and inverted diffusivity
    ierr = itctx_->tumor_->obs_->apply (itctx_->derivative_operators_->temp_, itctx_->tumor_->c_t_);               CHKERRQ (ierr);

    // mismatch between data and c
    ierr = VecAXPY (itctx_->derivative_operators_->temp_, -1.0, data_);     CHKERRQ (ierr);    // Oc(1) - d
    ierr = VecNorm (itctx_->derivative_operators_->temp_, NORM_2, &norm);   CHKERRQ (ierr);

    if (norm < min_norm) {
      min_norm = norm;
      idx = i;
    }
  }

  PCOUT << "Initial guess for reaction coefficient: " << rho_guess[idx] << std::endl;

  // x_ptr[nk] = x_in_ptr[itctx_->n_misc_->np_ + nk];  // rho
  // if (nr > 1) x_ptr[nk + 1] = x_in_ptr[itctx_->n_misc_->np_ + nk + 1];  // r2
  // if (nr > 2) x_ptr[nk + 2] = x_in_ptr[itctx_->n_misc_->np_ + nk + 2];  // r3

  x_ptr[nk] = rho_guess[idx];  // rho
  if (nr > 1) x_ptr[nk + 1] = 0;  // r2
  if (nr > 2) x_ptr[nk + 2] = 0;  // r3

  ierr = VecRestoreArray (x_in, &x_in_ptr);                                               CHKERRQ (ierr);
  ierr = VecRestoreArray (x, &x_ptr);                                                     CHKERRQ (ierr);
  ierr = TaoSetInitialVector (tao_, x);                                               CHKERRQ (ierr);

  // Lower and Upper Bounds
  Vec lower_bound;
  ierr = VecDuplicate (x, &lower_bound);                            CHKERRQ (ierr);
  ierr = VecSet (lower_bound, 0.);                                                CHKERRQ (ierr);
  Vec upper_bound;
  ierr = VecDuplicate (x, &upper_bound);                            CHKERRQ (ierr);
  ierr = VecSet (upper_bound, PETSC_INFINITY);                                    CHKERRQ (ierr);
  ierr = TaoSetVariableBounds(tao_, lower_bound, upper_bound);                    CHKERRQ (ierr);
  ierr = VecDestroy (&lower_bound);                                               CHKERRQ (ierr);
  ierr = VecDestroy (&upper_bound);                                               CHKERRQ (ierr);

  ierr = TaoSetObjectiveAndGradientRoutine (tao_, evaluateObjectiveAndGradientForParameters, (void*) ctx);       CHKERRQ (ierr);

  ctx->update_reference_gradient = true;

  ierr = TaoSetMonitor (tao_, optimizationMonitorForParameters, (void *) ctx, NULL);                                 CHKERRQ(ierr);
  ierr = TaoSetTolerances (tao_, ctx->optsettings_->gatol, ctx->optsettings_->grtol, ctx->optsettings_->opttolgrad); CHKERRQ(ierr);
  ierr = TaoSetMaximumIterations (tao_, ctx->optsettings_->newton_maxit);                                            CHKERRQ(ierr);
  ierr = TaoSetConvergenceTest (tao_, checkConvergenceGradForParameters, ctx);                                       CHKERRQ(ierr);

  // overwrite linesearch objects
  TaoLineSearch linesearch;        // line-search object
  PetscReal minstep;
  minstep = std::pow (2.0, 18.0);
  minstep = 1.0 / minstep;
  itctx_->optsettings_->ls_minstep = minstep;

  ierr = TaoGetLineSearch (tao_, &linesearch);                                        CHKERRQ(ierr);
  // ierr = TaoLineSearchSetType (linesearch, "armijo");                                 CHKERRQ(ierr);
  linesearch->stepmin = minstep;

  ierr = TaoLineSearchSetOptionsPrefix (linesearch,"tumor_");                    CHKERRQ(ierr);

  std::stringstream ss;
  tuMSGstd(" parameters (optimizer):");
  tuMSGstd(" tolerances (stopping conditions):");
  ss << "   gatol: "<< ctx->optsettings_->gatol;  /*pout(s.str(), cplctx->_fileOutput);*/ tuMSGstd(ss.str()); ss.str(""); ss.clear();
  ss << "   grtol: "<< ctx->optsettings_->grtol;  /*pout(s.str(), cplctx->_fileOutput);*/ tuMSGstd(ss.str()); ss.str(""); ss.clear();
  ss << "   gttol: "<< ctx->optsettings_->opttolgrad;  /*pout(s.str(), cplctx->_fileOutput);*/ tuMSGstd(ss.str()); ss.str(""); ss.clear();

  ierr = TaoSetFromOptions(tao_);                                                     CHKERRQ(ierr);

  itctx_->is_ksp_gradnorm_set        = false;
  itctx_->optfeedback_->converged    = false;
  itctx_->optfeedback_->solverstatus = "";
  itctx_->optfeedback_->nb_newton_it = 0;
  itctx_->optfeedback_->nb_krylov_it = 0;
  itctx_->optfeedback_->nb_matvecs   = 0;
  itctx_->optfeedback_->nb_objevals  = 0;
  itctx_->optfeedback_->nb_gradevals = 0;
  itctx_->data                       = data_;
  itctx_->data_gradeval              = data_gradeval_;


  ScalarType self_exec_time_tuninv = -MPI_Wtime(); ScalarType invtime = 0;
  itctx_->n_misc_->statistics_.reset();
  s << "Tumor regularization = "<< itctx_->n_misc_->beta_ << " type: " << itctx_->n_misc_->regularization_norm_;  ierr = tuMSGstd(s.str()); CHKERRQ(ierr); s.str(""); s.clear();

  ierr = TaoSolve (tao_);                                                             CHKERRQ(ierr);

  self_exec_time_tuninv += MPI_Wtime();
  MPI_Reduce(&self_exec_time_tuninv, &invtime, 1, MPIType, MPI_MAX, 0, MPI_COMM_WORLD);

  ierr = VecCopy (itctx_->x_old, xrec_);                                                          CHKERRQ(ierr);

  /* Get information on termination */
  TaoConvergedReason reason;
  TaoGetConvergedReason (tao_, &reason);
  /* get solution status */
  ScalarType xdiff;
  ierr = TaoGetSolutionStatus (tao_, NULL, &itctx_->optfeedback_->jval, &itctx_->optfeedback_->gradnorm, NULL, &xdiff, NULL);         CHKERRQ(ierr);
  /* display convergence reason: */
  ierr = dispTaoConvReason (reason, itctx_->optfeedback_->solverstatus);                 CHKERRQ(ierr);
  s << " optimization done: #N-it: " << itctx_->optfeedback_->nb_newton_it    << ", #K-it: " << itctx_->optfeedback_->nb_krylov_it
                      << ", #matvec: " << itctx_->optfeedback_->nb_matvecs    << ", #evalJ: " << itctx_->optfeedback_->nb_objevals
                      << ", #evaldJ: " << itctx_->optfeedback_->nb_gradevals  << ", exec time: " << invtime;
  ierr = tuMSGstd ("------------------------------------------------------------------------------------------------"); CHKERRQ(ierr);
  ierr = tuMSGstd (s.str());                                                                                            CHKERRQ(ierr);  s.str(""); s.clear();
  ierr = tuMSGstd ("------------------------------------------------------------------------------------------------"); CHKERRQ(ierr);
  itctx_->n_misc_->statistics_.print();
  itctx_->n_misc_->statistics_.reset();

  data_ = nullptr;
  data_gradeval_ = nullptr;
  tao_is_reset_ = false;

  ierr = VecDestroy (&itctx_->x_old);  CHKERRQ (ierr);
  itctx_->x_old = nullptr;

  ierr = VecDestroy (&noise); CHKERRQ (ierr);
  ierr = VecDestroy (&x);     CHKERRQ (ierr);

  PetscFunctionReturn (0);
}




PetscErrorCode InvSolver::solve () {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;

  TU_assert (initialized_, "InvSolver::solve (): InvSolver needs to be initialized.")
  TU_assert (data_ != nullptr, "InvSolver:solve (): requires non-null input data for inversion.");
  TU_assert (data_gradeval_ != nullptr, "InvSolver:solve (): requires non-null input data for gradient evaluation.");
  TU_assert (xrec_ != nullptr, "InvSolver:solve (): requires non-null p_rec vector to be set");
  TU_assert (optsettings_ != nullptr, "InvSolver:solve (): requires non-null optimizer settings to be passed.");
  /* === observed data === */
  // apply observer on ground truth, store observed data in d
  // ierr = itctx_->tumor_->obs_->apply (data_, data_);                                  CHKERRQ(ierr);
  // smooth observed data
  ScalarType *d_ptr;
  ScalarType sigma_smooth;
  sigma_smooth = 2.0 * M_PI / itctx_->n_misc_->n_[0];
  ierr = VecGetArray (data_, &d_ptr);                                                 CHKERRQ(ierr);

  /* === Add Noise === */
  Vec noise; ScalarType *noise_ptr;
  ierr = VecCreate (PETSC_COMM_WORLD, &noise);                                        CHKERRQ(ierr);
  ierr = VecSetSizes(noise, itctx_->n_misc_->n_local_, itctx_->n_misc_->n_global_);   CHKERRQ(ierr);
  ierr = setupVec(noise);                                                    CHKERRQ(ierr);
  ierr = VecSetRandom(noise, NULL);                                                   CHKERRQ(ierr);
  ierr = VecGetArray (noise, &noise_ptr);                                             CHKERRQ(ierr);
  for (int i = 0; i < itctx_->n_misc_->n_local_; i++) {
      d_ptr[i] += noise_ptr[i] * itctx_->n_misc_->noise_scale_;
      noise_ptr[i] = d_ptr[i];                                                        //just to measure d norm
  }
  ierr = VecRestoreArray (noise, &noise_ptr);                                         CHKERRQ(ierr);
  ierr = VecRestoreArray (data_, &d_ptr);                                             CHKERRQ(ierr);
	ScalarType max, min;                                                // compute d-norm
  ScalarType d_norm = 0., d_errorl2norm = 0., d_errorInfnorm = 0.;
  #ifdef POSITIVITY
    ierr = enforcePositivity (data_, itctx_->n_misc_);
    ierr = enforcePositivity (noise, itctx_->n_misc_);
  #endif
  ierr = VecNorm (noise, NORM_2, &d_norm);                                            CHKERRQ(ierr);
  ierr = VecMax (noise, NULL, &max);                                                  CHKERRQ(ierr);
  ierr = VecMin (noise, NULL, &min);                                                  CHKERRQ(ierr);
  ierr = VecAXPY (noise, -1.0, data_);                                                CHKERRQ(ierr);
  ierr = VecNorm (noise, NORM_2, &d_errorl2norm);                                     CHKERRQ(ierr);
  ierr = VecNorm (noise, NORM_INFINITY, &d_errorInfnorm);                             CHKERRQ(ierr);
  std::stringstream s;
  s << "data (ITP), with noise: l2norm = "<< d_norm <<" [max: "<<max<<", min: "<<min<<"]";  ierr = tuMSGstd(s.str()); CHKERRQ(ierr); s.str(""); s.clear();
  s << "IT data error due to thresholding and smoothing: l2norm = "<< d_errorl2norm <<", inf-norm = " <<d_errorInfnorm;  ierr = tuMSGstd(s.str()); CHKERRQ(ierr); s.str(""); s.clear();
  if (itctx_->n_misc_->writeOutput_) {
    dataOut (data_, itctx_->n_misc_, "data.nc");
  }
  /* === solve for a initial guess === */
  //solveForParameters(data.get(), itctx_->n_misc_, itctx_->tumor_->phi_, itctx_->n_misc_->timers_, &itctx_->tumor_->p_true_);
  //tumor_->t_history_->reset(); // TODO: no time history so far, if available, reset here
  /* === initialize inverse tumor context === */
  if (itctx_->c0old == nullptr) {
      ierr = VecDuplicate (data_, &itctx_->c0old);                                     CHKERRQ(ierr);
      ierr = VecSet (itctx_->c0old, 0.0);                                              CHKERRQ(ierr);
  }
  if (itctx_->tmp == nullptr) {
      ierr = VecDuplicate (data_, &itctx_->tmp);                                       CHKERRQ(ierr);
      ierr = VecSet (itctx_->tmp, 0.0);                                                CHKERRQ(ierr);
  }
  if (itctx_->x_old == nullptr)  {
      ierr = VecDuplicate (itctx_->tumor_->p_, &itctx_->x_old);                        CHKERRQ (ierr);
      ierr = VecCopy (itctx_->tumor_->p_, itctx_->x_old);                              CHKERRQ (ierr);
  }

  // reset with zero for new ITP solve
  ierr = VecSet (itctx_->c0old, 0.0);                                                  CHKERRQ(ierr);
  if (itctx_->n_misc_->beta_changed_)
      itctx_->optsettings_->beta      = itctx_->n_misc_->beta_;
  else
    itctx_->n_misc_->beta_             = itctx_->optsettings_->beta;                     // set beta for this inverse solver call
  itctx_->is_ksp_gradnorm_set        = false;
  itctx_->optfeedback_->converged    = false;
  itctx_->optfeedback_->solverstatus = "";
  itctx_->optfeedback_->nb_newton_it = 0;
  itctx_->optfeedback_->nb_krylov_it = 0;
  itctx_->optfeedback_->nb_matvecs   = 0;
  itctx_->optfeedback_->nb_objevals  = 0;
  itctx_->optfeedback_->nb_gradevals = 0;
  itctx_->data                       = data_;
  itctx_->data_gradeval              = data_gradeval_;

  // reset tao, if we want virgin TAO for every inverse solve
  if (itctx_->optsettings_->reset_tao) {
    ierr = resetTao(itctx_->n_misc_);                                                    CHKERRQ(ierr);
  }

  ScalarType *p_ptr, *w_ptr;
  ScalarType w = 1.;
  ScalarType p_max;
  if (itctx_->n_misc_->regularization_norm_ == wL2) {//set the weights for w-l2 solve
    itctx_->n_misc_->beta_ = 1.;
    ierr = VecGetArray (itctx_->tumor_->p_, &p_ptr);                                     CHKERRQ (ierr);
    ierr = VecSet (itctx_->weights, 0.0);                                                CHKERRQ (ierr);
    ierr = VecGetArray (itctx_->weights, &w_ptr);                                        CHKERRQ (ierr);
    ierr = VecMax (itctx_->tumor_->p_, NULL, &p_max);                                    CHKERRQ (ierr);

    for (int i = 0; i < itctx_->n_misc_->np_; i++) {
      if (p_ptr[i] > 0.1 * p_max) //nnz component (arbitrarily defined)
        w_ptr[i] = 0;
      else
        w_ptr[i] = w;
    }
    ierr = VecRestoreArray (itctx_->tumor_->p_, &p_ptr);                                 CHKERRQ (ierr);
    ierr = VecRestoreArray (itctx_->weights, &w_ptr);                                    CHKERRQ (ierr);
    ierr = VecCopy (itctx_->weights, itctx_->tumor_->weights_);                          CHKERRQ (ierr);
  }

  /* === set TAO options === */
  if (tao_is_reset_) {
    itctx_->update_reference_gradient = true;
    itctx_->update_reference_objective = true;
    ierr = setTaoOptions (tao_, itctx_.get());                                         CHKERRQ(ierr);

    if ((itctx_->optsettings_->newtonsolver == QUASINEWTON) &&
         itctx_->optsettings_->lmvm_set_hessian) {
      ierr = TaoLMVMSetH0 (tao_, H_);                                                    CHKERRQ(ierr);
    } else {
      ierr = TaoSetHessianRoutine (tao_, H_, H_, matfreeHessian, (void *) itctx_.get()); CHKERRQ(ierr);
    }
  // no TAO reset, re-use old LMVM data
  } else {
    if (itctx_->optsettings_->newtonsolver == QUASINEWTON) {
      // if TAO is not reset, prevent first step from being a gradeint descent step, use BFGS step
      //TAO_LMVM *lmP = (TAO_LMVM *)tao->data;   // get context
      //lmP->
    }
  }
  ScalarType self_exec_time_tuninv = -MPI_Wtime(); ScalarType invtime = 0;
  /* === solve === */
  // --------
  itctx_->n_misc_->statistics_.reset();
  //resetTimers(itctx->n_misc_->timers_);
  //Gradient check begin
  //    ierr = itctx_->derivative_operators_->checkGradient (itctx_->tumor_->p_, itctx_->data);
  //Gradient check end

  s << "Tumor regularization = "<< itctx_->n_misc_->beta_ << " type: " << itctx_->n_misc_->regularization_norm_;  ierr = tuMSGstd(s.str()); CHKERRQ(ierr); s.str(""); s.clear();

  ierr = TaoSolve (tao_);                                                                CHKERRQ(ierr);
  // --------
  self_exec_time_tuninv += MPI_Wtime();
  MPI_Reduce(&self_exec_time_tuninv, &invtime, 1, MPIType, MPI_MAX, 0, MPI_COMM_WORLD);

	/* === get solution === */
	Vec p; ierr = TaoGetSolutionVector (tao_, &p);                                          CHKERRQ(ierr);
	ierr = VecCopy (p, xrec_);                                                              CHKERRQ(ierr);
	/* Get information on termination */
	TaoConvergedReason reason;
	TaoGetConvergedReason (tao_, &reason);
	/* get solution status */
	ScalarType xdiff;
	ierr = TaoGetSolutionStatus (tao_, NULL, &itctx_->optfeedback_->jval, &itctx_->optfeedback_->gradnorm, NULL, &xdiff, NULL);         CHKERRQ(ierr);
	/* display convergence reason: */
	ierr = dispTaoConvReason (reason, itctx_->optfeedback_->solverstatus);                 CHKERRQ(ierr);
  s << " optimization done: #N-it: " << itctx_->optfeedback_->nb_newton_it    << ", #K-it: " << itctx_->optfeedback_->nb_krylov_it
                      << ", #matvec: " << itctx_->optfeedback_->nb_matvecs    << ", #evalJ: " << itctx_->optfeedback_->nb_objevals
                      << ", #evaldJ: " << itctx_->optfeedback_->nb_gradevals  << ", exec time: " << invtime;
  ierr = tuMSGstd ("------------------------------------------------------------------------------------------------"); CHKERRQ(ierr);
  ierr = tuMSGstd (s.str());                                                                                            CHKERRQ(ierr);  s.str(""); s.clear();
  ierr = tuMSGstd ("------------------------------------------------------------------------------------------------"); CHKERRQ(ierr);
  itctx_->n_misc_->statistics_.print();
  itctx_->n_misc_->statistics_.reset();

  out_params_.push_back (itctx_->optfeedback_->nb_newton_it);

	// only update if triggered from outside, i.e., if new information to the ITP solver is present
	itctx_->update_reference_gradient = false;
	// reset vectors (remember, memory managed on caller side):
	data_ = nullptr;
	data_gradeval_ = nullptr;
  tao_is_reset_ = false;

  ierr = VecDestroy (&itctx_->x_old);  CHKERRQ (ierr);
  itctx_->x_old = nullptr;

  ierr = VecDestroy (&noise); CHKERRQ (ierr);

	PetscFunctionReturn (0);
}

InvSolver::~InvSolver () {
    PetscErrorCode ierr = 0;
    ierr = VecDestroy (&xrec_);
    ierr = TaoDestroy (&tao_);
    ierr = MatDestroy (&H_);
}

// ============================= non-class methods used for TAO ============================
/* ------------------------------------------------------------------- */
/*
 evaluateObjectiveFunction - evaluates the objective function J(x).
 Input Parameters:
 .  tao - the Tao context
 .  x   - the input vector
 .  ptr - optional user-defined context, as set by TaoSetFunction()
 Output Parameters:
 .  J    - the newly evaluated function
 */
PetscErrorCode evaluateObjectiveFunction (Tao tao, Vec x, PetscReal *J, void *ptr) {
	PetscFunctionBegin;
	PetscErrorCode ierr = 0;
	Event e ("tao-eval-obj-tumor");
    std::array<ScalarType, 7> t = {0};
    ScalarType self_exec_time = -MPI_Wtime ();
	CtxInv *itctx = reinterpret_cast<CtxInv*>(ptr);
  itctx->optfeedback_->nb_objevals++;
	ierr = itctx->derivative_operators_->evaluateObjective (J, x, itctx->data);
	self_exec_time += MPI_Wtime ();
	accumulateTimers (itctx->n_misc_->timers_, t, self_exec_time);
    e.addTimings (t);
    e.stop ();
	PetscFunctionReturn (0);
}
/* ------------------------------------------------------------------- */
/*
 evaluateGradient evaluates the gradient g(m0)
 input parameters:
  . tao  - the Tao context
  . x   - input vector p (current estimate for the paramterized initial condition)
  . ptr  - optional user-defined context
 output parameters:
  . dJ    - vector containing the newly evaluated gradient
 */
PetscErrorCode evaluateGradient (Tao tao, Vec x, Vec dJ, void *ptr) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    Event e ("tao-eval-grad-tumor");
    std::array<ScalarType, 7> t = {0};
    ScalarType self_exec_time = -MPI_Wtime ();
    CtxInv *itctx = reinterpret_cast<CtxInv*>(ptr);
    ierr = VecCopy (x, itctx->derivative_operators_->p_current_);                       CHKERRQ (ierr);

    itctx->optfeedback_->nb_gradevals++;
    ierr = itctx->derivative_operators_->evaluateGradient (dJ, x, itctx->data_gradeval);
    std::stringstream s;
    if (itctx->optsettings_->verbosity > 1) {
        ScalarType gnorm;
        ierr = VecNorm (dJ, NORM_2, &gnorm);                                            CHKERRQ(ierr);
        s << " norm of gradient ||g||_2 = " << std::scientific << gnorm; ierr = tuMSGstd(s.str()); CHKERRQ(ierr); s.str(""); s.clear();
    }
    self_exec_time += MPI_Wtime ();
    accumulateTimers (itctx->n_misc_->timers_, t, self_exec_time);
    e.addTimings (t);
    e.stop ();
    PetscFunctionReturn (0);
}
/* ------------------------------------------------------------------- */
/*
 evaluateObjectiveFunctionAndGradient - evaluates the function and corresponding gradient
 Input Parameters:
  . tao - the Tao context
  . x   - the input vector
  . ptr - optional user-defined context, as set by TaoSetFunction()
 Output Parameters:
  . J   - the newly evaluated function
  . dJ   - the newly evaluated gradient
 */
PetscErrorCode evaluateObjectiveFunctionAndGradient (Tao tao, Vec x, PetscReal *J, Vec dJ, void *ptr){
	PetscFunctionBegin;
	PetscErrorCode ierr = 0;
  Event e ("tao-eval-obj/grad-tumor");
  std::array<ScalarType, 7> t = {0};
  ScalarType self_exec_time = -MPI_Wtime ();
  CtxInv *itctx = reinterpret_cast<CtxInv*>(ptr);
  ierr = VecCopy (x, itctx->derivative_operators_->p_current_);                       CHKERRQ (ierr);

  itctx->optfeedback_->nb_objevals++;
  itctx->optfeedback_->nb_gradevals++;
  ierr = itctx->derivative_operators_->evaluateObjectiveAndGradient (J, dJ, x, itctx->data_gradeval);
  std::stringstream s;
  if (itctx->optsettings_->verbosity > 1) {
      ScalarType gnorm;
      ierr = VecNorm (dJ, NORM_2, &gnorm);                                            CHKERRQ(ierr);
      s << " norm of gradient ||g||_2 = " << std::scientific << gnorm; ierr = tuMSGstd(s.str()); CHKERRQ(ierr); s.str(""); s.clear();
  }
  self_exec_time += MPI_Wtime ();
  accumulateTimers (itctx->n_misc_->timers_, t, self_exec_time);
  e.addTimings (t);
  e.stop ();

	//ierr = evaluateObjectiveFunction (tao, p, J, ptr);                     CHKERRQ(ierr);
	//ierr = evaluateGradient (tao, p, dJ, ptr);                             CHKERRQ(ierr);
	PetscFunctionReturn (0);
}

PetscErrorCode evaluateObjectiveAndGradientForParameters (Tao tao, Vec x, PetscReal *J, Vec dJ, void *ptr){
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  Event e ("tao-eval-obj/grad-tumor-params");
  std::array<ScalarType, 7> t = {0};
  ScalarType self_exec_time = -MPI_Wtime ();
  CtxInv *itctx = reinterpret_cast<CtxInv*>(ptr);

  #if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 9)
    int lock_state;
    ierr = VecLockGet (x, &lock_state);     CHKERRQ (ierr);
    if (lock_state != 0) {
      ierr = VecLockPop (x);                CHKERRQ (ierr);
    }
  #endif

  itctx->optfeedback_->nb_objevals++;
  itctx->optfeedback_->nb_gradevals++;

  // set the last 2-3 entries to the parameters obtained from tao and pass to derivativeoperators
  ScalarType *x_ptr, *x_full_ptr;
  ierr = VecGetArray (x, &x_ptr);       CHKERRQ (ierr);
  ierr = VecGetArray (itctx->x_old, &x_full_ptr);   CHKERRQ (ierr);

  x_full_ptr[itctx->n_misc_->np_] = x_ptr[0];   // k1
  if (itctx->n_misc_->nk_ > 1) x_full_ptr[itctx->n_misc_->np_ + 1] = x_ptr[1];  // k2
  if (itctx->n_misc_->nk_ > 2) x_full_ptr[itctx->n_misc_->np_ + 2] = x_ptr[2];  // k3
  x_full_ptr[itctx->n_misc_->np_ + itctx->n_misc_->nk_] = x_ptr[itctx->n_misc_->nk_];  // rho
  if (itctx->n_misc_->nr_ > 1) x_full_ptr[itctx->n_misc_->np_ + itctx->n_misc_->nk_ + 1] = x_ptr[itctx->n_misc_->nk_ + 1];  // r2
  if (itctx->n_misc_->nr_ > 2) x_full_ptr[itctx->n_misc_->np_ + itctx->n_misc_->nk_ + 2] = x_ptr[itctx->n_misc_->nk_ + 2];  // r2

  ierr = VecRestoreArray (x, &x_ptr);       CHKERRQ (ierr);
  ierr = VecRestoreArray (itctx->x_old, &x_full_ptr);   CHKERRQ (ierr);

  Vec dJ_full;
  ierr = VecDuplicate (itctx->x_old, &dJ_full);         CHKERRQ (ierr);

  #if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 9)
    if (lock_state != 0) {
      ierr = VecLockPush (x);     CHKERRQ (ierr);
    }
  #endif

  ierr = itctx->derivative_operators_->evaluateObjectiveAndGradient (J, dJ_full, itctx->x_old, itctx->data_gradeval);

  ScalarType *dj_ptr, *dj_full_ptr;
  ierr = VecGetArray (dJ, &dj_ptr);             CHKERRQ (ierr);
  ierr = VecGetArray (dJ_full, &dj_full_ptr);   CHKERRQ (ierr);

  dj_ptr[0] = dj_full_ptr[itctx->n_misc_->np_];
  if (itctx->n_misc_->nk_ > 1) dj_ptr[1] = dj_full_ptr[itctx->n_misc_->np_ + 1];  // k2
  if (itctx->n_misc_->nk_ > 2) dj_ptr[2] = dj_full_ptr[itctx->n_misc_->np_ + 2];  // k3
  dj_ptr[itctx->n_misc_->nk_] = dj_full_ptr[itctx->n_misc_->np_ + itctx->n_misc_->nk_];  // rho
  if (itctx->n_misc_->nr_ > 1) dj_ptr[itctx->n_misc_->nk_ + 1] = dj_full_ptr[itctx->n_misc_->np_ + itctx->n_misc_->nk_ + 1];  // r2
  if (itctx->n_misc_->nr_ > 2) dj_ptr[itctx->n_misc_->nk_ + 2] = dj_full_ptr[itctx->n_misc_->np_ + itctx->n_misc_->nk_ + 2];  // r2

  ierr = VecRestoreArray (dJ, &dj_ptr);             CHKERRQ (ierr);
  ierr = VecRestoreArray (dJ_full, &dj_full_ptr);   CHKERRQ (ierr);

  std::stringstream s;
  if (itctx->optsettings_->verbosity > 1) {
      ScalarType gnorm;
      ierr = VecNorm (dJ, NORM_2, &gnorm);                                            CHKERRQ(ierr);
      s << " norm of gradient ||g||_2 = " << std::scientific << gnorm; ierr = tuMSGstd(s.str()); CHKERRQ(ierr); s.str(""); s.clear();
  }
  self_exec_time += MPI_Wtime ();
  accumulateTimers (itctx->n_misc_->timers_, t, self_exec_time);
  e.addTimings (t);
  e.stop ();

  ierr = VecDestroy (&dJ_full);   CHKERRQ (ierr);

  PetscFunctionReturn (0);
}



/* ------------------------------------------------------------------- */
/*
 hessianMatVec    computes the Hessian matrix-vector product
 input parameters:
  . H       input matrix
  . s       input vector
 output parameters:
  . Hs      solution vector
 */
PetscErrorCode hessianMatVec (Mat A, Vec x, Vec y) {    //y = Ax
	PetscFunctionBegin;
	PetscErrorCode ierr = 0;

    Event e ("tao-hess-matvec-tumor");
    std::array<ScalarType, 7> t = {0}; ScalarType self_exec_time = -MPI_Wtime ();
    // get context
    void *ptr;
    ierr = MatShellGetContext (A, &ptr);						             CHKERRQ (ierr);
    CtxInv *itctx = reinterpret_cast<CtxInv*>( ptr);
    // eval hessian
    itctx->optfeedback_->nb_matvecs++;
    ierr = itctx->derivative_operators_->evaluateHessian (y, x);
    if (itctx->optsettings_->verbosity > 1) {
        PetscPrintf (MPI_COMM_WORLD, " applying hessian done!\n");
        ScalarType xnorm;
        ierr = VecNorm (x, NORM_2, &xnorm); CHKERRQ(ierr);
        PetscPrintf (MPI_COMM_WORLD, " norm of search direction ||x||_2 = %e\n", xnorm);
    }
    self_exec_time += MPI_Wtime ();
    accumulateTimers (itctx->n_misc_->timers_, t, self_exec_time);
    e.addTimings (t); e.stop ();
    PetscFunctionReturn (0);
}
/* ------------------------------------------------------------------- */
/*
 constApxHessianMatVec    computes the Hessian matrix-vector product for
                          a constant approximation \beta_p \Phi^T\Phi of the Hessian
 input parameters:
  . H       input matrix
  . s       input vector
 output parameters:
  . Hs      solution vector
 */
PetscErrorCode constApxHessianMatVec (Mat A, Vec x, Vec y) {    //y = Ax
	PetscFunctionBegin;
	PetscErrorCode ierr = 0;

    Event e ("tao-lmvm-init-hess--matvec");
    std::array<ScalarType, 7> t = {0}; ScalarType self_exec_time = -MPI_Wtime ();
    // get context
    void *ptr;
    ierr = MatShellGetContext (A, &ptr);						             CHKERRQ (ierr);
    CtxInv *itctx = reinterpret_cast<CtxInv*>( ptr);
    // eval hessian
    ierr = itctx->derivative_operators_->evaluateConstantHessianApproximation (y, x);
    self_exec_time += MPI_Wtime ();
    accumulateTimers (itctx->n_misc_->timers_, t, self_exec_time);
    e.addTimings (t); e.stop ();
    PetscFunctionReturn (0);
}
/* ------------------------------------------------------------------- */
PetscErrorCode matfreeHessian (Tao tao, Vec x, Mat H, Mat precH, void *ptr) {
	PetscFunctionBegin;
	PetscFunctionReturn (0);
}
/* ------------------------------------------------------------------- */
/*
 preconditionerMatVec    computes the matrix-vector product of inverse of
 preconditioner and some input vector
 input parameters:
 pinv    input matrix (shell context)
 x       input vector
 output parameters:
 .       pinvx   inverse of preconditioner applied to output vector
 */
PetscErrorCode preconditionerMatVec (PC pinv, Vec x, Vec pinvx) {
	PetscFunctionBegin;
	PetscErrorCode ierr = 0;
	void *ptr;
    // get shell context
    ierr = PCShellGetContext (pinv, &ptr);                  CHKERRQ(ierr);
    // apply the hessian
    ierr = applyPreconditioner (ptr, x, pinvx);             CHKERRQ(ierr);
    PetscFunctionReturn (0);
}
/* ------------------------------------------------------------------- */
/*
 applyPreconditioner  apply preconditioner to a given vector
 input parameters:
 ptr       pointer to user defined context
 x         input vector
 output parameters:
 .       pinvx     inverse of preconditioner applied to input vector
 */
PetscErrorCode applyPreconditioner (void *ptr, Vec x, Vec pinvx) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    Event e ("tao-apply-hess-precond");
    std::array<ScalarType, 7> t = {0};
    ScalarType self_exec_time = -MPI_Wtime ();
    ScalarType *ptr_pinvx = NULL, *ptr_x = NULL;
    CtxInv *itctx = reinterpret_cast<CtxInv*> (ptr);
    ierr = VecCopy (x, pinvx);
    // === PRECONDITIONER CURRENTLY DISABLED ===
    PetscFunctionReturn (0);
    // apply hessian preconditioner
    //ierr = itctx->derivative_operators_->evaluateHessian(pinvx, x);
    self_exec_time += MPI_Wtime ();
    accumulateTimers (itctx->n_misc_->timers_, t, self_exec_time);
    e.addTimings (t); e.stop ();
    PetscFunctionReturn (0);
}

/* ------------------------------------------------------------------- */
/*
 optimizationMonitor    mointors the inverse Gauß-Newton solve
 input parameters:
  . tao       TAO object
  . ptr       optional user defined context
 */
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
    TaoConvergedReason flag;
    CtxInv *itctx = reinterpret_cast<CtxInv*> (ptr);


    // get current iteration, objective value, norm of gradient, norm of
    // norm of contraint, step length / trust region readius of iteratore
    // and termination reason
    Vec tao_x;
    ierr = TaoGetSolutionStatus (tao, &its, &J, &gnorm, &cnorm, &step, &flag);      CHKERRQ(ierr);
    ierr = TaoGetSolutionVector(tao, &tao_x);                                       CHKERRQ(ierr);

    #if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 9)
    if (itctx->update_reference_gradient) {
      Vec dJ, p0;
      ScalarType norm_gref = 0.;
      ierr = VecDuplicate (itctx->tumor_->p_, &dJ);                               CHKERRQ(ierr);
      ierr = VecDuplicate (itctx->tumor_->p_, &p0);                               CHKERRQ(ierr);
      ierr = VecSet (dJ, 0.);                                                   CHKERRQ(ierr);
      ierr = VecSet (p0, 0.);                                                   CHKERRQ(ierr);

      if (itctx->n_misc_->flag_reaction_inv_) {
        norm_gref = gnorm;
      } else {
        ierr = evaluateGradient(tao, p0, dJ, (void*) itctx);
        ierr = VecNorm (dJ, NORM_2, &norm_gref);                                  CHKERRQ(ierr);
      }
      itctx->optfeedback_->gradnorm0 = norm_gref;
      //ctx->gradnorm0 = gnorm;
      itctx->update_reference_gradient = false;
      std::stringstream s; s <<"updated reference gradient for relative convergence criterion, Gauß-Newton solver: " << itctx->optfeedback_->gradnorm0;
      ierr = tuMSGstd(s.str());                                                 CHKERRQ(ierr);
      ierr = VecDestroy(&dJ);                                                   CHKERRQ(ierr);
      ierr = VecDestroy(&p0);                                                   CHKERRQ(ierr);
    }
    #endif

    // ierr = VecAXPY (itctx->x_old, -1.0, tao_x);                                     CHKERRQ (ierr);

    // ScalarType dp_norm, p_norm;
    // ierr = VecNorm (itctx->x_old, NORM_INFINITY, &dp_norm);                         CHKERRQ (ierr);
    // ierr = VecNorm (tao_x, NORM_INFINITY, &p_norm);                                 CHKERRQ (ierr);
    // accumulate number of newton iterations
    itctx->optfeedback_->nb_newton_it++;
    // print out Newton iteration information

    ierr = itctx->tumor_->phi_->apply (itctx->tumor_->c_0_, tao_x);                 CHKERRQ (ierr);
    //Prints a warning if tumor IC is clipped
    ierr = checkClipping (itctx->tumor_->c_0_, itctx->n_misc_);           CHKERRQ (ierr);

    ScalarType mx, mn;
    ierr = VecMax (itctx->tumor_->c_t_, NULL, &mx); CHKERRQ (ierr);
    ierr = VecMin (itctx->tumor_->c_t_, NULL, &mn); CHKERRQ (ierr);
    // this print helps determine if theres any large aliasing errors which is causing ls failure etc
    PCOUT << "[---------- Tumor final bounds: Max = " << mx << ", Min = " << mn << " -----------]" << std::endl;

    std::stringstream s;
    if (its == 0) {
        s << std::setw(4)  << " iter"              << "   " << std::setw(18) << "objective (abs)" << "   "
          << std::setw(18) << "||gradient||_2,rel" << "   " << std::setw(18) << "||gradient||_2"  << "   "
          << std::setw(18) << "step" << "   ";
          if (itctx->n_misc_->diffusivity_inversion_) {
            s << std::setw(18) << "k";
          }

        if(itctx->optsettings_->newtonsolver == QUASINEWTON) {
          ierr = tuMSGstd (" starting optimization, TAO's LMVM");                   CHKERRQ(ierr);
        } else {
          ierr = tuMSGstd (" starting optimization, TAO's Gauß-Newton");            CHKERRQ(ierr);
        }
        ierr = tuMSGstd ("-------------------------------------------------------------------------------------------------------------------------------------"); CHKERRQ(ierr);
        ierr = tuMSGwarn (s.str());                                                 CHKERRQ(ierr);
        ierr = tuMSGstd ("-------------------------------------------------------------------------------------------------------------------------------------"); CHKERRQ(ierr);
        s.str ("");
        s.clear ();
    }

    s << " "   << std::scientific << std::setprecision(5) << std::setfill('0') << std::setw(4) << its << std::setfill(' ')
      << "   " << std::scientific << std::setprecision(12) << std::setw(18) << J
      << "   " << std::scientific << std::setprecision(12) << std::setw(18) << gnorm/itctx->optfeedback_->gradnorm0
      << "   " << std::scientific << std::setprecision(12) << std::setw(18) << gnorm
      << "   " << std::scientific << std::setprecision(12) << std::setw(18) << step
      ;
      if (itctx->n_misc_->diffusivity_inversion_) {
        ScalarType *x_ptr;
        ierr = VecGetArray(tao_x, &x_ptr);                                         CHKERRQ(ierr);
        s << "   " << std::scientific << std::setprecision(12) << std::setw(18) << x_ptr[itctx->n_misc_->np_];
        if (itctx->n_misc_->nk_ > 1) {
          s << "   " << std::scientific << std::setprecision(12) << std::setw(18) << x_ptr[itctx->n_misc_->np_ + 1]; }
        ierr = VecRestoreArray(tao_x, &x_ptr);                                     CHKERRQ(ierr);
      }
    ierr = tuMSGwarn (s.str());                                                    CHKERRQ(ierr);
    s.str ("");
    s.clear ();


    //ierr = PetscPrintf (PETSC_COMM_WORLD, "\nKSP number of krylov iterations: %d\n", itctx->optfeedback_->nb_krylov_it);          CHKERRQ(ierr);
    //itctx->optfeedback_->nb_krylov_it = 0;

    //Gradient check begin
    // ierr = itctx->derivative_operators_->checkGradient (tao_x, itctx->data);
    //Gradient check end
    PetscFunctionReturn (0);
}

PetscErrorCode optimizationMonitorForParameters (Tao tao, void *ptr) {
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
    TaoConvergedReason flag;
    CtxInv *itctx = reinterpret_cast<CtxInv*> (ptr);

    ScalarType norm_gref;
    ierr = TaoGetSolutionStatus (tao, &its, &J, &gnorm, &cnorm, &step, &flag);      CHKERRQ(ierr);
    ierr = TaoGetSolutionVector(tao, &x);                                       CHKERRQ(ierr);

    #if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 9)
    if (itctx->update_reference_gradient) {
      norm_gref = gnorm;
      itctx->optfeedback_->gradnorm0 = norm_gref;
      itctx->update_reference_gradient = false;
      std::stringstream s; s <<"updated reference gradient for relative convergence criterion, Quasi-Newton solver: " << itctx->optfeedback_->gradnorm0;
      ierr = tuMSGstd(s.str());                                                 CHKERRQ(ierr);
    }
    #endif

    // accumulate number of newton iterations
    itctx->optfeedback_->nb_newton_it++;

    ScalarType *x_ptr, *x_full_ptr;
    ierr = VecGetArray (x, &x_ptr);       CHKERRQ (ierr);
    ierr = VecGetArray (itctx->x_old, &x_full_ptr);   CHKERRQ (ierr);

    x_full_ptr[itctx->n_misc_->np_] = x_ptr[0];   // k1
    if (itctx->n_misc_->nk_ > 1) x_full_ptr[itctx->n_misc_->np_ + 1] = x_ptr[1];  // k2
    if (itctx->n_misc_->nk_ > 2) x_full_ptr[itctx->n_misc_->np_ + 2] = x_ptr[2];  // k3
    x_full_ptr[itctx->n_misc_->np_ + itctx->n_misc_->nk_] = x_ptr[itctx->n_misc_->nk_];  // rho
    if (itctx->n_misc_->nr_ > 1) x_full_ptr[itctx->n_misc_->np_ + itctx->n_misc_->nk_ + 1] = x_ptr[itctx->n_misc_->nk_ + 1];  // r1
    if (itctx->n_misc_->nr_ > 2) x_full_ptr[itctx->n_misc_->np_ + itctx->n_misc_->nk_ + 2] = x_ptr[itctx->n_misc_->nk_ + 2];  // r1

    ierr = VecRestoreArray (x, &x_ptr);       CHKERRQ (ierr);
    ierr = VecRestoreArray (itctx->x_old, &x_full_ptr);   CHKERRQ (ierr);

    ierr = itctx->tumor_->phi_->apply (itctx->tumor_->c_0_, itctx->x_old);                 CHKERRQ (ierr);
    //Prints a warning if tumor IC is clipped
    ierr = checkClipping (itctx->tumor_->c_0_, itctx->n_misc_);           CHKERRQ (ierr);

    ScalarType mx, mn;
    ierr = VecMax (itctx->tumor_->c_t_, NULL, &mx); CHKERRQ (ierr);
    ierr = VecMin (itctx->tumor_->c_t_, NULL, &mn); CHKERRQ (ierr);
    // this print helps determine if theres any large aliasing errors which is causing ls failure etc
    PCOUT << "[---------- Tumor final bounds: Max = " << mx << ", Min = " << mn << " -----------]" << std::endl;

    std::stringstream s;
    if (its == 0) {
        s << std::setw(4)  << " iter"              << "   " << std::setw(18) << "objective (abs)" << "   "
          << std::setw(18) << "||gradient||_2,rel" << "   " << std::setw(18) << "||gradient||_2"  << "   "
          << std::setw(18) << "step" << "   ";

            s << std::setw(18) << "r1"; 
            if (itctx->n_misc_->nr_ > 1) s << std::setw(18) << "r2"; 
            if (itctx->n_misc_->nr_ > 2) s << std::setw(18) << "r3"; 
            s << std::setw(18) << "k1";
            if (itctx->n_misc_->nk_ > 1) s << std::setw(18) << "k2"; 
            if (itctx->n_misc_->nk_ > 2) s << std::setw(18) << "k3"; 
          

        ierr = tuMSGstd ("starting optimization for only biophysical parameters");                   CHKERRQ(ierr);
        
        ierr = tuMSGstd ("-------------------------------------------------------------------------------------------------------------------------------------"); CHKERRQ(ierr);
        ierr = tuMSGwarn (s.str());                                                 CHKERRQ(ierr);
        ierr = tuMSGstd ("-------------------------------------------------------------------------------------------------------------------------------------"); CHKERRQ(ierr);
        s.str ("");
        s.clear ();
    }

    s << " "   << std::scientific << std::setprecision(5) << std::setfill('0') << std::setw(4) << its << std::setfill(' ')
      << "   " << std::scientific << std::setprecision(12) << std::setw(18) << J
      << "   " << std::scientific << std::setprecision(12) << std::setw(18) << gnorm/itctx->optfeedback_->gradnorm0
      << "   " << std::scientific << std::setprecision(12) << std::setw(18) << gnorm
      << "   " << std::scientific << std::setprecision(12) << std::setw(18) << step
      ;

      ierr = VecGetArray(x, &x_ptr);                                         CHKERRQ(ierr);
      s << "   " << std::scientific << std::setprecision(12) << std::setw(18) << x_ptr[itctx->n_misc_->nk_]; 
      if (itctx->n_misc_->nr_ > 1) s << "   " << std::scientific << std::setprecision(12) << std::setw(18) << x_ptr[itctx->n_misc_->nk_ + 1]; 
      if (itctx->n_misc_->nr_ > 2) s << "   " << std::scientific << std::setprecision(12) << std::setw(18) << x_ptr[itctx->n_misc_->nk_ + 2]; 
      s << "   " << std::scientific << std::setprecision(12) << std::setw(18) << x_ptr[0];
      if (itctx->n_misc_->nk_ > 1) {
        s << "   " << std::scientific << std::setprecision(12) << std::setw(18) << x_ptr[1]; 
      }
      if (itctx->n_misc_->nk_ > 2) {
        s << "   " << std::scientific << std::setprecision(12) << std::setw(18) << x_ptr[2]; 
      }

      
      ierr = VecRestoreArray(x, &x_ptr);                                     CHKERRQ(ierr);
      
    ierr = tuMSGwarn (s.str());                                                    CHKERRQ(ierr);
    s.str ("");
    s.clear ();

    s << "c1guess_paraminvitr-" << its << ".nc";
    if (itctx->n_misc_->verbosity_ >= 4 && its % 5 == 0) {
        dataOut (itctx->tumor_->c_t_, itctx->n_misc_, s.str().c_str());
    }
    s.str(std::string()); s.clear();


    //ierr = PetscPrintf (PETSC_COMM_WORLD, "\nKSP number of krylov iterations: %d\n", itctx->optfeedback_->nb_krylov_it);          CHKERRQ(ierr);
    //itctx->optfeedback_->nb_krylov_it = 0;

    //Gradient check begin
    // ierr = itctx->derivative_operators_->checkGradient (itctx->x_old, itctx->data);
    //Gradient check end
    PetscFunctionReturn (0);
}

PetscErrorCode optimizationMonitorL1 (Tao tao, void *ptr) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    PetscInt its;
    ScalarType J = 0, gnorm = 0, cnorm = 0 , step = 0, D = 0, J0 = 0, D0 = 0, gnorm0 = 0;
    Vec x = nullptr;
    char msg[256];
    std::string statusmsg;
    TaoConvergedReason flag;
    CtxInv *itctx = reinterpret_cast<CtxInv*> (ptr);

    // get current iteration, objective value, norm of gradient, norm of
    // norm of contraint, step length / trust region readius of iteratore
    // and termination reason
    Vec tao_x;
    ierr = TaoGetSolutionStatus (tao, &its, &J, &gnorm, &cnorm, &step, &flag);      CHKERRQ(ierr);
    ierr = TaoGetSolutionVector(tao, &tao_x);                                   CHKERRQ(ierr);

    #if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 9)
    if (itctx->update_reference_gradient) {
      Vec dJ, p0;
      ScalarType norm_gref = 0.;
      ierr = VecDuplicate (itctx->tumor_->p_, &dJ);                               CHKERRQ(ierr);
      ierr = VecDuplicate (itctx->tumor_->p_, &p0);                               CHKERRQ(ierr);
      ierr = VecSet (dJ, 0.);                                                   CHKERRQ(ierr);
      ierr = VecSet (p0, 0.);                                                   CHKERRQ(ierr);

      if (itctx->n_misc_->flag_reaction_inv_) {
        norm_gref = gnorm;
      } else {
        ierr = evaluateGradient(tao, p0, dJ, (void*) itctx);
        ierr = VecNorm (dJ, NORM_2, &norm_gref);                                  CHKERRQ(ierr);
      }
      itctx->optfeedback_->gradnorm0 = norm_gref;
      //ctx->gradnorm0 = gnorm;
      itctx->update_reference_gradient = false;
      std::stringstream s; s <<"updated reference gradient for relative convergence criterion, Gauß-Newton solver: " << itctx->optfeedback_->gradnorm0;
      ierr = tuMSGstd(s.str());                                                 CHKERRQ(ierr);
      ierr = VecDestroy(&dJ);                                                   CHKERRQ(ierr);
      ierr = VecDestroy(&p0);                                                   CHKERRQ(ierr);
    }
    #endif

    TaoLineSearch ls = nullptr;
    ierr = TaoGetLineSearch(tao, &ls);                                          CHKERRQ (ierr);
    LSCtx *lsctx = (LSCtx*) ls->data;
    PetscReal J_old = lsctx->J_old;

    // accumulate number of newton iterations
    itctx->optfeedback_->nb_newton_it++;
    // print out Newton iteration information

    std::stringstream s;
    if (its == 0) {
        s << std::setw(4)  << " iter"              << "   " << std::setw(18) << "objective (abs)" << "   "
          << std::setw(18) << "|objective|_rel" << "   " << std::setw(18) << "||gradient||_2"  << "   "
          << std::setw(18) << "step" << "   " << std::setw(10) << "sparsity";
          if (itctx->n_misc_->diffusivity_inversion_) {
            s << std::setw(18) << "k";
          }

        ierr = tuMSGstd (" User defined L1 optimization");                          CHKERRQ(ierr);
        ierr = tuMSGstd ("------------------------------------------------------------------------------------------------"); CHKERRQ(ierr);
        ierr = tuMSGwarn (s.str());                                                 CHKERRQ(ierr);
        ierr = tuMSGstd ("------------------------------------------------------------------------------------------------"); CHKERRQ(ierr);
        s.str ("");
        s.clear ();
    }
    ScalarType sparsity;
    ierr = vecSparsity (tao_x, sparsity);                                  CHKERRQ (ierr);
    s << " "   << std::scientific << std::setprecision(5) << std::setfill('0') << std::setw(4) << its << std::setfill(' ')
      << "   " << std::scientific << std::setprecision(12) << std::setw(18) << J
      << "   " << std::scientific << std::setprecision(12) << std::setw(18) << PetscAbsReal (J - J_old)
      << "   " << std::scientific << std::setprecision(12) << std::setw(18) << gnorm
      << "   " << std::scientific << std::setprecision(12) << std::setw(18) << step
      << "   " << std::scientific << std::setprecision(12) << std::setw(18) << sparsity;
      if (itctx->n_misc_->diffusivity_inversion_) {
        ScalarType *x_ptr;
        ierr = VecGetArray(tao_x, &x_ptr);                                         CHKERRQ(ierr);
        s << "   " << std::scientific << std::setprecision(12) << std::setw(18) << x_ptr[itctx->n_misc_->np_];
        if (itctx->n_misc_->nk_ > 1) {
          s << "   " << std::scientific << std::setprecision(12) << std::setw(18) << x_ptr[itctx->n_misc_->np_ + 1]; }
        ierr = VecRestoreArray(tao_x, &x_ptr);                                     CHKERRQ(ierr);
      }


    ierr = tuMSGwarn (s.str());                                                    CHKERRQ(ierr);
    s.str ("");
    s.clear ();

    if (itctx->n_misc_->lambda_continuation_) {
      ScalarType sparsity = 0;
      ScalarType sparsity_old = 0;
      ScalarType threshold = itctx->n_misc_->target_sparsity_;
      if (its > 0) {
        //lambda continuation
        ierr = vecSparsity (tao_x, sparsity);
        ierr = vecSparsity (lsctx->x_work_2, sparsity_old);

        if (sparsity >= threshold) {//Sparse solution
          itctx->flag_sparse = true;
          itctx->lam_right = itctx->n_misc_->lambda_;
          itctx->n_misc_->lambda_ = 0.5 * (itctx->lam_left + itctx->lam_right);
          lsctx->lambda = itctx->n_misc_->lambda_;
        }

        if (sparsity < threshold && sparsity_old >= threshold && its > 1) {
          itctx->lam_left = itctx->n_misc_->lambda_;
          itctx->n_misc_->lambda_ = 0.5 * (itctx->lam_left + itctx->lam_right);
          lsctx->lambda = itctx->n_misc_->lambda_;
        }

        if (sparsity < threshold && sparsity_old < threshold && itctx->flag_sparse) {
          itctx->lam_left = itctx->n_misc_->lambda_;
          itctx->n_misc_->lambda_ = 0.5 * (itctx->lam_left + itctx->lam_right);
          lsctx->lambda = itctx->n_misc_->lambda_;
        }

        s << "Lambda continuation on: Lambda = : " << std::scientific << lsctx->lambda;
        ierr = tuMSGstd(s.str());                                             CHKERRQ(ierr);
        s.str(std::string());
        s.clear();
      }
    }

    //ierr = PetscPrintf (PETSC_COMM_WORLD, "\nKSP number of krylov iterations: %d\n", itctx->optfeedback_->nb_krylov_it);          CHKERRQ(ierr);
    //itctx->optfeedback_->nb_krylov_it = 0;

    //Gradient check begin
    //ierr = itctx->derivative_operators_->checkGradient (tao_x, itctx->data);
    //Gradient check end
    PetscFunctionReturn (0);
}

/* ------------------------------------------------------------------- */
/*
 optimizationMonitor    mointors the inner PCG Krylov solve to invert the Hessian
 input parameters:
  . KSP ksp          KSP solver object
	. PetscIntn        iteration number
	. PetscRela rnorm  l2-norm (preconditioned) of residual
  . void *ptr        optional user defined context
 */
PetscErrorCode hessianKSPMonitor (KSP ksp, PetscInt its, PetscReal rnorm, void *ptr) {
	PetscFunctionBegin;
	PetscErrorCode ierr = 0;

	Vec x; int maxit; ScalarType divtol, abstol, reltol;
	ierr = KSPBuildSolution (ksp,NULL,&x);
    ierr = KSPGetTolerances (ksp, &reltol, &abstol, &divtol, &maxit);             CHKERRQ(ierr);                                                             CHKERRQ(ierr);
	CtxInv *itctx = reinterpret_cast<CtxInv*>(ptr);     // get user context
	itctx->optfeedback_->nb_krylov_it++;                // accumulate number of krylov iterations

    std::stringstream s;
    if (its == 0) {
      s << std::setw(3)  << " PCG:" << " computing solution of hessian system (tol="
        << std::scientific << std::setprecision(5) << reltol << ")";
      ierr = tuMSGstd (s.str());                                                CHKERRQ(ierr);
      s.str (""); s.clear ();
    }
    s << std::setw(3)  << " PCG:" << std::setw(15) << " " << std::setfill('0') << std::setw(3)<< its
    << "   ||r||_2 = " << std::scientific << std::setprecision(5) << rnorm;
    ierr = tuMSGstd (s.str());                                                    CHKERRQ(ierr);
    s.str (""); s.clear ();

    // int ksp_itr;
    // ierr = KSPGetIterationNumber (ksp, &ksp_itr);                                 CHKERRQ (ierr);
    // ScalarType e_max, e_min;
    // if (ksp_itr % 10 == 0 || ksp_itr == maxit) {
    //   ierr = KSPComputeExtremeSingularValues (ksp, &e_max, &e_min);       CHKERRQ (ierr);
    //   s << "Condition number of hessian is: " << e_max / e_min << " | largest singular values is: " << e_max << ", smallest singular values is: " << e_min << std::endl;
    //   ierr = tuMSGstd (s.str());                                                    CHKERRQ(ierr);
    //   s.str (""); s.clear ();
    // }
	PetscFunctionReturn (0);
}

/* ------------------------------------------------------------------- */
/*
 constHessianKSPMonitor    mointors the PCG Krylov solve for the constant apx of initial hessian for lmvm
 input parameters:
  . KSP ksp          KSP solver object
	. PetscIntn        iteration number
	. PetscRela rnorm  l2-norm (preconditioned) of residual
  . void *ptr        optional user defined context
 */
PetscErrorCode constHessianKSPMonitor (KSP ksp, PetscInt its, PetscReal rnorm, void *ptr) {
	PetscFunctionBegin;
	PetscErrorCode ierr = 0;

	Vec x; int maxit; ScalarType divtol, abstol, reltol;
	ierr = KSPBuildSolution (ksp,NULL,&x);
  ierr = KSPGetTolerances (ksp, &reltol, &abstol, &divtol, &maxit);             CHKERRQ(ierr);                                                             CHKERRQ(ierr);
	CtxInv *itctx = reinterpret_cast<CtxInv*>(ptr);     // get user context

  std::stringstream s;
  if (its == 0) {
      s << std::setw(3)  << " PCG:" << " invert constant apx H = (beta Phi^T Phi) as initial guess for L-BFGS  (tol="
        << std::scientific << std::setprecision(5) << reltol << ")";
      ierr = tuMSGstd (s.str());                                                CHKERRQ(ierr);
      s.str (""); s.clear ();
  }
  s << std::setw(3)  << " PCG:" << std::setw(15) << " " << std::setfill('0') << std::setw(3)<< its
    << "   ||r||_2 = " << std::scientific << std::setprecision(5) << rnorm;
  ierr = tuMSGstd (s.str());                                                    CHKERRQ(ierr);
  s.str (""); s.clear ();
	PetscFunctionReturn (0);
}



/* ------------------------------------------------------------------- */
/*
 preKrylovSolve    preprocess right hand side and initial condition before entering
                   the krylov subspace method; in the context of numerical optimization
									 this means we preprocess the gradient and the incremental control variable

 input parameters:
  . KSP ksp       KSP solver object
	. Vec b         right hand side
	. Vec x         solution vector
  . void *ptr     optional user defined context
 */
PetscErrorCode preKrylovSolve (KSP ksp, Vec b, Vec x, void *ptr) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    PetscReal gnorm = 0., g0norm = 1., reltol, abstol = 0., divtol = 0., uppergradbound, lowergradbound;
    PetscInt maxit;
    int nprocs, procid;
    MPI_Comm_rank (PETSC_COMM_WORLD, &procid);
    MPI_Comm_size (PETSC_COMM_WORLD, &nprocs);

    CtxInv *itctx = reinterpret_cast<CtxInv*> (ptr);     // get user context
    ierr = VecNorm (b, NORM_2, &gnorm);                                                 CHKERRQ(ierr);   // compute gradient norm
    if(! itctx->is_ksp_gradnorm_set) {                // set initial gradient norm
    	itctx->ksp_gradnorm0 = gnorm;                 // for KSP Hessian solver
    	itctx->is_ksp_gradnorm_set = true;
    }
    g0norm = itctx->ksp_gradnorm0;                      // get reference gradient
    gnorm /= g0norm;                                    // normalize gradient
                                                        // get tolerances
    ierr = KSPGetTolerances (ksp, &reltol, &abstol, &divtol, &maxit);                   CHKERRQ(ierr);
    uppergradbound = 0.5;                               // assuming quadratic convergence
    lowergradbound = 1E-10;
    // user forcing sequence to estimate adequate tolerance for solution of
    //  KKT system (Eisenstat-Walker)
    if (itctx->optsettings_->fseqtype == QDFS) {
    	// assuming quadratic convergence (we do not solver more accurately than 12 digits)
    	reltol = PetscMax(lowergradbound, PetscMin(uppergradbound, gnorm));
    }
    else {
    	// assuming superlinear convergence (we do not solver  more accurately than 12 digitis)
    	reltol = PetscMax (lowergradbound, PetscMin (uppergradbound, std::sqrt(gnorm)));
    }
    // overwrite tolerances with estimate
    ierr = KSPSetTolerances (ksp, reltol, abstol, divtol, maxit);                       CHKERRQ(ierr);

    //if (procid == 0){
    //	std::cout << " ksp rel-tol (Eisenstat/Walker): " << reltol << ", grad0norm: " << g0norm<<", gnorm/grad0norm: " << gnorm << std::endl;
    //}
    PetscFunctionReturn (0);
}


/* Convergence tests used for L1 regularization: Relative change in objective and solution is
   monitored. Linesearch is user-defined with the tao solver */
PetscErrorCode checkConvergenceFun (Tao tao, void *ptr) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;

  PetscInt its, nl, ng;
  PetscInt iter, maxiter, miniter;
  PetscReal J, gnorm, step, gatol, grtol, gttol, g0norm, minstep, J_old;
  int verbosity;
  bool stop[2];
  std::stringstream ss, sc;
  Vec x = nullptr, g = nullptr;
  ierr = TaoGetSolutionVector(tao, &x);                                     CHKERRQ(ierr);
  TaoLineSearch ls = nullptr;
  TaoLineSearchConvergedReason ls_flag;
  ScalarType norm_g_inf;

  CtxInv *ctx = reinterpret_cast<CtxInv*> (ptr);     // get user context
  verbosity = ctx->optsettings_->verbosity;
  minstep = ctx->optsettings_->ls_minstep;
  miniter = ctx->optsettings_->newton_minit;
  // get tolerances
  #if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 7)
    ierr = TaoGetTolerances (tao, &gatol, &grtol, &gttol);                  CHKERRQ(ierr);
  #else
    ierr = TaoGetTolerances( tao, NULL, NULL, &gatol, &grtol, &gttol);      CHKERRQ(ierr);
  #endif

  // get line-search status
  ierr = TaoGetLineSearch(tao, &ls);                                          CHKERRQ (ierr);
  ierr = VecDuplicate (ctx->tumor_->p_, &g);                                  CHKERRQ(ierr);
  ierr = TaoLineSearchGetSolution (ls, x, &J, g, &step, &ls_flag);            CHKERRQ (ierr);
  // display line-search convergence reason
  ierr = dispLineSearchStatus (tao, ctx, ls_flag);                             CHKERRQ(ierr);
  ierr = TaoGetMaximumIterations (tao, &maxiter);                              CHKERRQ(ierr);
  ierr = TaoGetSolutionStatus (tao, &iter, &J, &gnorm, NULL, &step, NULL);     CHKERRQ(ierr);

  //Get ls context
  LSCtx *lsctx = (LSCtx*) ls->data;

  // update/set reference gradient (with p = initial-guess)
  #if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR < 9)
  if (ctx->update_reference_objective) {
    ScalarType g_percent = 0.1;
    Vec p0, dJ;
    ierr = VecDuplicate (ctx->tumor_->p_, &p0);                               CHKERRQ(ierr);
    ierr = VecDuplicate (ctx->tumor_->p_, &dJ);                               CHKERRQ(ierr);
    // ierr = VecSet (p0, ctx->n_misc_->p_scale_);                               CHKERRQ(ierr);
    ierr = VecSet (dJ, 0.);                                                   CHKERRQ(ierr);
    //Check for infeasible lambda values
    evaluateGradient(tao, x, dJ, (void*) ctx);
    ierr = VecNorm (dJ, NORM_INFINITY, &norm_g_inf);                           CHKERRQ (ierr);

    // if (ctx->n_misc_->lambda_ >= norm_g_inf) {
      ctx->n_misc_->lambda_ = norm_g_inf - g_percent * norm_g_inf;
      ctx->lam_right = ctx->n_misc_->lambda_;
      ctx->lam_left = 0;
      lsctx->lambda = ctx->n_misc_->lambda_;
    // }
    //Evaluate objective for reference using the correct lambda
    evaluateObjectiveFunction (tao, x, &ctx->optfeedback_->j0, (void*) ctx);
    std::stringstream s; s <<"updated reference objective for relative convergence criterion, ISTA L1 solver: " << ctx->optfeedback_->j0 << " ; regularization : " << ctx->n_misc_->lambda_;
    ctx->update_reference_objective = false;
    ierr = tuMSGstd(s.str());                                                 CHKERRQ(ierr);
    ierr = VecDestroy(&p0);                                                   CHKERRQ(ierr);
    ierr = VecDestroy(&dJ);                                                   CHKERRQ(ierr);
  }
  #endif

  // check for NaN value
  if (PetscIsInfOrNanReal(J)) {
    ierr = tuMSGwarn ("objective is NaN");                                      CHKERRQ(ierr);
    ierr = TaoSetConvergedReason (tao, TAO_DIVERGED_NAN);                       CHKERRQ(ierr);
    if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}
    PetscFunctionReturn (ierr);
  }
  // check for NaN value
  if (PetscIsInfOrNanReal(gnorm)) {
    ierr = tuMSGwarn("||g|| is NaN");                                           CHKERRQ(ierr);
    ierr = TaoSetConvergedReason(tao, TAO_DIVERGED_NAN);                        CHKERRQ(ierr);
    if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}
    PetscFunctionReturn(ierr);
  }
  // only check convergence criteria after a certain number of iterations
  ctx->optfeedback_->converged = false;
  //objective and solution convergence check
  //J_old : prev objective
  //J : current objective
  //J_ref : ref objective

  PetscReal J_ref = ctx->optfeedback_->j0;
  J_old = lsctx->J_old;
  ScalarType ftol = ctx->optsettings_->ftol;
  ScalarType norm_rel, norm;
  stop[0] = false; stop[1] = false;
  if (iter >= miniter) {
    if (verbosity > 1) {
      ss << "step size in linesearch: " << std::scientific << step;
      ierr = tuMSGstd(ss.str());                                             CHKERRQ(ierr);
      ss.str(std::string());
      ss.clear();
    }
    if (step < minstep) {
      ss << "step  = " << std::scientific << step << " < " << minstep << " = " << "bound";
      ierr = tuMSGwarn(ss.str()); CHKERRQ(ierr);
      ss.str(std::string());
      ss.clear();
      ierr = TaoSetConvergedReason(tao, TAO_CONVERGED_STEPTOL);              CHKERRQ(ierr);
      if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}
      PetscFunctionReturn(ierr);
    }
    if (ls_flag != 1 && ls_flag != 0 && ls_flag != 2) {
      ss << "step  = " << std::scientific << step << " < " << minstep << " = " << "bound";
      ierr = tuMSGwarn(ss.str()); CHKERRQ(ierr);
      ss.str(std::string());
      ss.clear();
      ierr = TaoSetConvergedReason(tao, TAO_DIVERGED_LS_FAILURE);
      if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}
      PetscFunctionReturn(ierr);
    }

    ierr = VecAXPY (lsctx->x_work_2, -1.0, x);                                CHKERRQ (ierr);
    ierr = VecNorm (lsctx->x_work_2, NORM_INFINITY, &norm_rel);               CHKERRQ (ierr);
    ierr = VecNorm (x, NORM_2, &norm);                                        CHKERRQ (ierr);

    ss.str(std::string());
    ss.clear();
    if (PetscAbsReal (J_old - J) < ftol * PetscAbsReal (1 + J_ref) && norm_rel < std::sqrt (ftol) * (1 + norm)) {  //L1 convergence check for objective and solution change
      ierr = TaoSetConvergedReason(tao, TAO_CONVERGED_USER);                  CHKERRQ(ierr);
      stop[0] = true;
    }
    ss  << "  " << stop[0] << "    |J_old - J| = " << std::setw(14)
        << std::right << std::scientific << std::abs (J_old - J) << "    <    "
        << std::left << std::setw(14) << ftol * PetscAbsReal (1 + J_ref) << " = " << "tol * |1 + J_ref| = " << ftol << " * |1 + " << J_ref << "|";
    ctx->convergence_message.push_back(ss.str());
    ss  << "  " << stop[0] << "    ||x_old - x||_inf = " << std::setw(14)
        << std::right << std::scientific << norm_rel << "    <    "
        << std::left << std::setw(14) << std::sqrt (ftol) * (1 + norm) << " = " << "sqrt (ftol) * (1 + ||x||_inf^k) = " << std::sqrt (ftol) << " * (1 + " << norm << ")";
    ctx->convergence_message.push_back(ss.str());
    if(verbosity >= 3) {
      ierr = tuMSGstd(ss.str());                                              CHKERRQ(ierr);
    }
    ss.str(std::string());
      ss.clear();
    // iteration number exceeds limit
    if (iter > maxiter) {
      ierr = TaoSetConvergedReason(tao, TAO_DIVERGED_MAXITS);                 CHKERRQ(ierr);
      stop[1] = true;
    }
    ss  << "  " << stop[1] << "     iter = " << std::setw(14)
        << std::right << iter  << "    >    "
        << std::left << std::setw(14) << maxiter << " = " << "maxiter";
    ctx->convergence_message.push_back(ss.str());
    if(verbosity >= 3) {
      ierr = tuMSGstd(ss.str());                                              CHKERRQ(ierr);
    }
    ss.str(std::string());
      ss.clear();
    // store objective function value
    ctx->jvalold = J;
    if (stop[0] || stop[1]) {
      ctx->optfeedback_->converged = true;
      if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}
      PetscFunctionReturn(ierr);
    }
  }

  // if we're here, we're good to go
  ierr = TaoSetConvergedReason (tao, TAO_CONTINUE_ITERATING);                 CHKERRQ(ierr);

  if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}

  PetscFunctionReturn (0);
}
/* ------------------------------------------------------------------- */
/*
 checkConvergenceGrad    checks convergence of the overall Gauß-Newton tumor inversion

 input parameters:
  . Tao tao       Tao solver object
  . void *ptr     optional user defined context
 */
PetscErrorCode checkConvergenceGrad (Tao tao, void *ptr) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    PetscInt its, nl, ng;
    PetscInt iter, maxiter, miniter;
    PetscReal J, gnorm, step, gatol, grtol, gttol, g0norm, minstep;
    bool stop[3];
    int verbosity;
    std::stringstream ss, sc;
    Vec x = nullptr, g = nullptr;
    ierr = TaoGetSolutionVector(tao, &x);                                     CHKERRQ(ierr);
    TaoLineSearch ls = nullptr;
    TaoLineSearchConvergedReason ls_flag;

    CtxInv *ctx = reinterpret_cast<CtxInv*> (ptr);     // get user context
    verbosity = ctx->optsettings_->verbosity;
    minstep = ctx->optsettings_->ls_minstep;
    miniter = ctx->optsettings_->newton_minit;
    // get tolerances
    #if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 7)
        ierr = TaoGetTolerances (tao, &gatol, &grtol, &gttol);                  CHKERRQ(ierr);
    #else
        ierr = TaoGetTolerances( tao, NULL, NULL, &gatol, &grtol, &gttol);      CHKERRQ(ierr);
    #endif

    // get line-search status
    nl = ctx->n_misc_->n_local_;
    ng = ctx->n_misc_->n_global_;
    ierr = TaoGetLineSearch(tao, &ls);                                          CHKERRQ (ierr);
    ierr = VecDuplicate (ctx->tumor_->p_, &g);                                  CHKERRQ(ierr);
    ierr = TaoLineSearchGetSolution(ls, x, &J, g, &step, &ls_flag);             CHKERRQ (ierr);
    // display line-search convergence reason
    ierr = dispLineSearchStatus(tao, ctx, ls_flag);                             CHKERRQ(ierr);
    ierr = TaoGetMaximumIterations(tao, &maxiter);                              CHKERRQ(ierr);
    ierr = TaoGetSolutionStatus(tao, &iter, &J, &gnorm, NULL, &step, NULL);     CHKERRQ(ierr);

    // update/set reference gradient (with p = zeros)
    #if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR < 9)
    if (ctx->update_reference_gradient) {
    	Vec dJ, p0;
    	ScalarType norm_gref = 0.;
    	ierr = VecDuplicate (ctx->tumor_->p_, &dJ);                               CHKERRQ(ierr);
      ierr = VecDuplicate (ctx->tumor_->p_, &p0);                               CHKERRQ(ierr);
    	ierr = VecSet (dJ, 0.);                                                   CHKERRQ(ierr);
      ierr = VecSet (p0, 0.);                                                   CHKERRQ(ierr);

      if (ctx->n_misc_->flag_reaction_inv_) {
        norm_gref = gnorm;
      } else {
    	  ierr = evaluateGradient(tao, p0, dJ, (void*) ctx);
    	  ierr = VecNorm (dJ, NORM_2, &norm_gref);                                  CHKERRQ(ierr);
      }
    	ctx->optfeedback_->gradnorm0 = norm_gref;
    	//ctx->gradnorm0 = gnorm;
    	ctx->update_reference_gradient = false;
      std::stringstream s; s <<"updated reference gradient for relative convergence criterion, Gauß-Newton solver: " << ctx->optfeedback_->gradnorm0;
      ierr = tuMSGstd(s.str());                                                 CHKERRQ(ierr);
    	ierr = VecDestroy(&dJ);                                                   CHKERRQ(ierr);
      ierr = VecDestroy(&p0);                                                   CHKERRQ(ierr);
    }
    #endif
    // get initial gradient
    g0norm = ctx->optfeedback_->gradnorm0;
    g0norm = (g0norm > 0.0) ? g0norm : 1.0;
    ctx->convergence_message.clear();
    // check for NaN value
    if (PetscIsInfOrNanReal(J)) {
  		ierr = tuMSGwarn ("objective is NaN");                                      CHKERRQ(ierr);
  		ierr = TaoSetConvergedReason (tao, TAO_DIVERGED_NAN);                       CHKERRQ(ierr);
      if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}
  		PetscFunctionReturn (ierr);
    }
    // check for NaN value
    if (PetscIsInfOrNanReal(gnorm)) {
  		ierr = tuMSGwarn("||g|| is NaN");                                           CHKERRQ(ierr);
  		ierr = TaoSetConvergedReason(tao, TAO_DIVERGED_NAN);                        CHKERRQ(ierr);
      if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}
  		PetscFunctionReturn(ierr);
    }
    //if(verbosity >= 1) {
    //	ierr = PetscPrintf (MPI_COMM_WORLD, "||g(x)|| / ||g(x0)|| = %6E, ||g(x0)|| = %6E \n", gnorm/g0norm, g0norm);
    //}
    // only check convergence criteria after a certain number of iterations
    stop[0] = false; stop[1] = false; stop[2] = false;
    ctx->optfeedback_->converged = false;
    if (iter >= miniter) {
    	if (verbosity > 1) {
    			ss << "step size in linesearch: " << std::scientific << step;
    			ierr = tuMSGstd(ss.str());                                            CHKERRQ(ierr);
    			ss.str(std::string());
                ss.clear();
    	}
    	if (step < minstep) {
    			ss << "step  = " << std::scientific << step << " < " << minstep << " = " << "bound";
    			ierr = tuMSGwarn(ss.str()); CHKERRQ(ierr);
    			ss.str(std::string());
                ss.clear();
    			ierr = TaoSetConvergedReason(tao, TAO_CONVERGED_STEPTOL);             CHKERRQ(ierr);
          if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}
    			PetscFunctionReturn(ierr);
    	}
      if (ls_flag != 1 && ls_flag != 0 && ls_flag != 2) {
        ss << "step  = " << std::scientific << step << " < " << minstep << " = " << "bound";
        ierr = tuMSGwarn(ss.str()); CHKERRQ(ierr);
        ss.str(std::string());
              ss.clear();
        ierr = TaoSetConvergedReason(tao, TAO_DIVERGED_LS_FAILURE);
        if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}
        PetscFunctionReturn(ierr);
      }
    	// ||g_k||_2 < tol*||g_0||
    	if (gnorm < gttol*g0norm) {
    			ierr = TaoSetConvergedReason(tao, TAO_CONVERGED_GTTOL);               CHKERRQ(ierr);
    			stop[0] = true;
    	}
    	ss << "  " << stop[0] << "    ||g|| = " << std::setw(14)
		     << std::right << std::scientific << gnorm << "    <    "
	       << std::left << std::setw(14) << gttol*g0norm << " = " << "tol";
    	ctx->convergence_message.push_back(ss.str());
      if(verbosity >= 3) {
    	  ierr = tuMSGstd(ss.str());                                              CHKERRQ(ierr);
      }
    	ss.str(std::string());
        ss.clear();
    	// ||g_k||_2 < tol
    	if (gnorm < gatol) {
			ierr = TaoSetConvergedReason(tao, TAO_CONVERGED_GATOL);                   CHKERRQ(ierr);
			stop[1] = true;
    	}
    	ss  << "  " << stop[1] << "    ||g|| = " << std::setw(14)
		      << std::right << std::scientific << gnorm << "    <    "
		      << std::left << std::setw(14) << gatol << " = " << "tol";
    	ctx->convergence_message.push_back(ss.str());
      if(verbosity >= 3) {
    	  ierr = tuMSGstd(ss.str());                                              CHKERRQ(ierr);
      }
    	ss.str(std::string());
        ss.clear();
    	// iteration number exceeds limit
    	if (iter > maxiter) {
			ierr = TaoSetConvergedReason(tao, TAO_DIVERGED_MAXITS);                    CHKERRQ(ierr);
			stop[2] = true;
    	}
    	ss  << "  " << stop[2] << "     iter = " << std::setw(14)
		      << std::right << iter  << "    >    "
		      << std::left << std::setw(14) << maxiter << " = " << "maxiter";
    	ctx->convergence_message.push_back(ss.str());
      if(verbosity >= 3) {
    	  ierr = tuMSGstd(ss.str());                                              CHKERRQ(ierr);
      }
    	ss.str(std::string());
        ss.clear();
    	// store objective function value
    	ctx->jvalold = J;
    	if (stop[0] || stop[1] || stop[2]) {
    		ctx->optfeedback_->converged = true;
        if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}
    		PetscFunctionReturn(ierr);
    	}

    }
    else {
		// if the gradient is zero, we should terminate immediately
		if (gnorm == 0) {
			ss << "||g|| = " << std::scientific << 0.0 << " < " << gatol  << " = " << "bound";
			ierr = tuMSGwarn(ss.str());                                               CHKERRQ(ierr);
			ss.str(std::string());
            ss.clear();
			ierr = TaoSetConvergedReason(tao, TAO_CONVERGED_GATOL);                   CHKERRQ(ierr);
      if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}
			PetscFunctionReturn(ierr);
		}
    }
    // if we're here, we're good to go
    ierr = TaoSetConvergedReason (tao, TAO_CONTINUE_ITERATING);                 CHKERRQ(ierr);

    if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}

    PetscFunctionReturn (0);
}

/* ------------------------------------------------------------------- */
/*
 checkConvergenceGradObj    checks convergence of the overall Gauß-Newton tumor inversion

 input parameters:
  . Tao tao       Tao solver object
  . void *ptr     optional user defined context
 */
PetscErrorCode checkConvergenceGradObj (Tao tao, void *ptr) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    int procid, nprocs;
    MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank (MPI_COMM_WORLD, &procid);

    PetscInt iter, maxiter, miniter, iterbound;
    PetscReal jx, jxold, gnorm, step, gatol, grtol, gttol, g0norm, gtolbound, minstep, theta, normx, normdx, tolj, tolx, tolg;
    const int nstop = 7;
    bool stop[nstop];
    std::stringstream ss;
    Vec x;

    CtxInv *ctx = reinterpret_cast<CtxInv*>(ptr);     // get user context
    // get minstep and miniter
    minstep = ctx->optsettings_->ls_minstep;
    miniter = ctx->optsettings_->newton_minit;
    iterbound = ctx->optsettings_->iterbound;
    // get lower bound for gradient
    gtolbound = ctx->optsettings_->gtolbound;

    #if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 7)
        ierr = TaoGetTolerances(tao, &gatol, &grtol, &gttol);                               CHKERRQ(ierr);
    #else
        ierr = TaoGetTolerances(tao, NULL, NULL, &gatol, &grtol, &gttol);                   CHKERRQ(ierr);
    #endif

    ierr = TaoGetMaximumIterations(tao, &maxiter);                                          CHKERRQ(ierr);
    if (maxiter > iterbound) iterbound = maxiter;

    ierr = TaoGetMaximumIterations(tao, &maxiter);                                          CHKERRQ(ierr);
    ierr = TaoGetSolutionStatus(tao, &iter, &jx, &gnorm, NULL, &step, NULL);                CHKERRQ(ierr);
    ierr = TaoGetSolutionVector(tao, &x);                                                   CHKERRQ(ierr);

    
    // update/set reference gradient (with p = zeros)
    #if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR < 9)
    if(ctx->update_reference_gradient) {
      Vec dJ, p0;
      ScalarType norm_gref = 0.;
      ierr = VecDuplicate (ctx->tumor_->p_, &dJ);                               CHKERRQ(ierr);
      ierr = VecDuplicate (ctx->tumor_->p_, &p0);                               CHKERRQ(ierr);
      ierr = VecSet (dJ, 0.);                                                   CHKERRQ(ierr);
      ierr = VecSet (p0, 0.);                                                   CHKERRQ(ierr);
      // evaluateGradient(tao, x, dJ, (void*) ctx);
      evaluateObjectiveFunctionAndGradient (tao, p0, &ctx->optfeedback_->j0, dJ, (void*) ctx);
    	ierr = VecNorm (dJ, NORM_2, &norm_gref); CHKERRQ(ierr);
    	ctx->optfeedback_->gradnorm0 = norm_gref;
      // evaluateObjectiveFunction (tao, x, &ctx->optfeedback_->j0, (void*) ctx);

      std::stringstream s; s <<"updated reference objective for relative convergence criterion: " << ctx->optfeedback_->j0;
    	ctx->update_reference_gradient = false;
      ierr = tuMSGstd(s.str());                                                     CHKERRQ(ierr);
      s.str(std::string());
      s.clear();
      s <<"updated reference gradient for relative convergence criterion, Gauß-Newton solver: " << ctx->optfeedback_->gradnorm0;
    	ierr = tuMSGstd(s.str());                                                          CHKERRQ(ierr);
      s.str(std::string());
      s.clear();
      ierr = VecDestroy(&dJ);                                                            CHKERRQ(ierr);
      ierr = VecDestroy(&p0);                                                            CHKERRQ(ierr);
    }
    #endif
    // get initial gradient
    g0norm = ctx->optfeedback_->gradnorm0;
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
    theta = 1.0 + std::abs(ctx->optfeedback_->j0);
    // compute norm(\Phi x^k - \Phi x^k-1) and norm(\Phi x^k)
    // ierr = ctx->tumor_->phi_->apply (ctx->tmp, x);                                         CHKERRQ(ierr);  // comp \Phi x
    // ierr = VecNorm (ctx->tmp, NORM_2, &normx);                                             CHKERRQ(ierr);  // comp norm \Phi x
    // ierr = VecAXPY(ctx->c0old, -1, ctx->tmp);                                              CHKERRQ(ierr);  // comp dx
    // ierr = VecNorm (ctx->c0old, NORM_2, &normdx);                                          CHKERRQ(ierr);  // comp norm \Phi dx
    // ierr = VecCopy(ctx->tmp, ctx->c0old);                                                  CHKERRQ(ierr);  // save \Phi x
    ierr = VecNorm (x, NORM_2, &normx);                                             CHKERRQ(ierr);  // comp norm x
    ierr = VecAXPY (ctx->x_old, -1.0, x);                                                  CHKERRQ(ierr);  // comp dx
    ierr = VecNorm (ctx->x_old, NORM_2, &normdx);                                   CHKERRQ(ierr);  // comp norm dx
    ierr = VecCopy (x, ctx->x_old);                                                        CHKERRQ(ierr);  // save old x
    // get old objective function value
    jxold = ctx->jvalold;

    ctx->convergence_message.clear();
    // check for NaN value
    if (PetscIsInfOrNanReal(jx)) {
  		ierr = tuMSGwarn("objective is NaN");                                             CHKERRQ(ierr);
  		ierr = TaoSetConvergedReason(tao, TAO_DIVERGED_NAN);                              CHKERRQ(ierr);
  		PetscFunctionReturn(ierr);
    }
    // check for NaN value
    if (PetscIsInfOrNanReal(gnorm)) {
  		ierr = tuMSGwarn("||g|| is NaN");                                                 CHKERRQ(ierr);
  		ierr = TaoSetConvergedReason(tao, TAO_DIVERGED_NAN);                              CHKERRQ(ierr);
  		PetscFunctionReturn(ierr);
    }

    // ierr = PetscPrintf (MPI_COMM_WORLD, "||g(x)|| / ||g(x0)|| = %6E, ||g(x0)|| = %6E \n", gnorm/g0norm, g0norm);
    ctx->optfeedback_->converged = false;
    // initialize flags for stopping conditions
    for (int i = 0; i < nstop; i++)
        stop[i] = false;
    // only check convergence criteria after a certain number of iterations
    if (iter >= miniter && iter > 1) {
    	if (step < minstep) {
			ierr = TaoSetConvergedReason(tao, TAO_CONVERGED_STEPTOL);                    CHKERRQ(ierr);
			PetscFunctionReturn(ierr);
    	}
    	// |j(x_{k-1}) - j(x_k)| < tolj*abs(1+J)
    	if (std::abs(jxold-jx) < tolj*theta) {
    		stop[0] = true;
    	}
    	ss << "  " << stop[0] << "    |dJ|  = " << std::setw(18)
        << std::right << std::scientific << std::abs(jxold-jx) << "    <    "
        << std::left << std::setw(18) << tolj*theta << " = " << "tol*|1+J|";
    	ctx->convergence_message.push_back(ss.str());
    	ierr = tuMSGstd(ss.str());                                                     CHKERRQ(ierr);
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
    	ierr = tuMSGstd(ss.str());                                                     CHKERRQ(ierr);
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
    	ierr = tuMSGstd(ss.str());                                                     CHKERRQ(ierr);
    	ss.str(std::string());
        ss.clear();
    	// ||g_k||_2 < tol
    	if (gnorm < gatol) {
    			stop[3] = true;
    	}
    	ss  << "  " << stop[3] << "    ||g|| = " << std::setw(18)
    	<< std::right << std::scientific << gnorm << "    <    "
    	<< std::left << std::setw(18) << gatol << " = " << "tol";
    	ctx->convergence_message.push_back(ss.str());
    	ierr = tuMSGstd(ss.str());                                                     CHKERRQ(ierr);
    	ss.str(std::string());
        ss.clear();

    	if (gnorm < gttol*g0norm) {
          stop[4] = true;
      }
    	ss << "  " << stop[0] << "    ||g|| = " << std::setw(18)
         << std::right << std::scientific << gnorm << "    <    "
         << std::left << std::setw(18) << gttol*g0norm << " = " << "tol";
      ctx->convergence_message.push_back(ss.str());
    	ierr = tuMSGstd(ss.str());                                                     CHKERRQ(ierr);
    	ss.str(std::string());
        ss.clear();

    	if (iter > maxiter) {
    			stop[5] = true;
    	}
    	ss  << "  " << stop[5] << "    iter  = " << std::setw(18)
		  << std::right << iter  << "    >    "
		  << std::left << std::setw(18) << maxiter << " = " << "maxiter";
    	ctx->convergence_message.push_back(ss.str());
    	ierr = tuMSGstd(ss.str());                                                     CHKERRQ(ierr);
    	ss.str(std::string());
        ss.clear();

    // 	if (iter > iterbound) {
    // 			stop[6] = true;
    // 	}
    // 	ss  << "  " << stop[6] << "    iter  = " << std::setw(14)
		  // << std::right << iter  << "    >    "
		  // << std::left << std::setw(14) << iterbound << " = " << "iterbound";
    // 	ctx->convergence_message.push_back(ss.str());
    // 	ierr = tuMSGstd(ss.str());                                                     CHKERRQ(ierr);
    // 	ss.str(std::string());
    //     ss.clear();

    	// store objective function value
    	ctx->jvalold = jx;

    	if (stop[0] && stop[1] && stop[2]) {
			ierr = TaoSetConvergedReason(tao, TAO_CONVERGED_USER);                      CHKERRQ(ierr);
			ctx->optfeedback_->converged = true;
			PetscFunctionReturn(ierr);
    	} else if (stop[3]) {
			ierr = TaoSetConvergedReason(tao, TAO_CONVERGED_GATOL);                     CHKERRQ(ierr);
			ctx->optfeedback_->converged = true;
			PetscFunctionReturn(ierr);
    	} else if (stop[4]) {
			ierr = TaoSetConvergedReason(tao, TAO_CONVERGED_GTTOL);               CHKERRQ(ierr);
			ctx->optfeedback_->converged = true;
			PetscFunctionReturn(ierr);
    	} else if (stop[5]) {
			ierr = TaoSetConvergedReason(tao, TAO_DIVERGED_MAXITS);                     CHKERRQ(ierr);
			ctx->optfeedback_->converged = true;
			PetscFunctionReturn(ierr);
    	}
    }
    else {
    	// if the gradient is zero, we should terminate immediately
    	if (gnorm < gatol) {
			ss << "||g|| = " << std::scientific << " < " << gatol;
			ierr = tuMSGwarn(ss.str());                                                 CHKERRQ(ierr);
			ss.str(std::string());
            ss.clear();
			ierr = TaoSetConvergedReason(tao, TAO_CONVERGED_GATOL);                     CHKERRQ(ierr);

			PetscFunctionReturn(ierr);
    	}
      // store objective function value
      ctx->jvalold = jx;
    }

    // if we're here, we're good to go
    ierr = TaoSetConvergedReason(tao, TAO_CONTINUE_ITERATING);                          CHKERRQ(ierr);

    PetscFunctionReturn (0);
}


PetscErrorCode dispTaoConvReason (TaoConvergedReason flag, std::string &msg) {
	PetscFunctionBegin;
	PetscErrorCode ierr = 0;
	switch (flag) {
      case TAO_CONVERGED_GATOL : {
          msg = "solver converged: ||g(x)|| <= tol";
          ierr = tuMSGwarn(msg);                                    CHKERRQ(ierr);
          break;
      }
      case TAO_CONVERGED_GRTOL : {
          msg = "solver converged: ||g(x)||/J(x) <= tol";
          ierr = tuMSGwarn(msg);                                    CHKERRQ(ierr);
          break;
      }
      case TAO_CONVERGED_GTTOL : {
          msg = "solver converged: ||g(x)||/||g(x0)|| <= tol";
          ierr = tuMSGwarn(msg);                                    CHKERRQ(ierr);
          break;
      }
      case TAO_CONVERGED_STEPTOL : {
          msg = "step size too small";
          ierr = tuMSGwarn(msg);                                    CHKERRQ(ierr);
          break;
      }
      case TAO_CONVERGED_MINF : {
          msg = "objective value to small";
          ierr = tuMSGwarn(msg);                                    CHKERRQ(ierr);
          break;
      }
      case TAO_CONVERGED_USER : {
          msg = "solver converged user";
          ierr = tuMSGwarn(msg);                                    CHKERRQ(ierr);
          break;
      }
      case TAO_DIVERGED_MAXITS : {
          msg = "maximum number of iterations reached";
          ierr = tuMSGwarn(msg);                                    CHKERRQ(ierr);
          break;
      }
      case TAO_DIVERGED_NAN : {
          msg = "numerical problems (NAN detected)";
          ierr = tuMSGwarn(msg);                                    CHKERRQ(ierr);
          break;
      }
      case TAO_DIVERGED_MAXFCN : {
          msg = "maximal number of function evaluations reached";
          ierr = tuMSGwarn(msg);                                    CHKERRQ(ierr);
          break;
      }
      case TAO_DIVERGED_LS_FAILURE : {
          msg = "line search failed";
          ierr = tuMSGwarn(msg);                                    CHKERRQ(ierr);
          break;
      }
      case TAO_DIVERGED_TR_REDUCTION : {
          msg = "trust region failed";
          ierr = tuMSGwarn(msg);                                    CHKERRQ(ierr);
          break;
      }
      case TAO_DIVERGED_USER : {
          msg = "user defined divergence criterion met";
          ierr = tuMSGwarn(msg);                                    CHKERRQ(ierr);
          break;
      }
      case TAO_CONTINUE_ITERATING : {
          // display nothing
          break;
      }
      default : {
          msg = "convergence reason not defined";
          ierr = tuMSGwarn(msg);                                    CHKERRQ(ierr);
          break;
      }
  }
	PetscFunctionReturn (0);
}

PetscErrorCode dispLineSearchStatus(Tao tao, void* ptr, TaoLineSearchConvergedReason flag) {
    PetscErrorCode ierr = 0;
    std::string msg;
    PetscFunctionBegin;

    switch(flag) {
        case TAOLINESEARCH_FAILED_INFORNAN:
        {
            msg = "linesearch: function evaluation gave INF or NaN";
            ierr = tuMSGwarn(msg); CHKERRQ(ierr);
            break;
        }
        case TAOLINESEARCH_FAILED_BADPARAMETER:
        {
            msg = "linesearch: bad parameter detected";
            ierr = tuMSGwarn(msg); CHKERRQ(ierr);
            break;
        }
        case TAOLINESEARCH_FAILED_ASCENT:
        {
            msg = "linesearch: search direction is not a descent direction";
            ierr = tuMSGwarn(msg); CHKERRQ(ierr);
            break;
        }
        case TAOLINESEARCH_HALTED_MAXFCN:
        {
            msg = "linesearch: maximum number of function evaluations reached";
            ierr = tuMSGwarn(msg); CHKERRQ(ierr);
            break;
        }
        case TAOLINESEARCH_HALTED_UPPERBOUND:
        {
            msg = "linesearch: step size reached upper bound";
            ierr = tuMSGwarn(msg); CHKERRQ(ierr);
            break;
        }
        case TAOLINESEARCH_HALTED_LOWERBOUND:
        {
            msg = "linesearch: step size reached lower bound";
            ierr = tuMSGwarn(msg); CHKERRQ(ierr);
            break;
        }
        case TAOLINESEARCH_HALTED_RTOL:
        {
            msg = "linesearch: range of uncertainty is smaller than given tolerance";
            ierr = tuMSGwarn(msg); CHKERRQ(ierr);
            break;
        }
        case TAOLINESEARCH_HALTED_OTHER:
        {
            msg = "linesearch: stopped (other)";
            ierr = tuMSGwarn(msg); CHKERRQ(ierr);
            break;
        }
        case TAOLINESEARCH_CONTINUE_ITERATING:
        {
            // do nothing, cause everything's fine
            break;
        }
        case TAOLINESEARCH_SUCCESS:
        {
            msg = "linesearch: successful";
            ierr = tuMSGstd(msg); CHKERRQ(ierr);
            break;
        }
        default:
        {
            msg = "linesearch: status not defined";
            ierr = tuMSGwarn(msg); CHKERRQ(ierr);
            break;
        }
    }
    PetscFunctionReturn(0);
}

PetscErrorCode InvSolver::setTaoOptions (Tao tao, CtxInv *ctx) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    TaoLineSearch linesearch;        // line-search object
    std::string msg;

    PetscReal minstep;
    minstep = std::pow (2.0, 20.0);
    minstep = 1.0 / minstep;
    itctx_->optsettings_->ls_minstep = minstep;

    if (itctx_->n_misc_->regularization_norm_ == L1) {
      ierr = TaoSetType (tao, "tao_L1");   CHKERRQ (ierr);
      TaoLineSearch ls;
      ierr = TaoGetLineSearch(tao_, &ls);                                          CHKERRQ (ierr);
      LSCtx *lsctx = (LSCtx*) ls->data;
      lsctx->lambda = ctx->n_misc_->lambda_;
      ls->stepmin = minstep;
    } else {
      if (itctx_->optsettings_->newtonsolver == QUASINEWTON)  {
        ierr = TaoSetType (tao, "blmvm");   CHKERRQ(ierr);   // set TAO solver type
      } else {
        ierr = TaoSetType (tao, "nls");    CHKERRQ(ierr);  // set TAO solver type
      }

      PetscBool flag = PETSC_FALSE;

      #if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 7)
          PetscOptionsHasName (NULL, NULL, "-tao_nls_pc_type", &flag);
          if (flag == PETSC_FALSE)
              PetscOptionsSetValue (NULL, "-tao_nls_pc_type", "none");
      #else
          PetscOptionsHasName (NULL, "-tao_nls_pc_type", &flag);
          if (flag == PETSC_FALSE)
              PetscOptionsSetValue ("-tao_nls_pc_type", "none");
      #endif
      flag = PETSC_FALSE;
      #if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 7)
          PetscOptionsHasName (NULL, NULL, "-tao_nls_ksp_type", &flag);
          if (flag == PETSC_FALSE)
          	PetscOptionsSetValue (NULL, "-tao_nls_ksp_type", "cg");
      #else
          PetscOptionsHasName (NULL, "-tao_nls_ksp_type", &flag);
          if (flag == PETSC_FALSE)
      	   PetscOptionsSetValue ("-tao_nls_ksp_type", "cg");
      #endif
      flag = PETSC_FALSE;
      #if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 7)
          PetscOptionsHasName (NULL, NULL, "-tao_ntr_pc_type", &flag);
          if (flag == PETSC_FALSE)
              PetscOptionsSetValue (NULL, "-tao_ntr_pc_type", "none");
      #else
          PetscOptionsHasName (NULL, "-tao_ntr_pc_type", &flag);
          if (flag == PETSC_FALSE)
              PetscOptionsSetValue ("-tao_ntr_pc_type", "none");
      #endif
    }

    // parse options user has set
    ierr = TaoSetFromOptions (tao);                                                 CHKERRQ(ierr);
    // set the initial vector
    ierr = TaoSetInitialVector (tao, ctx->tumor_->p_);                              CHKERRQ(ierr);
    // set routine for evaluating the objective
    ierr = TaoSetObjectiveRoutine (tao, evaluateObjectiveFunction, (void*) ctx);    CHKERRQ(ierr);
    // set routine for evaluating the Gradient
    ierr = TaoSetGradientRoutine (tao, evaluateGradient, (void*) ctx);              CHKERRQ(ierr);
    // set the routine to evaluate the objective and compute the gradient
    ierr = TaoSetObjectiveAndGradientRoutine (tao, evaluateObjectiveFunctionAndGradient, (void*) ctx);  CHKERRQ(ierr);
    // set monitor function
    if (itctx_->n_misc_->regularization_norm_ == L1) {
      ierr = TaoSetMonitor (tao, optimizationMonitorL1, (void *) ctx, NULL);        CHKERRQ(ierr);
    }
    else {
      ierr = TaoSetMonitor (tao, optimizationMonitor, (void *) ctx, NULL);          CHKERRQ(ierr);
    }
    // Lower and Upper Bounds
    Vec lower_bound;
    ierr = VecDuplicate (ctx->tumor_->p_, &lower_bound);                            CHKERRQ (ierr);
    ierr = VecSet (lower_bound, 0.);                                                CHKERRQ (ierr);
    Vec upper_bound;
    ierr = VecDuplicate (ctx->tumor_->p_, &upper_bound);                            CHKERRQ (ierr);
    ierr = VecSet (upper_bound, PETSC_INFINITY);                                    CHKERRQ (ierr);

    // alternate scheme bounds setting for max tumor location
    // ScalarType *lb_ptr, *ub_ptr;
    // if (itctx_->n_misc_->flag_reaction_inv_) {
    //   ierr = VecGetArray (upper_bound, &ub_ptr);                                    CHKERRQ (ierr);
    //   ierr = VecGetArray (lower_bound, &lb_ptr);                                    CHKERRQ (ierr);

    //   // set the ub to 1 for all p values
    //   for (int i = 0; i < itctx_->n_misc_->np_; i++) {
    //     ub_ptr[i] = itctx_->n_misc_->ic_max_;
    //   }

    //   // set the lower bound only for max tumor location to 0.9
    //   lb_ptr[itctx_->n_misc_->max_p_location_] = itctx_->n_misc_->ic_max_ - 0.1;

    //   // set upper bound of reaction coeff to current guess
    //   ub_ptr[itctx_->n_misc_->np_ + itctx_->n_misc_->nk_] = itctx_->n_misc_->rho_;

    //   ierr = VecRestoreArray (upper_bound, &ub_ptr);                                CHKERRQ (ierr);
    //   ierr = VecRestoreArray (lower_bound, &lb_ptr);                                CHKERRQ (ierr);
      
    // } 

    ierr = TaoSetVariableBounds(tao, lower_bound, upper_bound);                     CHKERRQ (ierr);
    ierr = VecDestroy (&lower_bound);                                               CHKERRQ (ierr);
    ierr = VecDestroy (&upper_bound);                                               CHKERRQ (ierr);



    // TAO type from user input
    #if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 9)
      TaoType taotype = NULL;
      ierr = TaoGetType (tao, &taotype);                                            CHKERRQ(ierr);
    #else
      const TaoType taotype;
      ierr = TaoGetType (tao, &taotype);                                            CHKERRQ(ierr);
    #endif
    if (strcmp(taotype, "nls") == 0) {
        msg = " limited memory variable metric method (unconstrained) selected\n";
    } else if (strcmp(taotype, "ntr") == 0) {
        msg = " Newton's method with trust region for unconstrained minimization\n";
    } else if (strcmp(taotype, "ntl") == 0) {
        msg = " Newton's method with trust region, line search for unconstrained minimization\n";
    } else if (strcmp(taotype, "nls") == 0) {
        msg = " Newton's method (line search; unconstrained) selected\n";
    } else if (strcmp(taotype, "ntr") == 0) {
        msg = " Newton's method (trust region; unconstrained) selected\n";
    } else if (strcmp(taotype, "fd_test") == 0) {
        msg = " gradient test selected\n";
    } else if (strcmp(taotype, "cg") == 0) {
        msg = " CG selected\n";
    } else if (strcmp(taotype, "tron") == 0) {
        msg = "  Newton Trust Region method chosen\n";
    } else if (strcmp(taotype, "blmvm") == 0) {
        msg = "  Limited memory variable metric method chosen\n";
    } else if (strcmp(taotype, "gpcg") == 0) {
        msg = " Newton Trust Region method for quadratic bound constrained minimization\n";
    } else if (strcmp(taotype, "tao_L1") == 0) {
        msg = " User defined solver for L1 minimization\n";
    } else {
        msg = " numerical optimization method not supported (setting default: LMVM)\n";
        ierr = TaoSetType (tao, "lmvm");                                          CHKERRQ(ierr);
    }
    // set tolerances
    // if (itctx_->n_misc_->regularization_norm_ == wL2) ctx->optsettings_->opttolgrad = 1e-6;
    #if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 7)
        ierr = TaoSetTolerances (tao, ctx->optsettings_->gatol, ctx->optsettings_->grtol, ctx->optsettings_->opttolgrad); CHKERRQ(ierr);
    #else
        ierr = TaoSetTolerances (tao, 1E-12, 1E-12, ctx->optsettings_->gatol, ctx->optsettings_->grtol, ctx->optsettings_->opttolgrad); CHKERRQ(ierr);
    #endif
    if (itctx_->n_misc_->regularization_norm_ == L1) {
      ierr = TaoSetMaximumIterations (tao, ctx->optsettings_->gist_maxit); CHKERRQ(ierr);
    }
    else {
      ierr = TaoSetMaximumIterations (tao, ctx->optsettings_->newton_maxit); CHKERRQ(ierr);
    }

    // if (itctx_->n_misc_->flag_reaction_inv_) {
    //   ierr = TaoSetMaximumIterations (tao, 200); CHKERRQ(ierr);
    // }

    if (itctx_->n_misc_->regularization_norm_ == L1) {
      ierr = TaoSetConvergenceTest (tao, checkConvergenceFun, ctx);                  CHKERRQ(ierr);
    }
    else {
      ierr = TaoSetConvergenceTest (tao, checkConvergenceGrad, ctx);                 CHKERRQ(ierr);
      // ierr = TaoSetConvergenceTest(tao, checkConvergenceGradObj, ctx);              CHKERRQ(ierr);
    }

    if (!itctx_->n_misc_->regularization_norm_ == L1) {  //Set other preferences for standard tao solvers
      // set linesearch (only for gauß-newton, lmvm uses more-thuente type line-search automatically)
      ierr = TaoGetLineSearch (tao, &linesearch);                                   CHKERRQ(ierr);
      linesearch->stepmin = minstep;
      if(itctx_->optsettings_->newtonsolver == GAUSSNEWTON) {
        ierr = TaoLineSearchSetType (linesearch, "armijo");                         CHKERRQ(ierr);
      }
      ierr = TaoLineSearchSetOptionsPrefix (linesearch,"tumor_");                    CHKERRQ(ierr);
      std::stringstream s;
      tuMSGstd(" parameters (optimizer):");
      tuMSGstd(" tolerances (stopping conditions):");
      s << "   gatol: "<< ctx->optsettings_->gatol;  /*pout(s.str(), cplctx->_fileOutput);*/ tuMSGstd(s.str()); s.str(""); s.clear();
      s << "   grtol: "<< ctx->optsettings_->grtol;  /*pout(s.str(), cplctx->_fileOutput);*/ tuMSGstd(s.str()); s.str(""); s.clear();
      s << "   gttol: "<< ctx->optsettings_->opttolgrad;  /*pout(s.str(), cplctx->_fileOutput);*/ tuMSGstd(s.str()); s.str(""); s.clear();

      ierr = TaoSetFromOptions(tao);                                                CHKERRQ(ierr);
      /* === set the KSP Krylov solver settings === */
      KSP ksp = PETSC_NULL;

      if (itctx_->optsettings_->newtonsolver == QUASINEWTON)  {
        // if (use_intial_hessian_lmvm_) {
        //   // get the ksp of H0 initial matrix
        //   ierr = TaoLMVMGetH0KSP(tao, &ksp);                                        CHKERRQ(ierr);
        //   if (ksp != PETSC_NULL) {
        //       ierr = KSPSetOptionsPrefix(ksp, "init-hessian_");                     CHKERRQ(ierr);
        //       // set default tolerance to 1E-6
        //       ierr = KSPSetTolerances(ksp, 1E-6, PETSC_DEFAULT, PETSC_DEFAULT, ctx->optsettings_->krylov_maxit); CHKERRQ(ierr);
        //       ierr = KSPMonitorSet(ksp, constHessianKSPMonitor,ctx, 0);              CHKERRQ(ierr);
        //   }
        //}
      } else {
        // get the ksp of the optimizer (use gauss-newton-krylov)
        ierr = TaoGetKSP(tao, &ksp);                                                CHKERRQ(ierr);
        if (ksp != PETSC_NULL) {
            ierr = KSPSetOptionsPrefix(ksp, "hessian_");                            CHKERRQ(ierr);
            // set default tolerance to 1E-6
            ierr = KSPSetTolerances(ksp, 1E-6, PETSC_DEFAULT, PETSC_DEFAULT, ctx->optsettings_->krylov_maxit); CHKERRQ(ierr);
            // to use Eisenstat/Walker convergence crit.
            KSPSetPreSolve (ksp, preKrylovSolve, ctx);                              CHKERRQ(ierr);
            ierr = KSPMonitorSet(ksp, hessianKSPMonitor,ctx, 0);                    CHKERRQ(ierr);
            // ierr = KSPSetComputeSingularValues(ksp, PETSC_TRUE);                CHKERRQ (ierr);  // To compute the condition number
            ierr = KSPSetFromOptions (ksp);                                     CHKERRQ (ierr);
        }
        // set the preconditioner (we check if KSP exists, as there are also
        // solvers that do not require a KSP solve (BFGS and friends))
        if (ksp != PETSC_NULL) {
        	PC pc;
        	ierr = KSPGetPC(ksp, &pc);                                                CHKERRQ(ierr);
        	ierr = PCSetType (pc, PCSHELL);                                           CHKERRQ(ierr);
        	ierr = PCShellSetApply(pc, preconditionerMatVec);                         CHKERRQ(ierr);
        	ierr = PCShellSetContext(pc, ctx);                                        CHKERRQ(ierr);
        }
      }
    }
    PetscFunctionReturn (0);
}
