#include "InvSolver.h"
#include "petsctao.h"
#include <iostream>
#include <limits>
#include "DerivativeOperators.h"
#include "PdeOperators.h"
#include "Utils.h"

InvSolver::InvSolver (std::shared_ptr <DerivativeOperators> derivative_operators, std::shared_ptr <NMisc> n_misc, std::shared_ptr <Tumor> tumor) :
initialized_(false),
data_(),
data_gradeval_(),
optsettings_(),
optfeedback_(),
itctx_() {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    tao_  = nullptr;
    H_    = nullptr;
    prec_ = nullptr;
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
    // allocate memory for prec_
    ierr = VecDuplicate (tumor->p_, &prec_);                                             CHKERRQ(ierr);
    ierr = VecSet (prec_, 0.0);                                                          CHKERRQ(ierr);
    // set up routine to compute the hessian matrix vector product
    if (H_ == nullptr) {
        int np = n_misc->np_;
        ierr = MatCreateShell (MPI_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, np, np, (void*) itctx_.get(), &H_);   CHKERRQ(ierr);
        ierr = MatShellSetOperation (H_, MATOP_MULT, (void (*)(void))hessianMatVec);     CHKERRQ(ierr);
        ierr = MatSetOption (H_, MAT_SYMMETRIC, PETSC_TRUE);                             CHKERRQ(ierr);
    }
    // create TAO solver object
    if( tao_ == nullptr) {
        ierr = TaoCreate (MPI_COMM_WORLD, &tao_);
    }
    initialized_ = true;
    PetscFunctionReturn (0);
}

PetscErrorCode InvSolver::setParams (std::shared_ptr<DerivativeOperators> derivative_operators, std::shared_ptr<NMisc> n_misc, std::shared_ptr<Tumor> tumor, bool npchanged) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    itctx_->derivative_operators_ = derivative_operators;
    itctx_->n_misc_ = n_misc;
    itctx_->tumor_ = tumor;
    // re-allocate memory
    if (npchanged){                                      // re-allocate memory for prec_
      if(prec_ != nullptr) {ierr = VecDestroy (&prec_);                                    CHKERRQ(ierr);}
      ierr = VecDuplicate (tumor->p_, &prec_);                                             CHKERRQ(ierr);
      ierr = VecSet (prec_, 0.0);                                                          CHKERRQ(ierr);
                                                        // re-allocate memory for H
      if(H_ != nullptr) {ierr = MatDestroy (&H_);                                          CHKERRQ(ierr);}
      ierr = MatDestroy (&H_);
      int np = n_misc->np_;
      ierr = MatCreateShell (MPI_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, np, np, (void*) itctx_.get(), &H_);   CHKERRQ(ierr);
      ierr = MatShellSetOperation (H_, MATOP_MULT, (void (*)(void))hessianMatVec);        CHKERRQ(ierr);
      ierr = MatSetOption (H_, MAT_SYMMETRIC, PETSC_TRUE);                                CHKERRQ(ierr);
    }
    PetscFunctionReturn (0);
}

PetscErrorCode InvSolver::solve () {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    TU_assert (initialized_, "InvSolver::solve (): InvSolver needs to be initialized.")
    TU_assert (data_ != nullptr, "InvSolver:solve (): requires non-null input data for inversion.");
    TU_assert (data_gradeval_ != nullptr, "InvSolver:solve (): requires non-null input data for gradient evaluation.");
    TU_assert (prec_ != nullptr, "InvSolver:solve (): requires non-null p_rec vector to be set");
    TU_assert (optsettings_ != nullptr, "InvSolver:solve (): requires non-null optimizer settings to be passed.");
    /* === observed data === */
    // apply observer on ground truth, store observed data in d
    ierr = itctx_->tumor_->obs_->apply (data_, data_);                                  CHKERRQ(ierr);
    // smooth observed data
    PetscScalar *d_ptr;
    double sigma_smooth;
    sigma_smooth = 2.0 * M_PI / itctx_->n_misc_->n_[0];
    ierr = VecGetArray (data_, &d_ptr);                                                 CHKERRQ(ierr);
    //SNAFU
    // ierr = weierstrassSmoother (d_ptr, d_ptr, itctx_->n_misc_, 0.0003);                 CHKERRQ(ierr);
    //static int it = 0; it++;
    //std::stringstream ss; ss<<"_it-"<<it;
    //std::string s("files/cpl/ITdata"+ss.str()+".nc");
    //DataOut(d_ptr, itctx_->n_misc_, s.c_str());
    /* === Add Noise === */
    Vec noise; double *noise_ptr;
    ierr = VecCreate (PETSC_COMM_WORLD, &noise);                                        CHKERRQ(ierr);
    ierr = VecSetSizes(noise, itctx_->n_misc_->n_local_, itctx_->n_misc_->n_global_);   CHKERRQ(ierr);
    ierr = VecSetFromOptions(noise);                                                    CHKERRQ(ierr);
    ierr = VecSetRandom(noise, NULL);                                                   CHKERRQ(ierr);
    ierr = VecGetArray (noise, &noise_ptr);                                             CHKERRQ(ierr);
    for (int i = 0; i < itctx_->n_misc_->n_local_; i++) {
        d_ptr[i] += noise_ptr[i] * itctx_->n_misc_->noise_scale_;
        noise_ptr[i] = d_ptr[i];                                                        //just to measure d norm
    }
    ierr = VecRestoreArray (noise, &noise_ptr);                                         CHKERRQ(ierr);
    ierr = VecRestoreArray (data_, &d_ptr);                                             CHKERRQ(ierr);
	PetscScalar max, min;                                                // compute d-norm
    PetscScalar d_norm = 0., d_errorl2norm = 0., d_errorInfnorm = 0.;
    ierr = VecNorm (noise, NORM_2, &d_norm);                                            CHKERRQ(ierr);
    ierr = VecMax (noise, NULL, &max);                                                  CHKERRQ(ierr);
    ierr = VecMin (noise, NULL, &min);                                                  CHKERRQ(ierr);
    ierr = VecAXPY (noise, -1.0, data_);                                                CHKERRQ(ierr);
    ierr = VecNorm (noise, NORM_2, &d_errorl2norm);                                     CHKERRQ(ierr);
    ierr = VecNorm (noise, NORM_INFINITY, &d_errorInfnorm);                             CHKERRQ(ierr);
    std::stringstream s;
    s << "data (ITP), with noise: l2norm = "<< d_norm <<" [max: "<<max<<", min: "<<min<<"]";  ierr = tuMSGstd(s.str()); CHKERRQ(ierr); s.str(""); s.clear();
    s << "IT data error due to thresholding and smoothing: l2norm = "<< d_errorl2norm <<", inf-norm = " <<d_errorInfnorm;  ierr = tuMSGstd(s.str()); CHKERRQ(ierr); s.str(""); s.clear();
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
    }                                                                                    // reset with zero for new ITP solve
    ierr = VecSet (itctx_->c0old, 0.0);                                                  CHKERRQ(ierr);
    itctx_->n_misc_->beta_             = itctx_->optsettings_->beta;                     // set beta for this inverse solver call
    itctx_->is_ksp_gradnorm_set        = false;
    itctx_->optfeedback_->converged    = false;
    itctx_->optfeedback_->solverstatus = "";
    itctx_->optfeedback_->nb_newton_it = 0;
    itctx_->optfeedback_->nb_krylov_it = 0;
    itctx_->data                       = data_;
    itctx_->data_gradeval              = data_gradeval_;
    /* === set TAO options === */
    ierr = setTaoOptions (tao_, itctx_.get());                                            CHKERRQ(ierr);
    ierr = TaoSetHessianRoutine (tao_, H_, H_, matfreeHessian, (void *) itctx_.get());    CHKERRQ(ierr);
    /* === solve === */
    // --------
    //resetTimers(itctx->n_misc_->timers_);
    ierr = TaoSolve (tao_);                                                               CHKERRQ(ierr);
    // --------
	/* === get solution === */
	Vec p; ierr = TaoGetSolutionVector (tao_, &p);                                         CHKERRQ(ierr);
	ierr = VecCopy (p, prec_);                                                             CHKERRQ(ierr);
	/* Get information on termination */
	TaoConvergedReason reason;
	TaoGetConvergedReason (tao_, &reason);
	/* get solution status */
	PetscScalar J, gnorm, xdiff;
	ierr = TaoGetSolutionStatus (tao_, NULL, &J, &itctx_->optfeedback_->gradnorm, NULL, &xdiff, NULL);         CHKERRQ(ierr);
	/* display convergence reason: */
	ierr = dispTaoConvReason (reason, itctx_->optfeedback_->solverstatus);                 CHKERRQ(ierr);
  ierr = tuMSGstd ("------------------------------------------------------------------------------------------------"); CHKERRQ(ierr);
  ierr = tuMSGstd (" optimization done"); CHKERRQ(ierr);
  ierr = tuMSGstd ("------------------------------------------------------------------------------------------------"); CHKERRQ(ierr);
	// only update if triggered from outside, i.e., if new information to the ITP solver is present
	itctx_->update_reference_gradient = false;
	// reset vectors (remember, memory managed on caller side):
	data_ = nullptr;
	data_gradeval_ = nullptr;

	PetscFunctionReturn (0);
}

InvSolver::~InvSolver () {
    PetscErrorCode ierr = 0;
    ierr = VecDestroy (&prec_);
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
	Event e ("tao-evaluate-objective-tumor");
    std::array<double, 7> t = {0};
    double self_exec_time = -MPI_Wtime ();
	CtxInv *itctx = reinterpret_cast<CtxInv*>(ptr);
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
    Event e ("tao-evaluate-gradient-tumor");
    std::array<double, 7> t = {0};
    double self_exec_time = -MPI_Wtime ();
    CtxInv *itctx = reinterpret_cast<CtxInv*>(ptr);
    ierr = itctx->derivative_operators_->evaluateGradient (dJ, x, itctx->data_gradeval);
    if (itctx->optsettings_->verbosity > 1) {
        double gnorm;
        ierr = VecNorm (dJ, NORM_2, &gnorm);                                            CHKERRQ(ierr);
        PetscPrintf (MPI_COMM_WORLD, " norm of gradient ||g||_2 = %e\n", gnorm);
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
PetscErrorCode evaluateObjectiveFunctionAndGradient (Tao tao, Vec p, PetscReal *J, Vec dJ, void *ptr){
	PetscFunctionBegin;
	PetscErrorCode ierr = 0;
	ierr = evaluateObjectiveFunction (tao, p, J, ptr);                     CHKERRQ(ierr);
	ierr = evaluateGradient (tao, p, dJ, ptr);                             CHKERRQ(ierr);
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

    Event e ("tao-hessian-matvec-tumor");
    std::array<double, 7> t = {0}; double self_exec_time = -MPI_Wtime ();
    // get context
    void *ptr;
    ierr = MatShellGetContext (A, &ptr);						             CHKERRQ (ierr);
    CtxInv *itctx = reinterpret_cast<CtxInv*>( ptr);
    // eval hessian
    ierr = itctx->derivative_operators_->evaluateHessian (y, x);
    if (itctx->optsettings_->verbosity > 1) {
        PetscPrintf (MPI_COMM_WORLD, " applying hessian done!\n");
        double xnorm;
        ierr = VecNorm (x, NORM_2, &xnorm); CHKERRQ(ierr);
        PetscPrintf (MPI_COMM_WORLD, " norm of search direction ||x||_2 = %e\n", xnorm);
    }
    self_exec_time += MPI_Wtime ();
    accumulateTimers (itctx->n_misc_->timers_, t, self_exec_time);
    e.addTimings (t); e.stop ();
    PetscFunctionReturn (0);
}

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
    Event e ("tao-apply-hessian-preconditioner");
    std::array<double, 7> t = {0};
    double self_exec_time = -MPI_Wtime ();
    double *ptr_pinvx = NULL, *ptr_x = NULL;
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

    PetscInt its;
    PetscScalar J, gnorm, cnorm, step, D, J0, D0, gnorm0;
    Vec x = nullptr;
    char msg[256];
    std::string statusmsg;
    TaoConvergedReason flag;
    CtxInv *itctx = reinterpret_cast<CtxInv*> (ptr);
    // get current iteration, objective value, norm of gradient, norm of
    // norm of contraint, step length / trust region readius of iteratore
    // and termination reason
    ierr = TaoGetSolutionStatus (tao, &its, &J, &gnorm, &cnorm, &step, &flag);      CHKERRQ(ierr);
    // accumulate number of newton iterations
    itctx->optfeedback_->nb_newton_it++;
    // print out Newton iteration information
    std::stringstream s;
    if (its == 0) {
        s << std::setw(4)  << " iter"              << "   " << std::setw(18) << "objective (abs)" << "   "
          << std::setw(18) << "||gradient||_2,rel" << "   " << std::setw(18) << "||gradient||_2"  << "   "
          << std::setw(18) << "step";
        ierr = tuMSG (" starting optimization, TAO's Gauß-Newton");                CHKERRQ(ierr);
        ierr = tuMSGstd ("------------------------------------------------------------------------------------------------"); CHKERRQ(ierr);
        ierr = tuMSGwarn (s.str());                                                CHKERRQ(ierr);
        ierr = tuMSGstd ("------------------------------------------------------------------------------------------------"); CHKERRQ(ierr);
        s.str ("");
        s.clear ();
    }
    s << " "     << std::scientific << std::setprecision(5) << std::setfill('0') << std::setw(4) << its << std::setfill(' ')
      << "   " << std::scientific << std::setprecision(12) << std::setw(18) << J
      << "   " << std::scientific << std::setprecision(12) << std::setw(18) << gnorm/itctx->optfeedback_->gradnorm0
      << "   " << std::scientific << std::setprecision(12) << std::setw(18) << gnorm
      << "   " << std::scientific << std::setprecision(12) << std::setw(18) << step;
    ierr = tuMSGwarn (s.str());                                                    CHKERRQ(ierr);
    s.str ("");
    s.clear ();
    //ierr = PetscPrintf (PETSC_COMM_WORLD, "\nKSP number of krylov iterations: %d\n", itctx->optfeedback_->nb_krylov_it);          CHKERRQ(ierr);
    //itctx->optfeedback_->nb_krylov_it = 0;
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

	Vec x; int maxit; PetscScalar divtol, abstol, reltol;
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
    lowergradbound = 1E-2;
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

    PetscInt iter, maxiter, miniter;
    PetscReal J, gnorm, step, gatol, grtol, gttol, g0norm, minstep;
    bool stop[3];
    int verbosity;
    std::stringstream ss, sc;

    CtxInv *ctx = reinterpret_cast<CtxInv*> (ptr);     // get user context
    verbosity = ctx->optsettings_->verbosity;
    minstep = std::pow (2.0, 10.0);
    minstep = 1.0 / minstep;
    miniter = ctx->optsettings_->newton_minit;
    // get tolerances
    #if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 7)
        ierr = TaoGetTolerances (tao, &gatol, &grtol, &gttol);                              CHKERRQ(ierr);
    #else
        ierr = TaoGetTolerances( tao, NULL, NULL, &gatol, &grtol, &gttol);                  CHKERRQ(ierr);
    #endif

    ierr = TaoGetMaximumIterations(tao, &maxiter);                                          CHKERRQ(ierr);
    ierr = TaoGetSolutionStatus(tao, &iter, &J, &gnorm, NULL, &step, NULL);                 CHKERRQ(ierr);

    // update/set reference gradient (with p = initial-guess)
    if (ctx->update_reference_gradient) {
    	Vec p0, dJ;
    	double norm_gref = 0.;
    	ierr = VecDuplicate (ctx->tumor_->p_, &p0);                                        CHKERRQ(ierr);
    	ierr = VecDuplicate (ctx->tumor_->p_, &dJ);                                        CHKERRQ(ierr);
    	ierr = VecSet (p0, 0.); CHKERRQ(ierr);
    	ierr = VecSet (dJ, 0.); CHKERRQ(ierr);
    	// evaluate reference gradient for initial guess p = 0 * ones(Np)
    	evaluateGradient(tao, p0, dJ, (void*) ctx);
    	ierr = VecNorm (dJ, NORM_2, &norm_gref); CHKERRQ(ierr);
    	ctx->optfeedback_->gradnorm0 = norm_gref;
    	//ctx->gradnorm0 = gnorm;
    	ctx->update_reference_gradient = false;
    	ierr = tuMSGstd("updated reference gradient for relative convergence criterion, Gauß-Newton solver."); CHKERRQ(ierr);
    	ierr = VecDestroy(&p0);                                                            CHKERRQ(ierr);
    	ierr = VecDestroy(&dJ);                                                            CHKERRQ(ierr);
    }
    // get initial gradient
    g0norm = ctx->optfeedback_->gradnorm0;
    g0norm = (g0norm > 0.0) ? g0norm : 1.0;
    ctx->convergence_message.clear();
    // check for NaN value
    if (PetscIsInfOrNanReal(J)) {
		ierr = tuMSGwarn ("objective is NaN");                                        CHKERRQ(ierr);
		ierr = TaoSetConvergedReason (tao, TAO_DIVERGED_NAN);                         CHKERRQ(ierr);
		PetscFunctionReturn (ierr);
    }
    // check for NaN value
    if (PetscIsInfOrNanReal(gnorm)) {
		ierr = tuMSGwarn("||g|| is NaN");                                             CHKERRQ(ierr);
		ierr = TaoSetConvergedReason(tao, TAO_DIVERGED_NAN);                          CHKERRQ(ierr);
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
    			ierr = tuMSGstd(ss.str());                                                CHKERRQ(ierr);
    			ss.str(std::string());
                ss.clear();
    	}
    	if (step < minstep) {
    			ss << "step  = " << std::scientific << step << " < " << minstep << " = " << "bound";
    			ierr = tuMSGwarn(ss.str()); CHKERRQ(ierr);
    			ss.str(std::string());
                ss.clear();
    			ierr = TaoSetConvergedReason(tao, TAO_CONVERGED_STEPTOL);                 CHKERRQ(ierr);
    			PetscFunctionReturn(ierr);
    	}
    	// ||g_k||_2 < tol*||g_0||
    	if (gnorm < gttol*g0norm) {
    			ierr = TaoSetConvergedReason(tao, TAO_CONVERGED_GTTOL);                   CHKERRQ(ierr);
    			stop[0] = true;
    	}
    	ss << "  " << stop[0] << "    ||g|| = " << std::setw(14)
		<< std::right << std::scientific << gnorm << "    <    "
		<< std::left << std::setw(14) << gttol*g0norm << " = " << "tol";
    	ctx->convergence_message.push_back(ss.str());
      if(verbosity >= 3) {
    	  ierr = tuMSGstd(ss.str());                                                  CHKERRQ(ierr);
      }
    	ss.str(std::string());
        ss.clear();
    	// ||g_k||_2 < tol
    	if (gnorm < gatol) {
			ierr = TaoSetConvergedReason(tao, TAO_CONVERGED_GATOL);                       CHKERRQ(ierr);
			stop[1] = true;
    	}
    	ss  << "  " << stop[1] << "    ||g|| = " << std::setw(14)
		<< std::right << std::scientific << gnorm << "    <    "
		<< std::left << std::setw(14) << gatol << " = " << "tol";
    	ctx->convergence_message.push_back(ss.str());
      if(verbosity >= 3) {
    	  ierr = tuMSGstd(ss.str());                                                  CHKERRQ(ierr);
      }
    	ss.str(std::string());
        ss.clear();
    	// iteration number exceeds limit
    	if (iter > maxiter) {
			ierr = TaoSetConvergedReason(tao, TAO_DIVERGED_MAXITS);                       CHKERRQ(ierr);
			stop[2] = true;
    	}
    	ss  << "  " << stop[2] << "     iter = " << std::setw(14)
		<< std::right << iter  << "    >    "
		<< std::left << std::setw(14) << maxiter << " = " << "maxiter";
    	ctx->convergence_message.push_back(ss.str());
      if(verbosity >= 3) {
    	  ierr = tuMSGstd(ss.str());                                                  CHKERRQ(ierr);
      }
    	ss.str(std::string());
        ss.clear();
    	// store objective function value
    	ctx->jvalold = J;
    	if (stop[0] || stop[1] || stop[2]) {
    		ctx->optfeedback_->converged = true;
    		PetscFunctionReturn(ierr);
    	}

    }
    else {
		// if the gradient is zero, we should terminate immediately
		if (gnorm == 0) {
			ss << "||g|| = " << std::scientific << 0.0 << " < " << gatol  << " = " << "bound";
			ierr = tuMSGwarn(ss.str());                                                   CHKERRQ(ierr);
			ss.str(std::string());
            ss.clear();
			ierr = TaoSetConvergedReason(tao, TAO_CONVERGED_GATOL);                       CHKERRQ(ierr);
			PetscFunctionReturn(ierr);
		}
    }
    // if we're here, we're good to go
    ierr = TaoSetConvergedReason (tao, TAO_CONTINUE_ITERATING);                           CHKERRQ(ierr);

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

    PetscInt iter, maxiter, miniter, iterbound;
    PetscReal jx, jxold, gnorm, step, gatol, grtol, gttol, g0norm, gtolbound, minstep, theta, normx, normdx, tolj, tolx, tolg;
    const int nstop = 7;
    bool stop[nstop];
    std::stringstream ss;
    Vec x;

    CtxInv *ctx = reinterpret_cast<CtxInv*>(ptr);     // get user context
    // get minstep and miniter
    minstep = std::pow(2.0, 10.0);
    minstep = 1.0 / minstep;
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

    // update/set reference gradient (with p = initial-guess)
    if(ctx->update_reference_gradient) {
    	Vec p0, dJ;
    	double norm_gref = 0.;
    	ierr = VecDuplicate (ctx->tumor_->p_, &p0);                                        CHKERRQ(ierr);
    	ierr = VecDuplicate (ctx->tumor_->p_, &dJ);                                        CHKERRQ(ierr);
    	ierr = VecSet (p0, 0.);                                                            CHKERRQ(ierr);
    	ierr = VecSet (dJ, 0.);                                                            CHKERRQ(ierr);
    	// evaluate reference gradient for initial guess p = 0 * ones(Np)
    	evaluateGradient(tao, p0, dJ, (void*) ctx);
    	ierr = VecNorm (dJ, NORM_2, &norm_gref); CHKERRQ(ierr);
    	ctx->optfeedback_->gradnorm0 = norm_gref;
    	ctx->update_reference_gradient = false;
    	ierr = tuMSGstd("updated reference gradient for relative convergence criterion, Gauß-Newton solver.");     CHKERRQ(ierr);
    	ierr = VecDestroy(&p0);                                                            CHKERRQ(ierr);
    	ierr = VecDestroy(&dJ);                                                            CHKERRQ(ierr);
    }
    // get initial gradient
    g0norm = ctx->optfeedback_->gradnorm0;
    g0norm = (g0norm > 0.0) ? g0norm : 1.0;
    // compute tolerances for stopping conditions
    tolj = gttol;
    tolx = std::sqrt(gttol);
    #if __cplusplus > 199711L
    tolg = std::cbrt(gttol);
    #else
    tolg = std::pow(gttol, 1.0/3.0);
    #endif
    // compute theta
    theta = 1.0 + std::abs(jx);
    // compute norm(\Phi x^k - \Phi x^k-1) and norm(\Phi x^k)
    ierr = ctx->tumor_->phi_->apply (ctx->tmp, x);                                         CHKERRQ(ierr);  // comp \Phi x
    ierr = VecNorm (ctx->tmp, NORM_2, &normx);                                             CHKERRQ(ierr);  // comp norm \Phi x
    ierr = VecAXPY(ctx->c0old, -1, ctx->tmp);                                              CHKERRQ(ierr);  // comp dx
    ierr = VecNorm (ctx->c0old, NORM_2, &normdx);                                          CHKERRQ(ierr);  // comp norm \Phi dx
    ierr = VecCopy(ctx->tmp, ctx->c0old);                                                  CHKERRQ(ierr);  // save \Phi x
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

    ierr = PetscPrintf (MPI_COMM_WORLD, "||g(x)|| / ||g(x0)|| = %6E, ||g(x0)|| = %6E \n", gnorm/g0norm, g0norm);
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
    	ss << "  " << stop[0] << "    |dJ|  = " << std::setw(14)
        << std::right << std::scientific << std::abs(jxold-jx) << "    <    "
        << std::left << std::setw(14) << tolj*theta << " = " << "tol*|1+J|";
    	ctx->convergence_message.push_back(ss.str());
    	ierr = tuMSGstd(ss.str());                                                     CHKERRQ(ierr);
    	ss.str(std::string());
        ss.clear();
    	// ||dx|| < sqrt(tolj)*(1+||x||)
    	if (normdx < tolx*(1+normx)) {
    	   stop[1] = true;
    	}
    	ss << "  " << stop[1] << "    |dx|  = " << std::setw(14)
        << std::right << std::scientific << normdx << "    <    "
        << std::left << std::setw(14) << tolx*(1+normx) << " = " << "sqrt(tol)*(1+||x||)";
    	ctx->convergence_message.push_back(ss.str());
    	ierr = tuMSGstd(ss.str());                                                     CHKERRQ(ierr);
    	ss.str(std::string());
        ss.clear();
    	// ||g_k||_2 < cbrt(tolj)*abs(1+Jc)
    	if (gnorm < tolg*theta) {
    		stop[2] = true;
    	}
    	ss  << "  " << stop[2] << "    ||g|| = " << std::setw(14)
        << std::right << std::scientific << gnorm << "    <    "
        << std::left << std::setw(14) << tolg*theta << " = " << "cbrt(tol)*|1+J|";
    	ctx->convergence_message.push_back(ss.str());
    	ierr = tuMSGstd(ss.str());                                                     CHKERRQ(ierr);
    	ss.str(std::string());
        ss.clear();
    	// ||g_k||_2 < tol
    	if (gnorm < gatol) {
    			stop[3] = true;
    	}
    	ss  << "  " << stop[3] << "    ||g|| = " << std::setw(14)
    	<< std::right << std::scientific << gnorm << "    <    "
    	<< std::left << std::setw(14) << gatol << " = " << "tol";
    	ctx->convergence_message.push_back(ss.str());
    	ierr = tuMSGstd(ss.str());                                                     CHKERRQ(ierr);
    	ss.str(std::string());
        ss.clear();

    	if (gnorm < gtolbound*g0norm) {
    			stop[4] = true;
    	}
    	ss  << "  " << stop[4] << "    ||g|| = " << std::setw(14)
		<< std::right << gnorm  << "    >    "
		<< std::left << std::setw(14) << gtolbound*g0norm << " = " << "kappa*||g0||";
    	ctx->convergence_message.push_back(ss.str());
    	ierr = tuMSGstd(ss.str());                                                     CHKERRQ(ierr);
    	ss.str(std::string());
        ss.clear();

    	if (iter > maxiter) {
    			stop[5] = true;
    	}
    	ss  << "  " << stop[5] << "    iter  = " << std::setw(14)
		<< std::right << iter  << "    >    "
		<< std::left << std::setw(14) << maxiter << " = " << "maxiter";
    	ctx->convergence_message.push_back(ss.str());
    	ierr = tuMSGstd(ss.str());                                                     CHKERRQ(ierr);
    	ss.str(std::string());
        ss.clear();

    	if (iter > iterbound) {
    			stop[6] = true;
    	}
    	ss  << "  " << stop[6] << "    iter  = " << std::setw(14)
		<< std::right << iter  << "    >    "
		<< std::left << std::setw(14) << iterbound << " = " << "iterbound";
    	ctx->convergence_message.push_back(ss.str());
    	ierr = tuMSGstd(ss.str());                                                     CHKERRQ(ierr);
    	ss.str(std::string());
        ss.clear();

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
    	} else if (stop[4] && stop[5]) {
			ierr = TaoSetConvergedReason(tao, TAO_DIVERGED_MAXITS);                     CHKERRQ(ierr);
			ctx->optfeedback_->converged = true;
			PetscFunctionReturn(ierr);
    	} else if (stop[6]) {
			ierr = TaoSetConvergedReason(tao, TAO_DIVERGED_MAXITS);                     CHKERRQ(ierr);
			ctx->optfeedback_->converged = true;
			PetscFunctionReturn(ierr);
    	}
    }
    else {
    	// if the gradient is zero, we should terminate immediately
    	if (gnorm == 0) {
			ss << "||g|| = " << std::scientific << 0.0 << " < " << gatol  << " = " << "bound";
			ierr = tuMSGwarn(ss.str());                                                 CHKERRQ(ierr);
			ss.str(std::string());
            ss.clear();
			ierr = TaoSetConvergedReason(tao, TAO_CONVERGED_GATOL);                     CHKERRQ(ierr);
			PetscFunctionReturn(ierr);
    	}
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
          msg = "solver converged";
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

PetscErrorCode setTaoOptions (Tao tao, CtxInv *ctx) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    TaoLineSearch linesearch;        // line-search object
    ierr = TaoSetType (tao, "nls");   // set TAO solver type
    PetscBool flag = PETSC_FALSE;
    std::string msg;
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

    // parse options user has set
    ierr = TaoSetFromOptions(tao);
    // set the initial vector
    ierr = TaoSetInitialVector(tao, ctx->tumor_->p_);
    // set routine for evaluating the objective
    ierr = TaoSetObjectiveRoutine (tao, evaluateObjectiveFunction, (void*) ctx);
    // set routine for evaluating the Gradient
    ierr = TaoSetGradientRoutine (tao, evaluateGradient, (void*) ctx);

    // TAO type from user input
    const TaoType taotype;
    ierr = TaoGetType (tao, &taotype);
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
    } else {
        msg = " numerical optimization method not supported (setting default: LMVM)\n";
        ierr = TaoSetType (tao, "lmvm");
    }
    // set tolerances
    #if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 7)
        ierr = TaoSetTolerances (tao, ctx->optsettings_->gatol, ctx->optsettings_->grtol, ctx->optsettings_->opttolgrad);
    #else
        ierr = TaoSetTolerances (tao, 1E-12, 1E-12, ctx->optsettings_->gatol, ctx->optsettings_->grtol, ctx->optsettings_->opttolgrad);
    #endif
        ierr = TaoSetMaximumIterations (tao, ctx->optsettings_->newton_maxit);

    // set adapted convergence test for warm-starts
    ierr = TaoSetConvergenceTest(tao, checkConvergenceGrad, ctx);
    //ierr = TaoSetConvergenceTest(tao, checkConvergenceGradObj, ctx);
    // set linesearch
    ierr = TaoGetLineSearch (tao, &linesearch);
    ierr = TaoLineSearchSetType (linesearch, "armijo");

    std::stringstream s;
    tuMSGstd(" parameters (optimizer):");
    tuMSGstd(" tolerances (stopping conditions):");
    s << "   gatol: "<< ctx->optsettings_->gatol;  /*pout(s.str(), cplctx->_fileOutput);*/ tuMSGstd(s.str()); s.str(""); s.clear();
    s << "   grtol: "<< ctx->optsettings_->grtol;  /*pout(s.str(), cplctx->_fileOutput);*/ tuMSGstd(s.str()); s.str(""); s.clear();
    s << "   gttol: "<< ctx->optsettings_->opttolgrad;  /*pout(s.str(), cplctx->_fileOutput);*/ tuMSGstd(s.str()); s.str(""); s.clear();

    ierr = TaoSetFromOptions(tao);
    /* === set the KSP Krylov solver settings === */
    KSP ksp = PETSC_NULL;
    ierr = TaoGetKSP(tao, &ksp);                          // get the ksp of the optimizer
    if (ksp != PETSC_NULL) {
        ierr = KSPSetOptionsPrefix(ksp, "opt_");            // set prefix to control sets and monitors   // set default tolerance to 1E-6
        ierr = KSPSetTolerances(ksp, 1E-6, PETSC_DEFAULT, PETSC_DEFAULT, ctx->optsettings_->krylov_maxit);
        KSPSetPreSolve (ksp, preKrylovSolve, ctx);           // to use Eisenstat/Walker convergence crit.
        ierr = KSPMonitorSet(ksp, hessianKSPMonitor,ctx, 0); // monitor
    }
    // set the preconditioner (we check if KSP exists, as there are also
    // solvers that do not require a KSP solve (BFGS and friends))
    if (ksp != PETSC_NULL) {
    	PC pc;
    	ierr = KSPGetPC(ksp, &pc);
    	ierr = PCSetType (pc, PCSHELL);
    	ierr = PCShellSetApply(pc, preconditionerMatVec);
    	ierr = PCShellSetContext(pc, ctx);
    }
    // set the routine to evaluate the objective and compute the gradient
    ierr = TaoSetObjectiveAndGradientRoutine (tao, evaluateObjectiveFunctionAndGradient, (void*) ctx);
    // set monitor function
    ierr = TaoSetMonitor(tao, optimizationMonitor, (void *) ctx, NULL);
    // Lower and Upper Bounds
    // ierr = TaoSetVariableBounds(tao, tumor->lowerb_, tumor->upperb_);
    PetscFunctionReturn (0);
}
