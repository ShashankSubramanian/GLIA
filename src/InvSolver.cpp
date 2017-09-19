#include "InvSolver.h"
#include "petsctao.h"
#include <iostream>
#include <limits>
#include "DerivativeOperators.h"
#include "PdeOperators.h"
#include "Utils.h"
#include "EventTimings.hpp"

/**
 *  ***********************************
 */
InvSolver::InvSolver (std::shared_ptr <DerivativeOperators> derivative_operators, std::shared_ptr <NMisc> n_misc)
:
  initialized_(false),
	optTolGrad_(1E-3),
	betap_(0),
	updateRefGradITPSolver_(true),
	refgradITPSolver_(1.),
	data_(),
	data_gradeval_(),
	solverstatus_(""),
	nbNewtonIt_(-1),
	nbKrylovIt_(-1),
	//_tumor(),
	itctx_(),
	tao_(),
	//params_(),
	tumor_(),
  itctx_()
  {
	  PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    if(derivative_operators_ != nullptr && n_misc != nullptr) {
      ierr = initialize(derivative_operators, n_misc) CHKERRQ(ierr);
	  }
}

/**
 *  ***********************************
 */
PetscErrorCode InvSolver::initialize(
	std::shared_ptr <DerivativeOperators> derivative_operators,
	std::shared_ptr <NMisc> n_misc) {

  PetscFunctionBegin;
	PetscErrorCode ierr = 0;
	if(initialized_) PetscFunctionReturn(0);

	itctx_ = std::make_shared<CtxInv>();
	itctx_->derivative_operators_ = derivative_operators;
	itctx_->n_misc_ = n_misc;

	ierr = MatCreateShell (PETSC_COMM_WORLD, n_misc->n_local_, n_misc->n_local_, n_misc->n_global_, n_misc->n_global_, itctx_.get(), &A_);
	ierr = MatShellSetOperation (A_, MATOP_MULT, (void(*)(void)) hessianMatVec);

	ierr = TaoCreate (PETSC_COMM_WORLD, &tao_);
	ierr = TaoSetType (tao_, "nls");
	ierr = TaoSetFromOptions (tao_);

  initialized_ = true;
	PetscFunctionReturn(0);
}

/**
 *  ***********************************
 */
PetscErrorCode InvSolver::solve () {
	PetscFunctionBegin;
	PetscErrorCode ierr = 0;
  TU_assert(initialized_, "InvSolver::solve(): InvSolver needs to be initialized.")
  TU_assert(data_ != nullptr, "InvSolver:solve(): requires non-null input data for inversion.");
	TU_assert(data_gradeval_ != nullptr, "InvSolver:solve(): requires non-null input data for gradient evaluation.");

  /* === observed data === */
	// apply observer on ground truth, store observed data in d
	err = tumor_->obs_->apply(data_, data_);                             CHKERRQ(ierr);
	// smooth observed data
  ScalarType *d_ptr;
	ierr = VecGetArray(data_, &d_ptr);                                   CHKERRQ(ierr);
  ierr = weierstrass_smoother(d_ptr, d_ptr, itctx_->n_misc_, 0.0003);  CHKERRQ(ierr);
	//static int it = 0; it++;
	//std::stringstream ss; ss<<"_it-"<<it;
	//std::string s("files/cpl/ITdata"+ss.str()+".nc");
	//DataOut(d_ptr, itctx_->n_misc_, s.c_str());

	/* Add Noise */
  Vec noise; double * noise_ptr;
  ierr = VecCreate(PETSC_COMM_WORLD, &noise);                          CHKERRQ(ierr);
  ierr = VecSetSizes(noise, itctx_->n_misc_->n_local_, itctx_->n_misc_->n_global_); CHKERRQ(ierr);
  ierr = VecSetFromOptions(noise);                                     CHKERRQ(ierr);
  ierr = VecSetRandom(noise, NULL);                                    CHKERRQ(ierr);
  ierr = VecGetArray(noise, &noise_ptr);                               CHKERRQ(ierr);
  for (int i = 0; i < itctx_->n_misc_->n_local_; ++i){
    d_ptr[i] += noise_ptr[i] * 0.0;
    noise_ptr[i] = d_ptr[i];                                           //just to measure d norm
  }
  ierr = VecRestoreArray(noise, &noise_ptr);                           CHKERRQ(ierr);
	ierr = VecRestoreArray(data_, &d_ptr);                               CHKERRQ(ierr);
	PetscScalar max, min;                                                // compute d-norm
  PetscScalar d_norm = 0., d_errorl2norm = 0., d_errorInfnorm = 0.;
  ierr = VecNorm(noise, NORM_2, &d_norm);                              CHKERRQ(ierr);
  ierr = VecMax(noise, NULL, &max);                                    CHKERRQ(ierr);
  ierr = VecMin(noise, NULL, &min);                                    CHKERRQ(ierr);
  ierr = VecAXPY(noise, -1.0, data_;                                   CHKERRQ(ierr);
  ierr = VecNorm(noise, NORM_2, &d_errorl2norm);                       CHKERRQ(ierr);
  ierr = VecNorm(noise, NORM_INFINITY, &d_errorInfnorm);               CHKERRQ(ierr);
	std::stringstream s;
  s << "data (ITP), with noise: l2norm = "<< d_norm <<" [max: "<<max<<", min: "<<min<<"]";  ierr = tuMSGstd(s.str()); CHKERRQ(ierr); s.str(""); s.clear();
  s << "IT data error due to thresholding and smoothing: l2norm = "<< d_errorl2norm <<", inf-norm = " <<d_errorInfnorm;  ierr = tuMSGstd(s.str()); CHKERRQ(ierr); s.str(""); s.clear();

	/* solve for a initial guess */
	//solveForParameters(data.get(), _tumor->N_Misc_, _tumor->phi_, timers, &_tumor->p_true_);

	_solverstatus = "";
  _nbNewtonIt = -1;
  _nbKrylovIt = -1;
  //tumor_->t_history_->reset(); // TODO: no time history so far, if available, reset here


/*
	CtxInv *itctx;
	ierr = MatShellGetContext (A_, &itctx);															 CHKERRQ (ierr);
	ierr = TaoSetObjectiveRoutine (tao_, formFunction, (void*) itctx);   CHKERRQ (ierr);
	ierr = TaoSetGradientRoutine (tao_, formGradient, (void*) itctx);    CHKERRQ (ierr);

	double ga_tol, gr_tol, gt_tol;
	ierr = TaoGetTolerances (tao_, &ga_tol, &gr_tol, &gt_tol);           CHKERRQ (ierr);
	ga_tol = 1e-6;
	gr_tol = 1e-6;
	gt_tol = 1e-6;
	ierr = TaoSetTolerances (tao_, ga_tol, gr_tol, gt_tol);              CHKERRQ (ierr);

	ierr = TaoSetHessianRoutine (tao_, A_, A_, formHessian, (void*) itctx); CHKERRQ (ierr);
*/
	// ierr = TaoSolve (tao);																			CHKERRQ (ierr);
	PetscFunctionReturn(0);
}

/**
 *  ***********************************
 */
InvSolver::~InvSolver () {
	PetscFunctionBegin;
	PetscErrorCode ierr = 0;
	ierr = TaoDestroy (&tao_);
	ierr = MatDestroy (&A_);
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
	// timing
	coupling::Event e("tao-evaluate-objective-tumor");
  double t[7] = {0}; double self_exec_time = -MPI_Wtime();

	CtxInv *itctx = reinterpret_cast<CtxInv*>(ptr);
	ierr = itctx->derivative_operators_->evaluateObjective (J, x, itctx->data->get());

  // timing
	self_exec_time += MPI_Wtime();
	accumulateTimers(itctx->n_misc_->timers_, t, self_exec_time);
  e.addTimings(t); e.stop();
	PetscFunctionReturn(0);
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
	// timing
	coupling::Event e("tao-evaluate-gradient-tumor");
	double t[7] = {0}; double self_exec_time = -MPI_Wtime();

	CtxInv *itctx = reinterpret_cast<CtxInv*>(ptr);
	ierr = ctx->derivative_operators_->evaluateGradient (dJ, x, itctx->data_gradeval->get());

	if (itctx->n_misc_->verbosity_ > 1) {
    double gnorm;
    ierr = VecNorm(dJ, NORM_2, &gnorm); CHKERRQ(ierr);
    PetscPrintf(MPI_COMM_WORLD, " norm of gradient ||g||_2 = %e\n", gnorm);
  }

	// timing
	self_exec_time += MPI_Wtime();
	accumulateTimers(itctx->n_misc_->timers_, t, self_exec_time);
	e.addTimings(t); e.stop();
	PetscFunctionReturn(0);
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
PetscErrorCode evaluateObjectiveFunctionAndGradient(Tao tao, Vec p, PetscReal *J, Vec dJ, void *ptr){
	PetscFunctionBegin;
	PetscErrorCode ierr = 0;

	ierr = evaluateObjectiveFunction(tao, p, J, ptr);  CHKERRQ(ierr);
	ierr = evaluateGradient(tao, p, dJ, ptr); CHKERRQ(ierr);
  /*
	int low1, high1, low2, high2;
	ierr = VecGetOwnershipRange(p, &low1, &high1); CHKERRQ(ierr);
	ierr = VecGetOwnershipRange(p, &low2, &high2); CHKERRQ(ierr);
	if (low1 != low2 || high1 != high2) {
		PetscPrintf(PETSC_COMM_SELF, "low1=%d high1=%d  \n", low1, high1);
		PetscPrintf(PETSC_COMM_SELF, "low2=%d high2=%d  \n", low2, high2);
	}
  */
	PetscFunctionReturn(0);
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
	// timing
	coupling::Event e("tao-hessian-matvec-tumor");
	double t[7] = {0}; double self_exec_time = -MPI_Wtime();

	// get context
	void *ptr;
	ierr = MatShellGetContext (A, &ptr);						CHKERRQ (ierr);
	CtxInv *itctx = reinterpret_cast<CtxInv*>(ptr);

	// eval hessian
	ierr = ctx->derivative_operators_->evaluateHessian (x, y);

	if (itctx->n_misc_->verbosity_ > 1) {
    PetscPrintf(MPI_COMM_WORLD, " applying hessian done!\n");
    double xnorm;
    ierr = VecNorm(x, NORM_2, &xnorm); CHKERRQ(ierr);
    PetscPrintf(MPI_COMM_WORLD, " norm of search direction ||x||_2 = %e\n", xnorm);
  }

	// timing
	self_exec_time += MPI_Wtime();
	accumulateTimers(itctx->n_misc_->timers_, t, self_exec_time);
	e.addTimings(t); e.stop();
	PetscFunctionReturn(0);
}

/**
 *  ***********************************
 */
PetscErrorCode matfreeHessian (Tao tao, Vec x, Mat H, Mat precH, void *ptr) {
	PetscFunctionBegin;
	PetscErrorCode ierr = 0;
  // empty
	PetscFunctionReturn(0);
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
PetscErrorCode preconditionerMatVec(PC pinv, Vec x, Vec pinvx){
	PetscFunctionBegin;
	PetscErrorCode ierr = 0;
	void *ptr;

	// get shell context
  ierr = PCShellGetContext(pinv, &ptr); CHKERRQ(ierr);
  // apply the hessian
  ierr = ApplyPreconditioner(ptr, x, pinvx); CHKERRQ(ierr);
	PetscFunctionReturn(0);
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
PetscErrorCode applyPreconditioner(void* ptr, Vec x, Vec pinvx){
	PetscFunctionBegin;
	PetscErrorCode ierr = 0;
	// timing
	coupling::Event e("tao-apply-hessian-preconditioner");
  double t[7] = {0}; double self_exec_time = -MPI_Wtime();

	double *ptr_pinvx = NULL, *ptr_x = NULL;
	CtxInv *itctx = reinterpret_cast<CtxInv*>(ptr);

	ierr = VecCopy(x, pinvx);
	// === PRECONDITIONER CURRENTLY DISABLED ===
  PetscFunctionReturn(0);

	// apply hessian
	ierr = ctx->derivative_operators_->evaluateHessian(pinvx, x);
	/*
  D2J_prec(pinvx, x, itctx->tumor->beta_, itctx->tumor->N_Misc_,
      itctx->tumor->K_, itctx->tumor->Rho_, itctx->tumor.get(),
      itctx->tumor->phi_, itctx->tumor->O_, t, itctx->tumor->t_history_);
  */

	// timing
	self_exec_time += MPI_Wtime();
	accumulateTimers(itctx->n_misc_->timers_, t, self_exec_time);
	e.addTimings(t); e.stop();
	PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
/*
 optimizationMonitor    mointors the inverse Gauß-Newton solve

 input parameters:
  . tao       TAO object
  . ptr       optional user defined context
 */
PetscErrorCode optimizationMonitor(Tao tao, void* ptr){
	PetscFunctionBegin;
	PetscErrorCode ierr = 0;

	IntType its;
	ScalarType J, gnorm, cnorm, step, D, J0, D0, gnorm0;
	Vec x = nullptr;
	char msg[256];
	std::string statusmsg;
	TaoConvergedReason flag;

	CtxInv *itctx = reinterpret_cast<CtxInv*>(ptr);
	// get current iteration, objective value, norm of gradient, norm of
  // norm of contraint, step length / trust region readius of iteratore
  // and termination reason
  ierr = TaoGetSolutionStatus(tao, &its, &J, &gnorm, &cnorm, &step, &flag); CHKERRQ(ierr);
	// accumulate number of newton iterations
  itctx->nbNewtonIt++;

	// print out Newton iteration information
  std::stringstream s;
  if (its == 0) {
    s << " Itr"  << "     J" << "            ||g||_2" << "      step";
    ierr = tuMSG(""); CHKERRQ(ierr);
    ierr = tuMSGwarn(s.str()); CHKERRQ(ierr); s.str(""); s.clear();
  }
  s << " " << std::scientific << std::setprecision(4) << std::setfill('0') << std::setw(3)<< its
    << "     " << std::scientific << std::setprecision(4) << J
    << "     " << std::scientific << std::setprecision(4) << gnorm
    << "     " << std::scientific << std::setprecision(4) << step;
  ierr = tuMSGwarn(s.str()); CHKERRQ(ierr); s.str(""); s.clear();

	/*
	#ifdef MONITOR_ITP_TIME_HISTORY
	  Vec p; ierr = TaoGetSolutionVector(tao, &p); CHKERRQ(ierr);
	  static int printoutcounter = 0;
	  printoutcounter++;
	  std::stringstream s; s<<"afterNewton_it_"<<printoutcounter;
	  Vec dJ, Hs;
	  // We can get dJ from Tao solver as well but it is better not to pollute that
	  ierr = VecDuplicate(p, &dJ);
	  ierr = VecDuplicate(p, &Hs);
	  // This is an optional step to avoid potential dirty data in t_history_
	  // itctx->tumor->t_history_->Reset();
	  // DJ(dJ, p, ptr_d, itctx->tumor->beta_, *itctx->tumor->N_Misc_, itctx->tumor->K_, itctx->tumor->Rho_, itctx->tumor->phi_,
	  //     itctx->tumor->O_, timings, itctx->tumor->t_history_);
	  // D2J(Hs, p, itctx->tumor->beta_, *itctx->tumor->N_Misc_, itctx->tumor->K_, itctx->tumor->Rho_, itctx->tumor->phi_,
	  //    itctx->tumor->O_, timings, itctx->tumor->t_history_);
	  itctx->tumor->t_history_->PrintOut(s.str().c_str());
	  itctx->tumor->t_history_->PrintNorms();
	  VecDestroy(&dJ);
	  VecDestroy(&Hs);
	#endif
	*/

	PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
/*
 optimizationMonitor    mointors the inner PCG Krylov solve to invert the Hessian

 input parameters:
  . KSP ksp          KSP solver object
	. PetscIntn        iteration number
	. PetscRela rnorm  l2-norm (preconditioned) of residual
  . void* ptr        optional user defined context
 */
PetscErrorCode hessianKSPMonitor(KSP ksp, PetscInt n,PetscReal rnorm, void *ptr){
	PetscFunctionBegin;
	PetscErrorCode ierr = 0;

	Vec x;
	ierr = KSPBuildSolution(ksp,NULL,&x);CHKERRQ(ierr); // build solution vector
	CtxInv *itctx = reinterpret_cast<CtxInv*>(ptr);     // get user context
	ctx->nbKrylovIt++;                                  // accumulate number of krylov iterations
	// ierr = PetscPrintf(PETSC_COMM_WORLD,"iteration %D solution vector:\n",n);CHKERRQ(ierr);
  // ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  //ierr = PetscPrintf(PETSC_COMM_WORLD,"iteration %D KSP Residual norm %14.12e \n",n,rnorm);CHKERRQ(ierr);
	PetscFunctionReturn(0);
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
  . void* ptr     optional user defined context
 */
PetscErrorCode preKrylovSolve(KSP ksp, Vec b, Vec x, void* ptr){
	PetscFunctionBegin;
	PetscErrorCode ierr = 0;

	PetscReal gnorm = 0., g0norm = 1., reltol, abstol = 0., divtol = 0.,
	          uppergradbound, lowergradbound;
	PetscInt maxit;
	int nprocs, procid;
	MPI_Comm_rank(PETSC_COMM_WORLD, &procid);
	MPI_Comm_size(PETSC_COMM_WORLD, &nprocs);

	CtxInv *itctx = reinterpret_cast<CtxInv*>(ptr);     // get user context
	ierr = VecNorm(b, NORM_2, &gnorm); CHKERRQ(ierr);   // compute gradient norm

	if(! itctx->isGradNorm0HessianSet) {                // set initial gradient norm
		itctx->gradnorm0_HessianCG = gnorm;               // for KSP Hessian solver
		itctx->isGradNorm0HessianSet = true;
	}
	g0norm = ctx->gradnorm0_HessianCG;                  // get reference gradient
	gnorm /= g0norm;                                    // normalize gradient
	                                                    // get tolerances
  ierr = KSPGetTolerances(ksp, &reltol, &abstol, &divtol, &maxit); CHKERRQ(ierr);
	uppergradbound = 0.5;                               // assuming quadratic convergence
	lowergradbound = 1E-2;

  // user forcing sequence to estimate adequate tolerance for solution of
	//  KKT system (Eisenstat-Walker)
	if (itctx->fseqtype == QDFS) {
		// assuming quadratic convergence (we do not solver more accurately than 12 digits)
		reltol = PetscMax(lowergradbound, PetscMin(uppergradbound, gnorm));
	} else {
		// assuming superlinear convergence (we do not solver  more accurately than 12 digitis)
		reltol = PetscMax(lowergradbound, PetscMin(uppergradbound, std::sqrt(gnorm)));
	}
	// overwrite tolerances with estimate
	ierr = KSPSetTolerances(ksp, reltol, abstol, divtol, maxit); CHKERRQ(ierr);

	if(procid == 0){
		std::cout<<" ksp rel-tol (Eisenstat/Walker): "<<reltol<<", grad0norm: "<<g0norm<<", gnorm/grad0norm: "<<gnorm<<std::endl;
	}
	PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
/*
 checkConvergenceGrad    checks convergence of the overall Gauß-Newton tumor inversion

 input parameters:
  . Tao tao       Tao solver object
  . void* ptr     optional user defined context
 */
PetscErrorCode checkConvergenceGrad(Tao tao, void* ptr){
	PetscFunctionBegin;
	PetscErrorCode ierr = 0;

	PetscInt iter, maxiter, miniter;
	PetscReal J, gnorm, step, gatol, grtol, gttol, g0norm, minstep;
	bool stop[3];
	int verbosity;
	std::stringstream ss, sc;

  CtxInv *ctx = reinterpret_cast<CtxInv*>(ptr);     // get user context
  verbosity = ctx->verbosity;
	minstep = std::pow(2.0, 10.0);
	minstep = 1.0 / minstep;
	miniter = ctx->newton_minit;
	// get tolerances
	#if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 7)
	    ierr = TaoGetTolerances(tao, &gatol, &grtol, &gttol); CHKERRQ(ierr);
	#else
	    ierr = TaoGetTolerances(tao, NULL, NULL, &gatol, &grtol, &gttol); CHKERRQ(ierr);
	#endif

	ierr = TaoGetMaximumIterations(tao, &maxiter); CHKERRQ(ierr);
	ierr = TaoGetSolutionStatus(tao, &iter, &J, &gnorm, NULL, &step, NULL); CHKERRQ(ierr);

	// update/set reference gradient (with p = initial-guess)
	if(ctx->updateGradNorm0) {
		Vec p0, dJ;
		double norm_gref = 0.;
		ierr = VecDuplicate(ctx->tumor->p_initguess_, &p0); CHKERRQ(ierr);
		ierr = VecDuplicate(ctx->tumor->p_initguess_, &dJ); CHKERRQ(ierr);
		ierr = VecSet(p0, 0.); CHKERRQ(ierr);
		ierr = VecSet(dJ, 0.); CHKERRQ(ierr);
		// evaluate reference gradient for initial guess p = 0 * ones(Np)
		evaluateGradient(tao, p0, dJ, (void*) ctx);
		ierr = VecNorm(dJ, NORM_2, &norm_gref); CHKERRQ(ierr);
		ctx->gradnorm0 = norm_gref;
		//ctx->gradnorm0 = gnorm;
		ctx->updateGradNorm0 = false;
		ierr = tuMSGstd("updated reference gradient for relative convergence criterion, Gauß-Newton solver."); CHKERRQ(ierr);
		ierr = VecDestroy(&p0); CHKERRQ(ierr);
		ierr = VecDestroy(&dJ); CHKERRQ(ierr);
	}

	// get initial gradient
	g0norm = ctx->gradnorm0;
	g0norm = (g0norm > 0.0) ? g0norm : 1.0;

	ctx->convergenceMessage.clear();
	// check for NaN value
	if (PetscIsInfOrNanReal(J)) {
			ierr = tuMSGwarn("objective is NaN"); CHKERRQ(ierr);
			ierr = TaoSetConvergedReason(tao, TAO_DIVERGED_NAN); CHKERRQ(ierr);
			PetscFunctionReturn(ierr);
	}

	// check for NaN value
	if (PetscIsInfOrNanReal(gnorm)) {
			ierr = tuMSGwarn("||g|| is NaN"); CHKERRQ(ierr);
			ierr = TaoSetConvergedReason(tao, TAO_DIVERGED_NAN); CHKERRQ(ierr);
			PetscFunctionReturn(ierr);
	}
	if(verbosity >= 2) {
		ierr = PetscPrintf(MPI_COMM_WORLD, "||g(x)|| / ||g(x0)|| = %6E, ||g(x0)|| = %6E \n", gnorm/g0norm, g0norm);
	}

	// only check convergence criteria after a certain number of iterations
	stop[0] = false; stop[1] = false; stop[2] = false;
	ctx->converged = false;
	if (iter >= miniter) {
		if (verbosity > 1) {
				ss << "step size in linesearch: " << std::scientific << step;
				ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
				ss.str(std::string()); ss.clear();
		}
		if (step < minstep) {
				ss << "step  = " << std::scientific << step << " < " << minstep << " = " << "bound";
				ierr = tuMSGwarn(ss.str()); CHKERRQ(ierr);
				ss.str(std::string()); ss.clear();
				ierr = TaoSetConvergedReason(tao, TAO_CONVERGED_STEPTOL); CHKERRQ(ierr);
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
		ctx->convergenceMessage.push_back(ss.str());
		ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
		ss.str(std::string()); ss.clear();

		// ||g_k||_2 < tol
		if (gnorm < gatol) {
				ierr = TaoSetConvergedReason(tao, TAO_CONVERGED_GATOL); CHKERRQ(ierr);
				stop[1] = true;
		}
		ss  << "  " << stop[1] << "    ||g|| = " << std::setw(14)
				<< std::right << std::scientific << gnorm << "    <    "
				<< std::left << std::setw(14) << gatol << " = " << "tol";
		ctx->convergenceMessage.push_back(ss.str());
		ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
		ss.str(std::string()); ss.clear();

		// iteration number exceeds limit
		if (iter > maxiter) {
				ierr = TaoSetConvergedReason(tao, TAO_DIVERGED_MAXITS); CHKERRQ(ierr);
				stop[2] = true;
		}
		ss  << "  " << stop[2] << "     iter = " << std::setw(14)
				<< std::right << iter  << "    >    "
				<< std::left << std::setw(14) << maxiter << " = " << "maxiter";
		ctx->convergenceMessage.push_back(ss.str());
		ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
		ss.str(std::string()); ss.clear();

		// store objective function value
		ctx->jvalold = J;

		if (stop[0] || stop[1] || stop[2]) {
				ctx->converged = true;
				PetscFunctionReturn(ierr);
		}

	} else {
			// if the gradient is zero, we should terminate immediately
			if (gnorm == 0) {
					ss << "||g|| = " << std::scientific << 0.0 << " < " << gatol  << " = " << "bound";
					ierr = tuMSGwarn(ss.str()); CHKERRQ(ierr);
					ss.str(std::string()); ss.clear();
					ierr = TaoSetConvergedReason(tao, TAO_CONVERGED_GATOL); CHKERRQ(ierr);
					PetscFunctionReturn(ierr);
			}
	}

	// if we're here, we're good to go
	ierr = TaoSetConvergedReason(tao, TAO_CONTINUE_ITERATING); CHKERRQ(ierr);

	PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
/*
 checkConvergenceGradObj    checks convergence of the overall Gauß-Newton tumor inversion

 input parameters:
  . Tao tao       Tao solver object
  . void* ptr     optional user defined context
 */
PetscErrorCode checkConvergenceGradObj(Tao tao, void* ptr){
	PetscFunctionBegin;
	PetscErrorCode ierr = 0;

	PetscInt iter, maxiter, miniter, iterbound;
	PetscReal jx, jxold, gnorm, step, gatol, grtol,
							gttol, g0norm, gtolbound, minstep, theta,
							normx, normdx, tolj, tolx, tolg;
	const int nstop = 7;
	bool stop[nstop];
	std::stringstream ss;
	Vec x;

  CtxInv *ctx = reinterpret_cast<CtxInv*>(ptr);     // get user context
	// get minstep and miniter
	minstep = std::pow(2.0, 10.0);
	minstep = 1.0 / minstep;
	miniter = ctx->newton_minit;
	iterbound = ctx->iterbound;
	// get lower bound for gradient
	gtolbound = ctx->gtolbound;

#if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 7)
	ierr = TaoGetTolerances(tao, &gatol, &grtol, &gttol); CHKERRQ(ierr);
#else
	ierr = TaoGetTolerances(tao, NULL, NULL, &gatol, &grtol, &gttol); CHKERRQ(ierr);
#endif

	ierr = TaoGetMaximumIterations(tao, &maxiter); CHKERRQ(ierr);
	if (maxiter > iterbound) iterbound = maxiter;

	ierr = TaoGetMaximumIterations(tao, &maxiter); CHKERRQ(ierr);
	ierr = TaoGetSolutionStatus(tao, &iter, &jx, &gnorm, NULL, &step, NULL); CHKERRQ(ierr);
	ierr = TaoGetSolutionVector(tao, &x); CHKERRQ(ierr);

	// update/set reference gradient (with p = initial-guess)
	if(ctx->updateGradNorm0) {
		Vec p0, dJ;
		double norm_gref = 0.;
		ierr = VecDuplicate(ctx->tumor->p_initguess_, &p0); CHKERRQ(ierr);
		ierr = VecDuplicate(ctx->tumor->p_initguess_, &dJ); CHKERRQ(ierr);
		ierr = VecSet(p0, 0.); CHKERRQ(ierr);
		ierr = VecSet(dJ, 0.); CHKERRQ(ierr);
		// evaluate reference gradient for initial guess p = 0 * ones(Np)
		evaluateGradient(tao, p0, dJ, (void*) ctx);
		ierr = VecNorm(dJ, NORM_2, &norm_gref); CHKERRQ(ierr);
		ctx->gradnorm0 = norm_gref;
		ctx->updateGradNorm0 = false;
		ierr = tuMSGstd("updated reference gradient for relative convergence criterion, Gauß-Newton solver."); CHKERRQ(ierr);
		ierr = VecDestroy(&p0); CHKERRQ(ierr);
		ierr = VecDestroy(&dJ); CHKERRQ(ierr);
	}

	// get initial gradient
	g0norm = ctx->gradnorm0;
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
	ierr = ctx->tumor->phi_->Apply(ctx->tmp, x);  CHKERRQ(ierr);  // comp \Phi x
	ierr = VecNorm(ctx->tmp, NORM_2, &normx);     CHKERRQ(ierr);  // comp norm \Phi x
	ierr = VecAXPY(ctx->c0old, -1, ctx->tmp);     CHKERRQ(ierr);  // comp dx
	ierr = VecNorm(ctx->c0old, NORM_2, &normdx);  CHKERRQ(ierr);  // comp norm \Phi dx
	ierr = VecCopy(ctx->tmp, ctx->c0old);         CHKERRQ(ierr);  // save \Phi x
	// get old objective function value
	jxold = ctx->jvalold;

	ctx->convergenceMessage.clear();
	// check for NaN value
	if (PetscIsInfOrNanReal(jx)) {
			ierr = tuMSGwarn("objective is NaN"); CHKERRQ(ierr);
			ierr = TaoSetConvergedReason(tao, TAO_DIVERGED_NAN); CHKERRQ(ierr);
			PetscFunctionReturn(ierr);
	}

	// check for NaN value
	if (PetscIsInfOrNanReal(gnorm)) {
			ierr = tuMSGwarn("||g|| is NaN"); CHKERRQ(ierr);
			ierr = TaoSetConvergedReason(tao, TAO_DIVERGED_NAN); CHKERRQ(ierr);
			PetscFunctionReturn(ierr);
	}

	ierr = PetscPrintf(MPI_COMM_WORLD, "||g(x)|| / ||g(x0)|| = %6E, ||g(x0)|| = %6E \n", gnorm/g0norm, g0norm);

	ctx->converged = false;
	// initialize flags for stopping conditions
	for (int i = 0; i < nstop; ++i) stop[i] = false;

	// only check convergence criteria after a certain number of iterations
	if (iter >= miniter && iter > 1) {
		if (step < minstep) {
				ierr = TaoSetConvergedReason(tao, TAO_CONVERGED_STEPTOL); CHKERRQ(ierr);
				PetscFunctionReturn(ierr);
		}

		// |j(x_{k-1}) - j(x_k)| < tolj*abs(1+J)
		if (std::abs(jxold-jx) < tolj*theta) {
				stop[0] = true;
		}
		ss << "  " << stop[0] << "    |dJ|  = " << std::setw(14)
			 << std::right << std::scientific << std::abs(jxold-jx) << "    <    "
			 << std::left << std::setw(14) << tolj*theta << " = " << "tol*|1+J|";
		ctx->convergenceMessage.push_back(ss.str());
		ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
		ss.str(std::string()); ss.clear();

		// ||dx|| < sqrt(tolj)*(1+||x||)
		if (normdx < tolx*(1+normx)) {
				stop[1] = true;
		}
		ss << "  " << stop[1] << "    |dx|  = " << std::setw(14)
			 << std::right << std::scientific << normdx << "    <    "
			 << std::left << std::setw(14) << tolx*(1+normx) << " = " << "sqrt(tol)*(1+||x||)";
		ctx->convergenceMessage.push_back(ss.str());
		ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
		ss.str(std::string()); ss.clear();

		// ||g_k||_2 < cbrt(tolj)*abs(1+Jc)
		if (gnorm < tolg*theta) {
				stop[2] = true;
		}
		ss  << "  " << stop[2] << "    ||g|| = " << std::setw(14)
				<< std::right << std::scientific << gnorm << "    <    "
				<< std::left << std::setw(14) << tolg*theta << " = " << "cbrt(tol)*|1+J|";
		ctx->convergenceMessage.push_back(ss.str());
		ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
		ss.str(std::string()); ss.clear();

		// ||g_k||_2 < tol
		if (gnorm < gatol) {
				stop[3] = true;
		}
		ss  << "  " << stop[3] << "    ||g|| = " << std::setw(14)
				<< std::right << std::scientific << gnorm << "    <    "
				<< std::left << std::setw(14) << gatol << " = " << "tol";
		ctx->convergenceMessage.push_back(ss.str());
		ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
		ss.str(std::string()); ss.clear();

		if (gnorm < gtolbound*g0norm) {
				stop[4] = true;
		}
		ss  << "  " << stop[4] << "    ||g|| = " << std::setw(14)
				<< std::right << gnorm  << "    >    "
				<< std::left << std::setw(14) << gtolbound*g0norm << " = " << "kappa*||g0||";
		ctx->convergenceMessage.push_back(ss.str());
		ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
		ss.str(std::string()); ss.clear();

		if (iter > maxiter) {
				stop[5] = true;
		}
		ss  << "  " << stop[5] << "    iter  = " << std::setw(14)
				<< std::right << iter  << "    >    "
				<< std::left << std::setw(14) << maxiter << " = " << "maxiter";
		ctx->convergenceMessage.push_back(ss.str());
		ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
		ss.str(std::string()); ss.clear();

		if (iter > iterbound) {
				stop[6] = true;
		}
		ss  << "  " << stop[6] << "    iter  = " << std::setw(14)
				<< std::right << iter  << "    >    "
				<< std::left << std::setw(14) << iterbound << " = " << "iterbound";
		ctx->convergenceMessage.push_back(ss.str());
		ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
		ss.str(std::string()); ss.clear();

		// store objective function value
		ctx->jvalold = jx;

		if (stop[0] && stop[1] && stop[2]) {
				ierr = TaoSetConvergedReason(tao, TAO_CONVERGED_USER); CHKERRQ(ierr);
				ctx->converged = true;
				PetscFunctionReturn(ierr);
		} else if (stop[3]) {
				ierr = TaoSetConvergedReason(tao, TAO_CONVERGED_GATOL); CHKERRQ(ierr);
				ctx->converged = true;
				PetscFunctionReturn(ierr);
		} else if (stop[4] && stop[5]) {
				ierr = TaoSetConvergedReason(tao, TAO_DIVERGED_MAXITS); CHKERRQ(ierr);
				ctx->converged = true;
				PetscFunctionReturn(ierr);
		} else if (stop[6]) {
				ierr = TaoSetConvergedReason(tao, TAO_DIVERGED_MAXITS); CHKERRQ(ierr);
				ctx->converged = true;
				PetscFunctionReturn(ierr);
		}
} else {
		// if the gradient is zero, we should terminate immediately
		if (gnorm == 0) {
				ss << "||g|| = " << std::scientific << 0.0 << " < " << gatol  << " = " << "bound";
				ierr = tuMSGwarn(ss.str()); CHKERRQ(ierr);
				ss.str(std::string()); ss.clear();
				ierr = TaoSetConvergedReason(tao, TAO_CONVERGED_GATOL); CHKERRQ(ierr);
				PetscFunctionReturn(ierr);
		}
}

	// if we're here, we're good to go
	ierr = TaoSetConvergedReason(tao, TAO_CONTINUE_ITERATING); CHKERRQ(ierr);

	PetscFunctionReturn(0);
}

/**
 *  ***********************************
 */
PetscErrorCode dispTaoConvReason(TaoConvergedReason flag, std::string& solverstatus){
	PetscFunctionBegin;
	PetscErrorCode ierr = 0;
	switch (flag) {
      case TAO_CONVERGED_GATOL:
      {
          msg = "solver converged: ||g(x)|| <= tol";
          ierr = tuMSGwarn(msg); CHKERRQ(ierr);
          break;
      }
      case TAO_CONVERGED_GRTOL:
      {
          msg = "solver converged: ||g(x)||/J(x) <= tol";
          ierr = tuMSGwarn(msg); CHKERRQ(ierr);
          break;
      }
      case TAO_CONVERGED_GTTOL:
      {
          msg = "solver converged: ||g(x)||/||g(x0)|| <= tol";
          ierr = tuMSGwarn(msg); CHKERRQ(ierr);
          break;
      }
      case TAO_CONVERGED_STEPTOL:
      {
          msg = "step size too small";
          ierr = tuMSGwarn(msg); CHKERRQ(ierr);
          break;
      }
      case TAO_CONVERGED_MINF:
      {
          msg = "objective value to small";
          ierr = tuMSGwarn(msg); CHKERRQ(ierr);
          break;
      }
      case TAO_CONVERGED_USER:
      {
          msg = "solver converged";
          ierr = tuMSGwarn(msg); CHKERRQ(ierr);
          break;
      }
      case TAO_DIVERGED_MAXITS:
      {
          msg = "maximum number of iterations reached";
          ierr = tuMSGwarn(msg); CHKERRQ(ierr);
          break;
      }
      case TAO_DIVERGED_NAN:
      {
          msg = "numerical problems (NAN detected)";
          ierr = tuMSGwarn(msg); CHKERRQ(ierr);
          break;
      }
      case TAO_DIVERGED_MAXFCN:
      {
          msg = "maximal number of function evaluations reached";
          ierr = tuMSGwarn(msg); CHKERRQ(ierr);
          break;
      }
      case TAO_DIVERGED_LS_FAILURE:
      {
          msg = "line search failed";
          ierr = tuMSGwarn(msg); CHKERRQ(ierr);
          break;
      }
      case TAO_DIVERGED_TR_REDUCTION:
      {
          msg = "trust region failed";
          ierr = tuMSGwarn(msg); CHKERRQ(ierr);
          break;
      }
      case TAO_DIVERGED_USER:
      {
          msg = "user defined divergence criterion met";
          ierr = tuMSGwarn(msg); CHKERRQ(ierr);
          break;
      }
      case TAO_CONTINUE_ITERATING:
      {
          // display nothing
          break;
      }
      default:
      {
          msg = "convergence reason not defined";
          ierr = tuMSGwarn(msg); CHKERRQ(ierr);
          break;
      }
  }
	PetscFunctionReturn(0);
}

/**
 *  ***********************************
 */
PetscErrorCode setTaoOptions(Tao* tao, CtxInv* ctx){
	PetscFunctionBegin;
	PetscErrorCode ierr = 0;

	TaoLineSearch linesearch;        // line-search object
	ierr = TaoSetType(tao, "nls");   // set TAO solver type
	PetscBool flag = PETSC_FALSE
	std::string msg;
#if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 7)
	  PetscOptionsHasName(NULL, NULL, "-tao_nls_pc_type", &flag);
	  if (flag == PETSC_FALSE)
	    PetscOptionsSetValue(NULL, "-tao_nls_pc_type", "none");
#else
	  PetscOptionsHasName(NULL, "-tao_nls_pc_type", &flag);
	  if (flag == PETSC_FALSE)
	    PetscOptionsSetValue("-tao_nls_pc_type", "none");
#endif
	flag = PETSC_FALSE;
#if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 7)
	PetscOptionsHasName(NULL, NULL, "-tao_nls_ksp_type", &flag);
	if (flag == PETSC_FALSE)
		PetscOptionsSetValue(NULL, "-tao_nls_ksp_type", "cg");
#else
	PetscOptionsHasName(NULL, "-tao_nls_ksp_type", &flag);
	if (flag == PETSC_FALSE)
		PetscOptionsSetValue("-tao_nls_ksp_type", "cg");
#endif
  flag = PETSC_FALSE;
#if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 7)
  PetscOptionsHasName(NULL, NULL, "-tao_ntr_pc_type", &flag);
  if (flag == PETSC_FALSE)
	  PetscOptionsSetValue(NULL, "-tao_ntr_pc_type", "none");
#else
  PetscOptionsHasName(NULL, "-tao_ntr_pc_type", &flag);
  if (flag == PETSC_FALSE)
	  PetscOptionsSetValue("-tao_ntr_pc_type", "none");
#endif

  // parse options user has set
  ierr = TaoSetFromOptions(tao);
  // set the initial vector
  ierr = TaoSetInitialVector(tao, tumor->p_initguess_);
  // set routine for evaluating the objective
  ierr = TaoSetObjectiveRoutine(tao, evaluateObjectiveFunction, (void*) ctx);
  // set routine for evaluating the Gradient
  ierr = TaoSetGradientRoutine(tao, evaluateGradient, (void*) ctx);

  // TAO type from user input
	const TaoType taotype;
  ierr = TaoGetType(tao, &taotype);
  if (strcmp(taotype, "nls") == 0) {
    msg =
      " limited memory variable metric method (unconstrained) selected\n";
  } else if (strcmp(taotype, "ntr") == 0) {
    msg =
      " Newton's method with trust region for unconstrained minimization\n";
  } else if (strcmp(taotype, "ntl") == 0) {
    msg =
      " Newton's method with trust region, line search for unconstrained minimization\n";
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
    msg =
      " Newton Trust Region method for quadratic bound constrained minimization\n";
  } else {
    msg =
      " numerical optimization method not supported (setting default: LMVM)\n";
    ierr = TaoSetType(tao, "lmvm");
    ;
  }

	// set tolerances
#if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 7)
  ierr = TaoSetTolerances(tao, ctx->gatol, ctx->grtol, ctx->gttol);
#else
  ierr = TaoSetTolerances(tao, 1E-12, 1E-12, ctx->gatol, ctx->grtol, ctx->gttol);
#endif
  ierr = TaoSetMaximumIterations(tao, ctx->newton_maxit);

  // set adapted convergence test for warm-starts
  ierr = TaoSetConvergenceTest(tao, CheckConvergenceGrad, ctx);
  //ierr = TaoSetConvergenceTest(tao, CheckConvergenceGradObj, ctx);

	// set linesearch
  ierr = TaoGetLineSearch(tao, &linesearch);
  ierr = TaoLineSearchSetType(linesearch, "armijo");

  std::stringstream s;
  tuMSGstd(" parameters (optimizer):");
  tuMSGstd(" tolerances (stopping conditions):");
  s << "   gatol: "<< ctx->gatol;  /*pout(s.str(), cplctx->_fileOutput);*/ tuMSGstd(s.str()); s.str(""); s.clear();
  s << "   grtol: "<< ctx->grtol;  /*pout(s.str(), cplctx->_fileOutput);*/ tuMSGstd(s.str()); s.str(""); s.clear();
  s << "   gttol: "<< ctx->gttol;  /*pout(s.str(), cplctx->_fileOutput);*/ tuMSGstd(s.str()); s.str(""); s.clear();

  ierr = TaoSetFromOptions(tao);

	/* === set the KSP Krylov solver settings === */
  KSP ksp = PETSC_NULL;
  ierr = TaoGetKSP(tao, &ksp);                          // get the ksp of the optimizer
  if (ksp != PETSC_NULL) {
    ierr = KSPSetOptionsPrefix(ksp, "opt_");            // set prefix to control sets and monitors
		                                                    // set default tolerance to 1E-6
    ierr = KSPSetTolerances(ksp, 1E-6, PETSC_DEFAULT, PETSC_DEFAULT, ctx->krylov_maxit);
    KSPSetPreSolve(ksp, PreKrylovSolve, ctx);           // to use Eisenstat/Walker convergence crit.
    ierr = KSPMonitorSet(ksp,hessianKSPMonitor,ctx, 0); // monitor
  }

	// set the preconditioner (we check if KSP exists, as there are also
	// solvers that do not require a KSP solve (BFGS and friends))
	if (ksp != PETSC_NULL) {
		PC pc;
		ierr = KSPGetPC(ksp, &pc);
		ierr = PCSetType(pc, PCSHELL);
		ierr = PCShellSetApply(pc, PrecondMatVec);
		ierr = PCShellSetContext(pc, ctx);
	}

	// set the routine to evaluate the objective and compute the gradient
  ierr = TaoSetObjectiveAndGradientRoutine(tao, evaluateObjectiveFunctionAndGradient, (void*) ctx);
  // set up routine to compute the hessian matrix vector product
  Mat H; int nH = tumor->N_Misc_->Np_;
	ierr = MatCreateShell(MPI_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, nH, nH, (void*) ctx, &H);
	ierr = MatShellSetOperation(H, MATOP_MULT, (void (*)(void))HessianMatVec);
  ierr = MatSetOption(H, MAT_SYMMETRIC, PETSC_TRUE);
  ierr = TaoSetHessianRoutine(tao, H, H, matfreeHessian, (void *) ctx);

  // set monitor function
  ierr = TaoSetMonitor(tao, OptimizationMonitor, (void *) ctx, NULL);
	// Lower and Upper Bounds
	// ierr = TaoSetVariableBounds(tao, tumor->lowerb_, tumor->upperb_);

	PetscFunctionReturn(0);
}
