#include "InvSolver.h"
#include "petsctao.h"
#include <iostream>
#include <limits>
#include "DerivativeOperators.h"
#include "PdeOperators.h"
#include "utils.h"
#include "EventTimings.hpp"
#include "IO.hpp"

PetscErrorCode hessianMatVec (Mat A, Vec x, Vec y);

InvSolver::InvSolver (std::shared_ptr <DerivativeOperators> derivative_operators, std::shared_ptr <NMisc> n_misc)
:
  initialized_(false),
  itctx_()
  {
	  PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    if(derivative_operators_ != nullptr && n_misc != nullptr) {
      ierr = initialize(derivative_operators, n_misc) CHKERRQ(ierr);
	  }
}

PetscErrorCode InvSolver::initialize(std::shared_ptr <DerivativeOperators> derivative_operators, std::shared_ptr <NMisc> n_misc) {
  PetscFunctionBegin;
	PetscErrorCode ierr = 0;
	if(initialized_) PetscFunctionReturn(0);

	itctx_ = std::make_shared<CtxInv>();
	ctx->derivative_operators_ = derivative_operators;
	ctx->n_misc_ = n_misc;

	ierr = MatCreateShell (PETSC_COMM_WORLD, n_misc->n_local_, n_misc->n_local_, n_misc->n_global_, n_misc->n_global_, ctx.get(), &A_);
	ierr = MatShellSetOperation (A_, MATOP_MULT, (void(*)(void)) hessianMatVec);

	ierr = TaoCreate (PETSC_COMM_WORLD, &tao_);
	ierr = TaoSetType (tao_, "nls");
	ierr = TaoSetFromOptions (tao_);

  initialized_ = true;
	PetscFunctionReturn(0);
}

PetscErrorCode InvSolver::solve () {
	PetscFunctionBegin;
	PetscErrorCode ierr = 0;
	CtxInv *ctx;
	ierr = MatShellGetContext (A_, &ctx);															CHKERRQ (ierr);
	ierr = TaoSetObjectiveRoutine (tao_, formFunction, (void*) ctx);								CHKERRQ (ierr);
	ierr = TaoSetGradientRoutine (tao_, formGradient, (void*) ctx);									CHKERRQ (ierr);

	double ga_tol, gr_tol, gt_tol;
	ierr = TaoGetTolerances (tao_, &ga_tol, &gr_tol, &gt_tol);										CHKERRQ (ierr);
	ga_tol = 1e-6;
	gr_tol = 1e-6;
	gt_tol = 1e-6;
	ierr = TaoSetTolerances (tao_, ga_tol, gr_tol, gt_tol);											CHKERRQ (ierr);

	ierr = TaoSetHessianRoutine (tao_, A_, A_, formHessian, (void*) ctx);							CHKERRQ (ierr);

	// ierr = TaoSolve (tao);																			CHKERRQ (ierr);
	PetscFunctionReturn(0);
}

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

/**
 *  ***********************************
 */
PetscErrorCode preconditionerMatVec(PC, Vec, Vec){
	PetscFunctionBegin;
	PetscErrorCode ierr = 0;

	PetscFunctionReturn(0);
}

/**
 *  ***********************************
 */
PetscErrorCode applyPreconditioner(void*, Vec, Vec){
	PetscFunctionBegin;
	PetscErrorCode ierr = 0;

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

	PetscFunctionReturn(0);
}

/**
 *  ***********************************
 */
PetscErrorCode hessianKSPMonitor(KSP ksp,PetscInt n,PetscReal rnorm, void *dummy){
	PetscFunctionBegin;
	PetscErrorCode ierr = 0;

	PetscFunctionReturn(0);
}

/**
 *  ***********************************
 */
PetscErrorCode preKrylovSolve(KSP ksp, Vec b, Vec x, void* ptr){
	PetscFunctionBegin;
	PetscErrorCode ierr = 0;

	PetscFunctionReturn(0);
}

/**
 *  ***********************************
 */
PetscErrorCode checkConvergenceGrad(Tao tao, void* ptr){
	PetscFunctionBegin;
	PetscErrorCode ierr = 0;

	PetscFunctionReturn(0);
}

/**
 *  ***********************************
 */
PetscErrorCode checkConvergenceGradObj(Tao tao, void* ptr){
	PetscFunctionBegin;
	PetscErrorCode ierr = 0;

	PetscFunctionReturn(0);
}

/**
 *  ***********************************
 */
PetscErrorCode dispTaoConvReason(TaoConvergedReason flag, std::string& solverstatus){
	PetscFunctionBegin;
	PetscErrorCode ierr = 0;

	PetscFunctionReturn(0);
}

/**
 *  ***********************************
 */
PetscErrorCode setTaoOptions(Tao* tao, Tumor* tumor, InverseTumorContext* ctx){
	PetscFunctionBegin;
	PetscErrorCode ierr = 0;

	PetscFunctionReturn(0);
}
