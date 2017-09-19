#include "InvSolver.h"
#include "petsctao.h"
#include <iostream>
#include <limits>
#include "DerivativeOperators.h"
#include "PdeOperators.h"
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

	CtxInv *ctx = reinterpret_cast<CtxInv*>(ptr);
	ierr = ctx->derivative_operators_->evaluateObjective (J, x); // TODO: add data used to evaluate

  // timing
	self_exec_time += MPI_Wtime();
  t[5] = self_exec_time;
  itctx->tumor->timers_[0] += t[0];
  itctx->tumor->timers_[1] += t[1];
  itctx->tumor->timers_[2] += t[2];
  itctx->tumor->timers_[3] += t[3];
  itctx->tumor->timers_[4] += t[4];
  itctx->tumor->timers_[5] += t[5];
  itctx->tumor->timers_[6] += t[6];
  e.addTimings(t);
  e.stop();

	PetscFunctionReturn(0);
}

/**
 *  ***********************************
 */
PetscErrorCode evaluateGradient (Tao tao, Vec x, Vec dJ, void *ptr) {
	PetscFunctionBegin;
	PetscErrorCode ierr = 0;
	CtxInv *ctx = reinterpret_cast<CtxInv*>(ptr);
	ierr = ctx->derivative_operators_->evaluateGradient (dJ, x);
	PetscFunctionReturn(0);
}

/**
 *  ***********************************
 */
PetscErrorCode evaluateObjectiveFunctionAndGradient(Tao, Vec, PetscReal *, Vec, void *){
	PetscFunctionBegin;
	PetscErrorCode ierr = 0;

	PetscFunctionReturn(0);
}

/**
 *  ***********************************
 */
PetscErrorCode hessianMatVec (Mat A, Vec x, Vec y) {    //y = Ax
	PetscFunctionBegin;
	PetscErrorCode ierr = 0;
	CtxInv *ctx;
	ierr = MatShellGetContext (A, &ctx);						CHKERRQ (ierr);
	ierr = ctx->derivative_operators_->evaluateHessian (x, y);
	PetscFunctionReturn(0);
}

/**
 *  ***********************************
 */
PetscErrorCode formHessian (Tao tao, Vec x, Mat H, Mat precH, void *ptr) {
	PetscFunctionBegin;
	PetscErrorCode ierr = 0;
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

/**
 *  ***********************************
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
