#include "InvSolver.h"

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

PetscErrorCode initialize(std::shared_ptr <DerivativeOperators> derivative_operators, std::shared_ptr <NMisc> n_misc) {
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

PetscErrorCode hessianMatVec (Mat A, Vec x, Vec y) {    //y = Ax
	PetscFunctionBegin;
	PetscErrorCode ierr = 0;
	CtxInv *ctx;
	ierr = MatShellGetContext (A, &ctx);						CHKERRQ (ierr);
	ierr = ctx->derivative_operators_->evaluateHessian (x, y);
	PetscFunctionReturn(0);
}

PetscErrorCode formHessian (Tao tao, Vec x, Mat H, Mat precH, void *ptr) {
	PetscFunctionBegin;
	PetscErrorCode ierr = 0;
	PetscFunctionReturn(0);
}

PetscErrorCode formGradient (Tao tao, Vec x, Vec dJ, void *ptr) {
	PetscFunctionBegin;
	PetscErrorCode ierr = 0;
	CtxInv *ctx = reinterpret_cast<CtxInv*>(ptr);
	ierr = ctx->derivative_operators_->evaluateGradient (dJ, x);
	PetscFunctionReturn(0);
}

PetscErrorCode formFunction (Tao tao, Vec x, PetscReal *J, void *ptr) {
	PetscFunctionBegin;
	PetscErrorCode ierr = 0;
	CtxInv *ctx = reinterpret_cast<CtxInv*>(ptr);
	ierr = ctx->derivative_operators_->evaluateObjective (J, x);
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
