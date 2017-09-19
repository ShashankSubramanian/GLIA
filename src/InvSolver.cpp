#include "InvSolver.h"

InvSolver::InvSolver (DerivativeOperators *derivative_operators, NMisc *n_misc) {
	PetscErrorCode ierr = 0;
	Ctx *ctx = new Ctx ();
	Ctx->derivative_operators_ = derivative_operators;
	Ctx->n_misc_ = n_misc;

	ierr = MatCreateShell (PETSC_COMM_WORLD, n_misc->n_local_, n_misc->n_local_, n_misc->n_global_, n_misc->n_global_, ctx, &A_);	
	ierr = MatShellSetOperation (A_, MATOP_MULT, (void(*)(void)) hessianMatVec);														 

	ierr = TaoCreate (PETSC_COMM_WORLD, &tao_);
	ierr = TaoSetType (tao_, "nls");
	ierr = TaoSetFromOptions (tao_);
}

PetscErrorCode hessianMatVec (Mat A, Vec x, Vec y) {    //y = Ax
	PetscErrorCode ierr = 0;
	Ctx *ctx;
	ierr = MatShellGetContext (A, &ctx);						CHKERRQ (ierr);

	ierr = ctx->derivative_operators_->evaluateHessian (x, y, ctx->n_misc_);

	return ierr;
}

PetscErrorCode formHessian (Tao tao, Vec x, Mat H, Mat precH, void *ptr) {    
	PetscErrorCode ierr = 0;
	return ierr;
}

PetscErrorCode formGradient (Tao tao, Vec x, PetscReal *dJ, void *ptr) {    
	PetscErrorCode ierr = 0;
	Ctx *ctx = (Ctx *) ptr;
	ierr = ctx->derivative_operators_->evaluateGradient (dJ, x, ctx->n_misc_);
	return ierr;
}

PetscErrorCode formFunction (Tao tao, Vec x, PetscReal *J, void *ptr) {    
	PetscErrorCode ierr = 0;
	Ctx *ctx = (Ctx *) ptr;
	ierr = ctx->derivative_operators_->evaluateObjective (J, x, ctx->n_misc_);
	return ierr;
}

PetscErrorCode InvSolver::solve () {
	PetscErrorCode ierr = 0;
	Ctx *ctx;
	ierr = MatShellGetContext (A, &ctx);															CHKERRQ (ierr);
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
	return ierr;
}

InvSolver::~InvSolver () {
	PetscErrorCode ierr = 0;
	ierr = TaoDestroy (&tao_);
	ierr = MatDestroy (&A_)
}