#include "DiffSolver.h"

DiffSolver::DiffSolver (std::shared_ptr<NMisc> n_misc, std::shared_ptr<DiffCoef> k) {
	PetscErrorCode ierr = 0;
	Ctx *ctx = new Ctx ();
	ctx->k_ = k;
	ctx->dt_ = n_misc->dt_;
	ctx->plan_ = n_misc->plan_;
	ctx->temp_ = k->temp_[0];

	ierr = MatCreateShell (PETSC_COMM_WORLD, n_misc->n_local_, n_misc->n_local_, n_misc->n_global_, n_misc->n_global_, ctx, &A_);
	ierr = MatShellSetOperation (A_, MATOP_MULT, (void(*)(void)) operatorA);

	ierr = KSPCreate (PETSC_COMM_WORLD, &ksp_);
	ierr = KSPSetOperators (ksp_, A_, A_);
	ierr = KSPSetTolerances (ksp_, 1e-6, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT);
	ierr = KSPSetType (ksp_, KSPCG);
	ierr = KSPSetFromOptions (ksp_);
	ierr = KSPSetUp (ksp_);

	ierr = VecCreate (PETSC_COMM_WORLD, &rhs_);
	ierr = VecSetSizes (rhs_, n_misc->n_local_, n_misc->n_global_);
	ierr = VecSetFromOptions (rhs_);
	ierr = VecSet (rhs_, 0);
}

PetscErrorCode DiffSolver::operatorA (Mat A, Vec x, Vec y) {    //y = Ax
	PetscErrorCode ierr = 0;
	Ctx *ctx;
	ierr = MatShellGetContext (A, &ctx);						CHKERRQ (ierr);
	ierr = VecCopy (x, y);										CHKERRQ (ierr);

	double alph = -1.0 / 2.0 * ctx->dt_;
	ierr = ctx->k_->applyD (ctx->temp_, y, ctx->plan_);
	ierr = VecAXPY (y, alph, ctx->temp_);						CHKERRQ (ierr);
	PetscFunctionReturn(0);;
}

PetscErrorCode DiffSolver::solve (Vec c, double dt) {
	PetscErrorCode ierr = 0;

	Ctx *ctx;
	ierr = MatShellGetContext (A_, &ctx);						CHKERRQ (ierr);
    ctx->dt_ = dt;
	if (ctx->k_->k_scale_ == 0) {
		return 0;
	}
	double alph = 1.0 / 2.0 * ctx->dt_;
	ierr = VecCopy (c, rhs_); 									CHKERRQ (ierr);
	ierr = ctx->k_->applyD (ctx->temp_, rhs_, ctx->plan_);
	ierr = VecAXPY (rhs_, alph, ctx->temp_);					CHKERRQ (ierr);

	//KSP solve
	ierr = KSPSolve (ksp_, rhs_, c);							CHKERRQ (ierr);

	//Debug
	int itr;
	ierr = KSPGetIterationNumber (ksp_, &itr);					CHKERRQ (ierr);
	
	PetscFunctionReturn(0);
}

DiffSolver::~DiffSolver () {
    PetscErrorCode ierr = 0;
    ierr = MatDestroy (&A_);
    ierr = KSPDestroy (&ksp_);
    ierr = VecDestroy (&rhs_);
}
