#include "AdvectionSolver.h"

AdvectionSolver::AdvectionSolver (std::shared_ptr<NMisc> n_misc, std::shared_ptr<Tumor> tumor) : ctx_ () {
	PetscErrorCode ierr = 0;
    ctx_ = std::make_shared<CtxAdv> ();
    ctx_->n_misc_ = n_misc;
    ctx_->dt_ = n_misc->dt_;
    ctx_->temp_.resize (3);
    for (int i = 0; i < 3; i++)
    	ctx_->temp_[i] = tumor->work_[11 - i]; 	// Choose some tumor work vector

    ctx_->velocity_ = std::make_shared<VecField> (n_misc->n_local_, n_misc->n_global_);

    ierr = MatCreateShell (PETSC_COMM_WORLD, n_misc->n_local_, n_misc->n_local_, n_misc->n_global_, n_misc->n_global_, ctx_.get(), &A_);
    ierr = MatShellSetOperation (A_, MATOP_MULT, (void(*)(void)) operatorAdv);

    ierr = KSPCreate (PETSC_COMM_WORLD, &ksp_);
    ierr = KSPSetOperators (ksp_, A_, A_);
    ierr = KSPSetTolerances (ksp_, 1e-6, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT);
    ierr = KSPSetType (ksp_, KSPGMRES);
    ierr = KSPSetFromOptions (ksp_);
    ierr = KSPSetUp (ksp_);

    ierr = VecCreate (PETSC_COMM_WORLD, &rhs_);
    ierr = VecSetSizes (rhs_, n_misc->n_local_, n_misc->n_global_);
    ierr = VecSetFromOptions (rhs_);
    ierr = VecSet (rhs_, 0);
}

// LHS for transport equation using Crank-Nicolson 
// y = Ax = (x + dt/2 div (xv))
PetscErrorCode operatorAdv (Mat A, Vec x, Vec y) {
	PetscFunctionBegin;
	PetscErrorCode ierr = 0;

	Event e ("tumor-advection-ksp-matvec");
    std::array<double, 7> t = {0};
    double self_exec_time = -MPI_Wtime ();
    CtxAdv *ctx;
    ierr = MatShellGetContext (A, &ctx);                        CHKERRQ (ierr);

    double alph = 1.0 / 2.0 * ctx->dt_;

    ierr = VecPointwiseMult (ctx->temp_[0], ctx->velocity_->x_, x);			CHKERRQ (ierr);
    ierr = VecPointwiseMult (ctx->temp_[1], ctx->velocity_->y_, x);			CHKERRQ (ierr);
    ierr = VecPointwiseMult (ctx->temp_[2], ctx->velocity_->z_, x);			CHKERRQ (ierr);

    accfft_divergence (y, ctx->temp_[0], ctx->temp_[1], ctx->temp_[2], ctx->n_misc_->plan_, t.data());

    ierr = VecScale (y, alph);									CHKERRQ (ierr);
    ierr = VecAXPY (y, 1.0, x);									CHKERRQ (ierr);

    self_exec_time += MPI_Wtime();
    accumulateTimers (ctx->n_misc_->timers_, t, self_exec_time);
    e.addTimings (t);
    e.stop ();
	PetscFunctionReturn (0);
}

PetscErrorCode TrapezoidalSolver::solve (Vec scalar, std::shared_ptr<VecField> velocity, double dt) {
	PetscFunctionBegin;
	PetscErrorCode ierr = 0;

	Event e ("tumor-advection-solve");
    std::array<double, 7> t = {0};
    double self_exec_time = -MPI_Wtime ();

    CtxAdv *ctx;
    ierr = MatShellGetContext (A_, &ctx);                       CHKERRQ (ierr);
    ctx->dt_ = dt;
    ctx->velocity_->x_ = velocity->x_;
    ctx->velocity_->y_ = velocity->y_;
    ctx->velocity_->z_ = velocity->z_;

    double alph = -1.0 / 2.0 * ctx->dt_;
    //rhs for advection solve: b = scalar - dt/2 div(scalar v)
    ierr = VecPointwiseMult (ctx->temp_[0], velocity->x_, scalar);			CHKERRQ (ierr);
    ierr = VecPointwiseMult (ctx->temp_[1], velocity->y_, scalar);			CHKERRQ (ierr);
    ierr = VecPointwiseMult (ctx->temp_[2], velocity->z_, scalar);			CHKERRQ (ierr);

    accfft_divergence (rhs_, ctx->temp_[0], ctx->temp_[1], ctx->temp_[2], ctx->n_misc_->plan_, t.data());

    ierr = VecScale (rhs_, alph);									CHKERRQ (ierr);
    ierr = VecAXPY (rhs_, 1.0, scalar);							    CHKERRQ (ierr);
    //KSP solve
    ierr = KSPSolve (ksp_, rhs_, scalar);                            CHKERRQ (ierr);

    self_exec_time += MPI_Wtime();
    accumulateTimers (ctx->n_misc_->timers_, t, self_exec_time);
    e.addTimings (t);
    e.stop ();
	PetscFunctionReturn (0);
}

AdvectionSolver::~AdvectionSolver () {
	PetscErrorCode ierr = 0;
	ierr = MatDestroy (&A_);
    ierr = KSPDestroy (&ksp_);
    ierr = VecDestroy (&rhs_);
}