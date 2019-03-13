#include <ElasticitySolver.h>

ElasiticitySolver::ElasiticitySolver (std::shared_ptr<NMisc> n_misc, std::shared_ptr<Tumor> tumor) : ctx_ () {
	PetscErrorCode ierr = 0;
    ctx_ = std::make_shared<CtxElasticity> ();
    ctx_->n_misc_ = n_misc;
    ctx_->tumor_ = tumor;

    // compute average coefficients
    ctx->mu_avg_ = (ctx->computeMu (n_misc->E_healthy_, n_misc->nu_healthy_) + ctx->computeMu (n_misc->E_bg_, n_misc->nu_bg_)
    				+ ctx->computeMu (n_misc->E_csf_, n_misc->nu_csf_) + ctx->computeMu (n_misc->E_tumor_, n_misc->nu_tumor_)) / 4;
    ctx->lam_avg_ = (ctx->computeLam (n_misc->E_healthy_, n_misc->nu_healthy_) + ctx->computeLam (n_misc->E_bg_, n_misc->nu_bg_)
    				+ ctx->computeLam (n_misc->E_csf_, n_misc->nu_csf_) + ctx->computeLam (n_misc->E_tumor_, n_misc->nu_tumor_)) / 4;
    ctx->screen_avg_ = (n_misc->screen_low_ + n_misc->screen_high_) / 2;


    int factor = 3;   // vector equations
    ierr = MatCreateShell (PETSC_COMM_WORLD, factor * n_misc->n_local_, factor * n_misc->n_local_, factor * n_misc->n_global_, factor * n_misc->n_global_, ctx_.get(), &A_);
    ierr = MatShellSetOperation (A_, MATOP_MULT, (void(*)(void)) operatorConstantCoefficients);

    ierr = KSPCreate (PETSC_COMM_WORLD, &ksp_);
    ierr = KSPSetOperators (ksp_, A_, A_);
    ierr = KSPSetTolerances (ksp_, 1e-3, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT);
    ierr = KSPSetType (ksp_, KSPGMRES);
    ierr = KSPSetFromOptions (ksp_);
    ierr = KSPSetUp (ksp_);

    ierr = VecCreate (PETSC_COMM_WORLD, &rhs_);
    ierr = VecSetSizes (rhs_, factor * n_misc->n_local_, factor * n_misc->n_global_);
    ierr = VecSetFromOptions (rhs_);
    ierr = VecSet (rhs_, 0);
}

PetscErrorCode operatorConstantCoefficients (Mat A, Vec x, Vec y) {
	PetscFunctionBegin;
	PetscErrorCode ierr = 0;

	Event e ("tumor-elasticity-constantcoefficients-ksp-matvec");
    std::array<double, 7> t = {0};
    double self_exec_time = -MPI_Wtime ();
    CtxElasticity *ctx;
    ierr = MatShellGetContext (A, &ctx);                        CHKERRQ (ierr);

    std::shared_ptr<NMisc> n_misc = ctx->n_misc_;
    std::shared_ptr<Tumor> tumor = ctx->tumor_;
    int factor = 3;

    std::shared_ptr<VecField> force = std::make_shared<VecField> (n_misc->n_local_, n_misc->n_global_);
    ierr = force->setIndividualComponents (y);		// sets components of y vector in f
    // FFT of each component
    Complex fx_hat = (Complex*) accfft_alloc (n_misc->accfft_alloc_max_);
    Complex fy_hat = (Complex*) accfft_alloc (n_misc->accfft_alloc_max_);
    Complex fz_hat = (Complex*) accfft_alloc (n_misc->accfft_alloc_max_);

    double *fx_ptr, *fy_ptr, *fz_ptr;
    ierr = force->getComponentArrays (fx_ptr, fy_ptr, fz_ptr);
    accfft_execute_r2c (n_misc->plan_, fx_ptr, fx_hat);
    accfft_execute_r2c (n_misc->plan_, fy_ptr, fy_hat);
    accfft_execute_r2c (n_misc->plan_, fz_ptr, fz_hat);

    PetscScalar s1, s2, s3;
    


    self_exec_time += MPI_Wtime();
    accumulateTimers (ctx->n_misc_->timers_, t, self_exec_time);
    e.addTimings (t);
    e.stop ();
	PetscFunctionReturn (0);
}

