#include <ElasticitySolver.h>

ElasticitySolver::ElasticitySolver (std::shared_ptr<NMisc> n_misc, std::shared_ptr<Tumor> tumor) : ctx_ () {
	PetscErrorCode ierr = 0;
    ctx_ = std::make_shared<CtxElasticity> ();
    ctx_->n_misc_ = n_misc;
    ctx_->tumor_ = tumor;

    // compute average coefficients
    ctx_->mu_avg_ = (ctx_->computeMu (n_misc->E_healthy_, n_misc->nu_healthy_) + ctx_->computeMu (n_misc->E_bg_, n_misc->nu_bg_)
    				+ ctx_->computeMu (n_misc->E_csf_, n_misc->nu_csf_) + ctx_->computeMu (n_misc->E_tumor_, n_misc->nu_tumor_)) / 4;
    ctx_->lam_avg_ = (ctx_->computeLam (n_misc->E_healthy_, n_misc->nu_healthy_) + ctx_->computeLam (n_misc->E_bg_, n_misc->nu_bg_)
    				+ ctx_->computeLam (n_misc->E_csf_, n_misc->nu_csf_) + ctx_->computeLam (n_misc->E_tumor_, n_misc->nu_tumor_)) / 4;
    ctx_->screen_avg_ = (n_misc->screen_low_ + n_misc->screen_high_) / 2;


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
    std::shared_ptr<VecField> displacement = std::make_shared<VecField> (n_misc->n_local_, n_misc->n_global_);
    ierr = force->setIndividualComponents (y);		// sets components of y vector in f
    ierr = displacement->setIndividualComponents (x);
    // FFT of each component
    Complex *fx_hat = (Complex*) accfft_alloc (n_misc->accfft_alloc_max_);
    Complex *fy_hat = (Complex*) accfft_alloc (n_misc->accfft_alloc_max_);
    Complex *fz_hat = (Complex*) accfft_alloc (n_misc->accfft_alloc_max_);

    Complex *ux_hat = (Complex*) accfft_alloc (n_misc->accfft_alloc_max_);
    Complex *uy_hat = (Complex*) accfft_alloc (n_misc->accfft_alloc_max_);
    Complex *uz_hat = (Complex*) accfft_alloc (n_misc->accfft_alloc_max_);

    double *fx_ptr, *fy_ptr, *fz_ptr;
    double *ux_ptr, *uy_ptr, *uz_ptr;

    ierr = force->getComponentArrays (fx_ptr, fy_ptr, fz_ptr);
    ierr = displacement->getComponentArrays (ux_ptr, uy_ptr, uz_ptr);

    accfft_execute_r2c (n_misc->plan_, fx_ptr, fx_hat);
    accfft_execute_r2c (n_misc->plan_, fy_ptr, fy_hat);
    accfft_execute_r2c (n_misc->plan_, fz_ptr, fz_hat);

    double s1, s2, s1_square, s3, scale;

    int64_t wx, wy, wz;
    double wTw;
    int64_t x_global, y_global, z_global;
    int64_t ptr;

    for (int i = 0; i < n_misc->osize_[0]; i++) {
    	for (int j = 0; j < n_misc->osize_[1]; j++) {
    		for (int k = 0; k < n_misc->osize_[2]; k++) {
    			ptr = i * n_misc->osize_[1] * n_misc->osize_[2] + j * n_misc->osize_[2] + k;

    			x_global = i + n_misc->ostart_[0];
    			y_global = j + n_misc->ostart_[1];
    			z_global = k + n_misc->ostart_[2];

    			wx = x_global;
    			if (x_global > n_misc->n_[0] / 2) // symmetric frequencies
    				wx -= n_misc->n_[0];
    			if (x_global == n_misc->n_[0]) // nyquist frequency
    				wx = 0;

    			wy = y_global;
    			if (y_global > n_misc->n_[1] / 2) // symmetric frequencies
    				wy -= n_misc->n_[1];
    			if (y_global == n_misc->n_[1]) // nyquist frequency
    				wy = 0;

    			wz = z_global;
    			if (z_global > n_misc->n_[2] / 2) // symmetric frequencies
    				wz -= n_misc->n_[2];
    			if (z_global == n_misc->n_[2]) // nyquist frequency
    				wz = 0;

    			wTw = -1.0 * (wx * wx + wy * wx + wz * wz);

    			s1 = -ctx->screen_avg_ + ctx->mu_avg_ * wTw;
    			s1_square = s1 * s1;
    			s2 = ctx->lam_avg_ + ctx->mu_avg_;
    			s3 = 1.0 / (1.0 + (wTw * s2) / s1);

    			// real part
    			scale = -1.0 * (wx * wx * fx_hat[ptr][0] + wx * wy * fy_hat[ptr][0] + wx * wz * fz_hat[ptr][0]);
    			ux_hat[ptr][0] = fx_hat[ptr][0] * ((1.0 / s1) - (1.0 / s1_square) * s2 * s3 * scale); 
    			// imaginary part
    			scale = -1.0 * (wx * wx * fx_hat[ptr][1] + wx * wy * fy_hat[ptr][1] + wx * wz * fz_hat[ptr][1]);
    			ux_hat[ptr][1] = fx_hat[ptr][1] * ((1.0 / s1) - (1.0 / s1_square) * s2 * s3 * scale); 

    			// real part
    			scale = -1.0 * (wy * wx * fx_hat[ptr][0] + wy * wy * fy_hat[ptr][0] + wy * wz * fz_hat[ptr][0]);
    			uy_hat[ptr][0] = fy_hat[ptr][0] * ((1.0 / s1) - (1.0 / s1_square) * s2 * s3 * scale); 
    			// imaginary part
    			scale = -1.0 * (wy * wx * fx_hat[ptr][1] + wy * wy * fy_hat[ptr][1] + wy * wz * fz_hat[ptr][1]);
    			uy_hat[ptr][1] = fy_hat[ptr][1] * ((1.0 / s1) - (1.0 / s1_square) * s2 * s3 * scale); 

    			// real part
    			scale = -1.0 * (wz * wx * fx_hat[ptr][0] + wz * wy * fy_hat[ptr][0] + wz * wz * fz_hat[ptr][0]);
    			uz_hat[ptr][0] = fz_hat[ptr][0] * ((1.0 / s1) - (1.0 / s1_square) * s2 * s3 * scale); 
    			// imaginary part
    			scale = -1.0 * (wz * wx * fx_hat[ptr][1] + wz * wy * fy_hat[ptr][1] + wz * wz * fz_hat[ptr][1]);
    			uz_hat[ptr][1] = fz_hat[ptr][1] * ((1.0 / s1) - (1.0 / s1_square) * s2 * s3 * scale); 
    		}
    	}
    }

    accfft_execute_c2r (n_misc->plan_, ux_hat, ux_ptr);
    accfft_execute_c2r (n_misc->plan_, uy_hat, uy_ptr);
    accfft_execute_c2r (n_misc->plan_, uz_hat, uz_ptr);
    
    ierr = force->restoreComponentArrays (fx_ptr, fy_ptr, fz_ptr);
    ierr = displacement->restoreComponentArrays (ux_ptr, uy_ptr, uz_ptr);

    ierr = displacement->getIndividualComponents (x);  // get the individual components of u and set it to x (input)

    self_exec_time += MPI_Wtime();
    accumulateTimers (ctx->n_misc_->timers_, t, self_exec_time);
    e.addTimings (t);
    e.stop ();
	PetscFunctionReturn (0);
}


ElasticitySolver::~ElasticitySolver () {
	PetscErrorCode ierr = 0;
	ierr = MatDestroy (&A_);
    ierr = KSPDestroy (&ksp_);
    ierr = VecDestroy (&rhs_);
}
