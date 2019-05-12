#include "DiffSolver.h"
#include "petsc/private/kspimpl.h"

DiffSolver::DiffSolver (std::shared_ptr<NMisc> n_misc, std::shared_ptr<DiffCoef> k)
:
ctx_() {
    PetscErrorCode ierr = 0;

    ksp_itr_ = 0;
    ctx_ = std::make_shared<Ctx> ();
    ctx_->k_ = k;
    ctx_->n_misc_ = n_misc;
    ctx_->dt_ = n_misc->dt_;
    ctx_->plan_ = n_misc->plan_;
    ctx_->temp_ = k->temp_[0];
    ctx_->precfactor_ = k->temp_accfft_;
    ierr = precFactor ();

    ierr = MatCreateShell (PETSC_COMM_WORLD, n_misc->n_local_, n_misc->n_local_, n_misc->n_global_, n_misc->n_global_, ctx_.get(), &A_);
    ierr = MatShellSetOperation (A_, MATOP_MULT, (void(*)(void)) operatorA);

    ierr = KSPCreate (PETSC_COMM_WORLD, &ksp_);
    ierr = KSPSetOperators (ksp_, A_, A_);
    ierr = KSPSetTolerances (ksp_, 1e-6, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT);
    ierr = KSPSetType (ksp_, KSPCG);
    ierr = KSPSetFromOptions (ksp_);
    ierr = KSPSetUp (ksp_);

    ierr = KSPGetPC (ksp_, &pc_);
    ierr = PCSetType (pc_, PCSHELL);
    ierr = PCShellSetApply (pc_, applyPC);
    ierr = PCShellSetContext (pc_, ctx_.get());
    ierr = KSPSetFromOptions (ksp_);
    ierr = KSPSetUp (ksp_);

    ierr = VecCreate (PETSC_COMM_WORLD, &rhs_);
    ierr = VecSetSizes (rhs_, n_misc->n_local_, n_misc->n_global_);
    ierr = setupVec (rhs_);
    ierr = VecSet (rhs_, 0);
}

PetscErrorCode operatorA (Mat A, Vec x, Vec y) {    //y = Ax
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    Event e ("tumor-diffusion-ksp-matvec");
    std::array<double, 7> t = {0};
    double self_exec_time = -MPI_Wtime ();
    Ctx *ctx;
    ierr = MatShellGetContext (A, &ctx);                        CHKERRQ (ierr);
    ierr = VecCopy (x, y);                                      CHKERRQ (ierr);

    double alph = -1.0 / 2.0 * ctx->dt_;
    ierr = ctx->k_->applyD (ctx->temp_, y, ctx->plan_);
    ierr = VecAXPY (y, alph, ctx->temp_);                       CHKERRQ (ierr);

    self_exec_time += MPI_Wtime();
    accumulateTimers (ctx->n_misc_->timers_, t, self_exec_time);
    e.addTimings (t);
    e.stop ();
    PetscFunctionReturn(0);;
}

PetscErrorCode DiffSolver::precFactor () {
    PetscFunctionBegin;
    Event e ("tumor-diffusion-prec-factor");
    std::array<double, 7> t = {0};
    double self_exec_time = -MPI_Wtime ();

    std::shared_ptr<NMisc> n_misc = ctx_->n_misc_;
    int64_t X, Y, Z, wx, wy, wz, index;
    double kxx_avg, kxy_avg, kxz_avg, kyy_avg, kyz_avg, kzz_avg;
    kxx_avg = ctx_->k_->kxx_avg_;
    kxy_avg = ctx_->k_->kxy_avg_;
    kxz_avg = ctx_->k_->kxz_avg_;
    kyy_avg = ctx_->k_->kyy_avg_;
    kyz_avg = ctx_->k_->kyz_avg_;
    kzz_avg = ctx_->k_->kzz_avg_;

    double factor = 1.0 / (n_misc->n_[0] * n_misc->n_[1] * n_misc->n_[2]);

    #ifdef CUDA
        double *work;   // work vector for cuda
        cudaMalloc ((void**)&work, 8 * sizeof(double));
        cudaMemcpy (&work[0], &ctx_->dt_, sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy (&work[1], &kxx_avg, sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy (&work[2], &kxy_avg, sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy (&work[3], &kxz_avg, sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy (&work[4], &kyz_avg, sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy (&work[5], &kyy_avg, sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy (&work[6], &kzz_avg, sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy (&work[7], &factor, sizeof(double), cudaMemcpyHostToDevice);
        precFactorDiffusionCuda (ctx_->precfactor_, work, n_misc->osize_);
    #else

    for (int x = 0; x < n_misc->osize_[0]; x++) {
        for (int y = 0; y < n_misc->osize_[1]; y++) {
            for (int z = 0; z < n_misc->osize_[2]; z++){
                X = n_misc->ostart_[0] + x;
                Y = n_misc->ostart_[1] + y;
                Z = n_misc->ostart_[2] + z;

                wx = X;
                wy = Y;
                wz = Z;

                if (X > n_misc->n_[0] / 2.0)
                    wx -= n_misc->n_[0];
                if (X == n_misc->n_[0] / 2.0)
                    wx = 0;

                if (Y > n_misc->n_[1] / 2.0)
                    wy -= n_misc->n_[1];
                if (Y == n_misc->n_[1] / 2.0)
                    wy = 0;

                if (Z > n_misc->n_[2] / 2.0)
                    wz -= n_misc->n_[2];
                if (Z == n_misc->n_[2] / 2.0)
                    wz = 0;

                index = x * n_misc->osize_[1] * n_misc->osize_[2] + y * n_misc->osize_[2] + z;
                ctx_->precfactor_[index] = (1 + 0.25 * ctx_->dt_ * (kxx_avg * wx * wx + 2.0 * kxy_avg * wx * wy
                                        + 2.0 * kxz_avg * wx * wz + 2.0 * kyz_avg * wy * wz + kyy_avg * wy * wy
                                                        + kzz_avg * wz *wz));
                if (ctx_->precfactor_[index] == 0)
                    ctx_->precfactor_[index] = factor;
                else
                    ctx_->precfactor_[index] = factor / ctx_->precfactor_[index];
            }
        }
    }
    #endif

    self_exec_time += MPI_Wtime();
    accumulateTimers (n_misc->timers_, t, self_exec_time);
    e.addTimings (t);
    e.stop ();
    PetscFunctionReturn (0);
}

PetscErrorCode applyPC (PC pc, Vec x, Vec y) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    Event e ("tumor-diffusion-precond");
    std::array<double, 7> t = {0};
    double self_exec_time = -MPI_Wtime ();

    Ctx *ctx;
    ierr = PCShellGetContext (pc, (void **) &ctx);              CHKERRQ (ierr);
    std::shared_ptr<NMisc> n_misc = ctx->n_misc_;
    ierr = VecCopy (x, y);                                      CHKERRQ (ierr);

    PetscScalar *y_ptr;
    #ifdef CUDA
        Complex *c_hat;
        ierr = VecCUDAGetArrayReadWrite (y, &y_ptr);                             CHKERRQ (ierr);
        cudaMalloc ((void**)&c_hat, n_misc->accfft_alloc_max_);
        accfft_execute_r2c (n_misc->plan_, y_ptr, c_hat, t.data());

        std::complex<double> *c_a = (std::complex<double> *) c_hat;
        hadamardComplexProductCuda ((cuDoubleComplex*) c_a, (cuDoubleComplex*) ctx->precfactor_);

        accfft_execute_c2r (n_misc->plan_, c_hat, y_ptr, t.data());

        ierr = VecCUDARestoreArrayReadWrite (y, &y_ptr);                         CHKERRQ (ierr);
    #else
        ierr = VecGetArray (y, &y_ptr);                             CHKERRQ (ierr);
        Complex *c_hat = (Complex *) accfft_alloc (n_misc->accfft_alloc_max_);
        accfft_execute_r2c (n_misc->plan_, y_ptr, c_hat, t.data());

        std::complex<double> *c_a = (std::complex<double> *) c_hat;
        for (int i = 0; i < n_misc->osize_[0] * n_misc->osize_[1] * n_misc->osize_[2]; i++) {
            c_a[i] *= ctx->precfactor_[i];
        }

        accfft_execute_c2r (n_misc->plan_, c_hat, y_ptr, t.data());

        ierr = VecRestoreArray (y, &y_ptr);                         CHKERRQ (ierr);
    #endif
    
    fft_free (c_hat);

    self_exec_time += MPI_Wtime();
    accumulateTimers (n_misc->timers_, t, self_exec_time);
    e.addTimings (t);
    e.stop ();
    PetscFunctionReturn (0);
}

PetscErrorCode DiffSolver::solve (Vec c, double dt) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    Event e ("tumor-diffusion-solve");
    std::array<double, 7> t = {0};
    double self_exec_time = -MPI_Wtime ();

    Ctx *ctx;
    ierr = MatShellGetContext (A_, &ctx);                       CHKERRQ (ierr);
    ctx->dt_ = dt;
    if (ctx->k_->k_scale_ == 0) {
        ksp_itr_ = 0;
        return 0;
    }
    double alph = 1.0 / 2.0 * ctx->dt_;
    ierr = VecCopy (c, rhs_);                                   CHKERRQ (ierr);
    ierr = ctx->k_->applyD (ctx->temp_, rhs_, ctx->plan_);
    ierr = VecAXPY (rhs_, alph, ctx->temp_);                    CHKERRQ (ierr);

    // hack -- explicity set the ksp_->work vecs as cuda vecs since
    // petsc does not do this for some reason. <follow-up with petsc>
    #ifdef CUDA
    for (int i = 0; i < 3; i++) {
        ierr = VecDestroy (&ksp_->work[i]); CHKERRQ(ierr);
        ierr = VecDuplicate (c, &ksp_->work[i]); CHKERRQ(ierr);
        ierr = VecSet (ksp_->work[i], 0.);  CHKERRQ(ierr);
    }
    #endif

    //KSP solve
    ierr = KSPSolve (ksp_, rhs_, c);                            CHKERRQ (ierr);

    ierr = KSPGetIterationNumber (ksp_, &ksp_itr_);             CHKERRQ (ierr);

    self_exec_time += MPI_Wtime();
    accumulateTimers (ctx->n_misc_->timers_, t, self_exec_time);
    e.addTimings (t);
    e.stop ();
    PetscFunctionReturn(0);
}

DiffSolver::~DiffSolver () {
    PetscErrorCode ierr = 0;
    ierr = MatDestroy (&A_);
    ierr = KSPDestroy (&ksp_);
    ierr = VecDestroy (&rhs_);
}
