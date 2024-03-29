#include "DiffusionSolver.h"

#include "petsc/private/kspimpl.h"

DiffusionSolver::DiffusionSolver(std::shared_ptr<Parameters> params, std::shared_ptr<SpectralOperators> spec_ops, std::shared_ptr<DiffCoef> k) : ctx_() {
  PetscErrorCode ierr = 0;
  ksp_itr_ = 0;
  ctx_ = std::make_shared<Ctx>();
  ctx_->k_ = k;
  ctx_->params_ = params;
  ctx_->dt_ = params->tu_->dt_;
  ctx_->plan_ = params->grid_->plan_;
  ctx_->spec_ops_ = spec_ops;
  ctx_->temp_ = k->temp_[0];
  ctx_->precfactor_ = k->temp_accfft_;
  ctx_->work_cuda_ = k->work_cuda_;
  ierr = precFactor();

  ierr = MatCreateShell(PETSC_COMM_WORLD, params->grid_->nl_, params->grid_->nl_, params->grid_->ng_, params->grid_->ng_, ctx_.get(), &A_);
  ierr = MatShellSetOperation(A_, MATOP_MULT, (void (*)(void))operatorA);
#if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 10)
  ierr = MatShellSetOperation(A_, MATOP_CREATE_VECS, (void (*)(void))operatorCreateVecs);
#endif

  ierr = KSPCreate(PETSC_COMM_WORLD, &ksp_);
  ierr = KSPSetOperators(ksp_, A_, A_);
  ierr = KSPSetTolerances(ksp_, 1e-6, PETSC_DEFAULT, PETSC_DEFAULT, 5000);
  ierr = KSPSetType(ksp_, KSPCG);
  ierr = KSPSetInitialGuessNonzero(ksp_, PETSC_TRUE);
  ierr = KSPSetFromOptions(ksp_);
  // ierr = KSPMonitorSet(ksp_, diffSolverKSPMonitor, ctx_.get(), 0);
  ierr = KSPSetUp(ksp_);

  ierr = KSPGetPC(ksp_, &pc_);
  ierr = PCSetType(pc_, PCSHELL);
  ierr = PCShellSetApply(pc_, applyPC);
  ierr = PCShellSetContext(pc_, ctx_.get());
  ierr = KSPSetFromOptions(ksp_);
  ierr = KSPSetUp(ksp_);

  ierr = VecCreate(PETSC_COMM_WORLD, &rhs_);
  ierr = VecSetSizes(rhs_, params->grid_->nl_, params->grid_->ng_);
  ierr = setupVec(rhs_);
  ierr = VecSet(rhs_, 0);
}

PetscErrorCode diffSolverKSPMonitor(KSP ksp, PetscInt its, PetscReal rnorm, void *ptr) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;

  Vec x;
  int maxit;
  ScalarType divtol, abstol, reltol;
  ierr = KSPBuildSolution(ksp, NULL, &x);
  ierr = KSPGetTolerances(ksp, &reltol, &abstol, &divtol, &maxit); CHKERRQ(ierr);
  CHKERRQ(ierr); CHKERRQ(ierr);
  Ctx *ctx = reinterpret_cast<Ctx *>(ptr);  // get user context

  std::stringstream s;
  if (its == 0) {
    s << std::setw(3) << " KSP:"
      << " computing solution of diffusion system (tol=" << std::scientific << std::setprecision(5) << reltol << ")";
    ierr = tuMSGstd(s.str()); CHKERRQ(ierr);
    s.str("");
    s.clear();
  }
  s << std::setw(3) << " KSP:" << std::setw(15) << " " << std::setfill('0') << std::setw(3) << its << "   ||r||_2 = " << std::scientific << std::setprecision(5) << rnorm;
  ierr = tuMSGstd(s.str()); CHKERRQ(ierr);
  s.str("");
  s.clear();

  // int ksp_itr;
  // ierr = KSPGetIterationNumber (ksp, &ksp_itr);                                 CHKERRQ (ierr);
  // ScalarType e_max, e_min;
  // if (ksp_itr % 10 == 0 || ksp_itr == maxit) {
  //   ierr = KSPComputeExtremeSingularValues (ksp, &e_max, &e_min);       CHKERRQ (ierr);
  //   s << "Condition number of matrix is: " << e_max / e_min << " | largest singular values is: " << e_max << ", smallest singular values is: " << e_min << std::endl; CHKERRQ(ierr);
  //   s.str (""); s.clear ();
  // }
  PetscFunctionReturn(ierr);
}

PetscErrorCode operatorCreateVecs(Mat A, Vec *left, Vec *right) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;

  Ctx *ctx;
  ierr = MatShellGetContext(A, &ctx); CHKERRQ(ierr);
  if (right) 
    ierr = VecDuplicate(ctx->k_->kxx_, right); CHKERRQ(ierr);
  if (left)
    ierr = VecDuplicate(ctx->k_->kxx_, left); CHKERRQ(ierr);

  PetscFunctionReturn(ierr);
}

PetscErrorCode operatorA(Mat A, Vec x, Vec y) {  // y = Ax
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  Event e("tumor-diffusion-ksp-matvec");
  std::array<double, 7> t = {0};
  double self_exec_time = -MPI_Wtime();
  Ctx *ctx;
  ierr = MatShellGetContext(A, &ctx); CHKERRQ(ierr);
  ierr = VecCopy(x, y); CHKERRQ(ierr);

  ScalarType alph = -1.0 / 2.0 * ctx->dt_;
  ierr = ctx->k_->applyD(ctx->temp_, y);
  ierr = VecAXPY(y, alph, ctx->temp_); CHKERRQ(ierr);

  self_exec_time += MPI_Wtime();
  accumulateTimers(ctx->params_->tu_->timers_, t, self_exec_time);
  e.addTimings(t);
  e.stop();
  PetscFunctionReturn(ierr);
  ;
}

PetscErrorCode DiffusionSolver::precFactor() {
  PetscFunctionBegin;
  Event e("tumor-diffusion-prec-factor");
  std::array<double, 7> t = {0};
  double self_exec_time = -MPI_Wtime();

  PetscErrorCode ierr = 0;

  std::shared_ptr<Parameters> params = ctx_->params_;
  int64_t X, Y, Z, wx, wy, wz, index;
  ScalarType kxx_avg, kxy_avg, kxz_avg, kyy_avg, kyz_avg, kzz_avg;
  kxx_avg = ctx_->k_->kxx_avg_;
  kxy_avg = ctx_->k_->kxy_avg_;
  kxz_avg = ctx_->k_->kxz_avg_;
  kyy_avg = ctx_->k_->kyy_avg_;
  kyz_avg = ctx_->k_->kyz_avg_;
  kzz_avg = ctx_->k_->kzz_avg_;

  ScalarType factor = 1.0 / (params->grid_->n_[0] * params->grid_->n_[1] * params->grid_->n_[2]);

#ifdef CUDA
  cudaMemcpy(&ctx_->work_cuda_[0], &ctx_->dt_, sizeof(ScalarType), cudaMemcpyHostToDevice);
  precFactorDiffusionCuda(ctx_->precfactor_, ctx_->work_cuda_, params->grid_->osize_);
#else

  for (int x = 0; x < params->grid_->osize_[0]; x++) {
    for (int y = 0; y < params->grid_->osize_[1]; y++) {
      for (int z = 0; z < params->grid_->osize_[2]; z++) {
        X = params->grid_->ostart_[0] + x;
        Y = params->grid_->ostart_[1] + y;
        Z = params->grid_->ostart_[2] + z;

        wx = X;
        wy = Y;
        wz = Z;

        if (X > params->grid_->n_[0] / 2.0) wx -= params->grid_->n_[0];
        if (X == params->grid_->n_[0] / 2.0) wx = 0;

        if (Y > params->grid_->n_[1] / 2.0) wy -= params->grid_->n_[1];
        if (Y == params->grid_->n_[1] / 2.0) wy = 0;

        if (Z > params->grid_->n_[2] / 2.0) wz -= params->grid_->n_[2];
        if (Z == params->grid_->n_[2] / 2.0) wz = 0;

        index = x * params->grid_->osize_[1] * params->grid_->osize_[2] + y * params->grid_->osize_[2] + z;
        ctx_->precfactor_[index] = (1 + 0.25 * ctx_->dt_ * (kxx_avg * wx * wx + 2.0 * kxy_avg * wx * wy + 2.0 * kxz_avg * wx * wz + 2.0 * kyz_avg * wy * wz + kyy_avg * wy * wy + kzz_avg * wz * wz));
        if (ctx_->precfactor_[index] == 0)
          ctx_->precfactor_[index] = factor;
        else
          ctx_->precfactor_[index] = factor / ctx_->precfactor_[index];
      }
    }
  }
#endif

  self_exec_time += MPI_Wtime();
  accumulateTimers(params->tu_->timers_, t, self_exec_time);
  e.addTimings(t);
  e.stop();
  PetscFunctionReturn(ierr);
}

PetscErrorCode applyPC(PC pc, Vec x, Vec y) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  Event e("tumor-diffusion-precond");
  std::array<double, 7> t = {0};
  double self_exec_time = -MPI_Wtime();

  Ctx *ctx;
  ierr = PCShellGetContext(pc, (void **)&ctx); CHKERRQ(ierr);
  std::shared_ptr<Parameters> params = ctx->params_;
  ierr = VecCopy(x, y); CHKERRQ(ierr);

  ScalarType *y_ptr;
  ierr = vecGetArray(y, &y_ptr); CHKERRQ(ierr);
#ifdef CUDA
  ctx->spec_ops_->executeFFTR2C(y_ptr, ctx->spec_ops_->x_hat_);
  hadamardComplexProductCuda((CudaComplexType *)ctx->spec_ops_->x_hat_, ctx->precfactor_, params->grid_->osize_);
  ctx->spec_ops_->executeFFTC2R(ctx->spec_ops_->x_hat_, y_ptr);
#else
  ctx->spec_ops_->executeFFTR2C(y_ptr, ctx->spec_ops_->x_hat_);
  std::complex<ScalarType> *c_a = (std::complex<ScalarType> *)ctx->spec_ops_->x_hat_;
  for (int i = 0; i < params->grid_->osize_[0] * params->grid_->osize_[1] * params->grid_->osize_[2]; i++) {
    c_a[i] *= ctx->precfactor_[i];
  }
  ctx->spec_ops_->executeFFTC2R(ctx->spec_ops_->x_hat_, y_ptr);
#endif
  ierr = vecRestoreArray(y, &y_ptr); CHKERRQ(ierr);

  self_exec_time += MPI_Wtime();
  accumulateTimers(params->tu_->timers_, t, self_exec_time);
  e.addTimings(t);
  e.stop();
  PetscFunctionReturn(ierr);
}

PetscErrorCode DiffusionSolver::solve(Vec c, ScalarType dt) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  Event e("tumor-diffusion-solve");
  std::array<double, 7> t = {0};
  double self_exec_time = -MPI_Wtime();

  Ctx *ctx;
  ierr = MatShellGetContext(A_, &ctx); CHKERRQ(ierr);
  ctx->dt_ = dt;
  if (ctx->k_->k_scale_ == 0) {
    ksp_itr_ = 0;
    self_exec_time += MPI_Wtime();
    accumulateTimers(ctx->params_->tu_->timers_, t, self_exec_time);
    e.addTimings(t);
    e.stop();
    PetscFunctionReturn(ierr);
  }
  ScalarType alph = 1.0 / 2.0 * ctx->dt_;
  ierr = VecCopy(c, rhs_); CHKERRQ(ierr);
  ierr = ctx->k_->applyD(ctx->temp_, rhs_);
  ierr = VecAXPY(rhs_, alph, ctx->temp_); CHKERRQ(ierr);

  // KSP solve
  ierr = KSPSolve(ksp_, rhs_, c); CHKERRQ(ierr);

  ierr = KSPGetIterationNumber(ksp_, &ksp_itr_); CHKERRQ(ierr);

  self_exec_time += MPI_Wtime();
  accumulateTimers(ctx->params_->tu_->timers_, t, self_exec_time);
  e.addTimings(t);
  e.stop();
  PetscFunctionReturn(ierr);
}

DiffusionSolver::~DiffusionSolver() {
  PetscErrorCode ierr = 0;
  ierr = MatDestroy(&A_);
  ierr = KSPDestroy(&ksp_);
  ierr = VecDestroy(&rhs_);
}
